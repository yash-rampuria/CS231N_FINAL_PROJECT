"""
Build a processed dataset from flightroom-format V-LEAD rollouts.

This builder handles the new trajectory format produced by the V-LEAD data
collection pipeline (vlead_flightroom).  Each per-index .pt file is a flat
Python list of trajectory dicts:

    trajs = torch.load("trajectories_val00000.pt")
    traj = trajs[0]
    traj["Tro"]         # (N+1,)   timestamps
    traj["Xro"]         # (10, N+1) state matrix (world frame, NED)
    traj["Uro"]         # (4, N)    body-rate expert commands
    traj["heading_vec"] # (2, N+1)  pre-computed goal heading unit vectors
    traj["dist"]        # (N+1,)    pre-computed distance to goal (meters)
    traj["goal_xy"]     # (2,)      2D XY position of semantic target

Two naming conventions are auto-detected:
  * Validation-mode  (full ~13 s rollouts):  trajectories_val{i}.pt
  * Training-mode    (2 s segments, domain randomised):  trajectories{i}.pt

Run assignment is explicit via the config:
  * train_runs  ->  all windows tagged split='train'
  * val_runs    ->  all windows tagged split='val'  (held entirely out)
  * test_runs   ->  NOT processed (used only for closed-loop eval)

Usage (inside the nav_policy Docker container):
    python scripts/build_dataset_flightroom.py --config configs/flightroom.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    import cv2  # type: ignore
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

# Reuse shared utilities from the main builder
from nav_policy.data.build_dataset import (
    CONTROL_HZ,
    write_cache,
    _resize_uint8,
    _quat_to_yaw,
    _read_video,
    _enumerate_windows,
    _compute_stats_from_entries,
)
from nav_policy.data.normalization import CommandStats


# ── filename patterns ──────────────────────────────────────────────────────────

# val-mode naming (e.g. trajectories_val00000.pt)
_VAL_TRAJ_RE   = re.compile(r"^trajectories_val(\d+)\.pt$")
_VAL_IMG_TMPL  = "imgdata_val{}.pt"
_VAL_VID_TMPL  = "video_val_rollout_images_rgb{}.mp4"

# training-mode naming (e.g. trajectories00000.pt)
_TRAIN_TRAJ_RE  = re.compile(r"^trajectories(\d+)\.pt$")
_TRAIN_IMG_TMPL = "imgdata{}.pt"
_TRAIN_VID_TMPL = "video_rgb{}.mp4"


@dataclass
class RunFilePaths:
    traj_pt: Path
    imgdata_pt: Optional[Path]
    rgb_video: Path
    raw_index: str      # zero-padded string id used in filenames


def _detect_naming(run_dir: Path) -> str:
    """Return 'val' or 'train' based on which naming pattern is found."""
    for name in os.listdir(run_dir):
        if _VAL_TRAJ_RE.match(name):
            return "val"
        if _TRAIN_TRAJ_RE.match(name):
            return "train"
    return "val"


def _iter_run_files(run_dir: Path) -> Iterator[RunFilePaths]:
    """Yield RunFilePaths for every trajectory file in the run directory."""
    naming = _detect_naming(run_dir)
    traj_re = _VAL_TRAJ_RE if naming == "val" else _TRAIN_TRAJ_RE
    img_tmpl = _VAL_IMG_TMPL if naming == "val" else _TRAIN_IMG_TMPL
    vid_tmpl = _VAL_VID_TMPL if naming == "val" else _TRAIN_VID_TMPL

    for name in sorted(os.listdir(run_dir)):
        m = traj_re.match(name)
        if not m:
            continue
        raw_id = m.group(1)

        traj_pt = run_dir / name
        imgdata_pt_path = run_dir / img_tmpl.format(raw_id)
        rgb_video_path = run_dir / vid_tmpl.format(raw_id)

        imgdata_pt = imgdata_pt_path if imgdata_pt_path.exists() else None
        if not rgb_video_path.exists():
            continue  # skip if no video

        yield RunFilePaths(
            traj_pt=traj_pt,
            imgdata_pt=imgdata_pt,
            rgb_video=rgb_video_path,
            raw_index=raw_id,
        )


# ── trajectory loading ─────────────────────────────────────────────────────────

def _load_trajs(path: Path) -> List[dict]:
    """
    Load a trajectory file, handling both the new flat-list and the legacy
    SINGER nested-dict format.  Returns a plain list of trajectory dicts.
    """
    raw = torch.load(path, weights_only=False, map_location="cpu")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    raise ValueError(
        f"Unrecognised trajectory file format in {path}: "
        f"expected list or dict-with-'data', got {type(raw).__name__}"
    )


def _load_frame_ranges(imgdata_pt: Optional[Path],
                       n_trajs: int) -> Optional[List[Tuple[int, int]]]:
    """
    Load (start_id, end_id) inclusive frame ranges from an imgdata file.

    Returns None if the file cannot be parsed, in which case the caller uses
    sequential frame assignment.
    """
    if imgdata_pt is None:
        return None
    try:
        raw = torch.load(imgdata_pt, weights_only=False, map_location="cpu")
        entries: List[dict]
        if isinstance(raw, dict) and "data" in raw:
            entries = raw["data"]
        elif isinstance(raw, list):
            entries = raw
        else:
            return None
        if len(entries) != n_trajs:
            return None
        ranges = []
        for e in entries:
            if isinstance(e, dict) and "start_id" in e and "end_id" in e:
                ranges.append((int(e["start_id"]), int(e["end_id"])))
            else:
                return None
        return ranges
    except Exception:
        return None


# ── command label extraction ───────────────────────────────────────────────────

def _extract_labels(xro: np.ndarray,
                    n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract velocity and yaw-rate labels from the state matrix.

    Args:
        xro: (10, N+1) state matrix; rows 3:6 are world-frame velocity,
             rows 6:10 are quaternion [qx, qy, qz, qw].
        n:   number of control steps to emit.

    Returns:
        vel:     (n, 3) float32 world-frame velocity labels [vx, vy, vz]
        psi_dot: (n,)   float32 yaw-rate labels (rad/s)
    """
    vel = xro[3:6, :n].T.astype(np.float32)         # (n, 3)
    quat_cols = xro[6:10, : n + 1]                  # (4, n+1)
    if quat_cols.shape[1] < n + 1:
        quat_cols = np.concatenate([quat_cols, quat_cols[:, -1:]], axis=1)
    quat = quat_cols.T                               # (n+1, 4)
    yaw = _quat_to_yaw(quat)                         # (n+1,)
    psi_dot = ((yaw[1:] - yaw[:-1]) * CONTROL_HZ).astype(np.float32)  # (n,)
    return vel, psi_dot


def _extract_goal(traj: dict, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Goal heading and distance toward the expert sub-trajectory endpoint Xro[0:2,-1].

    Matches closed-loop eval and DAgger.  Ignores semantic heading_vec/dist/goal_xy
    when present in flightroom rollouts.

    Returns:
        goal_heading: (n, 2) float32
        goal_dist:    (n,)   float32 in raw metres
    """
    from nav_policy.data.build_dataset import (
        _compute_goal_distance,
        _compute_goal_heading,
    )

    xro = np.asarray(traj["Xro"])
    goal_heading = _compute_goal_heading(xro, n)
    goal_dist = _compute_goal_distance(xro, n)
    return goal_heading, goal_dist


# ── core per-file processing ───────────────────────────────────────────────────

def _build_one_file(paths: RunFilePaths,
                    out_run_dir: Path,
                    image_size: int) -> List[Path]:
    """
    Process one trajectory file and its paired video into per-trajectory caches.

    Returns the list of cache paths actually written (may be empty).
    """
    cache_dir = out_run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    trajs = _load_trajs(paths.traj_pt)
    if not trajs:
        return []

    frame_ranges = _load_frame_ranges(paths.imgdata_pt, len(trajs))
    video_frames = _read_video(paths.rgb_video)   # (V, H, W, 3) uint8

    # If no frame-range map available, assign blocks sequentially.
    if frame_ranges is None:
        cursor = 0
        frame_ranges = []
        for traj in trajs:
            xro = np.asarray(traj["Xro"])
            nctl = int(xro.shape[1] - 1)
            frame_ranges.append((cursor, cursor + nctl - 1))
            cursor += nctl

    written: List[Path] = []
    for sub_idx, (traj, (start, end)) in enumerate(
        zip(trajs, frame_ranges)
    ):
        xro  = np.asarray(traj["Xro"])         # (10, N+1)
        nctl = xro.shape[1] - 1                # Uro.shape[1]

        n_vid = end - start + 1
        n_use = min(nctl, n_vid)
        if n_use <= 0:
            continue

        # ── visual frames ──
        sub_frames = video_frames[start : start + n_use]    # (n, H, W, 3)
        resized = np.stack(
            [_resize_uint8(f, image_size) for f in sub_frames], axis=0
        )
        rgb = torch.from_numpy(resized).permute(0, 3, 1, 2).contiguous()

        # ── command labels ──
        vel, psi_dot = _extract_labels(xro, n_use)

        # ── goal heading + distance ──
        goal_heading, goal_dist = _extract_goal(traj, n_use)

        # ── metadata ──
        frame_meta = traj.get("frame", {})
        meta = {
            "run":        out_run_dir.name,
            "file_index": paths.raw_index,
            "sub_idx":    sub_idx,
            "n_frames":   int(n_use),
            "control_hz": float(CONTROL_HZ),
            "source":     "bc_expert",
            "scene_name": str(frame_meta.get("scene_name", "")),
            "course_name": str(frame_meta.get("course_name", "")),
            "query":      str(frame_meta.get("query", "")),
        }

        cache_path = cache_dir / f"file{paths.raw_index}_sub{sub_idx}.pt"
        write_cache(
            cache_path,
            rgb_uint8=rgb,
            vel=torch.from_numpy(vel),
            psi_dot=torch.from_numpy(psi_dot),
            goal_heading=torch.from_numpy(goal_heading),
            goal_dist=torch.from_numpy(goal_dist),
            meta=meta,
        )
        written.append(cache_path)

    return written


# ── train / val manifest split ─────────────────────────────────────────────────

def _tag_explicit(entries: List[dict],
                  val_run_names: set) -> List[dict]:
    """
    Tag every window entry with split='train' or split='val' based on which
    processed sub-directory it came from.

    val_run_names: set of processed run directory names (e.g.
        {'flightroom_ssv_exp_2026-05-22_071733_trajs-110'})
    """
    for e in entries:
        # cache key is like "flightroom_ssv_exp_.../cache/file00000_sub0.pt"
        parts = Path(e["cache"]).parts
        run_name = parts[0] if parts else ""
        e["split"] = "val" if run_name in val_run_names else "train"
    return entries


# ── driver ─────────────────────────────────────────────────────────────────────

def build(config_path: Path) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base = config_path.resolve().parent.parent   # nav_policy/
    raw_root       = (base / cfg["data"]["raw_root"]).resolve()
    processed_root = (base / cfg["data"]["processed_root"]).resolve()

    train_runs: List[str] = cfg["data"].get("train_runs", [])
    val_runs:   List[str] = cfg["data"].get("val_runs",   [])
    # test_runs is purely for documentation / closed-loop eval; not processed here.

    all_runs = train_runs + val_runs
    if not all_runs:
        print("No runs specified under data.train_runs or data.val_runs.", file=sys.stderr)
        sys.exit(1)

    T:          int = int(cfg["window"]["T"])
    H:          int = int(cfg["window"]["H"])
    image_size: int = int(cfg["window"]["image_size"])

    processed_root.mkdir(parents=True, exist_ok=True)
    all_cache_paths: List[Path] = []

    for run in all_runs:
        run_dir = raw_root / run
        if not run_dir.is_dir():
            print(f"[skip] {run}: directory not found at {run_dir}", file=sys.stderr)
            continue

        out_run_dir = processed_root / run
        files = list(_iter_run_files(run_dir))
        mode = "val" if files and _detect_naming(run_dir) == "val" else "train"
        split_label = "val" if run in val_runs else "train"
        print(f"[{run}] {len(files)} files  naming={mode}  assigned_split={split_label}")

        for fp in tqdm(files, unit="file", leave=False):
            written = _build_one_file(fp, out_run_dir, image_size)
            all_cache_paths.extend(written)

    if not all_cache_paths:
        print("No caches were written.  Verify data paths.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[manifest] enumerating windows over {len(all_cache_paths)} caches...")
    entries = _enumerate_windows(all_cache_paths, T=T, H=H, processed_root=processed_root)
    print(f"[manifest] {len(entries)} raw windows")

    val_set = set(val_runs)
    entries = _tag_explicit(entries, val_set)
    n_train = sum(1 for e in entries if e["split"] == "train")
    n_val   = len(entries) - n_train
    print(f"[manifest] split: train={n_train}  val={n_val}")

    print("[stats] fitting CommandStats over train split...")
    stats = _compute_stats_from_entries(entries, processed_root, H=H)
    print(f"[stats] mean={stats.mean.tolist()}")
    print(f"[stats] std ={stats.std.tolist()}")

    manifest = {
        "T":            T,
        "H":            H,
        "image_size":   image_size,
        "control_hz":   CONTROL_HZ,
        "imagenet_mean": list(cfg["window"]["imagenet_mean"]),
        "imagenet_std":  list(cfg["window"]["imagenet_std"]),
        "samples":      entries,
    }
    (processed_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (processed_root / "stats.json").write_text(
        json.dumps(stats.to_dict(), indent=2), encoding="utf-8"
    )
    print(f"\n[done] {processed_root}/manifest.json")
    print(f"[done] {processed_root}/stats.json")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build nav_policy cache from flightroom V-LEAD rollouts."
    )
    p.add_argument("--config", type=Path, required=True,
                   help="Path to YAML config (e.g. configs/flightroom.yaml).")
    args = p.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
