"""
Offline evaluation of a trained RGBVelocityPolicy on the validation split.

Reports per-component MSE in physical units, broken down by horizon step
(so we can see how prediction quality degrades into the future), and writes
detailed per-window prediction logs for downstream plots.

Produced artifacts under ``<output_dir>``:

    summary.json         aggregate metrics in physical units
    per_horizon.csv      mse per [vx, vy, vz, psi_dot] x [0..H-1]
    predictions.npz      {u_hat_raw, u_star_raw, k, cache} arrays for the full val split
    cli.txt              the command line + config snapshot

The script never touches the FiGS simulator -- it is safe to run on any
machine with the processed cache + a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from nav_policy.data.normalization import CommandStats
from nav_policy.data.rgb_horizon_dataset import RGBHorizonDataset
from nav_policy.model.factory import build_model, model_uses_depth


CMD_NAMES = ("vx", "vy", "vz", "psi_dot")


def _build_model_from_ckpt(ckpt: dict) -> torch.nn.Module:
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model


def _collate(batch):
    rgbs, goals, u_stars, metas = zip(*batch)
    rgb = torch.stack(rgbs, dim=0)
    goal = torch.stack(goals, dim=0)
    u_star = torch.stack(u_stars, dim=0)
    u_raw = torch.stack([m["u_raw"] for m in metas], dim=0)
    depth = None
    if metas[0].get("depth") is not None:
        depth = torch.stack([m["depth"] for m in metas], dim=0)
    ks = np.array([int(m["k"]) for m in metas], dtype=np.int32)
    caches = [m["cache"] for m in metas]
    return rgb, goal, u_star, u_raw, depth, ks, caches


def evaluate(config_path: Path, checkpoint_path: Path, output_dir: Path) -> Dict[str, float]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    base = config_path.resolve().parent.parent
    processed_root = (base / cfg["data"]["processed_root"]).resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    stats = CommandStats.from_dict(ckpt["stats"])
    model = _build_model_from_ckpt(ckpt).eval()
    uses_depth = model_uses_depth(ckpt["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    H = model.H
    cmd_dim = model.cmd_dim
    assert cmd_dim == 4, f"expected cmd_dim=4, got {cmd_dim}"

    # Read the checkpoint's training-time goal-input configuration so we
    # evaluate the policy in the same regime it was trained in (otherwise a
    # 3-D goal-conditioned checkpoint evaluated with 2-D inputs, or with
    # zeroed goals, would look broken).
    ckpt_train_cfg = ckpt["config"].get("train", {})
    ckpt_model_cfg = ckpt["config"].get("model", {})
    zero_goal_eval = bool(ckpt_train_cfg.get("zero_goal_heading", False))
    goal_input_dim = int(ckpt_model_cfg.get("goal_input_dim", 2))
    goal_distance_scale = float(ckpt_model_cfg.get("goal_distance_scale", 5.0))
    data_cfg = cfg.get("data", {})
    ds = RGBHorizonDataset(
        processed_root, split="val",
        cache_blobs_in_memory=bool(data_cfg.get("cache_blobs_in_memory", False)),
        cache_lru_size=int(data_cfg.get("cache_lru_size", 64)),
        zero_goal_heading=zero_goal_eval,
        goal_input_dim=goal_input_dim,
        goal_distance_scale=goal_distance_scale,
        use_depth=uses_depth,
    )
    if ds.T != model.T or ds.H != model.H:
        raise ValueError(
            f"manifest T,H ({ds.T},{ds.H}) != model T,H ({model.T},{model.H})"
        )
    loader = DataLoader(
        ds, batch_size=int(cfg["train"].get("batch_size", 16)),
        shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available(),
        collate_fn=_collate, drop_last=False,
    )

    n = 0
    sum_sq_per_h: np.ndarray = np.zeros((H, cmd_dim), dtype=np.float64)
    sum_abs_per_h: np.ndarray = np.zeros((H, cmd_dim), dtype=np.float64)
    latencies: List[float] = []
    u_hat_all: List[np.ndarray] = []
    u_raw_all: List[np.ndarray] = []
    ks_all: List[np.ndarray] = []
    caches_all: List[List[str]] = []

    with torch.inference_mode():
        for rgb, goal, _u_star, u_raw, depth, ks, caches in loader:
            rgb = rgb.to(device, non_blocking=True)
            goal = goal.to(device, non_blocking=True)
            u_raw_np = u_raw.numpy()
            if depth is not None:
                depth = depth.to(device, non_blocking=True)

            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.time()
            if uses_depth:
                u_hat_z = model(rgb, goal, depth)
            else:
                u_hat_z = model(rgb, goal)
            torch.cuda.synchronize() if device.type == "cuda" else None
            latencies.append((time.time() - t0) / rgb.shape[0])  # per-sample seconds

            u_hat = stats.destandardize(u_hat_z).float().cpu().numpy()  # [B, H, 4]
            err = u_hat - u_raw_np                                       # [B, H, 4]

            sum_sq_per_h += (err ** 2).sum(axis=0)
            sum_abs_per_h += np.abs(err).sum(axis=0)
            n += err.shape[0]

            u_hat_all.append(u_hat.astype(np.float32))
            u_raw_all.append(u_raw_np.astype(np.float32))
            ks_all.append(ks)
            caches_all.append(caches)

    if n == 0:
        raise RuntimeError("validation set was empty")

    mse_per_h = sum_sq_per_h / n                # [H, 4]
    rmse_per_h = np.sqrt(mse_per_h)
    mae_per_h = sum_abs_per_h / n               # [H, 4]
    mse_overall = mse_per_h.mean(axis=0)        # [4]
    rmse_overall = np.sqrt(mse_overall)
    mae_overall = mae_per_h.mean(axis=0)        # [4]
    mse_lin_vel = mse_per_h[:, :3].mean()
    rmse_lin_vel = float(np.sqrt(mse_lin_vel))
    mse_psi_dot = mse_per_h[:, 3].mean()
    rmse_psi_dot = float(np.sqrt(mse_psi_dot))

    summary = {
        "n_samples": int(n),
        "device": str(device),
        "T": int(model.T),
        "H": int(model.H),
        "cmd_dim": int(cmd_dim),
        "image_size": int(ds.image_size),
        "stats_mean": stats.mean.tolist(),
        "stats_std": stats.std.tolist(),
        "rmse_overall": {name: float(v) for name, v in zip(CMD_NAMES, rmse_overall)},
        "mae_overall": {name: float(v) for name, v in zip(CMD_NAMES, mae_overall)},
        "rmse_lin_vel": rmse_lin_vel,
        "rmse_psi_dot": rmse_psi_dot,
        "latency_per_sample_ms": {
            "mean": float(np.mean(latencies)) * 1000.0,
            "median": float(np.median(latencies)) * 1000.0,
            "p95": float(np.percentile(latencies, 95)) * 1000.0,
        },
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        # Ablation-identification metadata for downstream collection.
        "run_tag": str(cfg.get("run_tag", output_dir.name)),
        "zero_goal_heading": bool(zero_goal_eval),
        "goal_input_dim": int(goal_input_dim),
        "goal_distance_scale": float(goal_distance_scale),
        "model_arch": str(ckpt_model_cfg.get("arch", "rgb_resnet18")),
        "train_epochs": int(ckpt_train_cfg.get("epochs", 0)),
    }
    print(json.dumps(summary, indent=2))

    # Per-horizon CSV
    import csv
    with open(output_dir / "per_horizon.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_step"] + [f"rmse_{n}" for n in CMD_NAMES] + [f"mae_{n}" for n in CMD_NAMES])
        for h in range(H):
            row = [h]
            row += [float(rmse_per_h[h, c]) for c in range(cmd_dim)]
            row += [float(mae_per_h[h, c]) for c in range(cmd_dim)]
            w.writerow(row)

    # Dense predictions for plots
    np.savez_compressed(
        output_dir / "predictions.npz",
        u_hat=np.concatenate(u_hat_all, axis=0),
        u_raw=np.concatenate(u_raw_all, axis=0),
        k=np.concatenate(ks_all, axis=0),
        caches=np.array([c for chunk in caches_all for c in chunk], dtype=object),
    )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "cli.txt", "w") as f:
        f.write("argv: " + " ".join(sys.argv) + "\n")
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"config:     {config_path}\n")

    print(f"[done] artifacts -> {output_dir}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Offline (no-FiGS) evaluation of a trained policy.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    evaluate(args.config, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
