"""
Quick inspection of a flightroom-format trajectory file to print its schema.

Usage (inside nav_policy Docker container):
    python scripts/inspect_flightroom_data.py \
        data/raw/flightroom_ssv_exp_2026-05-22_071733_trajs-110/trajectories_val00000.pt
    python scripts/inspect_flightroom_data.py \
        data/raw/flightroom_ssv_exp_2026-05-22_064652_training_mode_shuffled_trajs/trajectories00000.pt
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch


def _shape(v) -> str:
    if hasattr(v, "shape"):
        return str(tuple(v.shape))
    if isinstance(v, list):
        return f"list[{len(v)}]"
    if isinstance(v, dict):
        return f"dict[{list(v.keys())}]"
    return repr(v)


def inspect(path: Path) -> None:
    print(f"\n=== {path.name} ===")
    raw = torch.load(path, weights_only=False, map_location="cpu")

    # ── Detect format ─────────────────────────────────────────────────────────
    if isinstance(raw, dict) and "data" in raw:
        print("Format: SINGER nested dict  {'data': [sub_traj_0, ...]}")
        entries = raw["data"]
        print(f"  Number of sub-trajectories: {len(entries)}")
        if entries:
            e = entries[0]
            print(f"  Sub-traj keys: {list(e.keys())}")
            for k, v in e.items():
                print(f"    {k}: {_shape(v)}")
    elif isinstance(raw, list):
        print("Format: NEW flat list  [traj_0, traj_1, ...]")
        print(f"  Number of trajectories: {len(raw)}")
        if raw:
            e = raw[0]
            print(f"  Traj keys: {list(e.keys())}")
            for k, v in e.items():
                if isinstance(v, dict):
                    print(f"    {k}: dict[{list(v.keys())}]")
                else:
                    s = _shape(v)
                    print(f"    {k}: {s}")
    else:
        print(f"Format: unknown  type={type(raw).__name__}")
        print(f"  top-level: {_shape(raw)}")


def inspect_imgdata(path: Path) -> None:
    print(f"\n=== IMGDATA: {path.name} ===")
    raw = torch.load(path, weights_only=False, map_location="cpu")
    if isinstance(raw, dict) and "data" in raw:
        print(f"  Format: dict with 'data' key")
        d = raw["data"]
        print(f"  Entries: {len(d)}")
        if d:
            print(f"  First entry keys: {list(d[0].keys())}")
            for k, v in d[0].items():
                print(f"    {k}: {_shape(v)}")
    elif isinstance(raw, list):
        print(f"  Format: flat list  len={len(raw)}")
        if raw:
            e = raw[0]
            if isinstance(e, dict):
                print(f"  First entry keys: {list(e.keys())}")
            else:
                print(f"  First entry type: {type(e)}")
    else:
        print(f"  Format: {type(raw).__name__}  value={repr(raw)[:200]}")


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect flightroom-format .pt files.")
    p.add_argument("path", type=Path, nargs="+", help="Trajectory .pt files to inspect.")
    p.add_argument("--imgdata", action="store_true",
                   help="Also inspect the corresponding imgdata file.")
    args = p.parse_args()

    for pt in args.path:
        inspect(pt)
        if args.imgdata:
            # Try to find the matching imgdata file
            name = pt.name
            imgdata_name = name.replace("trajectories_val", "imgdata_val").replace(
                "trajectories", "imgdata"
            )
            imgdata_path = pt.parent / imgdata_name
            if imgdata_path.exists():
                inspect_imgdata(imgdata_path)
            else:
                print(f"  [imgdata not found: {imgdata_path}]")


if __name__ == "__main__":
    main()
