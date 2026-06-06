"""
Inspect any .pt file produced by the nav_policy pipeline.

Usage:
    python scripts/inspect_pt.py <path/to/file.pt> [path/to/file2.pt ...]

Handles:
  - Processed cache files  (keys: rgb, vel, psi_dot, goal_heading, goal_dist, meta)
  - Raw trajectories_val*.pt  (key: data -> list of sub-trajectories)
  - Raw imgdata_val*.pt        (key: data -> list of image metadata dicts)
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import torch


def _fmt(name: str, obj, depth: int = 0) -> None:
    pad = "  " * depth
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict  keys={list(obj.keys())}")
        for k, v in obj.items():
            _fmt(k, v, depth + 1)
    elif isinstance(obj, list):
        print(f"{pad}{name}: list  len={len(obj)}")
        if obj:
            _fmt("[0]", obj[0], depth + 1)
    elif isinstance(obj, np.ndarray):
        extra = ""
        if obj.size > 0:
            extra = f"  min={obj.min():.4f}  max={obj.max():.4f}"
        print(f"{pad}{name}: ndarray  shape={obj.shape}  dtype={obj.dtype}{extra}")
    elif isinstance(obj, torch.Tensor):
        extra = ""
        if obj.numel() > 0 and obj.is_floating_point():
            extra = f"  min={obj.float().min():.4f}  max={obj.float().max():.4f}"
        print(f"{pad}{name}: Tensor  shape={tuple(obj.shape)}  dtype={obj.dtype}{extra}")
    elif isinstance(obj, (int, float, str, bool, type(None))):
        print(f"{pad}{name}: {type(obj).__name__} = {obj!r}"[:120])
    else:
        print(f"{pad}{name}: {type(obj).__name__}")


def inspect(path: Path) -> None:
    size_mb = path.stat().st_size / 1e6
    print(f"\n{'=' * 64}")
    print(f"FILE : {path}")
    print(f"SIZE : {size_mb:.2f} MB")
    print(f"{'=' * 64}")

    blob = torch.load(path, weights_only=False, map_location="cpu")

    # ── Processed cache (produced by build_dataset / write_cache) ────────────
    if isinstance(blob, dict) and "rgb" in blob:
        print("TYPE : processed cache")
        keys = list(blob.keys())
        print(f"KEYS : {keys}")
        for k, v in blob.items():
            if k == "meta":
                print(f"  meta : {v}")
            elif isinstance(v, torch.Tensor):
                extra = ""
                if v.is_floating_point() and v.numel() > 0:
                    extra = f"  min={v.float().min():.4f}  max={v.float().max():.4f}"
                print(f"  {k:14s}: Tensor  shape={tuple(v.shape)}  dtype={v.dtype}{extra}")
            else:
                print(f"  {k:14s}: {type(v).__name__}")
        if "goal_heading" not in blob:
            print("\n  *** WARNING: 'goal_heading' key is MISSING.")
            print("  *** Re-run build_dataset.py to regenerate caches with goal heading. ***")
        if "goal_dist" not in blob:
            print("\n  *** WARNING: 'goal_dist' key is MISSING.")
            print("  *** Re-run build_dataset.py to add the scalar distance-to-goal channel. ***")
            print("  *** (Required for model.goal_input_dim=3; safe to skip for goal_input_dim=2.) ***")
        return

    # ── Raw trajectories_val*.pt ─────────────────────────────────────────────
    if isinstance(blob, dict) and "data" in blob and isinstance(blob["data"], list):
        sample = blob["data"][0] if blob["data"] else {}
        if isinstance(sample, dict) and "Xro" in sample:
            print("TYPE : raw trajectories_val*.pt")
            print(f"  n_sub_trajectories : {len(blob['data'])}")
            _fmt("data[0]", sample, depth=1)
            return
        if isinstance(sample, dict) and "start_id" in sample:
            print("TYPE : raw imgdata_val*.pt")
            print(f"  n_sub_trajectories : {len(blob['data'])}")
            _fmt("data[0]", sample, depth=1)
            return

    # ── Checkpoint (produced by train_bc) ────────────────────────────────────
    if isinstance(blob, dict) and "model" in blob and "config" in blob:
        print("TYPE : training checkpoint")
        print(f"  epoch        : {blob.get('epoch')}")
        print(f"  val_loss     : {blob.get('val_loss')}")
        print(f"  val_mse_overall : {blob.get('val_mse_overall')}")
        cfg = blob["config"]
        print(f"  T={cfg['window']['T']}  H={cfg['window']['H']}  "
              f"cmd_dim={cfg['model']['cmd_dim']}  "
              f"gru_hidden={cfg['model']['gru_hidden']}")
        stats = blob.get("stats", {})
        print(f"  stats mean   : {stats.get('mean')}")
        print(f"  stats std    : {stats.get('std')}")
        return

    # ── Fallback: generic ─────────────────────────────────────────────────────
    print("TYPE : unknown")
    _fmt("blob", blob)


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python scripts/inspect_pt.py <file.pt> [file2.pt ...]")
        sys.exit(1)
    for p in paths:
        inspect(Path(p))


if __name__ == "__main__":
    main()
