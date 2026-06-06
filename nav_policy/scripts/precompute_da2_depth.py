#!/usr/bin/env python3
"""
Add DA2-S depth maps to existing processed cache blobs.

Usage (inside Docker, from nav_policy/):
    python scripts/precompute_da2_depth.py --processed-root data/processed_flightroom
    python scripts/precompute_da2_depth.py --processed-root data/processed_flightroom --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nav_policy.model.depth_estimator import (  # noqa: E402
    DepthAnythingV2Small,
    depth_uint8_from_float,
)


def _process_cache(path: Path, model: DepthAnythingV2Small, device: torch.device) -> bool:
    blob = torch.load(path, weights_only=False, map_location="cpu")
    if "depth" in blob and blob["depth"].shape == blob["rgb"].shape[:1]:
        # Already has depth with matching frame count
        if blob["depth"].ndim == 4 and blob["depth"].shape[1] == 1:
            return False
    rgb = blob["rgb"]  # [N, 3, S, S] uint8
    n, _, s, _ = rgb.shape
    depths = []
    batch = 8
    for i in range(0, n, batch):
        chunk = rgb[i : i + batch].to(device)
        with torch.inference_mode():
            d = model(chunk)
        for j in range(d.shape[0]):
            arr = d[j, 0].cpu().numpy()
            depths.append(depth_uint8_from_float(arr))
    depth_t = torch.from_numpy(np.stack(depths, axis=0)).unsqueeze(1).contiguous()
    blob["depth"] = depth_t
    torch.save(blob, path)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute DA2-S depth for cache blobs.")
    p.add_argument("--processed-root", type=Path, required=True)
    p.add_argument("--split", type=str, default=None, help="train, val, or all (default)")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    processed = args.processed_root.resolve()
    manifest_path = processed / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    with open(manifest_path) as f:
        manifest = json.load(f)
    caches = sorted({s["cache"] for s in manifest["samples"]})
    if args.split:
        caches = sorted(
            {s["cache"] for s in manifest["samples"] if s["split"] == args.split}
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[da2] loading Depth Anything V2 Small on {device}...", flush=True)
    model = DepthAnythingV2Small(weights_path=args.weights).to(device)

    updated = 0
    for rel in tqdm(caches, desc="caches"):
        path = processed / rel
        if not path.exists():
            print(f"[skip] missing {path}", flush=True)
            continue
        if _process_cache(path, model, device):
            updated += 1
    print(f"[done] updated {updated}/{len(caches)} caches", flush=True)


if __name__ == "__main__":
    main()
