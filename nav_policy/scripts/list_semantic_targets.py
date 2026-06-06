#!/usr/bin/env python3
"""List semantic course/query labels in raw flightroom trajectory files."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load_trajs(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "data" in blob:
        return blob["data"]
    return blob


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path, help="e.g. data/raw/flightroom_ssv_exp_..._071733_trajs-110")
    p.add_argument("--max", type=int, default=20)
    args = p.parse_args()
    run = args.run_dir
    files = sorted(run.glob("trajectories_val*.pt"))[: args.max]
    if not files:
        files = sorted(run.glob("trajectories*.pt"))[: args.max]
    for path in files:
        trajs = _load_trajs(path)
        t0 = trajs[0]
        frame = t0.get("frame", {}) or {}
        idx = path.stem.replace("trajectories_val", "").replace("trajectories", "")
        print(
            f"{idx:>5s}  course={t0.get('course', '')!r}  "
            f"query={frame.get('query', '')!r}  "
            f"course_name={frame.get('course_name', '')!r}"
        )


if __name__ == "__main__":
    main()
