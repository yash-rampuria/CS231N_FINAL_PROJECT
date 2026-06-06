"""Helper: summarize a processed manifest (sample counts per split, caches, T/H)."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    args = p.parse_args()

    d = json.load(open(args.manifest))
    samples = d["samples"]
    c = Counter(s["split"] for s in samples)
    by_round = Counter(s.get("round", 0) for s in samples)
    unique_caches = sorted({s["cache"] for s in samples})
    print(f"manifest: {args.manifest}")
    print(f"  T,H            = {d['T']}, {d['H']}")
    print(f"  num caches     = {len(unique_caches)}")
    print(f"  splits         = {dict(c)}")
    print(f"  rounds         = {dict(by_round)}")
    print(f"  total samples  = {sum(c.values())}")


if __name__ == "__main__":
    main()
