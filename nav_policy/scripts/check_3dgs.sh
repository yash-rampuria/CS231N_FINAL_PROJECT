#!/usr/bin/env bash
# One-shot diagnostic: verify the 3DGS scene tree FiGS expects.

set -u

ROOT_VLEAD="/mnt/c/Users/rayan/Rahul/Github_Projects/V-LEAD"
EXPECTED="$ROOT_VLEAD/FiGS-Standalone/3dgs/workspace/outputs"
ALT="$ROOT_VLEAD/3dgs/workspace/outputs"

echo "=== expected location: $EXPECTED ==="
if [ -d "$EXPECTED" ]; then ls "$EXPECTED"; else echo "DOES NOT EXIST"; fi
echo
echo "=== alt location actually populated: $ALT ==="
if [ -d "$ALT" ]; then ls "$ALT"; else echo "DOES NOT EXIST"; fi

for ROOT in "$EXPECTED" "$ALT"; do
    [ -d "$ROOT" ] || continue
    echo
    echo "=========================================="
    echo "scanning: $ROOT"
    echo "=========================================="

    echo "* total size"
    du -sh "$ROOT" 2>/dev/null

    echo "* nerfstudio_models dirs"
    find "$ROOT" -name 'nerfstudio_models' -type d 2>/dev/null || echo "  (none)"

    echo "* .ckpt files"
    find "$ROOT" -name '*.ckpt' 2>/dev/null | head -20
    n=$(find "$ROOT" -name '*.ckpt' 2>/dev/null | wc -l)
    echo "  total .ckpt: $n"

    echo "* per-scene file counts and sizes"
    for d in "$ROOT"/*/; do
        nfiles=$(find "$d" -type f 2>/dev/null | wc -l)
        size=$(du -sh "$d" 2>/dev/null | cut -f1)
        printf "  %-8s files=%s  scene=%s\n" "$size" "$nfiles" "$d"
    done
done
