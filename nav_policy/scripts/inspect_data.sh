#!/usr/bin/env bash
# Quick inspector for nav_policy/data/raw/<run>
ROOT="/mnt/c/Users/rayan/Rahul/Github_Projects/V-LEAD/nav_policy/data/raw"
for d in backroom_exp_04-29-26 packardpark_exp_04-29-26; do
  echo "== $d =="
  cd "$ROOT/$d" || continue
  echo "trajectories_val: $(ls trajectories_val*.pt 2>/dev/null | wc -l)"
  echo "rgb videos:       $(ls video_val_rollout_images_rgb*.mp4 2>/dev/null | wc -l)"
  echo "depth videos:     $(ls video_val_rollout_images_depth*.mp4 2>/dev/null | wc -l)"
  echo "semantic videos:  $(ls video_val_rollout_images_semantic*.mp4 2>/dev/null | wc -l)"
  echo "total size:       $(du -sh . | cut -f1)"
  for k in trajectories_val rgb depth semantic; do
    if [ "$k" = trajectories_val ]; then pat='trajectories_val*.pt'; else pat="video_val_rollout_images_${k}*.mp4"; fi
    ids=$(ls $pat 2>/dev/null | grep -oE '[0-9]{5}' | sort -n | tr '\n' ' ')
    echo "  $k IDs: $ids"
  done
  echo
done
