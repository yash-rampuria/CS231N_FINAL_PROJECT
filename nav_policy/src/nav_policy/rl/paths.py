"""Filename helpers for RL rollout artifacts."""

from __future__ import annotations

import re

from nav_policy.evaluate.closed_loop import ExpertRef


def expert_semantic_slug(expert: ExpertRef) -> str:
    """Human-readable semantic target label for logs and video filenames."""
    for raw in (expert.course, expert.rollout_id):
        if not raw:
            continue
        slug = re.sub(r"[^\w\-.]+", "_", str(raw).strip().lower()).strip("_")
        if slug:
            return slug[:48]
    return "target"


def rollout_video_path(
    video_root,
    *,
    iteration: int,
    global_episode: int,
    rollout_name: str,
    semantic_slug: str,
) -> "Path":
    from pathlib import Path

    root = Path(video_root)
    safe_name = re.sub(r"[^\w\-.]+", "_", rollout_name)
    return root / f"iter{iteration:04d}_ep{global_episode:05d}_{safe_name}_{semantic_slug}.mp4"
