"""Lightweight deterministic closed-loop eval during RL training."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from figs.control.velocity_controller import VelocityController
from figs.simulator import Simulator

from nav_policy.data.normalization import CommandStats
from nav_policy.deploy.policy_controller import RGBVelocityController
from nav_policy.evaluate.closed_loop import load_expert_setup
from nav_policy.evaluate.sim_rollout import RolloutConfig, simulate_with_early_exit
from nav_policy.model.depth_estimator import DepthAnythingV2Small
from nav_policy.rl.stochastic_policy import StochasticVelocityPolicy


def eval_train_rollouts_goal_success(
    policy: StochasticVelocityPolicy,
    stats: CommandStats,
    rollouts: List[dict],
    rollout_sim_cfg: RolloutConfig,
    *,
    image_size: int = 224,
    frame_name: str = "carl",
    Kv: float = 2.0,
    Ka: float = 5.0,
    device: torch.device,
    depth_inference_stride: int = 5,
    depth_model: Optional[DepthAnythingV2Small] = None,
    zero_goal_heading: bool = False,
    goal_distance_scale: float = 5.0,
) -> Dict[str, Any]:
    """
    Run deterministic closed-loop rollouts (no artifact writes) and return
    per-query goal success for checkpoint selection.
    """
    was_training = policy.training
    policy.eval()

    inner = VelocityController(hz=20, Kv=Kv, Ka=Ka, frame_name=frame_name)
    controller = RGBVelocityController(
        model=policy.base,
        stats=stats,
        inner=inner,
        image_size=image_size,
        zero_goal_heading=zero_goal_heading,
        goal_distance_scale=goal_distance_scale,
        device=device,
        depth_model=depth_model,
        depth_inference_stride=depth_inference_stride,
    )

    per_query: Dict[str, bool] = {}
    for rcfg in rollouts:
        name = str(rcfg["name"])
        scene = rcfg["scene"]
        rollout = rcfg.get("rollout", "baseline")
        frame = rcfg.get("frame", frame_name)
        setup_from = Path(rcfg["setup_from"]).resolve()
        sub_idx = int(rcfg.get("sub_idx", 0))

        expert = load_expert_setup(setup_from, sub_idx)
        goal_pos_xy = expert.Xro[0:2, -1].astype(np.float64)
        goal_xyz = expert.Xro[0:3, -1].astype(np.float64)
        goal_yaw = float(
            Rotation.from_quat(expert.Xro[6:10, -1]).as_euler("xyz", degrees=False)[2]
        )
        controller.reset(goal_pos_xy=goal_pos_xy)

        sim = Simulator(scene, rollout, frame)
        try:
            result = simulate_with_early_exit(
                sim,
                controller,
                t0=float(expert.t0),
                expert_tf=expert.tf,
                x0=expert.x0,
                goal_xyz=goal_xyz,
                goal_yaw=goal_yaw,
                cfg=rollout_sim_cfg,
                goal_xy=goal_pos_xy,
                frame_name=frame,
                Kv=Kv,
                Ka=Ka,
            )
            ok = bool(result.goal_reached and not result.collision)
        finally:
            del sim
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        per_query[name] = ok

    if was_training:
        policy.train()

    n_success = int(sum(per_query.values()))
    n_rollouts = len(per_query)
    return {
        "goal_success_rate": float(n_success / max(n_rollouts, 1)),
        "n_success": n_success,
        "n_rollouts": n_rollouts,
        "per_query": per_query,
    }
