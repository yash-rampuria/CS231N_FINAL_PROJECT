"""Dense and terminal rewards for FiGS navigation RL (no relightable renderer)."""



from __future__ import annotations



from typing import Dict, List, Optional, Sequence



import numpy as np

from scipy.spatial.transform import Rotation





def _bbox_violation(positions: np.ndarray,
                    ref_positions: np.ndarray,
                    margin: float) -> np.ndarray:
    """Per-step bbox violation mask, shape (N,)."""
    ref = np.asarray(ref_positions, dtype=np.float64)
    if ref.ndim != 2:
        raise ValueError(f"ref_positions must be 2-D; got {ref.shape}")
    if ref.shape[0] != 3:
        if ref.shape[1] == 3:
            ref = ref.T
        else:
            raise ValueError(f"ref_positions must be (3, M) or (M, 3); got {ref.shape}")
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3); got {pos.shape}")
    lo = ref.min(axis=1) - margin
    hi = ref.max(axis=1) + margin
    out = (pos < lo) | (pos > hi)
    return out.any(axis=1)





def _goal_yaw_error(state: np.ndarray, goal_yaw: float) -> float:

    quat = np.asarray(state, dtype=np.float64)[6:10]

    yaw = Rotation.from_quat(quat).as_euler("xyz", degrees=False)[2]

    err = (yaw - goal_yaw + np.pi) % (2 * np.pi) - np.pi

    return float(abs(err))





def compute_episode_rewards(

    states: Sequence[np.ndarray],

    goal_xy: np.ndarray,

    expert_positions: np.ndarray,

    *,

    goal_yaw: float = 0.0,

    progress_weight: float = 1.0,

    heading_weight: float = 0.1,

    step_penalty: float = -0.01,

    collision_penalty: float = -50.0,

    bbox_penalty: float = -5.0,

    timeout_penalty: float = -20.0,

    success_bonus: float = 5.0,

    success_position_tol: float = 0.5,

    success_yaw_tol_rad: float = 0.5,

    bbox_margin: float = 2.0,

    collision_steps: Optional[Sequence[bool]] = None,

    actions: Optional[Sequence[np.ndarray]] = None,

    action_smooth_weight: float = 0.0,

    goal_settled: bool = False,

    termination: str = "",

) -> List[float]:

    """

    Compute per-step rewards from a list of 10-D state vectors ``xcr``.



    Rewards encourage progress toward the goal, heading alignment, staying in

    The final-step success bonus requires the same settled goal check used in
    sim rollouts (position + yaw + quiet hold), not merely ending near the goal
    on a timeout.

    """

    if not states:

        return []



    goal_xy = np.asarray(goal_xy, dtype=np.float64).ravel()[:2]

    positions = np.stack([np.asarray(s, dtype=np.float64)[0:3] for s in states], axis=0)

    velocities = np.stack([np.asarray(s, dtype=np.float64)[3:6] for s in states], axis=0)

    bbox_bad = _bbox_violation(positions, expert_positions, bbox_margin)

    if collision_steps is None:

        collision_bad = np.zeros(len(states), dtype=bool)

    else:

        collision_bad = np.asarray(list(collision_steps), dtype=bool)

        if collision_bad.shape[0] != len(states):

            raise ValueError("collision_steps length must match states length")



    rewards: List[float] = []

    prev_dist = float(np.linalg.norm(goal_xy - positions[0, 0:2]))

    for i, pos in enumerate(positions):

        dist = float(np.linalg.norm(goal_xy - pos[0:2]))

        progress = prev_dist - dist

        prev_dist = dist



        delta = goal_xy - pos[0:2]

        d_norm = float(np.linalg.norm(delta))

        if d_norm > 1e-6:

            heading = delta / d_norm

        else:

            heading = np.array([1.0, 0.0], dtype=np.float64)

        vel_xy = velocities[i, 0:2]

        v_norm = float(np.linalg.norm(vel_xy))

        align = float(np.dot(vel_xy / max(v_norm, 1e-6), heading)) if v_norm > 1e-3 else 0.0



        r = (

            progress_weight * progress

            + heading_weight * align

            + step_penalty

        )

        if bbox_bad[i]:

            r += bbox_penalty

        if collision_bad[i]:

            r += collision_penalty

        if (
            actions is not None
            and action_smooth_weight > 0.0
            and i > 0
            and i < len(actions)
        ):
            da = (
                np.asarray(actions[i], dtype=np.float64)
                - np.asarray(actions[i - 1], dtype=np.float64)
            )
            r -= action_smooth_weight * float(np.sum(da * da))

        rewards.append(float(r))



    if rewards:
        if (
            str(termination).lower() == "timeout"
            and not goal_settled
            and not collision_bad.any()
        ):
            rewards[-1] += timeout_penalty

    if (
        goal_settled
        and not bbox_bad.any()
        and not collision_bad.any()
    ):
        rewards[-1] += success_bonus

    return rewards





def reward_config_from_dict(cfg: Dict) -> Dict:

    """Extract reward kwargs from an RL YAML block."""

    r = cfg.get("rewards", {}) or {}

    return {

        "progress_weight": float(r.get("progress_weight", 1.0)),

        "heading_weight": float(r.get("heading_weight", 0.1)),

        "step_penalty": float(r.get("step_penalty", -0.01)),

        "collision_penalty": float(r.get("collision_penalty", -50.0)),

        "bbox_penalty": float(r.get("bbox_penalty", -5.0)),

        "timeout_penalty": float(r.get("timeout_penalty", -20.0)),

        "success_bonus": float(r.get("success_bonus", 5.0)),

        "success_position_tol": float(r.get("success_position_tol", 0.5)),

        "success_yaw_tol_rad": float(r.get("success_yaw_tol_rad", 0.5)),

        "bbox_margin": float(r.get("bbox_margin", 2.0)),

        "action_smooth_weight": float(r.get("action_smooth_weight", 0.05)),

    }


