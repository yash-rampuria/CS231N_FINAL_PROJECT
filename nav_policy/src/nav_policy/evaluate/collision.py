"""Depth-based collision checks using FiGS ``depth_raw`` renders."""



from __future__ import annotations



from typing import Tuple



import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation





def check_figs_depth_collision(

    depth_raw: np.ndarray,

    *,

    threshold_m: float = 0.25,

    roi_width_frac: float = 0.5,

    roi_height_frac: float = 0.6,

    roi_top_frac: float = 0.05,

    min_valid_depth_m: float = 0.05,

    max_valid_depth_m: float = 50.0,

) -> Tuple[bool, float, np.ndarray]:

    """

    Detect imminent collision from a metric depth image ``(H, W)``.



    Uses a forward-facing ROI (upper-center of the image) to reduce false

    positives from the ground plane in the lower image region.



    Returns ``(collision, min_depth_m, collision_mask)`` where ``collision_mask``

    marks pixels in the ROI below ``threshold_m``.

    """

    depth = np.asarray(depth_raw, dtype=np.float64)

    if depth.ndim != 2:

        raise ValueError(f"expected depth_raw (H,W), got {depth.shape}")



    h, w = depth.shape

    x0 = int(((1.0 - roi_width_frac) / 2.0) * w)

    x1 = int(((1.0 + roi_width_frac) / 2.0) * w)

    y0 = int(roi_top_frac * h)

    y1 = int(min(h, (roi_top_frac + roi_height_frac) * h))

    if x1 <= x0 or y1 <= y0:

        return False, float("inf"), np.zeros(depth.shape, dtype=bool)



    roi = depth[y0:y1, x0:x1]

    valid = np.isfinite(roi) & (roi > min_valid_depth_m) & (roi < max_valid_depth_m)

    if not valid.any():

        return False, float("inf"), np.zeros(depth.shape, dtype=bool)



    min_depth = float(roi[valid].min())

    collision = min_depth < threshold_m



    mask = np.zeros(depth.shape, dtype=bool)

    mask[y0:y1, x0:x1] = valid & (roi < threshold_m)

    return collision, min_depth, mask





def goal_reached(

    state: np.ndarray,

    goal_xyz: np.ndarray,

    goal_yaw: float,

    *,

    position_tol_m: float,

    yaw_tol_rad: float,

) -> Tuple[bool, float, float]:

    """Return ``(reached, position_error_m, yaw_error_rad)``."""

    pos = np.asarray(state, dtype=np.float64)[0:3]

    goal = np.asarray(goal_xyz, dtype=np.float64).ravel()[:3]

    pos_err = float(np.linalg.norm(pos - goal))

    quat = np.asarray(state, dtype=np.float64)[6:10]

    yaw = Rotation.from_quat(quat).as_euler("xyz", degrees=False)[2]

    yaw_err = float((yaw - goal_yaw + np.pi) % (2 * np.pi) - np.pi)



    reached = pos_err < position_tol_m and abs(yaw_err) < yaw_tol_rad

    return reached, pos_err, yaw_err


@dataclass
class GoalSettleConfig:
    """Thresholds for declaring the drone settled at the goal."""

    position_tol_m: float = 0.5
    yaw_tol_rad: float = 0.5
    max_vel_mps: float = 0.12
    max_body_rate_radps: float = 0.35
    max_yaw_rate_radps: float = 0.30
    settle_steps: int = 20


class GoalSettleDetector:
    """
    End the episode once pose, velocity, and body rates stay quiet at the goal.

    Calibrated from flightroom expert val tails (071733): last 10-20 steps at
    the endpoint typically have |v| < 0.08 m/s and body rates < 0.25 rad/s.
    """

    def __init__(self,
                 goal_xyz: np.ndarray,
                 goal_yaw: float,
                 cfg: GoalSettleConfig) -> None:
        self.goal_xyz = np.asarray(goal_xyz, dtype=np.float64).ravel()[:3]
        self.goal_yaw = float(goal_yaw)
        self.cfg = cfg
        self._consecutive = 0

    def reset(self) -> None:
        self._consecutive = 0

    def update(self, state: np.ndarray, u_cmd: np.ndarray) -> bool:
        """Return True after ``settle_steps`` consecutive settled control steps."""
        cfg = self.cfg
        at_goal, _, _ = goal_reached(
            state,
            self.goal_xyz,
            self.goal_yaw,
            position_tol_m=cfg.position_tol_m,
            yaw_tol_rad=cfg.yaw_tol_rad,
        )
        vel = float(np.linalg.norm(np.asarray(state[3:6], dtype=np.float64)))
        u = np.asarray(u_cmd, dtype=np.float64).ravel()[:4]
        body_rate = float(np.linalg.norm(u[1:4]))
        yaw_rate = float(abs(u[3]))
        quiet = (
            vel <= cfg.max_vel_mps
            and body_rate <= cfg.max_body_rate_radps
            and yaw_rate <= cfg.max_yaw_rate_radps
        )
        if at_goal and quiet:
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self._consecutive >= int(cfg.settle_steps)


