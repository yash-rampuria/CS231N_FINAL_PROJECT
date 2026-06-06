"""Hold-position stabilizer for rollout warm-up (no MPC / no policy)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def _yaw_from_quat(q: np.ndarray) -> float:
    return float(Rotation.from_quat(np.asarray(q, dtype=np.float64).ravel()[:4]).as_euler("xyz")[2])


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def goal_pose_state(goal_xyz: np.ndarray, goal_yaw: float) -> np.ndarray:
    """Build a 10-D state vector at the goal pose (zero velocity)."""
    x = np.zeros(10, dtype=np.float64)
    x[0:3] = np.asarray(goal_xyz, dtype=np.float64).ravel()[:3]
    x[6:10] = Rotation.from_euler("xyz", [0.0, 0.0, float(goal_yaw)]).as_quat()
    return x


class HoldPositionController:
    """PID hold at the reset pose, routed through FiGS VelocityController."""

    def __init__(self,
                 inner: Any,
                 *,
                 hz: float = 20.0,
                 kp_pos: float = 2.0,
                 kd_vel: float = 1.0,
                 kp_yaw: float = 2.0,
                 max_vel: float = 0.8,
                 max_yaw_rate: float = 1.0) -> None:
        self.inner = inner
        self.hz = float(hz)
        self.kp_pos = float(kp_pos)
        self.kd_vel = float(kd_vel)
        self.kp_yaw = float(kp_yaw)
        self.max_vel = float(max_vel)
        self.max_yaw_rate = float(max_yaw_rate)
        self._xyz_ref: Optional[np.ndarray] = None
        self._yaw_ref: Optional[float] = None
        self.nzcr = None
        self.name = "HoldPositionController"

    def reset(self, x0: np.ndarray) -> None:
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        self._xyz_ref = x0[0:3].copy()
        self._yaw_ref = _yaw_from_quat(x0[6:10])

    def control(self,
                tcr: float,
                xcr: np.ndarray,
                upr: Any,
                obj: Any,
                icr: Any,
                zcr: Any) -> Tuple[np.ndarray, None, np.ndarray, np.ndarray]:
        if self._xyz_ref is None or self._yaw_ref is None:
            raise RuntimeError("HoldPositionController.reset() must be called before control()")

        pos_err = self._xyz_ref - xcr[0:3].astype(np.float64)
        vel = xcr[3:6].astype(np.float64)
        vel_cmd = self.kp_pos * pos_err - self.kd_vel * vel
        speed = float(np.linalg.norm(vel_cmd))
        if speed > self.max_vel > 0.0:
            vel_cmd *= self.max_vel / speed

        yaw = _yaw_from_quat(xcr[6:10])
        yaw_err = _wrap_pi(self._yaw_ref - yaw)
        psi_dot = float(np.clip(self.kp_yaw * yaw_err, -self.max_yaw_rate, self.max_yaw_rate))

        cmd = np.array([vel_cmd[0], vel_cmd[1], vel_cmd[2], psi_dot], dtype=np.float64)
        ucr, _, _, _ = self.inner.control(
            tcr=tcr, xcr=xcr, upr=upr, obj=cmd, icr=None, zcr=None,
        )
        return ucr, None, cmd, np.zeros(4, dtype=np.float64)
