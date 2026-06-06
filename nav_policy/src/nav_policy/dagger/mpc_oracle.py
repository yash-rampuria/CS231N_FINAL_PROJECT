"""
MPC oracle for proper DAgger relabeling.

Reference-oracle (the default in run_dagger.py) labels every policy-visited
control step with the expert state-trajectory's recorded velocity at the same
time index.  That label was computed under the assumption the drone was at the
recorded reference state, so it cannot tell the policy how to recover when the
policy drifts off the reference.

The MPC oracle re-solves the FiGS VehicleRateMPC OCP starting from the policy's
actual state at every relabel step.  The resulting plan tells us what the
expert MPC would do from the drifted state, which is the canonical DAgger label
(Ross et al. 2011).  The label is then converted to velocity space so it can be
used as a drop-in replacement for the reference-oracle labels in our cache
format::

    vel(k)     = solver.get(1, 'x')[3:6]
    psi_dot(k) = (yaw(solver.get(1, 'x')[6:10]) - yaw(x_policy[6:10])) * hz

The relabeler is instantiated once per sub-trajectory (the recorded Xro/Uro is
given as the MPC's bypass reference trajectory) and reused for every
policy-visited step in that rollout.  Each ``label()`` call costs one OCP solve
(~30-50 ms with SQP, ~10 ms with RTI).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def mpc_reference_time(x: np.ndarray,
                       expert_xro: np.ndarray,
                       hz: float) -> float:
    """
    Map the current state to a reference time on the expert trajectory.

    VehicleRateMPC.get_ydes() centers its search window on ``int(hz * t)``.
    Wall-clock sim time is wrong after PID warm-up (time advances while the
    drone holds at x0) and when the policy drifts off the expert schedule.
    Nearest-neighbor position matching on the full expert path picks the
    correct reference segment instead.
    """
    pos = np.asarray(x[0:3], dtype=np.float64).reshape(3, 1)
    ref = np.asarray(expert_xro[0:3, :], dtype=np.float64)
    idx = int(np.argmin(np.linalg.norm(ref - pos, axis=0)))
    return idx / float(hz)


def _figs_configs_path() -> Path:
    """Locate FiGS-Standalone/configs in dev, Docker, and editable installs."""
    import figs

    if figs.__file__:
        return Path(figs.__file__).resolve().parent.parent.parent / "configs"

    data_path = os.environ.get("DATA_PATH")
    if data_path:
        candidate = Path(data_path).parents[1] / "configs"
        if candidate.exists():
            return candidate

    nav_root = Path(__file__).resolve().parents[3]
    candidate = nav_root.parent / "FiGS-Standalone" / "configs"
    if candidate.exists():
        return candidate

    raise FileNotFoundError("Could not locate FiGS configs directory")


def _load_mpc_horizon(policy: str) -> int:
    """Read the MPC prediction horizon from FiGS policy config."""
    with open(_figs_configs_path() / "policy" / f"{policy}.json") as f:
        return int(json.load(f)["horizon"])


def _pad_expert_for_mpc(Xro: np.ndarray,
                        Uro: np.ndarray,
                        horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hold the final expert pose so VehicleRateMPC.get_ydes() has enough
    reference length for its horizon lookup window.

    Choppy flightroom segments can be as short as 2 s (~41 states at 20 Hz),
    which is shorter than vrmpc_rrt's horizon (40).  Without padding,
    get_ydes() sees an empty nearest-neighbor slice and raises on argmin([]).
    """
    ncol = int(Xro.shape[1])
    if ncol > horizon + 1:
        return Xro, Uro
    pad = horizon
    Xro = np.hstack([Xro, np.tile(Xro[:, -1:], (1, pad))])
    Uro = np.hstack([Uro, np.tile(Uro[:, -1:], (1, pad))])
    return Xro, Uro


class MPCRelabeler:
    """Wraps a VehicleRateMPC to emit velocity-space DAgger labels."""

    def __init__(self, mpc, control_hz: float) -> None:
        self.mpc = mpc
        self.control_hz = float(control_hz)
        self.dt = 1.0 / self.control_hz

    @classmethod
    def from_expert(cls,
                    Xro: np.ndarray,
                    Uro: np.ndarray,
                    frame: str = "carl",
                    policy: str = "vrmpc_rrt",
                    control_hz: float = 20.0,
                    use_RTI: bool = False) -> "MPCRelabeler":
        """
        Build a relabeler that tracks the (Xro, Uro) reference.

        Args:
            Xro:        (10, N+1) recorded expert state trajectory (world frame).
            Uro:        (4, N)    recorded expert input trajectory.
            frame:      FiGS frame config name (e.g. "carl").
            policy:     FiGS MPC policy config name (e.g. "vrmpc_rrt").
            control_hz: control-loop frequency (must match the simulator).
            use_RTI:    if True, use SQP-RTI (single iteration, ~10 ms).  If
                        False, full SQP (more accurate, ~30-50 ms).  We default
                        to full SQP because DAgger labels prioritize accuracy
                        over throughput.

        Returns:
            MPCRelabeler ready to ``.label(t, x)``.
        """
        # Local import so this file can be imported on hosts without the FiGS env.
        from figs.control.vehicle_rate_mpc import VehicleRateMPC

        if Xro.ndim != 2 or Xro.shape[0] != 10:
            raise ValueError(f"Xro must be (10, N+1); got {Xro.shape}")
        if Uro.ndim != 2 or Uro.shape[0] != 4:
            raise ValueError(f"Uro must be (4, N); got {Uro.shape}")
        N = int(Uro.shape[1])
        if Xro.shape[1] != N + 1:
            raise ValueError(
                f"Xro should have N+1={N+1} columns to match Uro N={N}; got {Xro.shape[1]}"
            )

        horizon = _load_mpc_horizon(policy)
        Xro, Uro = _pad_expert_for_mpc(Xro, Uro, horizon=horizon)
        N = int(Uro.shape[1])

        # VehicleRateMPC accepts either a string config name or a "bypass" array
        # of shape (>=18, M) where:
        #   rows 0:11  -> [time, state(10)]
        #   rows 14:18 -> [input(4)]
        # We pack the recorded expert here; the MPC will track this as its
        # internal reference and the get_ydes() nearest-neighbor selection
        # recovers the appropriate sub-segment around the policy's drifted
        # state on every solve.
        T = np.arange(N + 1) / float(control_hz)
        Uro_pad = np.hstack([Uro, Uro[:, -1:]])  # (4, N+1)
        course_arr = np.vstack([
            T[None, :],                  # row 0
            Xro.astype(np.float64),      # rows 1-10
            np.zeros((3, N + 1)),        # rows 11-13 (padding required by FiGS slice)
            Uro_pad.astype(np.float64),  # rows 14-17
        ])

        mpc = VehicleRateMPC(
            course=course_arr,
            policy=policy,
            frame=frame,
            use_RTI=use_RTI,
        )
        return cls(mpc=mpc, control_hz=control_hz)

    def _plan_velocity_label(self,
                             x_policy_64: np.ndarray,
                             plan_x1: np.ndarray) -> Tuple[np.ndarray, float]:
        vel_world = np.asarray(plan_x1[3:6], dtype=np.float32)
        yaw0 = float(Rotation.from_quat(x_policy_64[6:10])
                     .as_euler("xyz", degrees=False)[2])
        yaw1 = float(Rotation.from_quat(np.asarray(plan_x1[6:10], dtype=np.float64))
                     .as_euler("xyz", degrees=False)[2])
        dyaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
        psi_dot = float(dyaw * self.control_hz)
        return vel_world, psi_dot

    def label(self,
              t: float,
              x_policy: np.ndarray,
              *,
              expert_xro: np.ndarray | None = None) -> Tuple[np.ndarray, float]:
        """
        Re-solve the OCP at (t, x_policy) and return the implied velocity-space
        DAgger label.

        Returns:
            vel_world: (3,) float32   commanded next-step world-frame velocity.
            psi_dot:   float          implied yaw rate (rad/s) over one dt.
        """
        if x_policy.shape != (10,):
            raise ValueError(f"x_policy must be (10,); got {x_policy.shape}")

        x_policy_64 = np.asarray(x_policy, dtype=np.float64)
        t_mpc = (
            mpc_reference_time(x_policy_64, expert_xro, self.control_hz)
            if expert_xro is not None else float(t)
        )
        self.mpc.control(t_mpc, x_policy_64)
        plan_x1 = self.mpc.solver.get(1, "x")
        return self._plan_velocity_label(x_policy_64, plan_x1)

    def control_and_label(self,
                          t: float,
                          x_policy: np.ndarray,
                          *,
                          expert_xro: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        One OCP solve: return FiGS motor command plus velocity-space label.

        Expert collection runs VehicleRateMPC directly (body-rate/thrust).
        Execution must use the same path, not a VelocityController wrapper
        around planned velocities.
        """
        if x_policy.shape != (10,):
            raise ValueError(f"x_policy must be (10,); got {x_policy.shape}")

        x_policy_64 = np.asarray(x_policy, dtype=np.float64)
        t_mpc = mpc_reference_time(x_policy_64, expert_xro, self.control_hz)
        ucr, _, _, tsol = self.mpc.control(t_mpc, x_policy_64)
        plan_x1 = self.mpc.solver.get(1, "x")
        vel_world, psi_dot = self._plan_velocity_label(x_policy_64, plan_x1)
        return (
            np.asarray(ucr, dtype=np.float64),
            vel_world,
            psi_dot,
            np.asarray(tsol, dtype=np.float64),
        )