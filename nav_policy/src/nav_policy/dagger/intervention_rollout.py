"""Intervention DAgger rollouts: policy flies until drift/oscillation, then MPC executes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import figs.utilities.trajectory_helper as th

from nav_policy.dagger.mpc_oracle import MPCRelabeler
from nav_policy.evaluate.collision import (
    GoalSettleDetector,
    check_figs_depth_collision,
)
from nav_policy.evaluate.sim_rollout import (
    RolloutConfig,
    RolloutResult,
    goal_settle_config_from_rollout,
    make_hold_controller,
    prime_policy_buffer,
)


@dataclass
class InterventionConfig:
    drift_threshold_m: float = 1.0
    min_policy_steps: int = 20
    oscillation_window_steps: int = 40
    oscillation_reversals: int = 4
    enabled: bool = True
    warmup_steps: int = 0
    warmup_kp_pos: float = 2.0
    warmup_kd_vel: float = 1.0
    warmup_kp_yaw: float = 2.0
    warmup_max_vel: float = 0.8
    warmup_max_yaw_rate: float = 1.0
    warmup_prime_policy_buffer: bool = True
    refresh_mpc_on_intervention: bool = True
    mpc_use_rti_for_execution: bool = False
    collision_early_intervention: bool = True
    mpc_immediate_after_warmup: bool = False


@dataclass
class InterventionRolloutResult(RolloutResult):
    intervention_step: int = -1
    intervention_reason: str = ""
    n_warmup_steps: int = 0
    n_policy_steps: int = 0
    n_expert_steps: int = 0
    vel_labels: Optional[np.ndarray] = None
    psi_dot_labels: Optional[np.ndarray] = None
    is_policy_step: Optional[np.ndarray] = None
    warmup_rgb: Optional[np.ndarray] = None


def intervention_config_from_dict(cfg: Dict) -> InterventionConfig:
    iv = cfg.get("intervention", {}) or {}
    return InterventionConfig(
        drift_threshold_m=float(iv.get("drift_threshold_m", 1.0)),
        min_policy_steps=int(iv.get("min_policy_steps", 20)),
        oscillation_window_steps=int(iv.get("oscillation_window_steps", 40)),
        oscillation_reversals=int(iv.get("oscillation_reversals", 4)),
        enabled=bool(iv.get("enabled", True)),
        warmup_steps=int(iv.get("warmup_steps", 0)),
        warmup_kp_pos=float(iv.get("warmup_kp_pos", 2.0)),
        warmup_kd_vel=float(iv.get("warmup_kd_vel", 1.0)),
        warmup_kp_yaw=float(iv.get("warmup_kp_yaw", 2.0)),
        warmup_max_vel=float(iv.get("warmup_max_vel", 0.8)),
        warmup_max_yaw_rate=float(iv.get("warmup_max_yaw_rate", 1.0)),
        warmup_prime_policy_buffer=bool(iv.get("warmup_prime_policy_buffer", True)),
        refresh_mpc_on_intervention=bool(iv.get("refresh_mpc_on_intervention", True)),
        mpc_use_rti_for_execution=bool(iv.get("mpc_use_rti_for_execution", False)),
        collision_early_intervention=bool(iv.get("collision_early_intervention", True)),
        mpc_immediate_after_warmup=bool(iv.get("mpc_immediate_after_warmup", False)),
    )


class MPCDirectExpert:
    """Execute VehicleRateMPC motor commands directly, matching FiGS expert collection."""

    def __init__(self, relabeler: MPCRelabeler, expert_xro: np.ndarray) -> None:
        self.relabeler = relabeler
        self.expert_xro = np.asarray(expert_xro, dtype=np.float64)
        self.hz = relabeler.control_hz
        self.nzcr = None
        self.name = "MPCDirectExpert"

    def control(self, tcr, xcr, upr, obj, icr, zcr):
        ucr, vel, psi_dot, tsol = self.relabeler.control_and_label(
            float(tcr), xcr, expert_xro=self.expert_xro,
        )
        cmd = np.array([vel[0], vel[1], vel[2], psi_dot], dtype=np.float64)
        return ucr, None, cmd, tsol


class _GoalDistOscillationDetector:
    """Count goal-distance derivative sign reversals (back-and-forth progress)."""

    def __init__(self, window_steps: int, reversal_threshold: int) -> None:
        self.window_steps = max(3, int(window_steps))
        self.reversal_threshold = max(2, int(reversal_threshold))
        self._history: List[float] = []

    def reset(self) -> None:
        self._history.clear()

    def update(self, goal_dist_m: float) -> int:
        self._history.append(float(goal_dist_m))
        if len(self._history) > self.window_steps:
            self._history.pop(0)
        if len(self._history) < 3:
            return 0
        reversals = 0
        for i in range(2, len(self._history)):
            d1 = self._history[i - 1] - self._history[i - 2]
            d2 = self._history[i] - self._history[i - 1]
            if d1 * d2 < 0.0:
                reversals += 1
        return reversals


def _expert_xy_drift_m(x: np.ndarray, expert_xro: np.ndarray, step_k: int) -> float:
    ei = min(int(step_k), int(expert_xro.shape[1]) - 1)
    delta = x[0:2].astype(np.float64) - expert_xro[0:2, ei].astype(np.float64)
    return float(np.linalg.norm(delta))


def _incremental_tracking_drift_m(x: np.ndarray,
                                expert_xro: np.ndarray,
                                k_policy: int,
                                handoff_xy: np.ndarray,
                                expert_handoff_xy: np.ndarray) -> float:
    """Drift relative to expert progress since policy handoff (not absolute time index)."""
    ei = min(int(k_policy), int(expert_xro.shape[1]) - 1)
    policy_delta = x[0:2].astype(np.float64) - handoff_xy.astype(np.float64)
    expert_delta = expert_xro[0:2, ei].astype(np.float64) - expert_handoff_xy.astype(np.float64)
    return float(np.linalg.norm(policy_delta - expert_delta))


def _goal_dist_xy(x: np.ndarray, goal_xy: np.ndarray) -> float:
    return float(np.linalg.norm(goal_xy - x[0:2].astype(np.float64)))



def _refresh_expert_mpc(expert_controller: MPCDirectExpert,
                        expert_xro: np.ndarray,
                        expert_uro: np.ndarray,
                        *,
                        mpc_frame: str,
                        mpc_policy: str,
                        control_hz: float,
                        use_rti: bool) -> None:
    expert_controller.relabeler = MPCRelabeler.from_expert(
        Xro=expert_xro,
        Uro=expert_uro,
        frame=mpc_frame,
        policy=mpc_policy,
        control_hz=control_hz,
        use_RTI=use_rti,
    )


def simulate_with_intervention(
    sim: Any,
    policy: Any,
    expert_controller: MPCDirectExpert,
    relabeler: MPCRelabeler,
    *,
    t0: float,
    expert_tf: float,
    x0: np.ndarray,
    expert_xro: np.ndarray,
    expert_uro: np.ndarray,
    goal_xyz: np.ndarray,
    goal_yaw: float,
    goal_xy: np.ndarray,
    rollout_cfg: RolloutConfig,
    intervention_cfg: InterventionConfig,
    hold_controller: Any = None,
    mpc_frame: str = "carl",
    mpc_policy: str = "vrmpc_rrt",
    Kv: float = 2.0,
    Ka: float = 5.0,
    obj: Any = None,
) -> InterventionRolloutResult:
    """Policy rollout with optional hold warm-up and MPC takeover after drift/oscillation."""
    if sim.solver is None:
        raise ValueError("Frame has not been loaded. Please load a frame before simulating.")

    expert_duration = max(float(expert_tf - t0), 1.0)
    if rollout_cfg.run_until_terminal:
        cap = (
            float(rollout_cfg.max_rollout_s)
            if rollout_cfg.max_rollout_s > 0
            else max(3.0 * expert_duration, 120.0)
        )
        max_tf = float(t0 + cap)
    else:
        max_tf = float(expert_tf + rollout_cfg.time_buffer_s)

    hz_sim = sim.conFiG["rollout"]["frequency"]
    t_dly = sim.conFiG["rollout"]["delay"]
    mu_md_s = np.array(sim.conFiG["rollout"]["model_noise"]["mean"])
    std_md_s = np.array(sim.conFiG["rollout"]["model_noise"]["std"])
    mu_sn = np.array(sim.conFiG["rollout"]["sensor_noise"]["mean"])
    std_sn = np.array(sim.conFiG["rollout"]["sensor_noise"]["std"])
    use_fusion = sim.conFiG["rollout"]["sensor_model_fusion"]["use_fusion"]
    Wf = np.diag(sim.conFiG["rollout"]["sensor_model_fusion"]["weights"])
    nx, nu = sim.conFiG["drone"]["nx"], sim.conFiG["drone"]["nu"]
    cam_cfg = sim.conFiG["drone"]["camera"]
    T_c2b = sim.conFiG["drone"]["T_c2b"]

    ctl_hz = float(policy.hz)
    n_sim2ctl = int(hz_sim / ctl_hz)
    mu_md = mu_md_s * (1 / n_sim2ctl)
    std_md = std_md_s * (1 / n_sim2ctl)
    dt = np.round(max_tf - t0)
    Nsim = int(dt * hz_sim)
    n_delay = int(t_dly * hz_sim)
    Wf_sn, Wf_md = Wf, 1 - Wf
    Nctl = int(dt * ctl_hz)

    Tro = np.zeros(Nctl + 1)
    Xro = np.zeros((nx, Nctl + 1))
    Uro = np.zeros((nu, Nctl))
    Iro_lists: Dict[str, list] = {}
    Tsol = np.zeros((4, Nctl))
    Adv = np.zeros((nu, Nctl))
    vel_labels = np.zeros((Nctl, 3), dtype=np.float32)
    psi_dot_labels = np.zeros((Nctl,), dtype=np.float32)
    is_policy_step = np.zeros((Nctl,), dtype=bool)

    Xro[:, 0] = x0
    xcr, xpr, xsn = x0.copy(), x0.copy(), x0.copy()
    ucm = np.array([-sim.conFiG["drone"]["m"] / sim.conFiG["drone"]["tn"], 0.0, 0.0, 0.0])
    udl = np.hstack((ucm.reshape(-1, 1), ucm.reshape(-1, 1)))
    zcr = torch.zeros(policy.nzcr) if isinstance(policy.nzcr, int) else None

    warmup_steps = max(0, int(intervention_cfg.warmup_steps))
    if warmup_steps > 0 and hold_controller is None:
        raise ValueError("warmup_steps > 0 requires hold_controller")

    camera = sim.gsplat.generate_output_camera(cam_cfg)
    osc = _GoalDistOscillationDetector(
        intervention_cfg.oscillation_window_steps,
        intervention_cfg.oscillation_reversals,
    )

    intervening = False
    intervention_step = -1
    intervention_reason = ""
    termination = "timeout"
    collision = False
    collision_step = -1
    reached_goal = False
    goal_reached_step = -1
    stored_steps = 0
    n_warmup_steps = 0
    n_policy_steps = 0
    n_expert_steps = 0
    warmup_frames: List[np.ndarray] = []
    policy_started = warmup_steps == 0
    policy_handoff_xy: Optional[np.ndarray] = None
    expert_handoff_xy: Optional[np.ndarray] = None
    expert_mpc_refreshed = False
    settle_detector = GoalSettleDetector(
        goal_xyz,
        goal_yaw,
        goal_settle_config_from_rollout(rollout_cfg),
    )

    if hold_controller is not None:
        hold_controller.reset(x0)

    if intervention_cfg.mpc_immediate_after_warmup and warmup_steps == 0:
        intervening = True
        intervention_reason = "mpc_verify"
        _refresh_expert_mpc(
            expert_controller,
            expert_xro,
            expert_uro,
            mpc_frame=mpc_frame,
            mpc_policy=mpc_policy,
            control_hz=ctl_hz,
            use_rti=intervention_cfg.mpc_use_rti_for_execution,
        )
        expert_mpc_refreshed = True

    for i in range(Nsim):
        tcr = t0 + i / hz_sim

        if i % n_sim2ctl == 0:
            k_ctl = i // n_sim2ctl
            in_warmup = warmup_steps > 0 and k_ctl < warmup_steps

            Tb2w = th.xv_to_T(xcr)
            T_c2w = Tb2w @ T_c2b
            image_dict = sim.gsplat.render_rgb(camera, T_c2w)
            icr = image_dict["rgb"]

            if use_fusion:
                xsn += np.random.normal(loc=mu_sn, scale=std_sn)
                xsn = Wf_sn @ xsn + Wf_md @ xcr
            else:
                xsn = xcr + np.random.normal(loc=mu_sn, scale=std_sn)
            xsn[6:10] = th.obedient_quaternion(xsn[6:10], xpr[6:10])

            if in_warmup:
                ucm, _, adv, tsol = hold_controller.control(
                    tcr, xsn, ucm, obj, icr, zcr,
                )
                n_warmup_steps += 1
                warmup_frames.append(np.asarray(icr).copy())
            elif not policy_started:
                policy_started = True
                policy_handoff_xy = xsn[0:2].astype(np.float64).copy()
                expert_handoff_xy = expert_xro[0:2, 0].astype(np.float64).copy()
                settle_detector.reset()
                if intervention_cfg.mpc_immediate_after_warmup:
                    intervening = True
                    intervention_reason = "mpc_verify"
                    if intervention_cfg.refresh_mpc_on_intervention and not expert_mpc_refreshed:
                        _refresh_expert_mpc(
                            expert_controller,
                            expert_xro,
                            expert_uro,
                            mpc_frame=mpc_frame,
                            mpc_policy=mpc_policy,
                            control_hz=ctl_hz,
                            use_rti=intervention_cfg.mpc_use_rti_for_execution,
                        )
                        expert_mpc_refreshed = True
                else:
                    reset_fn = getattr(policy, "reset", None)
                    if reset_fn is not None:
                        reset_fn(goal_pos_xy=goal_xy)
                    if intervention_cfg.warmup_prime_policy_buffer:
                        prime_n = int(getattr(getattr(policy, "buf", None), "T", 4))
                        prime_policy_buffer(policy, warmup_frames[-prime_n:])
                    osc.reset()

            if not in_warmup:
                k_policy = k_ctl - warmup_steps
                if policy_handoff_xy is None:
                    policy_handoff_xy = xsn[0:2].astype(np.float64).copy()
                    expert_handoff_xy = expert_xro[0:2, 0].astype(np.float64).copy()

                if not intervening and not intervention_cfg.mpc_immediate_after_warmup:
                    collision_risk = False
                    if (
                        intervention_cfg.collision_early_intervention
                        and rollout_cfg.check_collision
                        and "depth_raw" in image_dict
                    ):
                        collision_risk, _, _ = check_figs_depth_collision(
                            image_dict["depth_raw"],
                            threshold_m=rollout_cfg.collision_depth_threshold_m,
                        )
                    if collision_risk:
                        intervening = True
                        intervention_step = stored_steps
                        intervention_reason = "collision_risk"
                    elif k_policy >= intervention_cfg.min_policy_steps:
                        drift_m = _incremental_tracking_drift_m(
                            xsn,
                            expert_xro,
                            k_policy,
                            policy_handoff_xy,
                            expert_handoff_xy,
                        )
                        gdist = _goal_dist_xy(xsn, goal_xy)
                        reversals = osc.update(gdist)
                        if drift_m >= intervention_cfg.drift_threshold_m:
                            intervening = True
                            intervention_step = stored_steps
                            intervention_reason = "drift"
                        elif reversals >= intervention_cfg.oscillation_reversals:
                            intervening = True
                            intervention_step = stored_steps
                            intervention_reason = "oscillation"

                    if (
                        intervening
                        and intervention_cfg.refresh_mpc_on_intervention
                        and not expert_mpc_refreshed
                    ):
                        _refresh_expert_mpc(
                            expert_controller,
                            expert_xro,
                            expert_uro,
                            mpc_frame=mpc_frame,
                            mpc_policy=mpc_policy,
                            control_hz=ctl_hz,
                            use_rti=intervention_cfg.mpc_use_rti_for_execution,
                        )
                        expert_mpc_refreshed = True

                if intervening and intervention_step < 0:
                    intervention_step = stored_steps

                if intervening:
                    ucm, zcr_exp, adv, tsol = expert_controller.control(
                        tcr, xsn, ucm, obj, icr, zcr,
                    )
                    cmd = np.asarray(adv, dtype=np.float64).ravel()
                    vel_k = cmd[0:3].astype(np.float32)
                    psi_k = float(cmd[3])
                    n_expert_steps += 1
                    is_policy = False
                else:
                    ucm, zcr, adv, tsol = policy.control(tcr, xsn, ucm, obj, icr, zcr)
                    vel_k, psi_k = relabeler.label(
                        float(tcr), xsn, expert_xro=expert_xro,
                    )
                    n_policy_steps += 1
                    is_policy = True

                udl[:, 0] = udl[:, 1]
                udl[:, 1] = ucm

                for ch_name, ch_img in image_dict.items():
                    if ch_name == "depth_raw":
                        continue
                    Iro_lists.setdefault(ch_name, []).append(ch_img)

                k = stored_steps
                Tro[k] = tcr
                Xro[:, k + 1] = xcr
                Uro[:, k] = ucm
                Tsol[:, k] = tsol
                Adv[:, k] = adv
                vel_labels[k] = vel_k
                psi_dot_labels[k] = psi_k
                is_policy_step[k] = is_policy
                stored_steps = k + 1

                if rollout_cfg.check_collision and "depth_raw" in image_dict:
                    hit, _, _ = check_figs_depth_collision(
                        image_dict["depth_raw"],
                        threshold_m=rollout_cfg.collision_depth_threshold_m,
                    )
                    if hit:
                        collision = True
                        collision_step = k
                        termination = "collision"
                        break

                if rollout_cfg.check_goal and settle_detector.update(xcr, ucm):
                    reached_goal = True
                    goal_reached_step = k
                    termination = "goal_reached"
                    break
            else:
                udl[:, 0] = udl[:, 1]
                udl[:, 1] = ucm

                if rollout_cfg.check_collision and "depth_raw" in image_dict:
                    hit, _, _ = check_figs_depth_collision(
                        image_dict["depth_raw"],
                        threshold_m=rollout_cfg.collision_depth_threshold_m,
                    )
                    if hit:
                        collision = True
                        collision_step = -1
                        termination = "collision_warmup"
                        break

        uin = udl[:, 0] if i % n_sim2ctl < n_delay else udl[:, 1]
        xcr = sim.solver.simulate(x=xcr, u=uin)
        if use_fusion:
            xsn = sim.solver.simulate(x=xsn, u=uin)
        xcr = xcr + np.random.normal(loc=mu_md, scale=std_md)
        xcr[6:10] = th.obedient_quaternion(xcr[6:10], xpr[6:10])
        xpr = xcr

    n_out = stored_steps
    if n_out <= 0:
        term = termination
        raise RuntimeError(
            f"intervention rollout produced too few control steps (term={term}, warmup={n_warmup_steps})"
        )

    Tro_out = Tro[: n_out + 1].copy()
    Tro_out[n_out] = t0 + (n_out * n_sim2ctl) / hz_sim
    Imgs = {name: np.stack(frames) for name, frames in Iro_lists.items()}
    warmup_rgb = (
        np.stack(warmup_frames, axis=0).astype(np.uint8)
        if warmup_frames else None
    )

    return InterventionRolloutResult(
        Tro=Tro_out,
        Xro=Xro[:, : n_out + 1].copy(),
        Uro=Uro[:, :n_out].copy(),
        Imgs=Imgs,
        Tsol=Tsol[:, :n_out].copy(),
        Adv=Adv[:, :n_out].copy(),
        termination=termination,
        collision=collision,
        collision_step=collision_step,
        goal_reached=reached_goal,
        goal_reached_step=goal_reached_step,
        expert_tf=float(expert_tf),
        max_tf=max_tf,
        intervention_step=int(intervention_step),
        intervention_reason=intervention_reason,
        n_warmup_steps=int(n_warmup_steps),
        n_policy_steps=int(n_policy_steps),
        n_expert_steps=int(n_expert_steps),
        vel_labels=vel_labels[:n_out].copy(),
        psi_dot_labels=psi_dot_labels[:n_out].copy(),
        is_policy_step=is_policy_step[:n_out].copy(),
        warmup_rgb=warmup_rgb,
    )
