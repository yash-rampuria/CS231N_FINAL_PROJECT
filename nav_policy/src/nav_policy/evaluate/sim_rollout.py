"""Extended FiGS rollouts with goal/collision early termination and PID warm-up."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import figs.utilities.trajectory_helper as th

from nav_policy.evaluate.collision import (
    GoalSettleConfig,
    GoalSettleDetector,
    check_figs_depth_collision,
)


@dataclass
class RolloutConfig:
    time_buffer_s: float = 5.0
    collision_depth_threshold_m: float = 0.25
    goal_position_tol_m: float = 0.5
    goal_yaw_tol_rad: float = 0.5
    check_collision: bool = True
    check_goal: bool = True
    run_until_terminal: bool = False
    max_rollout_s: float = 180.0
    warmup_steps: int = 30
    warmup_kp_pos: float = 2.0
    warmup_kd_vel: float = 1.0
    warmup_kp_yaw: float = 2.0
    warmup_max_vel: float = 0.8
    warmup_max_yaw_rate: float = 1.0
    warmup_prime_policy_buffer: bool = True
    settle_vel_mps: float = 0.12
    settle_body_rate_radps: float = 0.35
    settle_yaw_rate_radps: float = 0.30
    settle_steps: int = 20


@dataclass
class RolloutResult:
    Tro: np.ndarray
    Xro: np.ndarray
    Uro: np.ndarray
    Imgs: Dict[str, np.ndarray]
    Tsol: np.ndarray
    Adv: np.ndarray
    termination: str
    collision: bool
    collision_step: int
    goal_reached: bool
    goal_reached_step: int
    expert_tf: float
    max_tf: float
    n_warmup_steps: int = 0
    warmup_rgb: Optional[np.ndarray] = None


def prime_policy_buffer(policy: Any, frames: List[np.ndarray]) -> None:
    """Push warm-up RGB frames into the policy buffer (no inference)."""
    buf = getattr(policy, "buf", None)
    if buf is None or not frames:
        return
    uses_depth = bool(getattr(policy, "uses_depth", False))
    infer_depth = getattr(policy, "_infer_depth", None)
    for frame in frames:
        depth_hw = infer_depth(frame) if uses_depth and infer_depth is not None else None
        buf.push(frame, depth_hw=depth_hw)


def goal_settle_config_from_rollout(cfg: RolloutConfig) -> GoalSettleConfig:
    return GoalSettleConfig(
        position_tol_m=cfg.goal_position_tol_m,
        yaw_tol_rad=cfg.goal_yaw_tol_rad,
        max_vel_mps=cfg.settle_vel_mps,
        max_body_rate_radps=cfg.settle_body_rate_radps,
        max_yaw_rate_radps=cfg.settle_yaw_rate_radps,
        settle_steps=cfg.settle_steps,
    )


def make_hold_controller(cfg: RolloutConfig,
                         *,
                         frame_name: str,
                         Kv: float,
                         Ka: float,
                         hz: float) -> Any:
    from figs.control.velocity_controller import VelocityController

    from nav_policy.deploy.hold_stabilizer import HoldPositionController

    inner = VelocityController(hz=hz, Kv=Kv, Ka=Ka, frame_name=frame_name)
    return HoldPositionController(
        inner,
        hz=hz,
        kp_pos=cfg.warmup_kp_pos,
        kd_vel=cfg.warmup_kd_vel,
        kp_yaw=cfg.warmup_kp_yaw,
        max_vel=cfg.warmup_max_vel,
        max_yaw_rate=cfg.warmup_max_yaw_rate,
    )


def simulate_with_early_exit(
    sim: Any,
    policy: Any,
    *,
    t0: float,
    expert_tf: float,
    x0: np.ndarray,
    goal_xyz: np.ndarray,
    goal_yaw: float,
    cfg: RolloutConfig,
    obj: Any = None,
    goal_xy: Optional[np.ndarray] = None,
    frame_name: str = "carl",
    Kv: float = 2.0,
    Ka: float = 5.0,
    hold_controller: Any = None,
) -> RolloutResult:
    """
    FiGS rollout with optional early exit on goal reach or depth collision.

    When ``cfg.warmup_steps > 0``, a PID hold stabilizer runs first (excluded
    from stored trajectory); the policy buffer is reset and primed at handoff.
    """
    if sim.solver is None:
        raise ValueError("Frame has not been loaded. Please load a frame before simulating.")

    expert_duration = max(float(expert_tf - t0), 1.0)
    if cfg.run_until_terminal:
        cap = float(cfg.max_rollout_s) if cfg.max_rollout_s > 0 else max(3.0 * expert_duration, 120.0)
        max_tf = float(t0 + cap)
    else:
        max_tf = float(expert_tf + cfg.time_buffer_s)

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

    Xro[:, 0] = x0
    xcr, xpr, xsn = x0.copy(), x0.copy(), x0.copy()
    ucm = np.array([-sim.conFiG["drone"]["m"] / sim.conFiG["drone"]["tn"], 0.0, 0.0, 0.0])
    udl = np.hstack((ucm.reshape(-1, 1), ucm.reshape(-1, 1)))
    zcr = torch.zeros(policy.nzcr) if isinstance(policy.nzcr, int) else None

    warmup_steps = max(0, int(cfg.warmup_steps))
    if warmup_steps > 0 and hold_controller is None:
        hold_controller = make_hold_controller(
            cfg, frame_name=frame_name, Kv=Kv, Ka=Ka, hz=ctl_hz,
        )
    if hold_controller is not None:
        hold_controller.reset(x0)

    goal_xy_arr = (
        np.asarray(goal_xy, dtype=np.float64).ravel()[:2]
        if goal_xy is not None
        else goal_xyz[0:2].astype(np.float64)
    )

    camera = sim.gsplat.generate_output_camera(cam_cfg)
    termination = "timeout"
    collision = False
    collision_step = -1
    reached_goal = False
    goal_reached_step = -1
    stored_steps = 0
    n_warmup_steps = 0
    warmup_frames: List[np.ndarray] = []
    policy_started = warmup_steps == 0
    settle_detector = GoalSettleDetector(
        goal_xyz,
        goal_yaw,
        goal_settle_config_from_rollout(cfg),
    )

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
                reset_fn = getattr(policy, "reset", None)
                if reset_fn is not None:
                    reset_fn(goal_pos_xy=goal_xy_arr)
                if cfg.warmup_prime_policy_buffer:
                    prime_n = int(getattr(getattr(policy, "buf", None), "T", 4))
                    prime_policy_buffer(policy, warmup_frames[-prime_n:])
                settle_detector.reset()

            if not in_warmup:
                ucm, zcr, adv, tsol = policy.control(tcr, xsn, ucm, obj, icr, zcr)

                udl[:, 0] = udl[:, 1]
                udl[:, 1] = ucm

                k = stored_steps
                for ch_name, ch_img in image_dict.items():
                    if ch_name == "depth_raw":
                        continue
                    Iro_lists.setdefault(ch_name, []).append(ch_img)

                Tro[k] = tcr
                Xro[:, k + 1] = xcr
                Uro[:, k] = ucm
                Tsol[:, k] = tsol
                Adv[:, k] = adv
                stored_steps = k + 1

                if cfg.check_collision and "depth_raw" in image_dict:
                    hit, _, _ = check_figs_depth_collision(
                        image_dict["depth_raw"],
                        threshold_m=cfg.collision_depth_threshold_m,
                    )
                    if hit:
                        collision = True
                        collision_step = k
                        termination = "collision"
                        break

                if cfg.check_goal and settle_detector.update(xcr, ucm):
                    reached_goal = True
                    goal_reached_step = k
                    termination = "goal_reached"
                    break
            else:
                udl[:, 0] = udl[:, 1]
                udl[:, 1] = ucm

                if cfg.check_collision and "depth_raw" in image_dict:
                    hit, _, _ = check_figs_depth_collision(
                        image_dict["depth_raw"],
                        threshold_m=cfg.collision_depth_threshold_m,
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
        raise RuntimeError(
            f"policy rollout produced too few control steps (term={termination}, warmup={n_warmup_steps})"
        )

    Tro_out = Tro[: n_out + 1].copy()
    Tro_out[n_out] = t0 + (n_out * n_sim2ctl) / hz_sim
    Imgs = {name: np.stack(frames) for name, frames in Iro_lists.items()}
    warmup_rgb = (
        np.stack(warmup_frames, axis=0).astype(np.uint8)
        if warmup_frames else None
    )

    return RolloutResult(
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
        n_warmup_steps=int(n_warmup_steps),
        warmup_rgb=warmup_rgb,
    )


def rollout_config_from_dict(cfg: Dict) -> RolloutConfig:
    m = cfg.get("metrics", {}) or cfg.get("collection", {}) or {}
    return RolloutConfig(
        time_buffer_s=float(m.get("time_buffer_s", 5.0)),
        collision_depth_threshold_m=float(m.get("collision_depth_threshold_m", 0.25)),
        goal_position_tol_m=float(m.get("goal_position_tol_m", 0.5)),
        goal_yaw_tol_rad=float(m.get("goal_yaw_tol_rad", 0.5)),
        check_collision=bool(m.get("check_collision", True)),
        check_goal=bool(m.get("check_goal", True)),
        run_until_terminal=bool(m.get("run_until_terminal", False)),
        max_rollout_s=float(m.get("max_rollout_s", 180.0)),
        warmup_steps=int(m.get("warmup_steps", 30)),
        warmup_kp_pos=float(m.get("warmup_kp_pos", 2.0)),
        warmup_kd_vel=float(m.get("warmup_kd_vel", 1.0)),
        warmup_kp_yaw=float(m.get("warmup_kp_yaw", 2.0)),
        warmup_max_vel=float(m.get("warmup_max_vel", 0.8)),
        warmup_max_yaw_rate=float(m.get("warmup_max_yaw_rate", 1.0)),
        warmup_prime_policy_buffer=bool(m.get("warmup_prime_policy_buffer", True)),
        settle_vel_mps=float(m.get("settle_vel_mps", 0.12)),
        settle_body_rate_radps=float(m.get("settle_body_rate_radps", 0.35)),
        settle_yaw_rate_radps=float(m.get("settle_yaw_rate_radps", 0.30)),
        settle_steps=int(m.get("settle_steps", 20)),
    )
