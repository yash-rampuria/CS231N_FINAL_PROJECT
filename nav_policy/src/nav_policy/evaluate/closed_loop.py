"""

Closed-loop evaluation of a trained RGBVelocityPolicy inside the FiGS

simulator.



This script REQUIRES a 3DGS scene checkpoint that FiGS knows how to load via

``figs.simulator.Simulator(scene, rollout, frame)``. The processed cache and

the .pt trajectories alone are not enough -- they contain the recorded RGB

videos but not the splat needed to re-render new viewpoints once the policy

drives the drone off the expert's trajectory.



Outputs land under ``<output_dir>``:

    summary.json                aggregate metrics across all rollouts

    per_rollout.csv             one row per rollout

    rollout_<name>/             one folder per rollout with

        video.mp4                 policy-driven RGB rollout

        trajectory.npz            {Tro, Xro, Uro, Tsol, Adv} from sim.simulate

        expert_reference.npz      {Tro, Xro, Uro} loaded from setup_from

        metrics.json              per-rollout metrics

"""



from __future__ import annotations



import argparse

import csv

import json

import sys

import time

import traceback

from dataclasses import dataclass

from pathlib import Path

from typing import Dict, List, Optional, Tuple



import numpy as np

import torch

import yaml

# PyTorch 2.6 changed torch.load default to weights_only=True, which breaks
# nerfstudio/gsplat checkpoints containing arbitrary numpy objects.
_orig_torch_load = torch.load
def _patched_torch_load(f, map_location=None, pickle_module=None, *,
                        weights_only=False, mmap=None, **kw):
    return _orig_torch_load(f, map_location=map_location,
                            weights_only=weights_only, mmap=mmap, **kw)
torch.load = _patched_torch_load

from scipy.spatial.transform import Rotation



from nav_policy.deploy.policy_controller import RGBVelocityController

from nav_policy.evaluate.sim_rollout import RolloutConfig, rollout_config_from_dict, simulate_with_early_exit





CMD_NAMES = ("vx", "vy", "vz", "psi_dot")





@dataclass

class ExpertRef:

    """Subset of a saved validation rollout used as an oracle for comparison."""

    Tro: np.ndarray   # (Nctl+1,)

    Xro: np.ndarray   # (10, Nctl+1)

    Uro: np.ndarray   # (4, Nctl)

    t0: float

    tf: float

    x0: np.ndarray    # (10,)

    course: str

    rollout_id: str

    setup_from: Path

    sub_idx: int





def load_expert_setup(setup_from: Path, sub_idx: int) -> ExpertRef:

    blob = torch.load(setup_from, weights_only=False, map_location="cpu")

    if "data" not in blob:

        raise ValueError(f"{setup_from}: missing 'data' key")

    if sub_idx >= len(blob["data"]):

        raise IndexError(

            f"{setup_from}: sub_idx={sub_idx} out of range ({len(blob['data'])} sub-trajs)"

        )

    traj = blob["data"][sub_idx]

    Tro = np.asarray(traj["Tro"], dtype=np.float64)

    Xro = np.asarray(traj["Xro"], dtype=np.float64)

    Uro = np.asarray(traj["Uro"], dtype=np.float64)

    return ExpertRef(

        Tro=Tro, Xro=Xro, Uro=Uro,

        t0=float(Tro[0]),

        tf=float(Tro[-1]),

        x0=Xro[:, 0].copy(),

        course=str(traj.get("course", "")),

        rollout_id=str(traj.get("rollout_id", "")),

        setup_from=setup_from,

        sub_idx=sub_idx,

    )





def _yaw_series(quat_cols: np.ndarray) -> np.ndarray:

    """quat_cols: (4, N) Hamilton scalar-last -> unwrapped yaw (N,)."""

    yaw = Rotation.from_quat(quat_cols.T).as_euler("xyz", degrees=False)[:, 2]

    return np.unwrap(yaw)





def _goal_yaw(expert: ExpertRef) -> float:

    return float(_yaw_series(expert.Xro[6:10, -1:])[-1])





def _path_length(positions: np.ndarray) -> float:

    """positions: (3, N) -> total path length in meters."""

    if positions.shape[1] < 2:

        return 0.0

    diffs = np.diff(positions, axis=1)

    return float(np.sum(np.linalg.norm(diffs, axis=0)))





def _bbox_violation(positions: np.ndarray,

                    ref_positions: np.ndarray,

                    margin: float = 2.0) -> Tuple[bool, int]:

    """Return (any_violation, first_violation_step) using a bbox grown from the expert."""

    pos = np.asarray(positions, dtype=np.float64)

    ref = np.asarray(ref_positions, dtype=np.float64)

    # Positions in closed_loop are (3, N); accept (N, 3) as well.

    if pos.ndim != 2 or ref.ndim != 2:

        raise ValueError(f"positions and ref_positions must be 2-D; got {pos.shape}, {ref.shape}")

    if pos.shape[0] != 3:

        if pos.shape[1] == 3:

            pos = pos.T

        else:

            raise ValueError(f"positions must be (3, N) or (N, 3); got {pos.shape}")

    if ref.shape[0] != 3:

        if ref.shape[1] == 3:

            ref = ref.T

        else:

            raise ValueError(f"ref_positions must be (3, M) or (M, 3); got {ref.shape}")

    lo = ref.min(axis=1, keepdims=True) - margin          # (3, 1)

    hi = ref.max(axis=1, keepdims=True) + margin          # (3, 1)

    out = (pos < lo) | (pos > hi)                           # (3, N)

    bad = out.any(axis=0)                                   # (N,)

    if bad.any():

        return True, int(np.argmax(bad))

    return False, -1





def _interp_expert_at_times(expert: ExpertRef, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Interpolate expert position, velocity, yaw at policy times."""

    t_ref = expert.Tro

    p_exp = np.zeros((3, len(times)), dtype=np.float64)

    v_exp = np.zeros((3, len(times)), dtype=np.float64)

    for d in range(3):

        p_exp[d] = np.interp(times, t_ref, expert.Xro[d])

        v_exp[d] = np.interp(times, t_ref, expert.Xro[3 + d])

    yaw_ref = _yaw_series(expert.Xro[6:10, :])

    yaw_exp = np.interp(times, t_ref, yaw_ref)

    return p_exp, v_exp, yaw_exp





def _wrap_yaw_err(yaw_a: np.ndarray, yaw_b: np.ndarray) -> np.ndarray:

    err = yaw_a - yaw_b

    return (err + np.pi) % (2 * np.pi) - np.pi





def compute_metrics(expert: ExpertRef,

                    Tpol: np.ndarray,

                    Xpol: np.ndarray,

                    Upol: np.ndarray,

                    Tsol: np.ndarray,

                    *,

                    rollout_meta: Optional[Dict] = None,

                    success_position_tol: float = 0.5,

                    success_tracking_tol: float = 1.0,

                    success_yaw_tol_rad: float = 0.5,

                    bbox_margin: float = 2.0) -> Dict[str, float]:

    """All quantities are in physical units (m, m/s, rad/s, seconds)."""

    rollout_meta = rollout_meta or {}

    n_pol = min(int(Tpol.shape[0]) - 1, int(Upol.shape[1]))

    if n_pol <= 1:

        raise RuntimeError("policy rollout produced too few control steps")



    times = Tpol[: n_pol + 1]

    p_pol = Xpol[0:3, : n_pol + 1]

    v_pol = Xpol[3:6, : n_pol + 1]

    yaw_pol = _yaw_series(Xpol[6:10, : n_pol + 1])



    p_exp, v_exp, yaw_exp = _interp_expert_at_times(expert, times)



    goal_xyz = expert.Xro[0:3, -1]

    goal_yaw = _goal_yaw(expert)



    pos_err = np.linalg.norm(p_pol - p_exp, axis=0)

    tracking_rmse = float(np.sqrt(np.mean(pos_err ** 2)))

    final_pos_err = float(pos_err[-1])

    max_pos_err = float(pos_err.max())



    vel_err = v_pol - v_exp

    vel_rmse_xyz = np.sqrt((vel_err ** 2).mean(axis=1))

    yaw_err = _wrap_yaw_err(yaw_pol, yaw_exp)

    yaw_rmse = float(np.sqrt(np.mean(yaw_err ** 2)))



    path_length_pol = _path_length(p_pol)

    path_length_exp_full = _path_length(expert.Xro[0:3, :])

    path_progress_ratio = float(

        min(1.0, path_length_pol / max(path_length_exp_full, 1e-6))

    )



    init_goal_dist = float(np.linalg.norm(p_pol[:, 0] - goal_xyz))

    final_goal_dist = float(np.linalg.norm(p_pol[:, -1] - goal_xyz))

    if init_goal_dist > 1e-6:

        goal_progress_ratio = float(

            np.clip((init_goal_dist - final_goal_dist) / init_goal_dist, 0.0, 1.0)

        )

    else:

        goal_progress_ratio = 1.0 if final_goal_dist < success_position_tol else 0.0



    final_goal_pos_err = final_goal_dist

    final_goal_yaw_err = float(abs(_wrap_yaw_err(yaw_pol[-1:], np.array([goal_yaw]))[0]))



    bbox_hit, bbox_step = _bbox_violation(p_pol, expert.Xro[0:3, :], margin=bbox_margin)



    collision = bool(rollout_meta.get("collision", False))

    termination = str(rollout_meta.get("termination", "timeout"))

    goal_reached_flag = bool(rollout_meta.get("goal_reached", False))



    expert_tracking_success = (

        (not collision)

        and (not bbox_hit)

        and final_pos_err < success_position_tol

        and tracking_rmse < success_tracking_tol

    )



    goal_success = (

        (not collision)

        and (not bbox_hit)

        and final_goal_pos_err < success_position_tol

        and final_goal_yaw_err < success_yaw_tol_rad

    )



    failure_reasons: List[str] = []

    if collision:

        failure_reasons.append("collision")

    if bbox_hit:

        failure_reasons.append("bbox_violation")

    if termination == "timeout" and not goal_reached_flag:

        failure_reasons.append("timeout")

    if final_goal_pos_err >= success_position_tol:

        failure_reasons.append("goal_position")

    if final_goal_yaw_err >= success_yaw_tol_rad:

        failure_reasons.append("goal_yaw")

    if final_pos_err >= success_position_tol:

        failure_reasons.append("expert_final_position")

    if tracking_rmse >= success_tracking_tol:

        failure_reasons.append("expert_tracking_rmse")



    if goal_success:

        primary_failure = "success"

    elif failure_reasons:

        primary_failure = failure_reasons[0]

    else:

        primary_failure = "unknown"



    model_latencies_ms = Tsol[1, :n_pol] * 1000.0

    inner_latencies_ms = Tsol[3, :n_pol] * 1000.0



    return {

        "n_steps": int(n_pol),

        "duration_s": float(Tpol[n_pol] - Tpol[0]),

        "expert_duration_s": float(expert.tf - expert.t0),

        "termination": termination,

        "collision": collision,

        "collision_step": int(rollout_meta.get("collision_step", -1)),

        "goal_reached": goal_reached_flag,

        "goal_reached_step": int(rollout_meta.get("goal_reached_step", -1)),

        "success": bool(goal_success),

        "goal_success": bool(goal_success),

        "expert_tracking_success": bool(expert_tracking_success),

        "failure_reason": primary_failure,

        "failure_reasons": failure_reasons,

        "bbox_violation": bool(bbox_hit),

        "bbox_violation_step": int(bbox_step),

        "tracking_rmse_m": tracking_rmse,

        "final_position_error_m": final_pos_err,

        "max_position_error_m": max_pos_err,

        "final_goal_position_error_m": final_goal_pos_err,

        "final_goal_yaw_error_rad": final_goal_yaw_err,

        "vel_rmse_x_mps": float(vel_rmse_xyz[0]),

        "vel_rmse_y_mps": float(vel_rmse_xyz[1]),

        "vel_rmse_z_mps": float(vel_rmse_xyz[2]),

        "vel_rmse_norm_mps": float(np.sqrt(np.mean((vel_err ** 2).sum(axis=0)))),

        "yaw_rmse_rad": yaw_rmse,

        "path_length_policy_m": path_length_pol,

        "path_length_expert_m": path_length_exp_full,

        "path_progress_ratio": path_progress_ratio,

        "goal_progress_ratio": goal_progress_ratio,

        "latency_model_ms_mean": float(model_latencies_ms.mean()),

        "latency_model_ms_p95": float(np.percentile(model_latencies_ms, 95)),

        "latency_inner_ms_mean": float(inner_latencies_ms.mean()),

    }





def _save_video(frames: np.ndarray, path: Path, fps: int = 20) -> None:

    import imageio.v3 as iio

    if frames.ndim != 4 or frames.shape[-1] != 3:

        raise ValueError(f"expected (N,H,W,3), got {frames.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)

    iio.imwrite(str(path), frames.astype(np.uint8), plugin="FFMPEG", fps=fps,
                macro_block_size=1)





def run_one(rollout_cfg: dict,

            controller: RGBVelocityController,

            output_dir: Path,

            rollout_sim_cfg: RolloutConfig,

            metrics_cfg: Dict,

            *,

            Kv: float = 2.0,

            Ka: float = 5.0,

            sim_cache: Optional[Dict] = None) -> Dict[str, float]:

    """Run a single closed-loop FiGS rollout and write artifacts."""

    import gc

    import torch

    from figs.simulator import Simulator



    name = rollout_cfg["name"]

    scene = rollout_cfg["scene"]

    rollout = rollout_cfg.get("rollout", "baseline")

    frame = rollout_cfg.get("frame", "carl")

    setup_from = Path(rollout_cfg["setup_from"]).resolve()

    sub_idx = int(rollout_cfg.get("sub_idx", 0))



    expert = load_expert_setup(setup_from, sub_idx)

    rollout_dir = output_dir / f"rollout_{name}"

    rollout_dir.mkdir(parents=True, exist_ok=True)



    goal_pos_xy = expert.Xro[0:2, -1].astype(np.float64)

    goal_xyz = expert.Xro[0:3, -1].astype(np.float64)

    goal_yaw = _goal_yaw(expert)

    controller.reset(goal_pos_xy=goal_pos_xy)



    _sim_key = (scene, rollout, frame)
    if sim_cache is not None:
        if _sim_key not in sim_cache:
            sim_cache[_sim_key] = Simulator(scene, rollout, frame)
        sim = sim_cache[_sim_key]
    else:
        sim = Simulator(scene, rollout, frame)

    t_start = time.time()

    try:

        result = simulate_with_early_exit(

            sim,

            controller,

            t0=expert.t0,

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

        Tpol, Xpol, Upol, Imgs, Tsol, Adv = (

            result.Tro, result.Xro, result.Uro, result.Imgs, result.Tsol, result.Adv,

        )

    finally:

        if sim_cache is None:
            del sim
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    wall_time = time.time() - t_start



    rollout_meta = {

        "collision": result.collision,

        "collision_step": result.collision_step,

        "goal_reached": result.goal_reached,

        "goal_reached_step": result.goal_reached_step,

        "termination": result.termination,

        "n_warmup_steps": int(result.n_warmup_steps),

    }

    metrics = compute_metrics(

        expert, Tpol, Xpol, Upol, Tsol,

        rollout_meta=rollout_meta,

        success_position_tol=float(metrics_cfg.get("goal_position_tol_m", 0.5)),

        success_tracking_tol=float(metrics_cfg.get("success_tracking_tol_m", 1.0)),

        success_yaw_tol_rad=float(metrics_cfg.get("goal_yaw_tol_rad", 0.5)),

        bbox_margin=float(metrics_cfg.get("bbox_margin", 2.0)),

    )

    metrics["wall_time_s"] = float(wall_time)

    metrics["scene"] = scene

    metrics["rollout"] = rollout

    metrics["frame"] = frame

    metrics["name"] = name

    metrics["course"] = expert.course

    metrics["max_tf_s"] = float(result.max_tf)

    metrics["n_warmup_steps"] = int(result.n_warmup_steps)



    np.savez_compressed(

        rollout_dir / "trajectory.npz",

        Tro=Tpol, Xro=Xpol, Uro=Upol, Tsol=Tsol, Adv=Adv,

    )

    np.savez_compressed(

        rollout_dir / "expert_reference.npz",

        Tro=expert.Tro, Xro=expert.Xro, Uro=expert.Uro,

    )

    if "rgb" in Imgs:

        video_frames = Imgs["rgb"]

        if result.warmup_rgb is not None and result.warmup_rgb.shape[0] > 0:

            video_frames = np.concatenate([result.warmup_rgb, video_frames], axis=0)

        _save_video(video_frames, rollout_dir / "video.mp4", fps=int(controller.hz))

    with open(rollout_dir / "metrics.json", "w") as f:

        json.dump(metrics, f, indent=2)



    print(

        f"  [{name}] goal={metrics['goal_success']}  expert={metrics['expert_tracking_success']}  "

        f"fail={metrics['failure_reason']}  tracking_rmse={metrics['tracking_rmse_m']:.3f}m  "

        f"goal_yaw_err={metrics['final_goal_yaw_error_rad']:.2f}rad  "

        f"latency={metrics['latency_model_ms_mean']:.1f}ms  "

        f"({metrics['n_steps']} steps in {wall_time:.1f}s)",

        flush=True,

    )

    return metrics





def aggregate(per_rollout: List[Dict[str, float]]) -> Dict[str, float]:

    if not per_rollout:

        return {}



    def _bootstrap_ci(values: np.ndarray,

                      n_boot: int = 2000,

                      alpha: float = 0.05) -> Tuple[float, float]:

        if values.size == 0:

            return float("nan"), float("nan")

        rng = np.random.default_rng(0)

        boots = []

        for _ in range(n_boot):

            sample = rng.choice(values, size=values.size, replace=True)

            boots.append(float(sample.mean()))

        lo = float(np.percentile(boots, 100 * alpha / 2))

        hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))

        return lo, hi



    def _rate(key: str) -> Tuple[float, Optional[float], Optional[float]]:

        vals = np.array(

            [1.0 if r.get(key) else 0.0 for r in per_rollout],

            dtype=np.float64,

        )

        if vals.size == 0:

            return 0.0, None, None

        mean = float(vals.mean())

        if vals.size >= 2:

            lo, hi = _bootstrap_ci(vals)

            return mean, lo, hi

        return mean, None, None



    keys_to_avg = [

        "tracking_rmse_m", "final_position_error_m", "max_position_error_m",

        "final_goal_position_error_m", "final_goal_yaw_error_rad",

        "vel_rmse_x_mps", "vel_rmse_y_mps", "vel_rmse_z_mps", "vel_rmse_norm_mps",

        "yaw_rmse_rad",

        "path_length_policy_m", "path_length_expert_m",

        "path_progress_ratio", "goal_progress_ratio",

        "latency_model_ms_mean", "latency_model_ms_p95",

        "latency_inner_ms_mean", "duration_s", "expert_duration_s",

    ]

    agg: Dict[str, float] = {}

    for k in keys_to_avg:

        vals = [r[k] for r in per_rollout if k in r]

        if vals:

            agg[f"mean_{k}"] = float(np.mean(vals))

            agg[f"std_{k}"] = float(np.std(vals))



    agg["n_rollouts"] = len(per_rollout)



    for rate_key, out_prefix in (

        ("goal_success", "goal_success"),

        ("expert_tracking_success", "expert_tracking_success"),

        ("collision", "collision"),

    ):

        mean, lo, hi = _rate(rate_key)

        agg[f"{out_prefix}_rate"] = mean

        if lo is not None:

            agg[f"{out_prefix}_rate_ci95_lo"] = lo

            agg[f"{out_prefix}_rate_ci95_hi"] = hi



    agg["success_rate"] = agg.get("goal_success_rate", 0.0)

    if "goal_success_rate_ci95_lo" in agg:

        agg["success_rate_ci95_lo"] = agg["goal_success_rate_ci95_lo"]

        agg["success_rate_ci95_hi"] = agg["goal_success_rate_ci95_hi"]



    agg["bbox_violation_rate"] = float(

        np.mean([1.0 if r.get("bbox_violation", True) else 0.0 for r in per_rollout])

    )

    agg["timeout_rate"] = float(

        np.mean([1.0 if r.get("termination") == "timeout" else 0.0 for r in per_rollout])

    )

    return agg





def evaluate(config_path: Path,

             checkpoint_override: Optional[Path] = None,

             output_dir_override: Optional[Path] = None,

             run_tag_override: Optional[str] = None) -> Dict[str, float]:

    with open(config_path, "r") as f:

        cfg = yaml.safe_load(f)



    if checkpoint_override is not None:

        cfg["checkpoint"] = str(checkpoint_override)

    if output_dir_override is not None:

        cfg["output_dir"] = str(output_dir_override)

    if run_tag_override is not None:

        cfg["run_tag"] = str(run_tag_override)



    base = config_path.resolve().parent.parent

    ckpt_path = (base / cfg["checkpoint"]).resolve()

    output_dir = (base / cfg["output_dir"]).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[checkpoint] {ckpt_path}")

    print(f"[output_dir] {output_dir}")



    for rcfg in cfg.get("rollouts", []):

        if "setup_from" in rcfg:

            rcfg["setup_from"] = str((base / rcfg["setup_from"]).resolve())



    metrics_cfg = cfg.get("metrics", {}) or {}

    rollout_sim_cfg = rollout_config_from_dict(cfg)

    depth_stride = int(metrics_cfg.get("depth_inference_stride", 3))

    Kv = float(cfg.get("Kv", 2.0))

    Ka = float(cfg.get("Ka", 5.0))

    print(

        f"[eval] warmup_steps={rollout_sim_cfg.warmup_steps}  "

        f"prime_buffer={rollout_sim_cfg.warmup_prime_policy_buffer}",

        flush=True,

    )



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stress = cfg.get("stress", {}) or {}

    controller = RGBVelocityController.from_checkpoint(

        ckpt_path,

        frame_name=cfg.get("frame", "carl"),

        Kv=float(cfg.get("Kv", 2.0)),

        Ka=float(cfg.get("Ka", 5.0)),

        device=device,

        observation_latency=int(stress.get("observation_latency", 0)),

        brightness_factor=float(stress.get("brightness_factor", 1.0)),

        blur_sigma=float(stress.get("blur_sigma", 0.0)),

        depth_inference_stride=depth_stride,

    )



    reuse_simulator = bool(cfg.get("reuse_simulator", False))
    sim_cache: Dict = {} if reuse_simulator else None

    per_rollout: List[Dict[str, float]] = []

    n_total = len(cfg["rollouts"])

    for rcfg in cfg["rollouts"]:

        try:

            metrics = run_one(

                rcfg, controller, output_dir, rollout_sim_cfg, metrics_cfg,

                Kv=Kv, Ka=Ka, sim_cache=sim_cache,

            )

            per_rollout.append(metrics)

        except Exception as exc:                              # pragma: no cover

            print(f"  [{rcfg.get('name', '?')}] FAILED: {exc}", file=sys.stderr)

            traceback.print_exc(file=sys.stderr)

            per_rollout.append({

                "name": rcfg.get("name", "?"),

                "goal_success": False,

                "expert_tracking_success": False,

                "success": False,

                "bbox_violation": True,

                "failure_reason": "exception",

                "error": str(exc),

            })

        n_done = len(per_rollout)
        n_success = sum(1 for r in per_rollout if r.get("goal_success"))
        print(
            f"  [cumulative] success {n_success}/{n_done} "
            f"({n_success / max(n_done, 1):.1%})  of {n_total} total",
            flush=True,
        )



    summary = aggregate([r for r in per_rollout if "error" not in r])

    summary["checkpoint"] = str(ckpt_path)

    summary["run_tag"] = str(cfg.get("run_tag", output_dir.name))

    summary["metrics_config"] = metrics_cfg

    try:

        ckpt_blob = torch.load(ckpt_path, weights_only=False, map_location="cpu")

        ckpt_train_cfg = ckpt_blob.get("config", {}).get("train", {})

        ckpt_model_cfg = ckpt_blob.get("config", {}).get("model", {})

        summary["zero_goal_heading"] = bool(ckpt_train_cfg.get("zero_goal_heading", False))

        summary["model_arch"] = str(ckpt_model_cfg.get("arch", "rgb_resnet18"))

    except Exception:

        summary["zero_goal_heading"] = False

        summary["model_arch"] = "unknown"

    print("\n=== SUMMARY ===")

    print(json.dumps(summary, indent=2))



    with open(output_dir / "summary.json", "w") as f:

        json.dump(summary, f, indent=2)

    if per_rollout:

        all_keys = sorted({k for r in per_rollout for k in r.keys()})

        with open(output_dir / "per_rollout.csv", "w", newline="") as f:

            w = csv.DictWriter(f, fieldnames=all_keys)

            w.writeheader()

            for r in per_rollout:
                row = dict(r)
                reasons = row.get("failure_reasons")
                if isinstance(reasons, list):
                    row["failure_reasons"] = ";".join(reasons)
                w.writerow(row)

    return summary





def main() -> None:

    p = argparse.ArgumentParser(description="Closed-loop FiGS evaluation of a trained policy.")

    p.add_argument("--config", type=Path, required=True)

    p.add_argument(

        "--checkpoint", type=Path, default=None,

        help="Override the YAML `checkpoint` field.  Useful for re-using one "

             "closed-loop config across BC and per-round DAgger checkpoints.",

    )

    p.add_argument(

        "--output-dir", type=Path, default=None,

        help="Override the YAML `output_dir` field so each ablation lands in "

             "its own directory and the collector can attribute the rows.",

    )

    p.add_argument(

        "--run-tag", type=str, default=None,

        help="Override the YAML `run_tag` field; persisted in summary.json.",

    )

    args = p.parse_args()

    evaluate(args.config,

             checkpoint_override=args.checkpoint,

             output_dir_override=args.output_dir,

             run_tag_override=args.run_tag)





if __name__ == "__main__":

    main()


