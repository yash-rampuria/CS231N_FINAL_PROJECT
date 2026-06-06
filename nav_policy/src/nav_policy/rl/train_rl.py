"""
RL fine-tuning for nav_policy (PPO default, SAC optional).

Warm-starts from a BC/DAgger checkpoint and collects on-policy rollouts in FiGS
(flightroom training scenes only). No relightable renderer changes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

# PyTorch 2.6 changed torch.load default to weights_only=True, which breaks
# nerfstudio/gsplat checkpoints that contain arbitrary numpy objects.
# Patch torch.load to keep weights_only=False for local files (trusted artifacts).
_orig_torch_load = torch.load
def _patched_torch_load(f, map_location=None, pickle_module=None, *,
                        weights_only=False, mmap=None, **kw):
    return _orig_torch_load(f, map_location=map_location,
                            weights_only=weights_only, mmap=mmap, **kw)
torch.load = _patched_torch_load

from figs.control.velocity_controller import VelocityController

from nav_policy.model.depth_estimator import DepthAnythingV2Small
from nav_policy.model.factory import model_uses_depth
from nav_policy.evaluate.sim_rollout import RolloutConfig, rollout_config_from_dict
from nav_policy.rl.buffer import ReplayBuffer, episodes_to_buffer
from nav_policy.rl.ppo import ppo_config_from_dict, ppo_update
from nav_policy.rl.rewards import reward_config_from_dict
from nav_policy.rl.paths import expert_semantic_slug, rollout_video_path
from nav_policy.rl.train_eval import eval_train_rollouts_goal_success
from nav_policy.evaluate.closed_loop import load_expert_setup
from nav_policy.rl.rollout import RLTrainingController, collect_episode
from nav_policy.rl.sac import SACTrainer, sac_config_from_dict
from nav_policy.rl.stochastic_policy import (
    load_stochastic_from_checkpoint,
    save_rl_checkpoint,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if key == "base_config":
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    nav_root = config_path.resolve().parent.parent
    if "base_config" in cfg:
        base_path = (nav_root / cfg["base_config"]).resolve()
        base_cfg = _load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def _resolve_rollouts(cfg: dict, nav_root: Path) -> List[dict]:
    rollouts = []
    for rcfg in cfg.get("rollouts", []):
        rc = dict(rcfg)
        if "setup_from" in rc:
            rc["setup_from"] = str((nav_root / rc["setup_from"]).resolve())
        rollouts.append(rc)
    return rollouts


def _load_eval_suite(
    eval_config_path: Path,
    nav_root: Path,
) -> tuple[List[dict], RolloutConfig, int]:
    """Load held-out eval rollouts + sim metrics from a closed-loop eval YAML."""
    with open(eval_config_path, "r") as f:
        eval_cfg = yaml.safe_load(f) or {}
    rollouts = _resolve_rollouts(eval_cfg, nav_root)
    if not rollouts:
        raise ValueError(f"eval_config has no rollouts: {eval_config_path}")
    metrics = eval_cfg.get("metrics", {}) or {}
    sim_cfg = rollout_config_from_dict(eval_cfg)
    depth_stride = int(metrics.get("depth_inference_stride", 3))
    return rollouts, sim_cfg, depth_stride


def _maybe_freeze_backbone(policy, freeze: bool) -> None:
    if not freeze:
        return
    for name, p in policy.base.named_parameters():
        if any(k in name for k in ("visual", "depth_enc", "fusion")):
            p.requires_grad = False


def _clear_sim_cache(sim_cache: Dict[tuple, Any]) -> None:
    """Drop cached FiGS Simulator instances to reclaim RAM."""
    if not sim_cache:
        return
    sim_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_writable_dir(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PermissionError(
            f"Cannot create {label} ({path}). "
            f"Parent directory may be root-owned: {path.parent}. "
            "On Docker Desktop (Windows), delete the folder on the host and rerun, e.g.:\n"
            "  Remove-Item -Recurse -Force nav_policy\\data\\rl_videos\n"
            "Videos default to checkpoint_dir/videos/ when video_dir is unset."
        ) from exc
    probe = path / ".write_probe"
    try:
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise PermissionError(
            f"Cannot write to {label} ({path}). "
            "On Docker Desktop (Windows), delete the folder on the host and rerun."
        ) from exc


def train(config_path: Path,
          *,
          resume_from: Optional[Path] = None,
          run_tag: Optional[str] = None,
          checkpoint_dir: Optional[Path] = None,
          n_iterations: Optional[int] = None,
          rollouts_per_iteration: Optional[int] = None,
          save_videos: Optional[bool] = None) -> Path:
    cfg = _load_config(config_path)

    nav_root = config_path.resolve().parent.parent
    rl_cfg = cfg.get("rl", {}) or {}
    algorithm = str(rl_cfg.get("algorithm", "ppo")).lower()
    if algorithm not in ("ppo", "sac"):
        raise ValueError(f"rl.algorithm must be 'ppo' or 'sac'; got {algorithm!r}")

    seed = int(rl_cfg.get("seed", 0))
    _set_seed(seed)

    init_ckpt = (nav_root / cfg["checkpoint"]).resolve()
    if resume_from is not None:
        init_ckpt = resume_from.resolve()

    ckpt_dir = checkpoint_dir.resolve() if checkpoint_dir is not None \
        else nav_root / rl_cfg.get("checkpoint_dir", "data/checkpoints_rl")
    _ensure_writable_dir(ckpt_dir, "checkpoint_dir")
    tag = run_tag or rl_cfg.get("run_tag", f"rl_{algorithm}")
    log_path = ckpt_dir / f"{tag}_log.csv"
    episode_log_path = ckpt_dir / f"{tag}_episodes.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_log_std = float(rl_cfg.get("init_log_std", -0.5))
    policy, stats, model_cfg = load_stochastic_from_checkpoint(
        init_ckpt, init_log_std=init_log_std, device=device,
    )
    _maybe_freeze_backbone(policy, bool(rl_cfg.get("freeze_backbone", False)))

    zero_goal_heading = bool(model_cfg.get("train", {}).get("zero_goal_heading", False))

    frame_name = str(cfg.get("frame", "carl"))
    inner = VelocityController(
        hz=20,
        Kv=float(cfg.get("Kv", 2.0)),
        Ka=float(cfg.get("Ka", 5.0)),
        frame_name=frame_name,
    )
    depth_model = DepthAnythingV2Small().to(device) if model_uses_depth(model_cfg) else None
    image_size = int(model_cfg["window"].get("image_size", 224))
    goal_distance_scale = float(model_cfg.get("model", {}).get("goal_distance_scale", 5.0))

    reward_cfg = reward_config_from_dict(rl_cfg)
    rollout_yaml = rl_cfg.get("rollout", {}) or rl_cfg.get("metrics", {}) or {}
    rollout_sim_cfg = rollout_config_from_dict({"metrics": rollout_yaml})
    depth_stride = int(rollout_yaml.get("depth_inference_stride", 3))
    start_cfg = {
        k: rollout_yaml[k]
        for k in (
            "start_mode",
            "random_start_prob",
            "random_start_min_frac",
            "random_start_max_frac",
        )
        if k in rollout_yaml
    }
    collect_rng = np.random.default_rng(seed)
    rollouts = _resolve_rollouts(cfg, nav_root)
    rollouts_per_iter = int(
        rollouts_per_iteration
        if rollouts_per_iteration is not None
        else rl_cfg.get("rollouts_per_iteration", min(4, len(rollouts)))
    )
    ppo_update_every = int(rl_cfg.get("ppo_update_every_episodes", rollouts_per_iter))
    total_episodes_cfg = rl_cfg.get("total_episodes")
    if total_episodes_cfg is not None:
        requested_episodes = int(total_episodes_cfg)
        if ppo_update_every > 0:
            total_episodes = (
                (requested_episodes + ppo_update_every - 1)
                // ppo_update_every
                * ppo_update_every
            )
        else:
            total_episodes = requested_episodes
        if total_episodes != requested_episodes:
            print(
                f"[rl] adjusted total_episodes {requested_episodes} -> {total_episodes} "
                f"(multiple of ppo_update_every={ppo_update_every})",
                flush=True,
            )
        n_iters = max(1, (total_episodes + rollouts_per_iter - 1) // rollouts_per_iter)
    else:
        total_episodes = None
        n_iters = int(n_iterations if n_iterations is not None else rl_cfg.get("n_iterations", 20))

    reuse_simulator = bool(rl_cfg.get("reuse_simulator", True))
    sim_cache_clear_every = int(rl_cfg.get("sim_cache_clear_every_episodes", 0))
    if not reuse_simulator:
        sim_cache_clear_every = 0
    compress_transitions = bool(rl_cfg.get("compress_transitions", True))
    save_every_n_iters = int(rl_cfg.get("save_every_n_iterations", 1))
    store_next_state = algorithm == "sac"
    record_videos = bool(save_videos if save_videos is not None else cfg.get("save_videos", False))
    video_root = nav_root / cfg["video_dir"] if cfg.get("video_dir") else ckpt_dir / "videos"
    if record_videos:
        _ensure_writable_dir(video_root, "video_dir")

    ppo_kw = ppo_config_from_dict(rl_cfg)
    sac_kw = sac_config_from_dict(rl_cfg)
    eval_every_episodes = int(rl_cfg.get("eval_every_episodes", 0))
    select_best_by_eval = eval_every_episodes > 0
    eval_rollouts = rollouts
    eval_sim_cfg = rollout_sim_cfg
    eval_depth_stride = depth_stride
    eval_suite_name = "train"
    eval_subset_size = int(rl_cfg.get("eval_subset_size", 0))
    if select_best_by_eval and rl_cfg.get("eval_config"):
        eval_cfg_path = (nav_root / rl_cfg["eval_config"]).resolve()
        eval_rollouts, eval_sim_cfg, eval_depth_stride = _load_eval_suite(
            eval_cfg_path, nav_root,
        )
        eval_suite_name = eval_cfg_path.stem

    reference_policy = None
    if algorithm == "ppo" and float(ppo_kw.get("ref_kl_coef", 0.0)) > 0.0:
        anchor_cfg = rl_cfg.get("bc_anchor", {}) or {}
        anchor_path = (nav_root / anchor_cfg.get("checkpoint", cfg["checkpoint"])).resolve()
        if anchor_path.resolve() == init_ckpt.resolve():
            reference_policy = policy.frozen_reference_copy()
        else:
            ref_policy, _, _ = load_stochastic_from_checkpoint(
                anchor_path, init_log_std=init_log_std, device=device,
            )
            reference_policy = ref_policy.frozen_reference_copy()
        print(
            f"[rl] BC KL anchor  coef={ppo_kw['ref_kl_coef']}  ref={anchor_path.name}",
            flush=True,
        )

    if algorithm == "ppo":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, policy.parameters()),
            lr=ppo_kw["lr"],
        )
        replay: Optional[ReplayBuffer] = None
        sac_trainer: Optional[SACTrainer] = None
    else:
        optimizer = None
        replay = ReplayBuffer(capacity=sac_kw["replay_capacity"])
        sac_trainer = SACTrainer(policy, device=device, **{
            k: sac_kw[k]
            for k in ("lr", "gamma", "tau", "alpha", "auto_alpha")
        })

    best_return = float("-inf")
    best_eval_success = float("-inf")
    log_fields = [
        "iteration", "algorithm", "mean_return", "mean_steps", "success_rate",
        "policy_loss", "value_loss", "entropy", "approx_kl", "ref_kl",
        "q1_loss", "q2_loss", "alpha",
    ]
    write_header = not log_path.exists()
    write_episode_header = not episode_log_path.exists()
    episode_fields = [
        "global_episode", "iteration", "rollout_name", "semantic_target", "course",
        "return", "steps", "success", "goal_settled", "final_pos_err_m",
        "start_idx", "collision", "termination",
    ]

    if total_episodes is not None:
        print(f"[rl] target episodes={total_episodes}  iters={n_iters}  "
              f"rollouts_per_iter={rollouts_per_iter}  ppo_update_every={ppo_update_every}")
    else:
        print(f"[rl] {len(rollouts)} rollout configs; {rollouts_per_iter} per iteration x {n_iters}")
    clear_msg = (
        f"  clear_every={sim_cache_clear_every} eps"
        if sim_cache_clear_every > 0
        else ""
    )
    print(
        f"[rl] reuse_simulator={reuse_simulator}{clear_msg}  "
        f"compress_transitions={compress_transitions}"
    )
    if record_videos:
        print(f"[rl] saving videos -> {video_root}")
    if select_best_by_eval:
        print(
            f"[rl] eval-based checkpointing every {eval_every_episodes} episodes "
            f"({len(eval_rollouts)} {eval_suite_name} queries, deterministic)",
            flush=True,
        )

    global_episode = 0
    total_rollouts = n_iters * rollouts_per_iter
    _ep_times: List[float] = []
    _train_start = time.monotonic()
    sim_cache: Dict[tuple, Any] = {}

    from torch.utils.tensorboard import SummaryWriter
    tb_dir = ckpt_dir / "tensorboard" / tag
    tb = SummaryWriter(log_dir=str(tb_dir))
    print(f"[rl] tensorboard -> {tb_dir}  (run: tensorboard --logdir {ckpt_dir}/tensorboard)", flush=True)
    controller = RLTrainingController(
        policy, stats, inner,
        image_size=image_size,
        goal_distance_scale=goal_distance_scale,
        device=device,
        depth_model=depth_model,
        deterministic=bool(rl_cfg.get("eval_deterministic", False)),
        depth_inference_stride=depth_stride,
        compress_transitions=compress_transitions,
    )
    pending_episodes: List[Any] = []

    def _maybe_eval_and_save_best(it: int, mean_return: float, success_rate: float, *, force: bool = False) -> None:
        nonlocal best_eval_success
        if not select_best_by_eval or global_episode <= 0:
            return
        if not force and global_episode % eval_every_episodes != 0:
            return

        _active_eval_rollouts = eval_rollouts
        if eval_subset_size > 0 and not force and len(eval_rollouts) > eval_subset_size:
            _active_eval_rollouts = random.sample(eval_rollouts, eval_subset_size)

        eval_out = eval_train_rollouts_goal_success(
            policy,
            stats,
            _active_eval_rollouts,
            eval_sim_cfg,
            image_size=image_size,
            frame_name=frame_name,
            Kv=float(cfg.get("Kv", 2.0)),
            Ka=float(cfg.get("Ka", 5.0)),
            device=device,
            depth_inference_stride=eval_depth_stride,
            depth_model=depth_model,
            zero_goal_heading=zero_goal_heading,
            goal_distance_scale=goal_distance_scale,
        )
        eval_rate = float(eval_out["goal_success_rate"])
        per_query = eval_out.get("per_query", {})
        print(
            f"[rl] {eval_suite_name} eval @ ep {global_episode}: "
            f"goal_success={eval_out['n_success']}/{eval_out['n_rollouts']} ({eval_rate:.0%})",
            flush=True,
        )
        for qname, ok in sorted(per_query.items()):
            print(f"    {qname}: {'ok' if ok else 'FAIL'}", flush=True)
        tb.add_scalar("eval/goal_success_rate", eval_rate, global_episode)
        tb.add_scalar("eval/n_success", eval_out["n_success"], global_episode)
        tb.add_scalar("eval/n_rollouts", eval_out["n_rollouts"], global_episode)

        meta = {
            "iteration": it,
            "global_episode": global_episode,
            "algorithm": algorithm,
            "mean_return": mean_return,
            "success_rate": success_rate,
            "eval_goal_success_rate": eval_rate,
            "eval_suite": eval_suite_name,
            "eval_per_query": per_query,
            "init_checkpoint": str(init_ckpt),
            "run_tag": tag,
        }
        if eval_rate > best_eval_success:
            best_eval_success = eval_rate
            best_path = ckpt_dir / f"{tag}_best.pt"
            save_rl_checkpoint(best_path, policy, stats, model_cfg, meta)
            print(
                f"[rl] new eval best  success={best_eval_success:.0%}  -> {best_path.name}",
                flush=True,
            )

    def _run_training_step(it: int, episodes: List[Any]) -> None:
        nonlocal best_return, write_header
        if not episodes:
            return

        mean_return = float(np.mean([e.total_return for e in episodes]))
        mean_steps = float(np.mean([e.n_steps for e in episodes]))
        success_rate = float(np.mean([1.0 if e.success else 0.0 for e in episodes]))

        row = {
            "iteration": it,
            "algorithm": algorithm,
            "mean_return": mean_return,
            "mean_steps": mean_steps,
            "success_rate": success_rate,
            "policy_loss": "",
            "value_loss": "",
            "entropy": "",
            "approx_kl": "",
            "ref_kl": "",
            "q1_loss": "",
            "q2_loss": "",
            "alpha": "",
        }

        if algorithm == "ppo":
            assert optimizer is not None
            buffer = episodes_to_buffer(
                episodes,
                gamma=ppo_kw["gamma"],
                gae_lambda=ppo_kw["gae_lambda"],
                buffer_fp16=compress_transitions,
            )
            stats_out = ppo_update(
                policy, buffer, optimizer, device=device,
                clip_eps=ppo_kw["clip_eps"],
                value_coef=ppo_kw["value_coef"],
                entropy_coef=ppo_kw["entropy_coef"],
                ref_kl_coef=ppo_kw["ref_kl_coef"],
                reference_policy=reference_policy,
                max_grad_norm=ppo_kw["max_grad_norm"],
                n_epochs=ppo_kw["n_epochs"],
                target_kl=ppo_kw["target_kl"],
                batch_size=min(ppo_kw["batch_size"], len(buffer)),
            )
            del buffer
            row.update({
                "policy_loss": stats_out.policy_loss,
                "value_loss": stats_out.value_loss,
                "entropy": stats_out.entropy,
                "approx_kl": stats_out.approx_kl,
                "ref_kl": stats_out.ref_kl,
            })
        else:
            assert replay is not None and sac_trainer is not None
            for ep in episodes:
                replay.add_episode(ep)
            sac_stats = None
            if len(replay) >= sac_kw["batch_size"]:
                for _ in range(sac_kw["updates_per_iter"]):
                    batch = replay.sample(sac_kw["batch_size"])
                    sac_stats = sac_trainer.update(batch)
            if sac_stats is not None:
                row.update({
                    "policy_loss": sac_stats.policy_loss,
                    "q1_loss": sac_stats.q1_loss,
                    "q2_loss": sac_stats.q2_loss,
                    "alpha": sac_stats.alpha,
                })

        meta = {
            "iteration": it,
            "global_episode": global_episode,
            "algorithm": algorithm,
            "mean_return": mean_return,
            "success_rate": success_rate,
            "init_checkpoint": str(init_ckpt),
            "run_tag": tag,
        }
        if (it + 1) % save_every_n_iters == 0 or (
            total_episodes is not None and global_episode >= total_episodes
        ):
            latest_path = ckpt_dir / f"{tag}_latest.pt"
            save_rl_checkpoint(latest_path, policy, stats, model_cfg, meta)
            if not select_best_by_eval and mean_return > best_return:
                best_return = mean_return
                best_path = ckpt_dir / f"{tag}_best.pt"
                save_rl_checkpoint(best_path, policy, stats, model_cfg, meta)
                print(f"[rl] new best  return={best_return:.2f}  -> {best_path.name}")

        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=log_fields)
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)

        tb.add_scalar("train/mean_return", mean_return, global_episode)
        tb.add_scalar("train/mean_steps", mean_steps, global_episode)
        tb.add_scalar("train/success_rate", success_rate, global_episode)
        if algorithm == "ppo" and row.get("policy_loss") != "":
            tb.add_scalar("ppo/policy_loss", float(row["policy_loss"]), global_episode)
            tb.add_scalar("ppo/value_loss", float(row["value_loss"]), global_episode)
            tb.add_scalar("ppo/entropy", float(row["entropy"]), global_episode)
            tb.add_scalar("ppo/approx_kl", float(row["approx_kl"]), global_episode)
            if row.get("ref_kl") != "":
                tb.add_scalar("ppo/ref_kl", float(row["ref_kl"]), global_episode)
        elif algorithm == "sac" and row.get("q1_loss") != "":
            tb.add_scalar("sac/policy_loss", float(row["policy_loss"]), global_episode)
            tb.add_scalar("sac/q1_loss", float(row["q1_loss"]), global_episode)
            tb.add_scalar("sac/q2_loss", float(row["q2_loss"]), global_episode)
            tb.add_scalar("sac/alpha", float(row["alpha"]), global_episode)

        print(
            f"[rl] iter {it}/{n_iters}  episodes={global_episode}  "
            f"return={mean_return:.2f}  success={success_rate:.0%}",
            flush=True,
        )

        del episodes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[rl] algorithm={algorithm}  init={init_ckpt.name}  device={device}")

    for it in range(n_iters):
        if total_episodes is not None and global_episode >= total_episodes:
            break

        n_collect = rollouts_per_iter
        if total_episodes is not None:
            n_collect = min(n_collect, total_episodes - global_episode)
        if n_collect <= 0:
            break

        subset = list(rollouts)
        random.shuffle(subset)
        if rollouts_per_iter < len(subset):
            subset = subset[:n_collect]
        iter_collected = 0

        for rcfg in subset:
            if total_episodes is not None and global_episode >= total_episodes:
                break

            expert_preview = load_expert_setup(
                Path(rcfg["setup_from"]).resolve(),
                int(rcfg.get("sub_idx", 0)),
            )
            semantic_slug = expert_semantic_slug(expert_preview)

            video_path = None
            if record_videos:
                video_path = rollout_video_path(
                    video_root,
                    iteration=it,
                    global_episode=global_episode,
                    rollout_name=str(rcfg["name"]),
                    semantic_slug=semantic_slug,
                )

            try:
                _ep_t0 = time.monotonic()
                ep = collect_episode(
                    rcfg, controller, reward_cfg, rollout_sim_cfg,
                    Kv=float(cfg.get("Kv", 2.0)),
                    Ka=float(cfg.get("Ka", 5.0)),
                    video_path=video_path,
                    start_cfg=start_cfg,
                    rng=collect_rng,
                    sim_cache=sim_cache if reuse_simulator else None,
                    store_next_state=store_next_state,
                )
                _ep_elapsed = time.monotonic() - _ep_t0
                _ep_times.append(_ep_elapsed)
                pending_episodes.append(ep)
                global_episode += 1
                iter_collected += 1
                _done = global_episode
                _remaining = total_rollouts - _done
                _avg_ep_s = sum(_ep_times[-20:]) / len(_ep_times[-20:])
                _eta_s = _avg_ep_s * _remaining
                _eta_str = (
                    f"{int(_eta_s // 3600)}h{int(_eta_s % 3600 // 60):02d}m"
                    if _eta_s >= 3600
                    else f"{int(_eta_s // 60)}m{int(_eta_s % 60):02d}s"
                )
                print(
                    f"  [ep {global_episode:05d}/{total_rollouts}] "
                    f"[iter {it+1}/{n_iters}] "
                    f"[{rcfg['name']}|{ep.semantic_target}] "
                    f"return={ep.total_return:.2f}  steps={ep.n_steps}  "
                    f"success={ep.success}  settled={ep.goal_settled}  "
                    f"final_dist={ep.final_pos_err_m:.2f}m  "
                    f"collision={ep.collision}  end={ep.termination}  "
                    f"ep_time={_ep_elapsed:.1f}s  eta={_eta_str}",
                    flush=True,
                )
                tb.add_scalar("episode/return", ep.total_return, global_episode)
                tb.add_scalar("episode/steps", ep.n_steps, global_episode)
                tb.add_scalar("episode/success", float(ep.success), global_episode)
                tb.add_scalar("episode/collision", float(ep.collision), global_episode)
                tb.add_scalar("episode/final_dist_xy_m", ep.final_pos_err_m, global_episode)
                tb.add_scalar("episode/final_dist_3d_m", ep.final_pos_err_3d_m, global_episode)
                tb.add_scalar("episode/final_yaw_err_rad", ep.final_yaw_err_rad, global_episode)
                tb.add_scalar("episode/final_speed_mps", ep.final_speed_mps, global_episode)
                tb.add_scalar("episode/final_cmd_body_rate", ep.final_cmd_body_rate, global_episode)
                tb.add_scalar("episode/ep_time_s", _ep_elapsed, global_episode)

                if not ep.success:
                    _settle_cfg = rollout_sim_cfg
                    _pos_ok  = ep.final_pos_err_3d_m  < _settle_cfg.goal_position_tol_m
                    _yaw_ok  = ep.final_yaw_err_rad   < _settle_cfg.goal_yaw_tol_rad
                    _vel_ok  = ep.final_speed_mps     <= _settle_cfg.settle_vel_mps
                    _rate_ok = ep.final_cmd_body_rate <= _settle_cfg.settle_body_rate_radps
                    _yr_ok   = ep.final_cmd_yaw_rate  <= _settle_cfg.settle_yaw_rate_radps
                    def _fmt(label, val, thr, passed, unit=""):
                        mark = "OK" if passed else "FAIL"
                        return f"{label}={val:.3f}{unit}(thr={thr}{unit}) [{mark}]"
                    print(
                        f"    -> failure breakdown:"
                        f"  {_fmt('pos3d',  ep.final_pos_err_3d_m,  _settle_cfg.goal_position_tol_m, _pos_ok,  'm')}"
                        f"  {_fmt('yaw',    ep.final_yaw_err_rad,   _settle_cfg.goal_yaw_tol_rad,    _yaw_ok,  'r')}"
                        f"  {_fmt('speed',  ep.final_speed_mps,     _settle_cfg.settle_vel_mps,      _vel_ok,  'm/s')}"
                        f"  {_fmt('brate',  ep.final_cmd_body_rate, _settle_cfg.settle_body_rate_radps, _rate_ok, 'r/s')}"
                        f"  {_fmt('yrate',  ep.final_cmd_yaw_rate,  _settle_cfg.settle_yaw_rate_radps,  _yr_ok,   'r/s')}"
                        f"  collision={ep.collision}",
                        flush=True,
                    )
                with open(episode_log_path, "a", newline="") as ef:
                    ew = csv.DictWriter(ef, fieldnames=episode_fields)
                    if write_episode_header:
                        ew.writeheader()
                        write_episode_header = False
                    ew.writerow({
                        "global_episode": global_episode,
                        "iteration": it,
                        "rollout_name": rcfg["name"],
                        "semantic_target": ep.semantic_target,
                        "course": ep.course,
                        "return": ep.total_return,
                        "steps": ep.n_steps,
                        "success": int(ep.success),
                        "goal_settled": int(ep.goal_settled),
                        "final_pos_err_m": ep.final_pos_err_m,
                        "start_idx": ep.start_index,
                        "collision": int(ep.collision),
                        "termination": ep.termination,
                    })

                if algorithm == "ppo" and len(pending_episodes) >= ppo_update_every:
                    batch_eps = list(pending_episodes)
                    pending_episodes.clear()
                    _run_training_step(it, batch_eps)

                if (
                    reuse_simulator
                    and sim_cache_clear_every > 0
                    and global_episode % sim_cache_clear_every == 0
                ):
                    print(
                        f"[rl] clearing sim cache after ep {global_episode} "
                        f"(reload on next episode)",
                        flush=True,
                    )
                    _clear_sim_cache(sim_cache)

                _maybe_eval_and_save_best(it, ep.total_return, float(ep.success))
            except Exception as exc:
                print(f"  [{rcfg.get('name', '?')}] FAILED: {exc}", file=sys.stderr)

        if iter_collected == 0:
            print(f"[rl] iter {it}: no episodes collected; skipping update", file=sys.stderr)

    if pending_episodes:
        if algorithm == "ppo" and len(pending_episodes) >= ppo_update_every:
            batch_eps = list(pending_episodes)
            pending_episodes.clear()
            _run_training_step(max(n_iters - 1, 0), batch_eps)
        elif pending_episodes:
            print(
                f"[rl] skipping final partial batch ({len(pending_episodes)} episodes; "
                f"need {ppo_update_every} for PPO update)",
                flush=True,
            )
            pending_episodes.clear()

    if global_episode > 0 and select_best_by_eval and global_episode % eval_every_episodes != 0:
        _maybe_eval_and_save_best(max(n_iters - 1, 0), 0.0, 0.0, force=True)

    if global_episode > 0:
        latest_path = ckpt_dir / f"{tag}_latest.pt"
        if not latest_path.exists():
            meta = {
                "iteration": n_iters - 1,
                "global_episode": global_episode,
                "algorithm": algorithm,
                "init_checkpoint": str(init_ckpt),
                "run_tag": tag,
            }
            save_rl_checkpoint(latest_path, policy, stats, model_cfg, meta)

    summary = {
        "run_tag": tag,
        "algorithm": algorithm,
        "total_episodes": global_episode,
        "best_mean_return": best_return,
        "best_eval_goal_success_rate": best_eval_success if select_best_by_eval else None,
        "init_checkpoint": str(init_ckpt),
        "episode_log": str(episode_log_path),
        "iteration_log": str(log_path),
    }
    with open(ckpt_dir / f"{tag}_summary.json", "w") as sf:
        json.dump(summary, sf, indent=2)

    best_path = ckpt_dir / f"{tag}_best.pt"
    latest_path = ckpt_dir / f"{tag}_latest.pt"
    if global_episode > 0 and not best_path.exists() and latest_path.exists():
        import shutil
        shutil.copy2(latest_path, best_path)
        print(f"[rl] no eval best saved; copied latest -> {best_path.name}", flush=True)

    tb.close()
    print(f"[rl] done. Best checkpoint: {best_path}")
    return best_path


def main() -> None:
    p = argparse.ArgumentParser(description="RL fine-tuning (PPO or SAC) from BC/DAgger checkpoint.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Optional RL checkpoint to resume (uses rl head + base weights).")
    p.add_argument("--run-tag", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Override rl.checkpoint_dir from the config.")
    p.add_argument("--save-videos", action="store_true",
                   help="Save rollout mp4s under video_dir (or checkpoint_dir/videos/<tag>).")
    p.add_argument("--n-iterations", type=int, default=None,
                   help="Override rl.n_iterations (use 1 for smoke tests).")
    p.add_argument("--rollouts-per-iteration", type=int, default=None,
                   help="Override rl.rollouts_per_iteration (use 2 for smoke tests).")
    args = p.parse_args()
    train(
        args.config,
        resume_from=args.resume_from,
        run_tag=args.run_tag,
        checkpoint_dir=args.checkpoint_dir,
        n_iterations=args.n_iterations,
        rollouts_per_iteration=args.rollouts_per_iteration,
        save_videos=args.save_videos if args.save_videos else None,
    )


if __name__ == "__main__":
    main()
