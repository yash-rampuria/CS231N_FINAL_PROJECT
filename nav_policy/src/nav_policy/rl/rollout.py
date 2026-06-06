"""On-policy rollout collection inside FiGS for RL fine-tuning."""



from __future__ import annotations



import gc
import sys

from dataclasses import dataclass, field

from pathlib import Path

from typing import Any, Dict, List, Optional



import numpy as np

import torch

import torch.nn as nn

from scipy.spatial.transform import Rotation



from figs.control.velocity_controller import VelocityController



from nav_policy.data.normalization import CommandStats

from nav_policy.deploy.frame_buffer import FrameBuffer

from nav_policy.evaluate.closed_loop import ExpertRef, load_expert_setup
from nav_policy.evaluate.sim_rollout import RolloutConfig, simulate_with_early_exit

from nav_policy.model.depth_estimator import DepthAnythingV2Small

from nav_policy.rl.rewards import compute_episode_rewards, reward_config_from_dict

from nav_policy.rl.stochastic_policy import StochasticVelocityPolicy





@dataclass

class Transition:

    rgb: torch.Tensor

    goal: torch.Tensor

    depth: Optional[torch.Tensor]

    action_z: torch.Tensor

    log_prob: torch.Tensor

    value: torch.Tensor

    reward: float = 0.0

    done: bool = False

    next_rgb: Optional[torch.Tensor] = None

    next_goal: Optional[torch.Tensor] = None

    next_depth: Optional[torch.Tensor] = None





@dataclass

class EpisodeBatch:

    transitions: List[Transition] = field(default_factory=list)

    rollout_name: str = ""

    total_return: float = 0.0

    n_steps: int = 0

    success: bool = False

    collision: bool = False

    termination: str = "timeout"

    final_pos_err_m: float = float("inf")

    final_pos_err_3d_m: float = float("inf")

    final_yaw_err_rad: float = float("inf")

    final_speed_mps: float = float("inf")

    final_cmd_body_rate: float = float("inf")

    final_cmd_yaw_rate: float = float("inf")

    goal_settled: bool = False

    start_index: int = 0

    semantic_target: str = ""

    course: str = ""





class RLTrainingController:

    """FiGS-compatible controller that samples stochastic actions and logs transitions."""



    def __init__(self,

                 policy: StochasticVelocityPolicy,

                 stats: CommandStats,

                 inner: VelocityController,

                 *,

                 image_size: int = 224,

                 goal_distance_scale: float = 5.0,

                 device: Optional[torch.device] = None,

                 depth_model: Optional[DepthAnythingV2Small] = None,

                 deterministic: bool = False,

                 depth_inference_stride: int = 3,

                 compress_transitions: bool = True) -> None:

        self.policy = policy

        self.stats = stats

        self.inner = inner

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_size = image_size

        self.goal_distance_scale = goal_distance_scale

        self.deterministic = deterministic

        self.uses_depth = bool(getattr(policy.base, "use_depth", False))

        self._depth_model = depth_model

        if self.uses_depth and self._depth_model is None:

            self._depth_model = DepthAnythingV2Small().to(self.device)

        elif self._depth_model is not None:

            self._depth_model = self._depth_model.eval().to(self.device)



        if depth_inference_stride < 1:

            raise ValueError(f"depth_inference_stride must be >= 1; got {depth_inference_stride}")

        self.depth_inference_stride = int(depth_inference_stride)

        self.compress_transitions = bool(compress_transitions)

        self._control_step = 0

        self._last_depth_hw: Optional[np.ndarray] = None



        self.buf = FrameBuffer(T=int(policy.base.T), image_size=image_size, device=self.device)

        self._goal_pos_xy: Optional[np.ndarray] = None

        self._goal_input_dim = int(policy.base.goal_input_dim)

        self.hz = inner.hz

        self.nzcr = None

        self.name = "RLTrainingController"



        self._states: List[np.ndarray] = []

        self._actions: List[np.ndarray] = []

        self._episode: List[Transition] = []

        self._collision_steps: List[bool] = []



    def set_goal(self, goal_pos_xy: np.ndarray) -> None:

        self._goal_pos_xy = np.asarray(goal_pos_xy, dtype=np.float64).ravel()[:2]



    def reset(self, goal_pos_xy: Optional[np.ndarray] = None) -> None:

        self.buf.reset()

        self._control_step = 0

        self._last_depth_hw = None

        self._states.clear()

        self._actions.clear()

        self._episode.clear()

        self._collision_steps.clear()

        if goal_pos_xy is not None:

            self.set_goal(goal_pos_xy)



    def _compute_goal_vector(self, xcr: np.ndarray) -> np.ndarray:

        if self._goal_pos_xy is None:

            return np.array([1.0, 0.0, 0.0], dtype=np.float32)[: self._goal_input_dim]

        delta = self._goal_pos_xy - xcr[0:2].astype(np.float64)

        norm = float(np.linalg.norm(delta))

        if norm < 1e-6:

            heading = np.array([1.0, 0.0], dtype=np.float32)

            d_norm = 0.0

        else:

            heading = (delta / norm).astype(np.float32)

            d_norm = float(norm / self.goal_distance_scale)

        if self._goal_input_dim == 2:

            return heading

        return np.array([heading[0], heading[1], d_norm], dtype=np.float32)



    @torch.no_grad()

    def _infer_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:

        if not self.uses_depth or self._depth_model is None:

            return None

        t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)

        d = self._depth_model(t)

        return d[0, 0].cpu().numpy()



    def note_collision(self, step_idx: int) -> None:

        while len(self._collision_steps) <= step_idx:

            self._collision_steps.append(False)

        self._collision_steps[step_idx] = True



    def control(self,

                tcr: float,

                xcr: np.ndarray,

                upr: Any,

                obj: Any,

                icr: np.ndarray,

                zcr: Any):

        self._states.append(np.asarray(xcr, dtype=np.float64).copy())
        self._collision_steps.append(False)



        goal_np = self._compute_goal_vector(xcr)

        if self.uses_depth:

            if (

                self._last_depth_hw is None

                or self._control_step % self.depth_inference_stride == 0

            ):

                self._last_depth_hw = self._infer_depth(icr)

            depth_hw = self._last_depth_hw

        else:

            depth_hw = None

        self._control_step += 1

        self.buf.push(icr, depth_hw=depth_hw)

        rgb_seq, depth_seq = self.buf.tensors()

        goal_t = torch.from_numpy(goal_np).unsqueeze(0).to(self.device, non_blocking=True)



        was_training = self.policy.training

        self.policy.eval()

        action_z, log_prob, value = self.policy.act(

            rgb_seq, goal_t, depth_seq, deterministic=self.deterministic,

        )

        if was_training:

            self.policy.train()



        cmd0 = self.stats.destandardize(
            action_z.view(1, 1, -1)
        )[0, 0].detach().cpu().numpy().astype(np.float64)

        self._actions.append(cmd0.copy())



        rgb_cpu = rgb_seq.detach().cpu()
        if self.compress_transitions:
            rgb_cpu = rgb_cpu.half()
        depth_cpu = None
        if depth_seq is not None:
            depth_cpu = depth_seq.detach().cpu()
            if self.compress_transitions:
                depth_cpu = depth_cpu.half()

        self._episode.append(

            Transition(

                rgb=rgb_cpu,

                goal=goal_t.detach().cpu(),

                depth=depth_cpu,

                action_z=action_z.detach().cpu(),

                log_prob=log_prob.detach().cpu(),

                value=value.detach().cpu(),

            )

        )



        ucr, _, _, _ = self.inner.control(

            tcr=tcr, xcr=xcr, upr=upr, obj=cmd0, icr=None, zcr=None,

        )

        adv = cmd0.astype(np.float64)

        tsol = np.zeros(4, dtype=np.float64)

        return ucr, None, adv, tsol



    def finalize_episode(self,

                         expert: ExpertRef,

                         reward_cfg: Dict,

                         rollout_result: Optional[Any] = None,

                         *, store_next_state: bool = False) -> EpisodeBatch:

        goal_yaw = float(

            Rotation.from_quat(expert.Xro[6:10, -1]).as_euler("xyz", degrees=False)[2]

        )

        if rollout_result is not None and rollout_result.collision:

            cstep = int(rollout_result.collision_step)

            if 0 <= cstep < len(self._collision_steps):

                self._collision_steps[cstep] = True

        goal_settled = bool(
            rollout_result is not None and getattr(rollout_result, "goal_reached", False)
        )

        collision = bool(rollout_result.collision) if rollout_result is not None else any(self._collision_steps)
        termination = str(rollout_result.termination) if rollout_result is not None else "timeout"

        rewards = compute_episode_rewards(
            self._states,
            expert.Xro[0:2, -1],
            expert.Xro[0:3, :],
            goal_yaw=goal_yaw,
            collision_steps=self._collision_steps,
            actions=self._actions,
            goal_settled=goal_settled,
            termination=termination,
            **reward_cfg,
        )

        for tr, r in zip(self._episode, rewards):

            tr.reward = float(r)

            tr.done = False

        if store_next_state:
            for i, tr in enumerate(self._episode):
                if i + 1 < len(self._episode):
                    nxt = self._episode[i + 1]
                    tr.next_rgb = nxt.rgb
                    tr.next_goal = nxt.goal
                    tr.next_depth = nxt.depth
                else:
                    tr.next_rgb = tr.rgb
                    tr.next_goal = tr.goal
                    tr.next_depth = tr.depth

        if self._episode:

            self._episode[-1].done = True



        total = float(sum(rewards))

        final_pos_err = float(
            np.linalg.norm(self._states[-1][0:2] - expert.Xro[0:2, -1])
        ) if self._states else float("inf")

        final_pos_err_3d = float(
            np.linalg.norm(self._states[-1][0:3] - expert.Xro[0:3, -1])
        ) if self._states else float("inf")

        final_yaw_err = float("inf")
        final_speed = float("inf")
        if self._states:
            yaw = Rotation.from_quat(self._states[-1][6:10]).as_euler("xyz", degrees=False)[2]
            final_yaw_err = float(abs((yaw - goal_yaw + np.pi) % (2 * np.pi) - np.pi))
            final_speed = float(np.linalg.norm(self._states[-1][3:6]))

        final_cmd_body_rate = float("inf")
        final_cmd_yaw_rate = float("inf")
        if self._actions:
            last_u = np.asarray(self._actions[-1], dtype=np.float64).ravel()[:4]
            final_cmd_body_rate = float(np.linalg.norm(last_u[1:4]))
            final_cmd_yaw_rate = float(abs(last_u[3]))

        success = goal_settled and not collision

        return EpisodeBatch(
            transitions=list(self._episode),
            total_return=total,
            n_steps=len(self._episode),
            success=success,
            collision=collision,
            termination=termination,
            final_pos_err_m=final_pos_err,
            final_pos_err_3d_m=final_pos_err_3d,
            final_yaw_err_rad=final_yaw_err,
            final_speed_mps=final_speed,
            final_cmd_body_rate=final_cmd_body_rate,
            final_cmd_yaw_rate=final_cmd_yaw_rate,
            goal_settled=goal_settled,
        )





def resolve_episode_start(
    expert: ExpertRef,
    rollout_cfg: dict,
    start_cfg: dict,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, int]:
    """
    Choose sim start time/state for an RL episode.

    start_mode:
      expert_x0          — full val traj start (default)
      random_along_traj  — random index along the full expert path
      mixed              — random_start_prob chance of random_along_traj
    Goal is always the terminal pose of the full expert trajectory.
    """
    mode = str(
        rollout_cfg.get("start_mode", start_cfg.get("start_mode", "expert_x0"))
    ).lower()
    if mode == "mixed":
        prob = float(start_cfg.get("random_start_prob", 0.3))
        mode = "random_along_traj" if rng.random() < prob else "expert_x0"

    n = int(expert.Xro.shape[1])
    if mode == "random_along_traj" and n > 1:
        min_frac = float(start_cfg.get("random_start_min_frac", 0.0))
        max_frac = float(start_cfg.get("random_start_max_frac", 0.85))
        i_min = int(min_frac * (n - 1))
        i_max = max(i_min, int(max_frac * (n - 1)))
        idx = int(rng.integers(i_min, i_max + 1))
        return float(expert.Tro[idx]), expert.Xro[:, idx].copy(), idx

    return float(expert.t0), expert.x0.copy(), 0


def collect_episode(
    rollout_cfg: dict,
    controller: RLTrainingController,
    reward_cfg: Dict,
    rollout_sim_cfg: RolloutConfig,
    *,
    Kv: float = 2.0,
    Ka: float = 5.0,
    video_path: Optional[Path] = None,
    start_cfg: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    sim_cache: Optional[Dict[tuple, Any]] = None,
    store_next_state: bool = False,
) -> EpisodeBatch:

    from figs.simulator import Simulator

    from nav_policy.rl.paths import expert_semantic_slug



    name = rollout_cfg["name"]

    scene = rollout_cfg["scene"]

    rollout = rollout_cfg.get("rollout", "baseline")

    frame = rollout_cfg.get("frame", "carl")

    setup_from = Path(rollout_cfg["setup_from"]).resolve()

    sub_idx = int(rollout_cfg.get("sub_idx", 0))



    expert = load_expert_setup(setup_from, sub_idx)

    start_cfg = start_cfg or {}
    rng = rng if rng is not None else np.random.default_rng()
    t0, x0, start_index = resolve_episode_start(expert, rollout_cfg, start_cfg, rng)

    goal_pos_xy = expert.Xro[0:2, -1].astype(np.float64)

    goal_xyz = expert.Xro[0:3, -1].astype(np.float64)

    goal_yaw = float(

        Rotation.from_quat(expert.Xro[6:10, -1]).as_euler("xyz", degrees=False)[2]

    )

    controller.reset(goal_pos_xy=goal_pos_xy)

    cache_key = (scene, rollout, frame)
    owned_sim = False
    if sim_cache is not None and cache_key in sim_cache:
        sim = sim_cache[cache_key]
    else:
        sim = Simulator(scene, rollout, frame)
        owned_sim = sim_cache is None
        if sim_cache is not None:
            sim_cache[cache_key] = sim

    rollout_result = None

    try:

        rollout_result = simulate_with_early_exit(

            sim,

            controller,

            t0=t0,

            expert_tf=expert.tf,

            x0=x0,

            goal_xyz=goal_xyz,

            goal_yaw=goal_yaw,

            cfg=rollout_sim_cfg,

            goal_xy=goal_pos_xy,

            frame_name=frame,

            Kv=Kv,

            Ka=Ka,

        )

    finally:

        if owned_sim:
            del sim
            gc.collect()



    batch = controller.finalize_episode(
        expert, reward_cfg, rollout_result, store_next_state=store_next_state,
    )
    batch.rollout_name = name
    batch.start_index = start_index
    batch.semantic_target = expert_semantic_slug(expert)
    batch.course = expert.course

    if video_path is not None and rollout_result is not None and "rgb" in rollout_result.Imgs:
        from nav_policy.evaluate.closed_loop import _save_video

        frames = rollout_result.Imgs["rgb"]
        if rollout_result.warmup_rgb is not None and rollout_result.warmup_rgb.shape[0] > 0:
            frames = np.concatenate([rollout_result.warmup_rgb, frames], axis=0)
        try:
            _save_video(frames, video_path, fps=int(controller.hz))
            print(f"  [{name}] video -> {video_path}", flush=True)
        except OSError as exc:
            print(f"  [{name}] video FAILED ({video_path}): {exc}", file=sys.stderr)

    return batch





__all__ = [

    "Transition",

    "EpisodeBatch",

    "RLTrainingController",

    "collect_episode",

    "resolve_episode_start",

    "reward_config_from_dict",

]


