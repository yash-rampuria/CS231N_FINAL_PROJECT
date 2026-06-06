"""Experience buffers for PPO and SAC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from nav_policy.rl.rollout import EpisodeBatch, Transition


@dataclass
class RolloutBuffer:
    """Flattened on-policy buffer with GAE advantages."""

    rgb: torch.Tensor
    goal: torch.Tensor
    depth: Optional[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return int(self.rewards.shape[0])


def _compute_gae(rewards: np.ndarray,
                 values: np.ndarray,
                 dones: np.ndarray,
                 *,
                 gamma: float,
                 gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_non_terminal = 1.0 - float(dones[t])
        next_value = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def _subsample_transitions(transitions: List[Transition],
                           max_steps: int) -> List[Transition]:
    if max_steps <= 0 or len(transitions) <= max_steps:
        return transitions
    idx = np.linspace(0, len(transitions) - 1, max_steps, dtype=np.int64)
    out = [transitions[int(i)] for i in idx]
    out[-1].done = True
    return out


def _cat_obs(tensors: List[torch.Tensor], *, buffer_fp16: bool) -> torch.Tensor:
    out = torch.cat(tensors, dim=0)
    if buffer_fp16:
        return out.half() if out.dtype != torch.float16 else out
    return out.float() if out.dtype != torch.float32 else out


def episodes_to_buffer(episodes: List[EpisodeBatch],
                       *,
                       gamma: float = 0.99,
                       gae_lambda: float = 0.95,
                       max_steps_per_episode: int = 0,
                       buffer_fp16: bool = True) -> RolloutBuffer:
    transitions: List[Transition] = []
    for ep in episodes:
        trs = _subsample_transitions(ep.transitions, max_steps_per_episode)
        if trs:
            trs[-1].done = True
        transitions.extend(trs)
        ep.transitions.clear()
    if not transitions:
        raise RuntimeError("no transitions collected")

    rgb = _cat_obs([t.rgb for t in transitions], buffer_fp16=buffer_fp16)
    goal = torch.cat([t.goal for t in transitions], dim=0)
    depth = (
        _cat_obs([t.depth for t in transitions if t.depth is not None], buffer_fp16=buffer_fp16)
        if transitions[0].depth is not None
        else None
    )
    actions = torch.cat([t.action_z for t in transitions], dim=0)
    log_probs = torch.cat([t.log_prob for t in transitions], dim=0)
    values = torch.cat([t.value for t in transitions], dim=0)
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
    dones = torch.tensor([float(t.done) for t in transitions], dtype=torch.float32)

    adv_np, ret_np = _compute_gae(
        rewards.numpy(),
        values.numpy(),
        dones.numpy(),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    advantages = torch.from_numpy(adv_np)
    returns = torch.from_numpy(ret_np)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return RolloutBuffer(
        rgb=rgb,
        goal=goal,
        depth=depth,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
        advantages=advantages,
        returns=returns,
    )


class ReplayBuffer:
    """Off-policy replay buffer for SAC."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self._data: List[Transition] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self._data)

    def add_episode(self, episode: EpisodeBatch) -> None:
        for tr in episode.transitions:
            if len(self._data) < self.capacity:
                self._data.append(tr)
            else:
                self._data[self._pos] = tr
                self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> dict:
        if len(self._data) < batch_size:
            raise RuntimeError(
                f"replay buffer has {len(self._data)} samples; need {batch_size}"
            )
        idx = np.random.choice(len(self._data), size=batch_size, replace=False)
        items = [self._data[i] for i in idx]
        depth = (
            torch.cat([t.depth for t in items if t.depth is not None], dim=0)
            if items[0].depth is not None
            else None
        )
        next_depth = (
            torch.cat([t.next_depth for t in items if t.next_depth is not None], dim=0)
            if items[0].next_depth is not None
            else None
        )
        return {
            "rgb": torch.cat([t.rgb for t in items], dim=0),
            "goal": torch.cat([t.goal for t in items], dim=0),
            "depth": depth,
            "next_rgb": torch.cat([t.next_rgb for t in items], dim=0),
            "next_goal": torch.cat([t.next_goal for t in items], dim=0),
            "next_depth": next_depth,
            "actions": torch.cat([t.action_z for t in items], dim=0),
            "rewards": torch.tensor([t.reward for t in items], dtype=torch.float32),
            "dones": torch.tensor([float(t.done) for t in items], dtype=torch.float32),
        }
