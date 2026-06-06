"""Gaussian actor + value critic for RL fine-tuning on BC/DAgger initializers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from nav_policy.data.normalization import CommandStats
from nav_policy.model.factory import build_model


class StochasticVelocityPolicy(nn.Module):
    """
    Wraps a deterministic BC policy with a diagonal Gaussian exploration head
    and a scalar value critic on the shared latent features.

    Actions are sampled in z-score space (same as BC training targets); the
    caller de-standardizes with ``CommandStats`` before sending to FiGS.
    """

    def __init__(self,
                 base: nn.Module,
                 init_log_std: float = -0.5,
                 critic_hidden: int = 256) -> None:
        super().__init__()
        self.base = base
        self.cmd_dim = int(base.cmd_dim)
        # FiLM conditioning changes the latent size; fall back to the original
        # concat dim for models that predate the latent_dim attribute.
        latent_dim = int(getattr(base, "latent_dim",
                                 int(base.gru_hidden) + int(base.goal_emb_dim)))
        self.log_std = nn.Parameter(
            torch.full((self.cmd_dim,), float(init_log_std))
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, critic_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(critic_hidden, 1),
        )
        self.use_depth = bool(getattr(base, "use_depth", False))

    def _latent(self,
                rgb_seq: torch.Tensor,
                goal: torch.Tensor,
                depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_depth:
            return self.base.forward_latent(rgb_seq, goal, depth_seq)
        return self.base.forward_latent(rgb_seq, goal)

    def _mean(self,
              rgb_seq: torch.Tensor,
              goal: torch.Tensor,
              depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_depth:
            return self.base.predict_mean_first(rgb_seq, goal, depth_seq)
        return self.base.predict_mean_first(rgb_seq, goal)

    def _distribution(self, mean: torch.Tensor) -> Normal:
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self,
            rgb_seq: torch.Tensor,
            goal: torch.Tensor,
            depth_seq: Optional[torch.Tensor] = None,
            deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample first-step action; returns (action_z [B,4], log_prob [B], value [B])."""
        latent = self._latent(rgb_seq, goal, depth_seq)
        mean = self._mean(rgb_seq, goal, depth_seq)
        dist = self._distribution(mean)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(latent).squeeze(-1)
        return action, log_prob, value

    def evaluate(self,
                 rgb_seq: torch.Tensor,
                 goal: torch.Tensor,
                 action_z: torch.Tensor,
                 depth_seq: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO/SAC update: log_prob, value, entropy for stored actions."""
        latent = self._latent(rgb_seq, goal, depth_seq)
        mean = self._mean(rgb_seq, goal, depth_seq)
        dist = self._distribution(mean)
        log_prob = dist.log_prob(action_z).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(latent).squeeze(-1)
        return log_prob, value, entropy

    def kl_to(self,
              other: "StochasticVelocityPolicy",
              rgb_seq: torch.Tensor,
              goal: torch.Tensor,
              depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """KL(other || self) per sample, shape [B]. Keeps policy close to reference BC."""
        mean = self._mean(rgb_seq, goal, depth_seq)
        other_mean = other._mean(rgb_seq, goal, depth_seq)
        dist = self._distribution(mean)
        other_dist = other._distribution(other_mean)
        return torch.distributions.kl_divergence(other_dist, dist).sum(dim=-1)

    def frozen_reference_copy(self) -> "StochasticVelocityPolicy":
        """Deep copy for BC/KL anchoring; parameters do not receive gradients."""
        ref = copy.deepcopy(self)
        ref.eval()
        for param in ref.parameters():
            param.requires_grad = False
        return ref

    def q_input(self,
                rgb_seq: torch.Tensor,
                goal: torch.Tensor,
                depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Latent features for SAC Q-networks."""
        return self._latent(rgb_seq, goal, depth_seq)


class TwinQCritic(nn.Module):
    """Twin Q-networks for SAC on (latent, action_z)."""

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        in_dim = latent_dim + action_dim

        def _q() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )

        self.q1 = _q()
        self.q2 = _q()

    def forward(self, latent: torch.Tensor, action_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([latent, action_z], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def load_stochastic_from_checkpoint(
    ckpt_path: Path,
    *,
    init_log_std: float = -0.5,
    device: Optional[torch.device] = None,
) -> Tuple[StochasticVelocityPolicy, CommandStats, dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    base = build_model(cfg)
    base.load_state_dict(ckpt["model"])
    policy = StochasticVelocityPolicy(base, init_log_std=init_log_std)
    if "rl_head" in ckpt:
        policy.log_std.data.copy_(ckpt["rl_head"]["log_std"])
        policy.critic.load_state_dict(ckpt["rl_head"]["critic"])
    stats = CommandStats.from_dict(ckpt["stats"])
    return policy.to(device), stats, cfg


def save_rl_checkpoint(path: Path,
                       policy: StochasticVelocityPolicy,
                       stats: CommandStats,
                       cfg: dict,
                       meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.base.state_dict(),
            "rl_head": {
                "log_std": policy.log_std.detach().cpu(),
                "critic": policy.critic.state_dict(),
            },
            "stats": stats.to_dict(),
            "config": cfg,
            "rl_meta": meta,
        },
        path,
    )
