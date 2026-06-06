"""Soft Actor-Critic updates for nav_policy RL fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from nav_policy.rl.stochastic_policy import StochasticVelocityPolicy, TwinQCritic


@dataclass
class SACStats:
    q1_loss: float
    q2_loss: float
    policy_loss: float
    alpha: float


class SACTrainer:
    """Minimal SAC trainer operating in z-score action space."""

    def __init__(self,
                 policy: StochasticVelocityPolicy,
                 *,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 auto_alpha: bool = True,
                 device: torch.device) -> None:
        self.policy = policy
        self.device = device
        self.gamma = gamma
        self.tau = tau
        latent_dim = int(policy.base.gru_hidden) + int(policy.base.goal_emb_dim)
        self.critic = TwinQCritic(latent_dim, policy.cmd_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
            self.target_entropy = -float(policy.cmd_dim)
        else:
            self.log_alpha = None
            self.alpha_opt = None
            self.target_entropy = 0.0
            self._fixed_alpha = alpha

    @property
    def alpha(self) -> torch.Tensor:
        if self.log_alpha is not None:
            return self.log_alpha.exp()
        return torch.tensor(self._fixed_alpha, device=self.device)

    def _soft_update(self) -> None:
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update(self, batch: dict) -> SACStats:
        rgb = batch["rgb"].to(self.device)
        goal = batch["goal"].to(self.device)
        depth = batch["depth"].to(self.device) if batch.get("depth") is not None else None
        next_rgb = batch["next_rgb"].to(self.device)
        next_goal = batch["next_goal"].to(self.device)
        next_depth = (
            batch["next_depth"].to(self.device)
            if batch.get("next_depth") is not None
            else None
        )
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.policy.act(
                next_rgb, next_goal, next_depth,
            )
            latent_next = self.policy.q_input(next_rgb, next_goal, next_depth)
            q1_next, q2_next = self.critic_target(latent_next, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_prob
            target = rewards + (1.0 - dones) * self.gamma * q_next

        latent = self.policy.q_input(rgb, goal, depth)
        q1, q2 = self.critic(latent, actions)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        latent = self.policy.q_input(rgb, goal, depth)
        new_actions, log_prob, _ = self.policy.act(rgb, goal, depth)
        q1_pi, q2_pi = self.critic(latent, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.alpha.detach()
        policy_loss = (alpha * log_prob - q_pi).mean()

        self.policy_opt.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_opt.step()

        alpha_val = float(alpha.item())
        if self.log_alpha is not None and self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_val = float(self.alpha.item())

        self._soft_update()
        return SACStats(
            q1_loss=float(q1_loss.item()),
            q2_loss=float(q2_loss.item()),
            policy_loss=float(policy_loss.item()),
            alpha=alpha_val,
        )


def sac_config_from_dict(cfg: Dict) -> Dict:
    s = cfg.get("sac", {}) or {}
    return {
        "lr": float(s.get("lr", 3e-4)),
        "gamma": float(s.get("gamma", 0.99)),
        "tau": float(s.get("tau", 0.005)),
        "alpha": float(s.get("alpha", 0.2)),
        "auto_alpha": bool(s.get("auto_alpha", True)),
        "batch_size": int(s.get("batch_size", 256)),
        "updates_per_iter": int(s.get("updates_per_iter", 4)),
        "replay_capacity": int(s.get("replay_capacity", 100_000)),
    }
