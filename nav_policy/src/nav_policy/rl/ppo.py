"""Proximal Policy Optimization updates for nav_policy RL fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from nav_policy.rl.buffer import RolloutBuffer
from nav_policy.rl.stochastic_policy import StochasticVelocityPolicy


@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    ref_kl: float = 0.0


def ppo_update(policy: StochasticVelocityPolicy,
               buffer: RolloutBuffer,
               optimizer: torch.optim.Optimizer,
               *,
               clip_eps: float = 0.2,
               value_coef: float = 0.5,
               entropy_coef: float = 0.01,
               ref_kl_coef: float = 0.0,
               reference_policy: Optional[StochasticVelocityPolicy] = None,
               max_grad_norm: float = 1.0,
               n_epochs: int = 4,
               batch_size: int = 256,
               target_kl: float = 0.05,
               device: torch.device) -> PPOStats:
    policy.train()
    n = len(buffer)
    idx = torch.arange(n)

    total_pl = total_vl = total_ent = total_kl = total_ref_kl = 0.0
    n_updates = 0

    for epoch in range(n_epochs):
        perm = idx[torch.randperm(n)]
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            rgb = buffer.rgb[batch_idx].float().to(device, non_blocking=True)
            goal = buffer.goal[batch_idx].to(device, non_blocking=True)
            depth = (
                buffer.depth[batch_idx].float().to(device, non_blocking=True)
                if buffer.depth is not None
                else None
            )
            actions = buffer.actions[batch_idx].to(device, non_blocking=True)
            old_log_probs = buffer.log_probs[batch_idx].to(device, non_blocking=True)
            advantages = buffer.advantages[batch_idx].to(device, non_blocking=True)
            returns = buffer.returns[batch_idx].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                new_log_probs, values, entropy = policy.evaluate(rgb, goal, actions, depth)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, returns)
                ent = entropy.mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * ent
                ref_kl_val = 0.0
                if reference_policy is not None and ref_kl_coef > 0.0:
                    ref_kl = policy.kl_to(reference_policy, rgb, goal, depth).mean()
                    loss = loss + ref_kl_coef * ref_kl
                    ref_kl_val = float(ref_kl.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean().item()

            total_pl += float(policy_loss.item())
            total_vl += float(value_loss.item())
            total_ent += float(ent.item())
            total_kl += approx_kl
            total_ref_kl += ref_kl_val
            n_updates += 1

        if target_kl > 0 and n_updates > 0 and (total_kl / n_updates) > target_kl:
            print(f"  [ppo] early stop at epoch {epoch+1}/{n_epochs}: "
                  f"kl={total_kl/n_updates:.4f} > target={target_kl}", flush=True)
            break

    n_updates = max(n_updates, 1)
    return PPOStats(
        policy_loss=total_pl / n_updates,
        value_loss=total_vl / n_updates,
        entropy=total_ent / n_updates,
        approx_kl=total_kl / n_updates,
        ref_kl=total_ref_kl / n_updates,
    )


def ppo_config_from_dict(cfg: Dict) -> Dict:
    p = cfg.get("ppo", {}) or {}
    anchor = cfg.get("bc_anchor", {}) or {}
    ref_kl_coef = p.get("ref_kl_coef", anchor.get("kl_coef", 0.0))
    return {
        "clip_eps": float(p.get("clip_eps", 0.2)),
        "value_coef": float(p.get("value_coef", 0.5)),
        "entropy_coef": float(p.get("entropy_coef", 0.01)),
        "ref_kl_coef": float(ref_kl_coef),
        "max_grad_norm": float(p.get("max_grad_norm", 1.0)),
        "n_epochs": int(p.get("n_epochs", 4)),
        "target_kl": float(p.get("target_kl", 0.05)),
        "batch_size": int(p.get("batch_size", 256)),
        "gamma": float(p.get("gamma", 0.99)),
        "gae_lambda": float(p.get("gae_lambda", 0.95)),
        "lr": float(p.get("lr", 3e-4)),
    }
