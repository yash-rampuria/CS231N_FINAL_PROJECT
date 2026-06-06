"""
Behavior-cloning loss and metric utilities.

Notation:
    u_hat:   predicted command horizon, z-scored [B, H, cmd_dim]
    u_star:  expert command horizon, z-scored [B, H, cmd_dim]
    u_raw:   un-normalized targets [B, H, cmd_dim] (for reporting in physical units)

L_cmd    = mean over (B, H, cmd_dim) of (u_hat - u_star)^2
L_smooth = mean over (B, H-1, cmd_dim) of (u_hat[:, 1:] - u_hat[:, :-1])^2
L_total  = L_cmd + lambda_smooth * L_smooth
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from nav_policy.data.normalization import CommandStats


@dataclass
class LossOutputs:
    total: torch.Tensor
    cmd: torch.Tensor
    smooth: torch.Tensor


def bc_loss(u_hat: torch.Tensor,
            u_star: torch.Tensor,
            lambda_smooth: float = 0.05) -> LossOutputs:
    if u_hat.shape != u_star.shape:
        raise ValueError(f"shape mismatch: u_hat {tuple(u_hat.shape)} vs u_star {tuple(u_star.shape)}")
    l_cmd = F.mse_loss(u_hat, u_star)
    if u_hat.shape[1] > 1:
        diff = u_hat[:, 1:] - u_hat[:, :-1]
        l_smooth = (diff ** 2).mean()
    else:
        l_smooth = torch.zeros((), device=u_hat.device, dtype=u_hat.dtype)
    return LossOutputs(total=l_cmd + lambda_smooth * l_smooth, cmd=l_cmd, smooth=l_smooth)


@torch.no_grad()
def per_component_mse(u_hat_z: torch.Tensor,
                      u_raw: torch.Tensor,
                      stats: CommandStats) -> Dict[str, torch.Tensor]:
    """
    Report MSE in **physical units** broken down by [vx, vy, vz, psi_dot].

    u_hat_z: model output [B, H, 4] in z-scored space.
    u_raw:   ground-truth target [B, H, 4] in raw units.
    """
    u_hat = stats.destandardize(u_hat_z)
    err2 = (u_hat - u_raw) ** 2                      # [B, H, 4]
    per_chan = err2.mean(dim=(0, 1))                 # [4]
    return {
        "mse_vx": per_chan[0],
        "mse_vy": per_chan[1],
        "mse_vz": per_chan[2],
        "mse_psi_dot": per_chan[3],
        "mse_lin_vel": per_chan[:3].mean(),
        "mse_overall": per_chan.mean(),
    }
