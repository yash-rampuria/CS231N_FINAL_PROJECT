"""
ResNet-18 + Depth Anything V2 Small fusion policies.

Architectures (``fusion``):
    crossattn  — LayerNorm + multi-head cross-attention (primary, A2)
    concat     — raw concat + linear, no pre-fusion LayerNorm (ablation A1)

Both share the same GRU + goal-conditioned MLP head as RGBVelocityPolicy.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn

from nav_policy.model.rgb_velocity_policy import (
    RGBVelocityPolicy,
    _MLPHead,
    _PerFrameResNet18,
    count_parameters,
)

FusionMode = Literal["crossattn", "concat"]


class _DepthEncoder(nn.Module):
    """Small CNN: [B, 1, S, S] -> [B, depth_feat_dim]."""

    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CrossAttnFusion(nn.Module):
    def __init__(self, rgb_dim: int = 512, dep_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.dep_norm = nn.LayerNorm(dep_dim)
        self.dep_to_rgb = nn.Linear(dep_dim, rgb_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=rgb_dim, num_heads=num_heads, batch_first=True
        )
        self.out_norm = nn.LayerNorm(rgb_dim)

    def forward(self, f_rgb: torch.Tensor, f_dep: torch.Tensor) -> torch.Tensor:
        q = self.rgb_norm(f_rgb).unsqueeze(1)
        kv = self.dep_to_rgb(self.dep_norm(f_dep)).unsqueeze(1)
        attn_out, _ = self.attn(q, kv, kv)
        fused = self.out_norm(attn_out.squeeze(1) + f_rgb)
        return fused


class _ConcatFusion(nn.Module):
    """Ablation A1: no LayerNorm before fusion."""

    def __init__(self, rgb_dim: int = 512, dep_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(rgb_dim + dep_dim, rgb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_rgb: torch.Tensor, f_dep: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([f_rgb, f_dep], dim=-1))


class RGBDA2VelocityPolicy(nn.Module):
    """
    RGB + depth sequence + goal -> velocity horizon.

    ``depth_seq`` must be [B, T, 1, S, S] float32 in [0, 1].
    """

    def __init__(self,
                 T: int = 4,
                 H: int = 10,
                 cmd_dim: int = 4,
                 gru_hidden: int = 256,
                 gru_layers: int = 1,
                 mlp_hidden: Sequence[int] = (256, 128),
                 mlp_dropout: float = 0.1,
                 goal_emb_dim: int = 32,
                 goal_input_dim: int = 3,
                 freeze_stem_and_layer1: bool = True,
                 fusion: FusionMode = "crossattn",
                 depth_feat_dim: int = 256,
                 cross_attn_heads: int = 4,
                 goal_conditioning: str = "concat") -> None:
        super().__init__()
        if goal_input_dim not in (2, 3):
            raise ValueError(f"goal_input_dim must be 2 or 3; got {goal_input_dim}")
        if fusion not in ("crossattn", "concat"):
            raise ValueError(f"fusion must be crossattn or concat; got {fusion!r}")
        if goal_conditioning not in ("concat", "film"):
            raise ValueError(
                f"goal_conditioning must be concat or film; got {goal_conditioning!r}"
            )

        self.T = T
        self.H = H
        self.cmd_dim = cmd_dim
        self.gru_hidden = gru_hidden
        self.goal_emb_dim = goal_emb_dim
        self.goal_input_dim = int(goal_input_dim)
        self.fusion_mode = fusion
        # "concat" (default) = exactly the current architecture; "film" = strong
        # goal conditioning (goal modulates GRU features + fed into GRU each step).
        self.goal_conditioning = goal_conditioning
        self.use_depth = True

        self.visual = _PerFrameResNet18(freeze_stem_and_layer1=freeze_stem_and_layer1)
        self.depth_enc = _DepthEncoder(out_dim=depth_feat_dim)
        if fusion == "crossattn":
            self.fusion = _CrossAttnFusion(
                rgb_dim=self.visual.out_dim,
                dep_dim=depth_feat_dim,
                num_heads=cross_attn_heads,
            )
            fused_dim = self.visual.out_dim
        else:
            self.fusion = _ConcatFusion(
                rgb_dim=self.visual.out_dim, dep_dim=depth_feat_dim
            )
            fused_dim = self.visual.out_dim

        # In FiLM mode the goal embedding is also fed into the GRU at every step,
        # so the GRU input is widened by goal_emb_dim.
        gru_input = fused_dim + (goal_emb_dim if goal_conditioning == "film" else 0)
        self.gru = nn.GRU(
            input_size=gru_input,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.gru_norm = nn.LayerNorm(gru_hidden)
        self.goal_embed = nn.Sequential(
            nn.Linear(self.goal_input_dim, goal_emb_dim),
            nn.ReLU(inplace=True),
        )

        if goal_conditioning == "film":
            # Goal -> per-channel (gamma, beta) that modulate the GRU context:
            #   h' = (1 + gamma) * h + beta
            # Final layer zero-init so it starts as identity (stable warm-start).
            self.film = nn.Linear(goal_emb_dim, 2 * gru_hidden)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.latent_dim = gru_hidden
        else:
            self.film = None
            self.latent_dim = gru_hidden + goal_emb_dim

        self.head = _MLPHead(
            in_dim=self.latent_dim,
            hidden=tuple(mlp_hidden),
            out_dim=H * cmd_dim,
            dropout=mlp_dropout,
        )

    def _fuse_frame(self, f_rgb: torch.Tensor, f_dep: torch.Tensor) -> torch.Tensor:
        return self.fusion(f_rgb, f_dep)

    def _encode_latent(self,
                       rgb_seq: torch.Tensor,
                       goal: torch.Tensor,
                       depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode (rgb, depth, goal) into the latent fed to the MLP head [B, latent_dim]."""
        if rgb_seq.ndim != 5:
            raise ValueError(f"expected rgb_seq [B,T,3,S,S], got {tuple(rgb_seq.shape)}")
        B, T, C, S1, S2 = rgb_seq.shape
        if T != self.T:
            raise ValueError(f"T mismatch: config={self.T}, input={T}")
        if goal.shape != (B, self.goal_input_dim):
            raise ValueError(
                f"goal must be [B,{self.goal_input_dim}], got {tuple(goal.shape)}"
            )
        if depth_seq is None:
            raise ValueError("depth_seq is required for RGBDA2VelocityPolicy")
        if depth_seq.shape != (B, T, 1, S1, S2):
            raise ValueError(
                f"depth_seq must be [B,T,1,{S1},{S2}], got {tuple(depth_seq.shape)}"
            )

        flat_rgb = rgb_seq.reshape(B * T, C, S1, S2)
        flat_dep = depth_seq.reshape(B * T, 1, S1, S2)
        f_rgb = self.visual(flat_rgb)
        f_dep = self.depth_enc(flat_dep)
        fused = self._fuse_frame(f_rgb, f_dep)
        seq = fused.view(B, T, -1)
        g = self.goal_embed(goal)

        if self.goal_conditioning == "film":
            # Feed goal into the GRU at every timestep ...
            g_seq = g.unsqueeze(1).expand(B, T, -1)
            seq = torch.cat([seq, g_seq], dim=-1)
            _, h_n = self.gru(seq)
            h = self.gru_norm(h_n[-1])
            # ... and FiLM-modulate the final context: h' = (1+gamma)*h + beta.
            gamma, beta = self.film(g).chunk(2, dim=-1)
            return (1.0 + gamma) * h + beta

        # concat (default, original behaviour)
        _, h_n = self.gru(seq)
        h = self.gru_norm(h_n[-1])
        return torch.cat([h, g], dim=-1)

    def forward(self,
                rgb_seq: torch.Tensor,
                goal: torch.Tensor,
                depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        latent = self._encode_latent(rgb_seq, goal, depth_seq)
        out = self.head(latent)
        return out.view(rgb_seq.shape[0], self.H, self.cmd_dim)

    def forward_latent(self,
                       rgb_seq: torch.Tensor,
                       goal: torch.Tensor,
                       depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the latent features [B, latent_dim] fed to the MLP head."""
        return self._encode_latent(rgb_seq, goal, depth_seq)

    def predict_mean_first(self,
                           rgb_seq: torch.Tensor,
                           goal: torch.Tensor,
                           depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Deterministic BC mean for the first horizon step [B, cmd_dim] in z-space."""
        h_aug = self.forward_latent(rgb_seq, goal, depth_seq)
        out = self.head(h_aug)
        return out.view(-1, self.H, self.cmd_dim)[:, 0, :]


__all__ = [
    "RGBDA2VelocityPolicy",
    "count_parameters",
]
