"""
ResNet-18 + GRU + MLP policy that maps an RGB frame sequence and a goal vector
to a horizon of velocity commands [vx, vy, vz, psi_dot].

Forward shapes:
    rgb_seq: [B, T, 3, S, S]  (S = 224 by default, ImageNet-normalized)
    goal:    [B, goal_input_dim]
                goal_input_dim = 2 -> [hx, hy]               (heading only)
                goal_input_dim = 3 -> [hx, hy, d_normalized] (heading + distance)
    output:  [B, H, cmd_dim]  (z-scored; caller de-standardizes with CommandStats)

Architectural notes:
    - ResNet-18 (shared weights across time) encodes each frame to a 512-D vector.
    - A GRU processes the T-frame visual sequence and yields a context vector h.
    - LayerNorm is applied to h before concatenation, normalizing its magnitude so
      the goal embedding (which is naturally unit-scale from a unit-vector input)
      contributes meaningfully and neither stream drowns out the other.
    - The goal vector is projected through a small linear embedding (goal_emb_dim).
    - [h_normed ; goal_emb] are concatenated and fed to the MLP head.
    - Dropout after each hidden ReLU in the MLP head regularizes the prediction head
      without touching the backbone or GRU.
    - The stem (`conv1` + `bn1`) and `layer1` can optionally be frozen for early
      training stability on small datasets / small GPUs.
    - The 3-dim goal-input variant adds a scale-normalized distance-to-goal scalar
      alongside the unit heading vector, giving the network a direct signal for
      deceleration as it approaches the goal.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class _PerFrameResNet18(nn.Module):
    """ResNet-18 with the final classifier removed; returns a 512-D pooled feature per image."""

    def __init__(self, freeze_stem_and_layer1: bool = True) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace the FC head with identity -> .forward() returns the 512-D pooled feature.
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.out_dim = 512

        if freeze_stem_and_layer1:
            for m in (self.backbone.conv1, self.backbone.bn1, self.backbone.layer1):
                for p in m.parameters():
                    p.requires_grad = False
            # Keep BN in eval mode so its running stats don't drift on small batches.
            self.backbone.bn1.eval()
            for bn_module in self.backbone.layer1.modules():
                if isinstance(bn_module, nn.BatchNorm2d):
                    bn_module.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _MLPHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden: Sequence[int],
                 out_dim: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RGBVelocityPolicy(nn.Module):
    """
    RGB sequence + goal vector -> velocity command horizon.

    Goal vector is one of:
        * [hx, hy]                     -- unit heading toward the goal (goal_input_dim=2)
        * [hx, hy, d_normalized]       -- unit heading + scale-normalized distance (3)

    It is projected through a small linear embedding and concatenated with
    the LayerNorm'd GRU context before the MLP prediction head, giving the
    policy an explicit, direction-aware navigation objective without changing
    the visual backbone or recurrent structure.

    Outputs z-scored values; de-standardize with CommandStats.
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
                 goal_input_dim: int = 2,
                 freeze_stem_and_layer1: bool = True) -> None:
        super().__init__()
        if goal_input_dim not in (2, 3):
            raise ValueError(f"goal_input_dim must be 2 or 3; got {goal_input_dim}")
        self.T = T
        self.H = H
        self.cmd_dim = cmd_dim
        self.gru_hidden = gru_hidden
        self.goal_emb_dim = goal_emb_dim
        self.goal_input_dim = int(goal_input_dim)
        self.use_depth = False

        self.visual = _PerFrameResNet18(freeze_stem_and_layer1=freeze_stem_and_layer1)
        self.gru = nn.GRU(
            input_size=self.visual.out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        # LayerNorm on the GRU output normalizes its magnitude so the goal
        # embedding (naturally ~unit-scale from a unit-vector input) contributes
        # proportionally when the two streams are concatenated.
        self.gru_norm = nn.LayerNorm(gru_hidden)

        # Goal embedding: goal_input_dim -> goal_emb_dim.
        self.goal_embed = nn.Sequential(
            nn.Linear(self.goal_input_dim, goal_emb_dim),
            nn.ReLU(inplace=True),
        )
        # MLP head: concatenation of normalized GRU context + goal embedding.
        self.head = _MLPHead(
            in_dim=gru_hidden + goal_emb_dim,
            hidden=tuple(mlp_hidden),
            out_dim=H * cmd_dim,
            dropout=mlp_dropout,
        )

    def forward(self,
                rgb_seq: torch.Tensor,
                goal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_seq: [B, T, 3, S, S] float32, ImageNet-normalized.
            goal:    [B, goal_input_dim] float32 -- [hx, hy] or [hx, hy, d/scale].

        Returns:
            commands: [B, H, cmd_dim] float32, z-scored.
        """
        if rgb_seq.ndim != 5:
            raise ValueError(f"expected rgb_seq [B,T,3,S,S], got {tuple(rgb_seq.shape)}")
        B, T, C, S1, S2 = rgb_seq.shape
        if T != self.T:
            raise ValueError(f"T mismatch: config={self.T}, input={T}")
        if goal.shape != (B, self.goal_input_dim):
            raise ValueError(
                f"goal must be [B,{self.goal_input_dim}], got {tuple(goal.shape)}"
            )

        flat = rgb_seq.reshape(B * T, C, S1, S2)             # [B*T, 3, S, S]
        feats = self.visual(flat)                            # [B*T, 512]
        seq = feats.view(B, T, self.visual.out_dim)          # [B, T, 512]

        _, h_n = self.gru(seq)                                # [num_layers, B, gru_hidden]
        h = self.gru_norm(h_n[-1])                           # [B, gru_hidden], unit-scale

        g = self.goal_embed(goal)                            # [B, goal_emb_dim]
        h_aug = torch.cat([h, g], dim=-1)                    # [B, gru_hidden + goal_emb_dim]

        out = self.head(h_aug)                               # [B, H * cmd_dim]
        return out.view(B, self.H, self.cmd_dim)

    def forward_latent(self,
                       rgb_seq: torch.Tensor,
                       goal: torch.Tensor) -> torch.Tensor:
        """Return fused GRU+goal features [B, gru_hidden + goal_emb_dim] before the MLP head."""
        if rgb_seq.ndim != 5:
            raise ValueError(f"expected rgb_seq [B,T,3,S,S], got {tuple(rgb_seq.shape)}")
        B, T, C, S1, S2 = rgb_seq.shape
        if T != self.T:
            raise ValueError(f"T mismatch: config={self.T}, input={T}")
        if goal.shape != (B, self.goal_input_dim):
            raise ValueError(
                f"goal must be [B,{self.goal_input_dim}], got {tuple(goal.shape)}"
            )
        flat = rgb_seq.reshape(B * T, C, S1, S2)
        feats = self.visual(flat)
        seq = feats.view(B, T, self.visual.out_dim)
        _, h_n = self.gru(seq)
        h = self.gru_norm(h_n[-1])
        g = self.goal_embed(goal)
        return torch.cat([h, g], dim=-1)

    def predict_mean_first(self,
                           rgb_seq: torch.Tensor,
                           goal: torch.Tensor) -> torch.Tensor:
        """Deterministic BC mean for the first horizon step [B, cmd_dim] in z-space."""
        h_aug = self.forward_latent(rgb_seq, goal)
        out = self.head(h_aug)
        return out.view(-1, self.H, self.cmd_dim)[:, 0, :]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
