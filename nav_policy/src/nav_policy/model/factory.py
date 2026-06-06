"""Build a policy module from a training/eval YAML config."""

from __future__ import annotations

from typing import Union

import torch.nn as nn

from nav_policy.model.rgb_da2_policy import RGBDA2VelocityPolicy
from nav_policy.model.rgb_velocity_policy import RGBVelocityPolicy


def build_model(cfg: dict) -> nn.Module:
    mcfg = cfg["model"]
    arch = str(mcfg.get("arch", "rgb_resnet18"))
    common = dict(
        T=int(cfg["window"]["T"]),
        H=int(cfg["window"]["H"]),
        cmd_dim=int(mcfg["cmd_dim"]),
        gru_hidden=int(mcfg["gru_hidden"]),
        gru_layers=int(mcfg["gru_layers"]),
        mlp_hidden=tuple(mcfg["mlp_hidden"]),
        mlp_dropout=float(mcfg.get("mlp_dropout", 0.1)),
        goal_emb_dim=int(mcfg.get("goal_emb_dim", 32)),
        goal_input_dim=int(mcfg.get("goal_input_dim", 2)),
        freeze_stem_and_layer1=bool(mcfg.get("freeze_stem_and_layer1", True)),
    )

    if arch == "rgb_resnet18":
        return RGBVelocityPolicy(**common)

    goal_conditioning = str(mcfg.get("goal_conditioning", "concat"))

    if arch == "rgb_da2_crossattn_v1":
        return RGBDA2VelocityPolicy(
            **common,
            fusion="crossattn",
            depth_feat_dim=int(mcfg.get("depth_feat_dim", 256)),
            cross_attn_heads=int(mcfg.get("cross_attn_heads", 4)),
            goal_conditioning=goal_conditioning,
        )

    if arch == "rgb_da2_concat_v1":
        return RGBDA2VelocityPolicy(
            **common,
            fusion="concat",
            depth_feat_dim=int(mcfg.get("depth_feat_dim", 256)),
            goal_conditioning=goal_conditioning,
        )

    raise ValueError(
        f"unknown model.arch={arch!r}; expected rgb_resnet18, "
        "rgb_da2_crossattn_v1, or rgb_da2_concat_v1"
    )


def model_uses_depth(cfg: dict) -> bool:
    arch = str(cfg.get("model", {}).get("arch", "rgb_resnet18"))
    return arch.startswith("rgb_da2_")
