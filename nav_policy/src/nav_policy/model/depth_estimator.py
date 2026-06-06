"""
Frozen Depth Anything V2 Small (ViT-S) monocular depth for train + deploy.

Weights: place ``depth_anything_v2_vits.pth`` under
``nav_policy/data/weights/`` or set ``DEPTH_ANYTHING_V2_VITS_WEIGHTS``.

Reference: https://github.com/DepthAnything/Depth-Anything-V2
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

_DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[3] / "data" / "weights" / "depth_anything_v2_vits.pth"
)


def _resolve_weights(path: Optional[Union[str, Path]]) -> Path:
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
    env = __import__("os").environ.get("DEPTH_ANYTHING_V2_VITS_WEIGHTS")
    if env and Path(env).is_file():
        return Path(env)
    if _DEFAULT_WEIGHTS.is_file():
        return _DEFAULT_WEIGHTS
    raise FileNotFoundError(
        "Depth Anything V2 ViT-S weights not found. Download "
        "depth_anything_v2_vits.pth to nav_policy/data/weights/ or set "
        "DEPTH_ANYTHING_V2_VITS_WEIGHTS."
    )


class DepthAnythingV2Small(nn.Module):
    """Wraps frozen DA2-S; outputs single-channel depth in [0, 1]."""

    def __init__(self, weights_path: Optional[Union[str, Path]] = None) -> None:
        super().__init__()
        from nav_policy.vendor.depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            }
        }
        self._model = DepthAnythingV2(**model_configs["vits"])
        wpath = _resolve_weights(weights_path)
        state = torch.load(wpath, map_location="cpu", weights_only=False)
        self._model.load_state_dict(state)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_uint8: [B, 3, H, W] uint8 RGB (0–255), any H=W multiple of 14.

        Returns:
            depth: [B, 1, H, W] float32 in [0, 1] (per-image min-max normalized).
        """
        if rgb_uint8.dtype != torch.uint8:
            raise TypeError(f"expected uint8 RGB, got {rgb_uint8.dtype}")
        b, _, h, w = rgb_uint8.shape
        rgb_np = rgb_uint8.permute(0, 2, 3, 1).cpu().numpy()
        outs = []
        for i in range(b):
            # DA2 expects BGR uint8 HxWx3
            bgr = rgb_np[i][:, :, ::-1].copy()
            raw = self._model.infer_image(bgr)
            raw = np.asarray(raw, dtype=np.float32)
            lo, hi = float(raw.min()), float(raw.max())
            if hi - lo < 1e-6:
                norm = np.zeros_like(raw)
            else:
                norm = (raw - lo) / (hi - lo)
            outs.append(torch.from_numpy(norm))
        depth = torch.stack(outs, dim=0).unsqueeze(1)
        return depth.to(device=rgb_uint8.device, dtype=torch.float32)

    @torch.inference_mode()
    def infer_numpy(self, rgb_hwc: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 -> HxW float32 depth in [0, 1]."""
        if rgb_hwc.dtype != np.uint8:
            rgb_hwc = rgb_hwc.astype(np.uint8)
        t = torch.from_numpy(rgb_hwc).permute(2, 0, 1).unsqueeze(0)
        d = self.forward(t)
        return d[0, 0].cpu().numpy()


def depth_uint8_from_float(depth01: np.ndarray) -> np.ndarray:
    """HxW float [0,1] -> HxW uint8 for compact cache storage."""
    return (np.clip(depth01, 0.0, 1.0) * 255.0).astype(np.uint8)


def depth_float_from_uint8(depth_u8: np.ndarray) -> np.ndarray:
    """HxW uint8 -> HxW float32 [0,1]."""
    return depth_u8.astype(np.float32) / 255.0
