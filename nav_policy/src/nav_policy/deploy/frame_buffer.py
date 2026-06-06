"""
Rolling frame buffer for online inference.

Holds the last T resized + ImageNet-normalized RGB tensors (and optional depth).
When the buffer is partially full it replicates the oldest available frame at the
front.  Optional observation latency delays which frames the policy sees.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from nav_policy.data.normalization import IMAGENET_MEAN, IMAGENET_STD, imagenet_normalize

try:
    import cv2  # type: ignore
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


def _resize_uint8(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 uint8, got {frame.shape}")
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if _HAVE_CV2:
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    from PIL import Image
    return np.asarray(
        Image.fromarray(frame, mode="RGB").resize((size, size), Image.BILINEAR)
    )


def _apply_brightness_hwc(frame: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3:
        return frame
    out = np.clip(frame.astype(np.float32) * factor, 0, 255)
    return out.round().astype(np.uint8)


def _apply_blur_hwc(frame: np.ndarray, sigma: float) -> np.ndarray:
    if not _HAVE_CV2 or sigma <= 0:
        return frame
    k = max(3, int(6 * sigma + 1) | 1)
    return cv2.GaussianBlur(frame, (k, k), sigmaX=sigma, sigmaY=sigma)


class FrameBuffer:
    """Maintain a rolling window of T preprocessed frames (+ optional depth)."""

    def __init__(self,
                 T: int = 4,
                 image_size: int = 224,
                 mean: Sequence[float] = IMAGENET_MEAN,
                 std: Sequence[float] = IMAGENET_STD,
                 device: Optional[torch.device] = None,
                 observation_latency: int = 0,
                 brightness_factor: float = 1.0,
                 blur_sigma: float = 0.0) -> None:
        self.T = T
        self.image_size = image_size
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.observation_latency = max(0, int(observation_latency))
        self.brightness_factor = float(brightness_factor)
        self.blur_sigma = float(blur_sigma)
        self._buf_uint8: list[torch.Tensor] = []
        self._depth_buf: list[torch.Tensor] = []

    def reset(self) -> None:
        self._buf_uint8.clear()
        self._depth_buf.clear()

    def push(self,
             frame: np.ndarray,
             depth_hw: Optional[np.ndarray] = None) -> None:
        """`frame` is HxWx3 uint8 RGB; optional depth_hw is HxW float32 in [0,1]."""
        resized = _resize_uint8(frame, self.image_size)
        if self.brightness_factor != 1.0:
            resized = _apply_brightness_hwc(resized, self.brightness_factor)
        if self.blur_sigma > 0:
            resized = _apply_blur_hwc(resized, self.blur_sigma)
        chw = torch.from_numpy(resized).permute(2, 0, 1).contiguous()
        if len(self._buf_uint8) >= self.T + self.observation_latency:
            self._buf_uint8.pop(0)
            if self._depth_buf:
                self._depth_buf.pop(0)
        self._buf_uint8.append(chw)
        if depth_hw is not None:
            if _HAVE_CV2:
                d = cv2.resize(depth_hw, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            else:
                from PIL import Image
                d = np.asarray(
                    Image.fromarray((depth_hw * 255).astype(np.uint8)).resize(
                        (self.image_size, self.image_size), Image.BILINEAR
                    )
                ).astype(np.float32) / 255.0
            if self.blur_sigma > 0 and _HAVE_CV2:
                d_u8 = (np.clip(d, 0, 1) * 255).astype(np.uint8)
                k = max(3, int(6 * self.blur_sigma + 1) | 1)
                d_u8 = cv2.GaussianBlur(d_u8, (k, k), sigmaX=self.blur_sigma, sigmaY=self.blur_sigma)
                d = d_u8.astype(np.float32) / 255.0
            self._depth_buf.append(
                torch.from_numpy(d).unsqueeze(0).contiguous()
            )

    def is_ready(self) -> bool:
        return len(self._buf_uint8) > 0

    def _window(self, buf: list[torch.Tensor]) -> list[torch.Tensor]:
        if not buf:
            raise RuntimeError("FrameBuffer is empty")
        end = max(0, len(buf) - 1 - self.observation_latency)
        start = max(0, end - self.T + 1)
        chunk = buf[start : end + 1]
        if len(chunk) == self.T:
            return chunk
        if not chunk:
            chunk = [buf[0]]
        pad_n = self.T - len(chunk)
        pad = [chunk[0]] * pad_n
        return pad + chunk

    def tensor(self) -> torch.Tensor:
        """Return [1, T, 3, S, S] float32 ImageNet-normalized RGB."""
        seq = self._window(self._buf_uint8)
        stacked = torch.stack(seq, dim=0)
        normed = imagenet_normalize(stacked, mean=self.mean, std=self.std)
        return normed.unsqueeze(0).to(self.device, non_blocking=True)

    def depth_tensor(self) -> Optional[torch.Tensor]:
        """Return [1, T, 1, S, S] float32 depth in [0,1], or None if unavailable."""
        if not self._depth_buf:
            return None
        seq = self._window(self._depth_buf)
        stacked = torch.stack(seq, dim=0)
        return stacked.unsqueeze(0).to(self.device, non_blocking=True)

    def tensors(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.tensor(), self.depth_tensor()
