"""
FiGS-compatible controller wrapping trained RGB / RGB+DA2 policies.

Implements the duck-typed contract used by figs.simulator.Simulator.simulate().
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from figs.control.velocity_controller import VelocityController

from nav_policy.data.normalization import CommandStats
from nav_policy.deploy.frame_buffer import FrameBuffer
from nav_policy.model.depth_estimator import DepthAnythingV2Small
from nav_policy.model.factory import build_model, model_uses_depth


class RGBVelocityController:
    """Wraps a trained policy + FiGS VelocityController for closed-loop deployment."""

    def __init__(self,
                 model: nn.Module,
                 stats: CommandStats,
                 inner: VelocityController,
                 image_size: int = 224,
                 goal_pos_xy: Optional[np.ndarray] = None,
                 zero_goal_heading: bool = False,
                 goal_distance_scale: float = 5.0,
                 device: Optional[torch.device] = None,
                 depth_model: Optional[DepthAnythingV2Small] = None,
                 observation_latency: int = 0,
                 brightness_factor: float = 1.0,
                 blur_sigma: float = 0.0,
                 depth_inference_stride: int = 1) -> None:
        if int(getattr(model, "cmd_dim", 4)) != 4:
            raise ValueError(f"expected cmd_dim=4, got {getattr(model, 'cmd_dim', None)}")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.eval().to(self.device)
        self.stats = stats
        self.inner = inner
        self.uses_depth = bool(getattr(model, "use_depth", False))
        self._depth_model = depth_model
        if self.uses_depth and self._depth_model is None:
            self._depth_model = DepthAnythingV2Small().to(self.device)
        elif self._depth_model is not None:
            self._depth_model = self._depth_model.eval().to(self.device)

        self.buf = FrameBuffer(
            T=int(model.T),
            image_size=image_size,
            device=self.device,
            observation_latency=observation_latency,
            brightness_factor=brightness_factor,
            blur_sigma=blur_sigma,
        )
        self._goal_pos_xy: Optional[np.ndarray] = (
            np.asarray(goal_pos_xy, dtype=np.float64).ravel()[:2]
            if goal_pos_xy is not None else None
        )
        self._zero_goal_heading: bool = bool(zero_goal_heading)
        self._goal_input_dim: int = int(model.goal_input_dim)
        if goal_distance_scale <= 0.0:
            raise ValueError(f"goal_distance_scale must be > 0; got {goal_distance_scale}")
        self._goal_distance_scale: float = float(goal_distance_scale)
        if depth_inference_stride < 1:
            raise ValueError(f"depth_inference_stride must be >= 1; got {depth_inference_stride}")
        self.depth_inference_stride = int(depth_inference_stride)
        self._control_step = 0
        self._last_depth_hw: Optional[np.ndarray] = None

        self.hz = inner.hz
        self.nzcr = None
        self.name = "RGBVelocityController"

    def set_goal(self, goal_pos_xy: np.ndarray) -> None:
        self._goal_pos_xy = np.asarray(goal_pos_xy, dtype=np.float64).ravel()[:2]

    def _compute_goal_vector(self, xcr: np.ndarray) -> np.ndarray:
        if self._zero_goal_heading:
            return np.zeros(self._goal_input_dim, dtype=np.float32)
        if self._goal_pos_xy is None:
            heading = np.array([1.0, 0.0], dtype=np.float32)
            d_norm = 0.0
        else:
            delta = self._goal_pos_xy - xcr[0:2].astype(np.float64)
            norm = float(np.linalg.norm(delta))
            if norm < 1e-6:
                heading = np.array([1.0, 0.0], dtype=np.float32)
                d_norm = 0.0
            else:
                heading = (delta / norm).astype(np.float32)
                d_norm = float(norm / self._goal_distance_scale)
        if self._goal_input_dim == 2:
            return heading
        return np.array([heading[0], heading[1], d_norm], dtype=np.float32)

    @classmethod
    def from_checkpoint(cls,
                        ckpt_path: Path,
                        frame_name: str = "carl",
                        Kv: float = 2.0,
                        Ka: float = 5.0,
                        goal_pos_xy: Optional[np.ndarray] = None,
                        device: Optional[torch.device] = None,
                        observation_latency: int = 0,
                        brightness_factor: float = 1.0,
                        blur_sigma: float = 0.0,
                        depth_inference_stride: int = 3) -> "RGBVelocityController":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        cfg = ckpt["config"]
        model = build_model(cfg)
        model.load_state_dict(ckpt["model"])
        stats = CommandStats.from_dict(ckpt["stats"])
        inner = VelocityController(hz=20, Kv=Kv, Ka=Ka, frame_name=frame_name)
        zero_goal = bool(cfg.get("train", {}).get("zero_goal_heading", False))
        goal_distance_scale = float(cfg.get("model", {}).get("goal_distance_scale", 5.0))
        depth_model = DepthAnythingV2Small().to(device) if model_uses_depth(cfg) else None
        return cls(
            model=model,
            stats=stats,
            inner=inner,
            image_size=int(cfg["window"].get("image_size", 224)),
            goal_pos_xy=goal_pos_xy,
            zero_goal_heading=zero_goal,
            goal_distance_scale=goal_distance_scale,
            device=device,
            depth_model=depth_model,
            observation_latency=observation_latency,
            brightness_factor=brightness_factor,
            blur_sigma=blur_sigma,
            depth_inference_stride=depth_inference_stride,
        )

    def reset(self, goal_pos_xy: Optional[np.ndarray] = None) -> None:
        self.buf.reset()
        self._control_step = 0
        self._last_depth_hw = None
        if goal_pos_xy is not None:
            self.set_goal(goal_pos_xy)

    @torch.inference_mode()
    def _infer_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.uses_depth or self._depth_model is None:
            return None
        t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
        d = self._depth_model(t)
        return d[0, 0].cpu().numpy()

    @torch.inference_mode()
    def encode_depth_batch(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        """rgb [N,3,S,S] uint8 -> depth [N,1,S,S] uint8 for cache writing."""
        if self._depth_model is None:
            raise RuntimeError("depth model not loaded")
        from nav_policy.model.depth_estimator import depth_uint8_from_float

        outs = []
        batch = 8
        for i in range(0, rgb_uint8.shape[0], batch):
            chunk = rgb_uint8[i : i + batch].to(self.device)
            d = self._depth_model(chunk)
            for j in range(d.shape[0]):
                outs.append(depth_uint8_from_float(d[j, 0].cpu().numpy()))
        stacked = np.stack(outs, axis=0)
        return torch.from_numpy(stacked).unsqueeze(1).contiguous()

    @torch.inference_mode()
    def _predict_first_command(self,
                               icr: np.ndarray,
                               goal_np: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.time()
        if self.uses_depth:
            if (
                self._last_depth_hw is None
                or self._control_step % self.depth_inference_stride == 0
            ):
                self._last_depth_hw = self._infer_depth(icr)
            depth_hw = self._last_depth_hw
        else:
            depth_hw = None
        self._control_step += 1
        self.buf.push(icr, depth_hw=depth_hw)
        rgb_seq, depth_seq = self.buf.tensors()
        goal_t = torch.from_numpy(goal_np).unsqueeze(0).to(
            self.device, non_blocking=True
        )
        if self.uses_depth:
            if depth_seq is None:
                raise RuntimeError("depth buffer empty for DA2 policy")
            u_hat_z = self.model(rgb_seq, goal_t, depth_seq)
        else:
            u_hat_z = self.model(rgb_seq, goal_t)
        u_hat = self.stats.destandardize(u_hat_z)
        cmd0 = u_hat[0, 0].cpu().numpy().astype(np.float64)
        return cmd0, time.time() - t0

    def control(self,
                tcr: float,
                xcr: np.ndarray,
                upr: Any,
                obj: Any,
                icr: np.ndarray,
                zcr: Any) -> Tuple[np.ndarray, None, np.ndarray, np.ndarray]:
        goal_np = self._compute_goal_vector(xcr)
        cmd0, dt_model = self._predict_first_command(icr, goal_np)

        t1 = time.time()
        ucr, _, _, _ = self.inner.control(
            tcr=tcr,
            xcr=xcr,
            upr=upr,
            obj=cmd0,
            icr=None,
            zcr=None,
        )
        dt_inner = time.time() - t1

        adv = cmd0.astype(np.float64)
        tsol = np.array([0.0, float(dt_model), 0.0, float(dt_inner)], dtype=np.float64)
        return ucr, None, adv, tsol
