#!/usr/bin/env python3
"""
Generate Grad-CAM saliency overlays for ResNet layer4 in RGB / RGB+DA2 policies.

Usage (inside Docker, from nav_policy/):
    python scripts/gradcam_saliency.py \\
        --checkpoint data/checkpoints_flightroom/bc_best.pt \\
        --cache data/processed_flightroom/flightroom_ssv_exp_2026-05-22_071733_trajs-110/cache/file00000_sub0.pt \\
        --k 250 \\
        --target goal_align \\
        --method gradcam++ \\
        --output-dir data/eval/saliency_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nav_policy.data.normalization import imagenet_normalize
from nav_policy.model.factory import build_model, model_uses_depth


def _load_goal(blob: dict, k: int, goal_input_dim: int, scale: float) -> torch.Tensor:
    heading = blob["goal_heading"][k].float()
    if goal_input_dim == 2:
        return heading
    d = float(blob["goal_dist"][k].item()) / scale
    return torch.cat([heading, torch.tensor([d], dtype=torch.float32)])


def _scalar_target(cmd: torch.Tensor, goal: torch.Tensor, target: str) -> torch.Tensor:
    """First-horizon-step command [B,4] -> scalar [B,1] for Grad-CAM."""
    if target == "vx":
        return cmd[:, 0:1]
    if target == "vy":
        return cmd[:, 1:2]
    if target == "vz":
        return cmd[:, 2:3]
    if target == "yaw":
        return cmd[:, 3:4]
    if target == "speed":
        return cmd[:, :3].norm(dim=-1, keepdim=True)
    if target == "goal_align":
        heading = goal[:, :2]
        heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (cmd[:, :2] * heading).sum(dim=-1, keepdim=True)
    raise ValueError(f"unknown target: {target}")


def _build_cam(method: str, model: torch.nn.Module, target_layers: list):
    if method == "gradcam":
        from pytorch_grad_cam import GradCAM
        return GradCAM(model=model, target_layers=target_layers)
    if method == "gradcam++":
        from pytorch_grad_cam import GradCAMPlusPlus
        return GradCAMPlusPlus(model=model, target_layers=target_layers)
    raise ValueError(f"unknown method: {method}")


def main() -> None:
    p = argparse.ArgumentParser(description="Grad-CAM saliency for nav_policy checkpoints.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--k", type=int, default=50, help="Window end index in cache")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frame-index", type=int, default=-1, help="Which of T frames (-1=last)")
    p.add_argument(
        "--target",
        choices=("goal_align", "vx", "vy", "vz", "yaw", "speed"),
        default="goal_align",
        help="Policy output to explain (default: velocity aligned with goal heading)",
    )
    p.add_argument(
        "--method",
        choices=("gradcam", "gradcam++"),
        default="gradcam++",
        help="Grad-CAM variant (gradcam++ is usually sharper)",
    )
    p.add_argument(
        "--layer",
        type=int,
        choices=(3, 4),
        default=4,
        help="ResNet layer to hook (4=semantic, 3=more spatial detail)",
    )
    args = p.parse_args()

    try:
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except Exception as exc:
        raise SystemExit(
            f"grad-cam is required for this script.\n"
            f"  Python: {sys.executable}\n"
            f"  Install: {sys.executable} -m pip install grad-cam\n"
            f"  Original error: {exc}"
        ) from exc

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    T = int(cfg["window"]["T"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).eval().to(device)
    model.load_state_dict(ckpt["model"])
    uses_depth = model_uses_depth(cfg)
    goal_input_dim = int(cfg["model"].get("goal_input_dim", 3))
    goal_scale = float(cfg["model"].get("goal_distance_scale", 5.0))

    blob = torch.load(args.cache, weights_only=False, map_location="cpu")
    k = int(args.k)
    n_frames = int(blob["rgb"].shape[0])
    if k < T - 1 or k >= n_frames:
        raise SystemExit(f"--k={k} out of range for cache with {n_frames} frames (need {T - 1}..{n_frames - 1})")

    rgb_u8 = blob["rgb"][k - T + 1 : k + 1]
    fi = args.frame_index if args.frame_index >= 0 else T - 1
    if fi < 0 or fi >= T:
        raise SystemExit(f"--frame-index={fi} out of range for T={T}")

    frame_u8 = rgb_u8[fi].float() / 255.0
    rgb_norm = imagenet_normalize(rgb_u8, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    rgb_seq = rgb_norm.unsqueeze(0).to(device)
    goal = _load_goal(blob, k, goal_input_dim, goal_scale).unsqueeze(0).to(device)

    depth_seq = None
    if uses_depth:
        if "depth" not in blob:
            raise RuntimeError("cache missing depth; run precompute_da2_depth.py")
        dep = blob["depth"][k - T + 1 : k + 1].float() / 255.0
        depth_seq = dep.unsqueeze(0).to(device)

    layer_name = f"layer{args.layer}"
    target_layer = getattr(model.visual.backbone, layer_name)

    class _PolicyFrameWrap(torch.nn.Module):
        def __init__(
            self,
            policy: torch.nn.Module,
            rgb_seq: torch.Tensor,
            goal: torch.Tensor,
            depth_seq: torch.Tensor | None,
            frame_index: int,
            target: str,
        ) -> None:
            super().__init__()
            self.policy = policy
            self.rgb_seq = rgb_seq.detach()
            self.goal = goal
            self.depth_seq = depth_seq
            self.frame_index = frame_index
            self.target = target

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seq = self.rgb_seq.clone()
            seq[:, self.frame_index] = x
            if self.depth_seq is not None:
                out = self.policy(seq, self.goal, self.depth_seq)
            else:
                out = self.policy(seq, self.goal)
            return _scalar_target(out[:, 0, :], self.goal, self.target)

    wrap = _PolicyFrameWrap(model, rgb_seq, goal, depth_seq, fi, args.target)
    cam = _build_cam(args.method, wrap, [target_layer])

    input_tensor = rgb_seq[:, fi]
    # cuDNN GRU backward requires training mode; keep frozen BN in eval.
    model.train()
    backbone = model.visual.backbone
    if hasattr(backbone, "bn1"):
        backbone.bn1.eval()
    for bn_module in backbone.layer1.modules():
        if isinstance(bn_module, torch.nn.BatchNorm2d):
            bn_module.eval()

    try:
        with torch.enable_grad():
            grayscale = cam(input_tensor=input_tensor, eigen_smooth=True)[0]
    finally:
        model.eval()

    rgb_hwc = frame_u8.permute(1, 2, 0).numpy()
    overlay = show_cam_on_image(rgb_hwc, grayscale, use_rgb=True)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio
        iio.imwrite(str(out_dir / "gradcam_overlay.png"), overlay.astype(np.uint8))
        iio.imwrite(str(out_dir / "rgb_frame.png"), (rgb_hwc * 255).astype(np.uint8))
    except Exception:
        from PIL import Image
        Image.fromarray(overlay.astype(np.uint8)).save(out_dir / "gradcam_overlay.png")
        Image.fromarray((rgb_hwc * 255).astype(np.uint8)).save(out_dir / "rgb_frame.png")

    meta = {
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "k": k,
        "frame_index": fi,
        "target": args.target,
        "method": args.method,
        "layer": args.layer,
        "arch": cfg.get("model", {}).get("arch", "rgb_resnet18"),
        "goal_heading": goal[0, :2].detach().cpu().tolist(),
        "goal_dist_norm": float(goal[0, 2].item()) if goal.shape[-1] > 2 else None,
    }
    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta))
    print(f"[gradcam] target={args.target} method={args.method} layer={args.layer} k={k} -> {out_dir}")


if __name__ == "__main__":
    main()
