# FiGS-Standalone: Complete Operator Guide

FiGS (Flight in Gaussian Splats) is a drone simulation and data-generation framework built on top of Nerfstudio. It trains Gaussian Splat (3DGS) models from video captures, then simulates quadcopter flight inside those splats with RGB, depth, and CLIP-semantic rendering. This guide covers every capability: training, rendering, simulation, data processing, scene management, and export.

---

## Directory Structure

```
FiGS-Standalone/
├── src/figs/
│   ├── simulator.py                    # Simulator class (main entry point for simulation)
│   ├── render/
│   │   ├── capture_generation.py       # generate_gsplat() — full train pipeline
│   │   └── gsplat_semantic.py          # GSplat class — rendering engine
│   ├── control/
│   │   ├── vehicle_rate_mpc.py         # VehicleRateMPC — default MPC controller
│   │   ├── velocity_controller.py      # VelocityController — cascaded P-controller: (vx,vy,vz,ψ̇) → body rates
│   │   └── base_controller.py          # BaseController abstract class
│   ├── dynamics/
│   │   ├── model_equations.py          # ACADOS ODE quadcopter model
│   │   └── model_specifications.py     # Frame parameter parsing
│   ├── tsampling/
│   │   ├── build_rrt_dataset.py        # RRT* dataset builder — generates collision-free trajectories toward semantic targets
│   │   └── rrt_datagen_v10.py          # RRT* core algorithm (used by build_rrt_dataset.py)
│   ├── tsplines/
│   │   └── min_snap.py                 # Minimum-snap trajectory optimization
│   ├── utilities/
│   │   ├── trajectory_helper.py        # State/pose conversion utilities
│   │   └── capture_helper.py           # RANSAC transform, ArUco utilities
│   ├── scene_editing/
│   │   └── scene_editing_utils.py      # Point cloud filtering, CLIP semantic filtering
│   └── visualize/
│       ├── generate_videos.py          # images_to_mp4()
│       └── plot_trajectories.py        # 3D trajectory plotting (plotly)
├── notebooks/
│   ├── figs_3dgs_oneliner.py           # Minimal: train 3DGS only
│   ├── figs_generate_3dgs_example.py   # Full: train + verify
│   └── figs_simulate_flight_example.py # Full: simulate + render video
├── configs/
│   ├── capture/                        # Camera intrinsics + capture mode
│   ├── frame/                          # Drone physical specs (mass, inertia, etc.)
│   ├── rollout/                        # Simulation noise, delay, frequency
│   ├── policy/                         # MPC cost weights
│   ├── course/                         # Reference trajectory (min-snap waypoints)
│   ├── perception/
│   │   └── perception_mode.yml         # CRITICAL: controls what the sim renders
│   └── method/                         # SINGER data-gen parameters (used by SINGER)
├── 3dgs/                               # Symlink → /data/kothari1/singer_figs_data/3dgs  ⚠ must be created: ln -s /data/kothari1/singer_figs_data/3dgs 3dgs
│   └── workspace/
│       ├── {scene_name}/               # SfM data + transforms.json
│       └── outputs/{scene_name}/       # Trained model checkpoints
└── docker-compose.base.yml
```

---

## 1. Docker: Getting Into the Container

All FiGS Python code runs inside Docker (Nerfstudio, ACADOS, COLMAP, tiny-cuda-nn are not on the host).

```bash
# From the LEAD root:
cd FiGS-Standalone
docker compose -f docker-compose.base.yml run --rm figs
```

Inside the container you get a shell with:
- `figs` and `gemsplat` installed in editable mode
- All `ns-*` CLI commands available
- CUDA, COLMAP, FFmpeg, Open3D

**Build the image** (only if Dockerfile changed):
```bash
# L40S GPU (this server):
CUDA_ARCHITECTURES=89 docker compose -f docker-compose.base.yml build

# A6000 / RTX 3090:
CUDA_ARCHITECTURES=86 docker compose -f docker-compose.base.yml build
```

**Key env vars set by `.env`:**
```
DATA_PATH=/data/kothari1/singer_figs_data
CUDA_VISIBLE_DEVICES=1
DISPLAY=:1
```

The `DATA_PATH` is mounted at the same absolute path inside Docker so symlinks (e.g. `3dgs/` → `/data/.../3dgs`) resolve identically.

---

## 2. Perception Mode (Critical Config)

**File:** `configs/perception/perception_mode.yml`

This file is read by both `Simulator` and `GSplat` at load time. **Change it before running any simulation or rollout generation.**

```yaml
# Option A — RGB only (fast, no CLIP inference):
visual_mode: "rgb"
perception_type: null

# Option B — RGB + depth + CLIP semantic heatmap (needed for Semantic_HSM training):
visual_mode: "semantic_depth"
perception_type: "similarity"   # uses nerfstudio's built-in CLIP cosine similarity

# NEVER use:
# perception_type: "clipseg"   # broken tuple-return bug, slow HuggingFace download
```

`visual_mode` determines what channels `Simulator.simulate()` returns in `Iro`:
| `visual_mode` | `Iro` keys |
|---|---|
| `"rgb"` | `rgb` |
| `"semantic_depth"` | `rgb`, `depth`, `semantic` |

---

## 3. Training a 3DGS Model

### 3a. One-liner (Python, inside Docker)

```python
import figs.render.capture_generation as pg

pg.generate_gsplat(
    scene_file_name="flightroom_ssv_exp",  # must match video in 3dgs/captures/
    capture_cfg_name="iphone15pro_142",    # camera config (see §3c)
    force_recompute=False                  # True to redo all stages
)
```

**Source:** `src/figs/render/capture_generation.py` → `generate_gsplat()`

The pipeline has 5 checkpointed stages (skipped automatically if already complete):
1. Load capture config
2. Extract PNG frames from video (`3dgs/captures/{scene}.mp4`)
3. Run COLMAP SfM → `transforms.json` + sparse point cloud
4. RANSAC coordinate alignment (only in `"semantic"` mode, uses ArUco markers)
5. `ns-train` (splatfacto or gemsplat)

**Notebook:** `notebooks/figs_3dgs_oneliner.py` or `notebooks/figs_generate_3dgs_example.py`

### 3b. Direct `ns-train` commands (inside Docker, inside `3dgs/workspace/`)

**Standard splatfacto** (RGB-only, no CLIP features):
```bash
cd /workspace/FiGS-Standalone/3dgs/workspace
ns-train splatfacto \
  --data {scene_name} \
  --pipeline.model.camera-optimizer.mode SO3xR3 \
  --pipeline.model.rasterize-mode antialiased
```

**gemsplat** (with CLIP semantic features — required for `semantic_depth` mode):
```bash
cd /workspace/FiGS-Standalone/3dgs/workspace
ns-train gemsplat \
  --data {scene_name} \
  --pipeline.model.camera-optimizer.mode SO3xR3 \
  --viewer.quit-on-train-completion True \
  --output-dir outputs \
  nerfstudio-data \
    --orientation-method none \
    --center-method none
```

Output goes to: `3dgs/workspace/outputs/{scene_name}/{model_type}/{timestamp}/`

### 3c. Capture Config Files

**Location:** `configs/capture/`

| File | Use |
|---|---|
| `iphone15pro_142.json` | iPhone 15 Pro, 142° horizontal sweep |
| `iphone15pro_341.json` | iPhone 15 Pro, 341° sweep |
| `pixel8pro.json` | Google Pixel 8 Pro |
| `semantic_iphone15pro_142.json` | iPhone 15 Pro + ArUco marker detection (semantic mode) |
| `semantic_iphone15pro.json` | Same, alternate sweep |
| `rgb_iphone15pro.json` | iPhone 15 Pro, RGB-only mode |
| `default.json` | Generic default intrinsics |
| `camera_front.json` | Forward-facing camera |

**Config structure:**
```json
{
  "camera": {
    "height": 360, "width": 640, "channels": 3,
    "fx": 462.956, "fy": 463.002, "cx": 323.076, "cy": 181.184
  },
  "extractor": {
    "num_images": 150,
    "aruco_dict_name": "6X6_250",
    "marker_length": 0.0525
  },
  "mode": "semantic"  // "rgb" | "semantic" | "no_ransac"
}
```

**`mode` values:**
- `"rgb"` — splatfacto training, no coordinate alignment
- `"semantic"` — gemsplat training + RANSAC world-frame alignment via ArUco markers
- `"no_ransac"` — gemsplat training, skip RANSAC even if markers detected

### 3d. Raw Data Requirements

Place captures at: `3dgs/captures/{scene_name}.mp4`

For `"semantic"` mode the video must contain ArUco 6×6_250 markers of size 52.5mm visible in multiple frames for RANSAC alignment.

---

## 4. Scene Model Discovery

`Simulator` and `GSplat` find the trained model automatically:

**Search order:**
1. `3dgs/workspace/outputs/{scene_name}/**/*.yml`
2. `$DATA_PATH/trained_gsplats/{scene_name}/**/*.yml`

**If multiple `*.yml` found → error.** You must either:
- Specify the exact sub-path as scene_name: `"flightroom_ssv_exp/gemsplat/2026-02-28_205058"`
- Or delete old checkpoints

**Available trained models on this server:**
```
# gemsplat (preferred):
3dgs/workspace/outputs/flightroom_ssv_exp/gemsplat/2026-02-28_205058/
  → step-000029999.ckpt

# gemsplat (older fallback):
$DATA_PATH/trained_gsplats/flightroom_ssv_exp/gemsplat/2026-02-03_115017/
  → step-000028000.ckpt

# splatfacto:
3dgs/workspace/outputs/flightroom/splatfacto/2024-07-12_145513/
  → step-000029999.ckpt
```

---

## 5. Simulating Drone Flight

### 5a. Simulator class

**File:** `src/figs/simulator.py`

```python
from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC

sim = Simulator(
    scene_name="flightroom_ssv_exp",   # or exact sub-path if ambiguous
    rollout_name="baseline",            # rollout config (noise, freq, delay)
    frame_name="carl"                   # drone frame (mass, inertia, camera)
)
ctl = VehicleRateMPC(
    course_name="track_spiral",         # reference trajectory
    policy_name="vrmpc_rrt",            # MPC cost weights
    frame_name="carl"
)

t0, tf, x0 = ctl.tXUd[0,0], ctl.tXUd[0,-1], ctl.tXUd[1:11, 0]

Tro, Xro, Uro, Iro, Tsol, Adv = sim.simulate(
    policy=ctl,
    t0=t0,
    tf=tf,
    x0=x0,
    query="ladder",            # optional: CLIP semantic query string
    vision_processor=None,     # optional: external vision backend
    validation=False,          # if True, also runs clipseg for comparison
    verbose=False              # if True, prints per-frame timing
)
```

**Returns:**
| Variable | Shape | Description |
|---|---|---|
| `Tro` | `(Nctl+1,)` | Time stamps at control rate |
| `Xro` | `(10, Nctl+1)` | State: `[px,py,pz, vx,vy,vz, qx,qy,qz,qw]` |
| `Uro` | `(4, Nctl)` | Controls: `[thrust, roll_rate, pitch_rate, yaw_rate]` |
| `Iro` | `dict` | Rendered images; keys depend on `perception_mode.yml` |
| `Tsol` | `(4, Nctl)` | Timing: `[solver, control, render, total]` per step |
| `Adv` | `(4, Nctl)` | Adaptive terms from controller |

**`Iro` dict keys:**
- `"rgb"` — `(Nctl, H, W, 3)` uint8 always present
- `"depth"` — `(Nctl, H, W, 3)` uint8 JET colormap (if `semantic_depth`)
- `"semantic"` — `(Nctl, H, W, 3)` uint8 turbo colormap (if `semantic_depth` + `query`)
- `"depth_raw"` — raw float depth (not stacked, not in output array)

**Note:** `del ctl` after each policy to avoid ACADOS re-initialization errors.

### 5b. Simulator runtime swap methods

```python
# Swap scene without recreating simulator:
sim.load_scene("backroom")

# Swap rollout config (noise, delay, frequency):
sim.load_rollout("baseline")          # by name (loads from configs/rollout/)
sim.load_rollout({"frequency": 50, ...})  # or pass dict directly

# Swap drone frame (rebuilds ACADOS solver):
sim.load_frame("carl")
sim.load_frame("carl_hires")

# Reload perception config from disk:
sim.load_perception()
```

### 5c. Notebook example

```bash
# Inside container:
cd /workspace/FiGS-Standalone
python3 notebooks/figs_simulate_flight_example.py
```

---

## 6. Direct Rendering (without simulation)

Use `GSplat` directly to render images at arbitrary camera poses.

```python
from figs.render.gsplat_semantic import GSplat
import numpy as np

gsplat = GSplat({"name": "flightroom_ssv_exp",
                 "path": "<path-to-config.yml>"})

# Generate output camera from frame config
camera = gsplat.generate_output_camera({
    "height": 360, "width": 640, "channels": 3,
    "fx": 462.956, "fy": 463.002, "cx": 323.076, "cy": 181.184
})

# T_c2w: 4x4 camera-to-world transform (OpenCV convention)
T_c2w = np.eye(4)   # replace with actual pose

# RGB only:
out = gsplat.render_rgb(camera, T_c2w)
# out["rgb"]       → (H, W, 3) uint8
# out["depth"]     → (H, W, 3) uint8  JET colormap
# out["depth_raw"] → (H, W)    float  metric depth

# RGB + semantic heatmap for a query:
out = gsplat.render_rgb(camera, T_c2w, query="green clock")
# out["semantic"]  → (H, W, 3) uint8  turbo colormap CLIP similarity
```

**Note:** `GSplat` reads `perception_mode.yml` at construction time and raises `ValueError` if `visual_mode` is not `"rgb"` or `"semantic_depth"`.

---

## 7. Video Generation

```python
import figs.visualize.generate_videos as gv

# Iro["rgb"] has shape (N, H, W, 3)
gv.images_to_mp4(
    images=Iro["rgb"],
    filename="output/my_flight.mp4",
    fps=ctl.hz                  # match controller rate (typically 50 or 100 Hz)
)
```

**File:** `src/figs/visualize/generate_videos.py`
**Backend:** FFMPEG via imageio. Creates parent directories automatically.

---

## 8. Point Cloud Extraction and Scene Editing

```python
import figs.scene_editing.scene_editing_utils as scdt

# Extract Gaussian means as colored point cloud
pcd, mask, env_attr = sim.gsplat.generate_point_cloud(
    use_bounding_box=True,
    bounding_box_min=(-3.7, -8.0, -2.0),
    bounding_box_max=( 3.7,  8.0,  0.0),
    densify_scene=False,   # split Gaussians to densify; use split_params to configure
    cull_scene=False       # remove low-opacity/small Gaussians; use cull_params
)
# pcd → open3d.geometry.PointCloud
# env_attr → dict with keys: means, scales, quats, opacities, clip_embeds

# Optional split / cull params:
split_params = {"n_split_samples": 2}
cull_params  = {"cull_alpha_thresh": 0.1, "cull_scale_thresh": 0.5}

# Filter point cloud by CLIP semantics:
semantic_pcd = sim.gsplat.get_semantic_point_cloud(
    positives="ladder",
    negatives="",
    pcd_attr=env_attr
)

# Visualize in plotly (works inside Jupyter):
scdt.plot_point_cloud(sim, (Tro, Xro, Uro), n_points=50)
```

---

## 9. Nerfstudio CLI Reference (inside container)

### Processing raw data

```bash
# Video → frames → SfM (Nerfstudio format):
cd /workspace/FiGS-Standalone/3dgs/workspace
ns-process-data video \
  --data <video-directory-name> \
  --output-dir .

# Images → SfM:
ns-process-data images \
  --data <images-dir> \
  --output-dir <output-dir>
```

### Training

```bash
# See all available models:
ns-train --help

# splatfacto (RGB):
ns-train splatfacto \
  --data <data-dir> \
  --pipeline.model.camera-optimizer.mode SO3xR3 \
  --pipeline.model.rasterize-mode antialiased

# gemsplat (semantic):
ns-train gemsplat \
  --data <data-dir> \
  --pipeline.model.camera-optimizer.mode SO3xR3 \
  --viewer.quit-on-train-completion True \
  --output-dir outputs \
  nerfstudio-data --orientation-method none --center-method none

# Resume from checkpoint:
ns-train splatfacto \
  --data <data-dir> \
  --load-checkpoint outputs/<scene>/<model>/<timestamp>/checkpoints/step-000XXXXX.ckpt
```

### Rendering (camera-path or spiral)

```bash
# Render a camera path (defined in nerfstudio viewer):
ns-render camera-path \
  --load-config outputs/{scene}/{model}/{timestamp}/config.yml \
  --camera-path-filename camera_path.json \
  --output-path renders/output.mp4

# Render interpolated path between poses:
ns-render interpolate \
  --load-config outputs/{scene}/{model}/{timestamp}/config.yml \
  --output-path renders/output.mp4 \
  --pose-source train                  # or "eval"

# Render at dataset camera poses:
ns-render dataset \
  --load-config outputs/{scene}/{model}/{timestamp}/config.yml \
  --output-path renders/
```

### Export

```bash
# Export .ply Gaussian splats file (for viewers like Supersplat, 3DGS viewer):
ns-export gaussian-splat \
  --load-config outputs/{scene}/{model}/{timestamp}/config.yml \
  --output-dir outputs/{scene}/{model}/{timestamp}/exports/

# Export Poisson mesh:
ns-export poisson \
  --load-config outputs/{scene}/{model}/{timestamp}/config.yml \
  --output-dir outputs/{scene}/{model}/{timestamp}/exports/ \
  --normal-method open3d

# Export point cloud (.ply):
ns-export pointcloud \
  --load-config outputs/{scene}/{model}/{timestamp}/config.yml \
  --output-dir outputs/{scene}/{model}/{timestamp}/exports/ \
  --num-points 1000000
```

### Viewer / Evaluation

```bash
# Launch interactive viewer on a trained model:
ns-viewer --load-config outputs/{scene}/{model}/{timestamp}/config.yml

# Evaluate PSNR/SSIM/LPIPS on held-out frames:
ns-eval \
  --load-config outputs/{scene}/{model}/{timestamp}/config.yml \
  --output-path eval_results.json
```

---

## 10. Config Reference

### `configs/rollout/baseline.json`
```json
{
  "frequency": 100,       // simulation Hz (also determines image capture rate)
  "delay": 0.0,           // control delay in seconds
  "model_noise": {
    "mean": [...],        // 10-vector: zero-mean Gaussian on dynamics state
    "std":  [...]         // non-zero values add process noise to each state dim
  },
  "sensor_noise": {
    "mean": [...],
    "std":  [...]         // non-zero adds noise to the state estimate seen by controller
  },
  "sensor_model_fusion": {
    "use_fusion": false,
    "weights": [...]      // per-state fusion weight between sensor and model
  }
}
```

### `configs/frame/` — Available Frames

| File | Description |
|---|---|
| `carl.json` | 300g research quadcopter, 360×640 forward camera |
| `carl_hires.json` | Carl with higher-resolution camera |
| `hires_zed.json` | ZED stereo camera variant |

**Key fields in frame config:**
```json
{
  "mass": 1.144,
  "massless_inertia": [Ixx, Iyy, Izz],
  "arm_front": [dx, dy], "arm_back": [dx, dy],
  "force_normalized": 6.90,   // thrust-to-weight normalization
  "torque_gain": 0.040,
  "number_of_rotors": 4,
  "camera_to_body_transform": [...],  // 4x4 T_c2b
  "camera": {"height": 360, "width": 640, "fx": ..., "fy": ..., "cx": ..., "cy": ...}
}
```

### `configs/policy/` — Available Policies

| File | Description |
|---|---|
| `vrmpc_rrt.json` | MPC tuned for RRT-based trajectories (used by SINGER) |
| `vrmpc_fr.json` | MPC tuned for free-flight / fixed routes |
| `vrmpc_br.json` | MPC with back-reference cost |

### `configs/course/` — Available Courses

| File | Description |
|---|---|
| `track_spiral.json` | Spiral inspection trajectory |

Course configs define min-snap waypoints fed to `VehicleRateMPC`.

---

## 11. Trajectory Utilities

```python
import figs.utilities.trajectory_helper as th

# State vector → 4x4 pose matrix:
T_b2w = th.xv_to_T(x)            # x is 10-vector [pos, vel, quat_xyzw]

# Quaternion continuity enforcement (avoids flips):
q_new = th.obedient_quaternion(q_new, q_prev)

# Plot spatial trajectory (plotly):
import figs.visualize.plot_trajectories as pt
pt.plot_RO_spatial((Tro, Xro, Uro))
```

---

## 12. Common Workflows

### A. Train a new scene from scratch
```bash
# 1. Place video: 3dgs/captures/my_scene.mp4
# 2. Enter container:
docker compose -f docker-compose.base.yml run --rm figs
# 3. Inside container:
python3 - <<'EOF'
import figs.render.capture_generation as pg
pg.generate_gsplat("my_scene", capture_cfg_name="iphone15pro_142", force_recompute=False)
EOF
```

### B. Simulate flight and save video
```bash
docker compose -f docker-compose.base.yml run --rm figs
# Inside container:
python3 notebooks/figs_simulate_flight_example.py
# Output: test_space/track_spiral_flightroom_ssv_exp.mp4
```

### C. Render semantic video at arbitrary pose sequence
```python
# Inside container (Python):
from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC
import figs.visualize.generate_videos as gv
import numpy as np

sim = Simulator("flightroom_ssv_exp", "baseline", "carl")
ctl = VehicleRateMPC("track_spiral", "vrmpc_rrt", "carl")
t0, tf, x0 = ctl.tXUd[0,0], ctl.tXUd[0,-1], ctl.tXUd[1:11,0]
_, _, _, Iro, _, _ = sim.simulate(ctl, t0, tf, x0, query="green clock")
gv.images_to_mp4(Iro["rgb"],      "out/rgb.mp4",      ctl.hz)
gv.images_to_mp4(Iro["semantic"], "out/semantic.mp4",  ctl.hz)
gv.images_to_mp4(Iro["depth"],    "out/depth.mp4",     ctl.hz)
del ctl
```

### D. Export splat for external viewer
```bash
# Inside container, from 3dgs/workspace/:
ns-export gaussian-splat \
  --load-config outputs/flightroom_ssv_exp/gemsplat/2026-02-28_205058/config.yml \
  --output-dir outputs/flightroom_ssv_exp/gemsplat/2026-02-28_205058/exports/
# Produces: exports/gaussian_splat.ply  (open in Supersplat or 3DGS viewer)
```

### E. Evaluate a trained model
```bash
# Inside container, from 3dgs/workspace/:
ns-eval \
  --load-config outputs/flightroom_ssv_exp/gemsplat/2026-02-28_205058/config.yml \
  --output-path eval.json
```

---

## 13. Known Gotchas

| Issue | Cause | Fix |
|---|---|---|
| `ValueError: search path returned multiple configurations` | Multiple `config.yml` files under scene name | Pass exact sub-path: `"flightroom_ssv_exp/gemsplat/2026-02-28_205058"` |
| `FileNotFoundError: transforms.json not found` | `visual_mode: semantic_depth` can't find SfM data | Ensure `3dgs/workspace/{scene_name}/transforms.json` exists |
| Simulation hangs or crashes | ACADOS solver not re-initialized after `del ctl` | Always `del ctl` after each simulation loop iteration |
| Dark/wrong semantic heatmap | `perception_type: clipseg` in `perception_mode.yml` | Change to `perception_type: similarity` |
| OOM on GPU | Running on GPU 0 (full) | Set `CUDA_VISIBLE_DEVICES=1` in `.env` |
| `/home` partition full | Writing outputs to home | All large outputs must go to `/data/kothari1/singer_figs_data/` |
