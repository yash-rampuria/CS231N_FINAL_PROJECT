# V-LEAD Framework: User Instructions

> **Last updated:** 2026-05-22
> **Audience:** Developers and researchers using the V-LEAD repo for quadcopter autonomy research.
> For AI agent context (architecture, gotchas, file maps), see `AGENT_CONTEXT.md`.

---

## What is V-LEAD?

V-LEAD integrates two systems (as git submodules):

- **FiGS-Standalone** (*Flying in Gaussian Splats*): Physics-accurate quadcopter simulator flying through 3D Gaussian Splat environments. Provides dynamics, MPC expert controller, RRT* trajectory planning, and rendering.

- **SINGER** (*Scene Understanding via Synthesized Visual Inertial Data from Experts*): Learned autonomy layer on top of FiGS. Trains vision-language navigation policies via DAgger-style imitation learning from expert MPC demonstrations.

> **Note:** `Semantic_HSM` (Semantic Hierarchical State Machine) is **not** a submodule of V-LEAD. Clone it separately if needed. Sections below that reference it assume it is checked out as a sibling directory.

The full pipeline: capture real environment → train 3DGS → generate expert rollouts → train neural policy → deploy in simulation.

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| Host | `coruscant`, user `kothari1` |
| GPU | Available via Docker (no `runtime: nvidia`; use `deploy.resources.reservations.devices`) |
| Docker | No sudo needed — `kothari1` is in `docker` group |
| Python | 3.10 (inside containers only — do not run pipeline scripts on host) |
| Legacy data drive | `/data/kothari1/singer_figs_data` — old SINGER data + 3dgs scene (nearly full) |
| **Primary output drive** | `/project/kothari1/` — 7 TB, 6.6 TB free — **all new V-LEAD outputs go here** |
| Home disk | `/home/kothari1` is near capacity — never write large files there |

### Critical Directory Symlinks
These symlinks redirect large data to the correct drives automatically:
- `FiGS-Standalone/3dgs` → `/data/kothari1/singer_figs_data/3dgs` (3DGS scenes; legacy drive)
- `SINGER/cohorts` → `/project/kothari1/vlead_data/rollouts_vlead` (V-LEAD rollout outputs; project drive)

**Do not delete these symlinks.** They work inside Docker because both `/data/` and `/project/` are mounted at the same absolute paths inside the container (`docker-compose.yml` mounts both).

---

## Environment Setup

### 1. Build the FiGS Docker Image (one-time, ~20 min)
Only needed once. Already built on coruscant as `figs:latest`.
```bash
cd /home/kothari1/autonomy_projects/V-LEAD/FiGS-Standalone
git submodule update --init gemsplat
CUDA_ARCHITECTURES=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.') \
  docker compose build
```

### 2. Install Python Dependencies (one-time per Docker volume)
The `site-packages` Docker volume persists across runs. Run this once:
```bash
cd /home/kothari1/autonomy_projects/V-LEAD/SINGER
docker compose run --rm singer bash -lc "
  python3 -m pip install typer 'transformers==4.40.0' 'huggingface_hub==0.23.0' shapely scikit-image imageio
"
```
> **Version pinning is critical:** `transformers==4.40.0` is the highest version compatible with PyTorch 2.1.2+cu118 in the container.

### 3. Verify `.env` Files
Both repos have `.env` files that must point to the data drive:

**`FiGS-Standalone/.env`:**
```
DATA_PATH=/data/kothari1/singer_figs_data
```

**`SINGER/.env`:**
```
DATA_PATH=/data/kothari1/singer_figs_data
FIGS_PATH=../FiGS-Standalone
```

---

## Container Entry Points

### FiGS Container
```bash
cd /home/kothari1/autonomy_projects/V-LEAD/FiGS-Standalone
docker compose -f docker-compose.base.yml run --rm figs
# Working directory inside: /workspace/FiGS-Standalone
```

### SINGER Container (primary development environment)
```bash
cd /home/kothari1/autonomy_projects/V-LEAD/SINGER
docker compose run --rm singer
# Working directory inside: /workspace/SINGER
# Editable installs: figs, gemsplat, sousvide
# Bind mounts: FiGS-Standalone (Semantic_HSM mounted only if present at ../Semantic_HSM)
```

---

## Full Pipeline: Training a Neural Pilot

All steps run inside the SINGER container at `/workspace/SINGER`. Use `smoke_test.yml` for testing (33 batches, 5 epochs) or `ssv_multi3dgs.yml` for production.

### Step 1: Generate Training Rollouts
MPC expert flies RRT* paths toward semantic targets. Applies domain randomization (mass, force perturbations).
```bash
python3 notebooks/ssv_multi3dgs_campaign.py generate-rollouts \
    --config-file configs/experiment/smoke_test.yml
```
**Output:** `trajectories{id:05d}.pt`, `imgdata{id:05d}.pt`, `video{id:05d}.mp4` in `.../rollout_data/<scene>/`

### Step 2: Generate Validation Rollouts
Same as Step 1 but without domain randomization. Also renders multi-channel video (RGB + depth + semantic) needed for `Semantic_HSM` training.
```bash
python3 notebooks/ssv_multi3dgs_campaign.py generate-rollouts \
    --config-file configs/experiment/smoke_test.yml --validation-mode
```
**Output:** `trajectories_val{id:05d}.pt`, `video_val_rollout_images_{semantic,rgb,depth}{id:05d}.mp4` in `.../rollout_data/<scene>/`

> **Perception mode required:** `FiGS-Standalone/configs/perception/perception_mode.yml` must have:
> ```yaml
> visual_mode: "semantic_depth"
> perception_type: "similarity"
> ```

### Step 3: Generate Observations
Replays expert trajectories through the Pilot's observation pipeline. Produces `(state, image, expert_action)` training tuples.
```bash
python3 notebooks/ssv_multi3dgs_campaign.py generate-observations \
    --config-file configs/experiment/smoke_test.yml
```
**Output:** `observations_{id:05d}.pt` in `.../observation_data/<pilot_name>/`

### Step 4: Train History Encoder
Trains `HistoryEncoder` (Parameter module) to predict drone model parameters from delta-state history.
```bash
python3 notebooks/ssv_multi3dgs_campaign.py train-history \
    --config-file configs/experiment/smoke_test.yml
```
**Output:** History encoder checkpoint in `.../roster/<pilot_name>/`

### Step 5: Train Commander Module
Locks the trained HistoryEncoder. Trains `VisionMLP` + `CommanderSV` jointly.
```bash
python3 notebooks/ssv_multi3dgs_campaign.py train-command \
    --config-file configs/experiment/smoke_test.yml
```
**Output:** Full policy checkpoint in `.../roster/<pilot_name>/`

### Step 6: Evaluate
Deploy trained pilot in the gsplat scene and evaluate performance.
```bash
python3 notebooks/ssv_multi3dgs_campaign.py simulate \
    --config-file configs/experiment/smoke_test.yml
```

---

## V-LEAD Visuomotor Data Generation (CS231N)

This pipeline generates expert demonstration data for training a goal-conditioned visuomotor policy: `(RGB frames, goal heading, distance) → velocity commands`. It extends the standard SINGER rollout generator with two key additions:

1. **Goal heading supervision** — each timestep stores a unit 2D vector pointing from the drone toward the target object, plus scalar distance.
2. **Multi-channel video output** — RGB, depth (JET colormap), and CLIP semantic heatmap videos saved alongside trajectory data.

All steps run **inside the SINGER container**.

---

### Prerequisites

Before running, confirm:

1. **Perception mode** is set correctly in `FiGS-Standalone/configs/perception/perception_mode.yml`:
   ```yaml
   visual_mode: "semantic_depth"
   perception_type: "similarity"
   ```

2. **Symlinks** exist (see Prerequisites section above):
   ```bash
   ls -la SINGER/cohorts          # → /project/kothari1/vlead_data/rollouts_vlead
   ls -la FiGS-Standalone/3dgs    # → /data/kothari1/singer_figs_data/3dgs
   ```

3. **Project drive** is mounted in Docker. Check `SINGER/docker-compose.yml` — it must include:
   ```yaml
   - /project/kothari1:/project/kothari1
   ```

---

### Step 0: Dry Run (Recommended First)

Before committing to a full run, verify the pipeline works end-to-end with 1 trajectory per target object:

```bash
cd ~/autonomy_projects/V-LEAD/SINGER
CUDA_VISIBLE_DEVICES=1 docker compose run --rm singer bash -lc \
  "CUDA_VISIBLE_DEVICES=1 python3 notebooks/generate_training_data.py \
   --config-file configs/experiment/vlead_dryrun.yml \
   --validation-mode"
```

Expected output: 5 batches (one per target object), ~2 min total. Check:
```bash
ls /project/kothari1/vlead_data/rollouts_vlead/vlead_dryrun/rollout_data/
```

---

### Step 1: Generate Full-Length Validation Trajectories

These are complete spawn-to-goal trajectories with no domain randomization. Good for training via behavior cloning.

**Single container (sequential, ~3–4 hours for all 550 trajectories):**
```bash
cd ~/autonomy_projects/V-LEAD/SINGER
CUDA_VISIBLE_DEVICES=1 docker compose run --rm singer bash -lc \
  "CUDA_VISIBLE_DEVICES=1 python3 notebooks/generate_training_data.py \
   --config-file configs/experiment/vlead_flightroom.yml \
   --validation-mode"
```

**Parallel (3–4× faster — see "Parallel Execution" below).**

---

### Step 2: Generate Training-Mode Trajectories (Optional)

Short 2-second segments with domain randomization (mass/force perturbations, 4 reps per branch). Higher volume, more diversity.

```bash
cd ~/autonomy_projects/V-LEAD/SINGER
CUDA_VISIBLE_DEVICES=1 docker compose run --rm singer bash -lc \
  "CUDA_VISIBLE_DEVICES=1 python3 notebooks/generate_training_data.py \
   --config-file configs/experiment/vlead_flightroom.yml"
```

---

### Parallel Execution

The server has multiple GPUs. Only L40S GPUs (sm_89) work with the current Docker image — the Blackwell GPUs (sm_120) are **not compatible** with the PyTorch version in the container.

Available GPUs for data gen (as of 2026-05-22):
- **GPU 0 (L40S):** usually occupied — check `nvidia-smi` first
- **GPU 1 (L40S, 46 GB):** primary workhorse
- **GPU 2/3 (Blackwell):** incompatible with current image — skip

The `--query-indices` flag lets each container process a subset of target objects. Run multiple containers simultaneously on GPU 1 — each compiles its ACADOS solvers into an isolated temp directory (race-condition-safe after a [fix in FiGS-Standalone](#acados-parallel-fix)):

```bash
cd ~/autonomy_projects/V-LEAD/SINGER

# Val mode — split 5 queries across 3 containers on GPU 1
# Pane 1: queries 0+1 (green clock, leafblower)
CUDA_VISIBLE_DEVICES=1 docker compose run --rm singer bash -lc \
  "CUDA_VISIBLE_DEVICES=1 python3 notebooks/generate_training_data.py \
   --config-file configs/experiment/vlead_flightroom.yml --validation-mode --query-indices 0,1"

# Pane 2: queries 2+3 (drill, mannequin)
CUDA_VISIBLE_DEVICES=1 docker compose run --rm singer bash -lc \
  "CUDA_VISIBLE_DEVICES=1 python3 notebooks/generate_training_data.py \
   --config-file configs/experiment/vlead_flightroom.yml --validation-mode --query-indices 2,3"

# Pane 3: query 4 (ladder)
CUDA_VISIBLE_DEVICES=1 docker compose run --rm singer bash -lc \
  "CUDA_VISIBLE_DEVICES=1 python3 notebooks/generate_training_data.py \
   --config-file configs/experiment/vlead_flightroom.yml --validation-mode --query-indices 4"
```

Run training-mode containers in additional panes simultaneously if desired. Each run writes to its own timestamped output directory — no conflicts.

Monitor progress:
```bash
watch -n 30 'ls /project/kothari1/vlead_data/rollouts_vlead/vlead_flightroom/rollout_data/'
```

---

### Output Structure

All outputs land at:
```
/project/kothari1/vlead_data/rollouts_vlead/<cohort>/rollout_data/<YYYY-MM-DD_HHMMSS>/<scene>/
```

Each timestamped directory is one complete run. Files per batch index `id`:

| File | Content |
|------|---------|
| `trajectories_val{id:05d}.pt` | Trajectory dict — see fields below |
| `imgdata_val{id:05d}.pt` | Frame index metadata |
| `video_val_rollout_images_semantic{id:05d}.mp4` | CLIP semantic heatmap video |
| `video_val_rollout_images_rgb{id:05d}.mp4` | RGB drone-eye video |
| `video_val_rollout_images_depth{id:05d}.mp4` | Depth video (JET colormap) |

Training-mode files use the same names without `_val`.

**Trajectory dict fields** (loaded via `torch.load(...)`):

| Field | Shape | Description |
|-------|-------|-------------|
| `Tro` | `(N+1,)` | Timestamps |
| `Xro` | `(10, N)` | State: `[px,py,pz, vx,vy,vz, qx,qy,qz,qw]` |
| `Uro` | `(4, N)` | Controls: `[thrust, ωx, ωy, ωz]` |
| `goal_xy` | `(2,)` | 2D XY centroid of target object (world frame) |
| `heading_vec` | `(2, N)` | Unit vector from drone to goal at each timestep |
| `dist` | `(N,)` | Scalar XY distance from drone to goal |
| `course` | `str` | Target object query string |
| `Ndata` | `int` | Number of timesteps |

---

### Configuration Files

| File | Purpose |
|------|---------|
| `SINGER/configs/experiment/vlead_flightroom.yml` | Full production run (5 queries, 110 branches each) |
| `SINGER/configs/experiment/vlead_dryrun.yml` | Dry run (1 branch per query, fast) |
| `SINGER/configs/scenes/flightroom_ssv_exp.yml` | Scene config: queries, altitudes, RRT* params |
| `SINGER/configs/scenes/flightroom_ssv_exp_dryrun.yml` | Same scene, `nbranches: 1` for dry runs |

**Current target objects** (in `flightroom_ssv_exp.yml`):
- Index 0: `"green clock"`
- Index 1: `"green and pink leafblower"`
- Index 2: `"yellow handheld cordless drill on two boxes"`
- Index 3: `"human mannequin"`
- Index 4: `"ladder"`

To add a new object: append to `queries`, `radii`, `altitudes`, `similarities`, and `nbranches` (one entry each, in order) in the scene config.

---

## Training Semantic_HSM

> **Note:** `Semantic_HSM/` is not a submodule of V-LEAD. Clone it separately and adjust paths below accordingly.

`Semantic_HSM/scripts/train_03.py` trains `VisualNavPolicy_Sequence` on the **validation rollouts** from Step 2. It has no FiGS/SINGER dependencies — runs directly on the host using the `lead_ml` conda env (no Docker needed).

### Python Environment

The `lead_ml` env lives on the data drive (keeps `/home` free). Activate it with:
```bash
conda activate /data/kothari1/singer_figs_data/conda_envs/lead_ml
```
Packages: `torch 2.6.0+cu124`, `torchvision`, `tqdm`, `matplotlib`, `scipy`, `opencv-python`.

To add new packages (always route caches to data drive):
```bash
TMPDIR=/data/kothari1/singer_figs_data/.pip_tmp \
  pip install --cache-dir /data/kothari1/singer_figs_data/.pip_cache <package>
```

### Running Training

**Before running, verify CONFIG settings in `train_03.py`:**
```python
CONFIG = {
    'data_root': '/data/kothari1/singer_figs_data/rollouts_singer/smoke_test/rollout_data/flightroom_ssv_exp',
    'start_idx': 0,
    'end_idx': 32,   # inclusive; covers indices 00000-00032 (33 validation batches)
    ...
}
```

**Run directly on the host** (interactive matplotlib loss plot included):
```bash
conda activate /data/kothari1/singer_figs_data/conda_envs/lead_ml
cd <path-to-Semantic_HSM>/scripts
TORCH_HOME=/data/kothari1/singer_figs_data/.torch_hub \
CUDA_VISIBLE_DEVICES=1 \
  python3 train_03.py
```

> **GPU note:** GPU 0 is typically occupied by other users. `CUDA_VISIBLE_DEVICES=1` targets GPU 1. Verify with `nvidia-smi` before running.

> **`TORCH_HOME`:** Redirects MobileNetV3 weight downloads to the data drive. Required on first run; cached after that.

**Stopping training early** (all options finish the current epoch then save the model):
| Method | How |
|--------|-----|
| Ctrl+C in the terminal | Most convenient — sends SIGINT, handled gracefully |
| `q` or `Esc` in the plot window | Requires the matplotlib window to have keyboard focus |
| Sentinel file | `touch <path-to-Semantic_HSM>/scripts/STOP_TRAINING` from any terminal |

**Checkpoints** → `Semantic_HSM/scripts/checkpoints_v3/` (every 5 epochs + `final_model_seq.pth` on exit).

**Expected input files per index `id`:**
- `trajectories_val{id:05d}.pt` — trajectory state/action sequence
- `video_val_rollout_images_rgb{id:05d}.mp4` — RGB video
- `video_val_rollout_images_semantic{id:05d}.mp4` — semantic heatmap video
- `video_val_rollout_images_depth{id:05d}.mp4` — depth map video

---

## V2C State-Machine Simulation (`Semantic_HSM/sim/`)

> **Note:** `Semantic_HSM/` is not a submodule of V-LEAD. Clone it separately; adjust paths below accordingly.

An alternative to `SINGER/notebooks/simulate_v2c_adi.py` that warm-starts the V2C
policy with a rule-based state machine before handing control to the neural policy.
This prevents the immediate `COLLISION_BOUNDS` failure that occurs when the policy
fires from a cold start at a random position.

**Run inside the SINGER container:**
```bash
python3 /workspace/Semantic_HSM/sim/simulate.py \
    --model /workspace/Semantic_HSM/scripts/trained_models/t5_altframes_seq_final_model.pth \
    --object "green clock"
```

**State machine sequence:**
```
STABILIZE → SCAN → CENTER → NAVIGATE → SUCCESS / OOB / TIMEOUT
```

| Phase | What it does |
|-------|--------------|
| STABILIZE | P-control altitude to -1.0 m NED; holds x/y |
| SCAN | Rotates 360° at 0.05 rad/step; records semantic peak yaw; fills history buffer |
| CENTER | P-control yaw toward best semantic hit; waits until object centred in FOV |
| NAVIGATE | Runs `VisualNavPolicy_Sequence` with warm history buffer |

**CLI options:**
```
--scene     STR    3DGS scene name (default: flightroom_ssv_exp)
--object    STR    Language query  (default: green clock)
--model     STR    Path to .pth checkpoint [REQUIRED]
--renderer  STR    gemsplat | splatfacto   (default: gemsplat)
--timeout   FLOAT  Hard time limit in s    (default: 180.0)
--max-speed FLOAT  Max drone speed m/s     (default: 2.0)
```

Output videos → `Semantic_HSM/sim/outputs/` (small files, local storage OK).
For full documentation see `Semantic_HSM/sim/README.md`.

---

## Configuration Reference

### Experiment Config (`configs/experiment/*.yml`)
Controls the top-level pipeline: which scenes, which pilot, how many epochs.
```yaml
cohort: "smoke_test"       # Output directory name under cohorts/
method: "rrt"              # Trajectory generation method (configs/method/rrt.json)
Nep_his: 5                 # History encoder training epochs
Nep_com: 5                 # Commander training epochs
flights:
  - ["flightroom_ssv_exp", "flightroom_ssv_exp"]   # [scene_3dgs, scene_config]
roster:
  - "InstinctJester"       # Pilot architecture (configs/pilots/InstinctJester.json)
```

### Perception Config (`FiGS-Standalone/configs/perception/perception_mode.yml`)
Controls what the simulator renders at each timestep.
```yaml
visual_mode: "semantic_depth"   # "rgb" for fast single-channel, "semantic_depth" for all 3
perception_type: "similarity"   # Always use "similarity" — "clipseg" is slow and buggy
extra_channels: []
```

### Method Config (`configs/method/rrt.json`)
Controls trajectory generation, domain randomization, and frame sets.
- `trajectory_set.initial` — perturbation bounds for initial drone state (defines n_domain_rand_perturbations)
- `frame_set` — list of drone frame configs for domain randomization
- `sample_set` — rollout simulation parameters (duration, rate, noise profile)

### Scene Config (`configs/scenes/<scene>.yml`)
Per-scene parameters: semantic target queries, flight altitude/radius, RRT* obstacle params.

### Pilot Config (`configs/pilots/InstinctJester.json`)
Neural network architecture: HistoryEncoder hidden dims, VisionMLP backbone (SqueezeNet1_1), CommanderSV hidden dims, state/observation indices.

---

## Available 3DGS Models

For the `flightroom_ssv_exp` scene (used by all smoke tests):

| Model | Path | Checkpoint |
|-------|------|------------|
| gemsplat (preferred, user-trained) | `3dgs/workspace/outputs/flightroom_ssv_exp/gemsplat/2026-02-28_205058/` | step-000029999.ckpt |
| gemsplat (older) | `trained_gsplats/flightroom_ssv_exp/gemsplat/2026-02-03_115017/` | step-000028000.ckpt |
| splatfacto | `3dgs/workspace/outputs/flightroom/splatfacto/2024-07-12_145513/` | step-000029999.ckpt |

Paths are relative to `/data/kothari1/singer_figs_data/`. The Simulator searches `FiGS-Standalone/3dgs/workspace/outputs/<scene>` first, then falls back to `$DATA_PATH/trained_gsplats/<scene>`.

---

## Output File Nomenclature

All rollout data goes to `.../rollouts_singer/<cohort>/rollout_data/<scene>/`.

Each file uses a zero-padded 5-digit batch index. The index maps to one rollout: one RRT branch toward one target object, with one domain-randomized drone configuration.

**Training files** (330 files for smoke_test: 3 targets × 11 branches × 10 perturbations):

| File | Content |
|------|---------|
| `trajectories{id:05d}.pt` | Expert MPC trajectory: state `x` (10-dim), control `u` (4-dim), timestamps `t`, metadata |
| `imgdata{id:05d}.pt` | Per-frame semantic similarity maps (tensors) from gemsplat rendering |
| `video{id:05d}.mp4` | Semantic heatmap video — CLIP similarity scores as colored overlay |

**Validation files** (33 files for smoke_test: 3 targets × 11 branches):

| File | Content |
|------|---------|
| `trajectories_val{id:05d}.pt` | Validation trajectory (same format, no domain randomization) |
| `imgdata_val{id:05d}.pt` | Validation image tensors |
| `video_val_rollout_images_semantic{id:05d}.mp4` | Semantic heatmap — target object highlighted |
| `video_val_rollout_images_rgb{id:05d}.mp4` | Raw RGB from drone forward camera |
| `video_val_rollout_images_depth{id:05d}.mp4` | Depth map (JET colormap: near=blue, far=red) |

---

## Common Issues

### "python not found"
Use `python3` explicitly inside SINGER container. Never use bare `python`.

### "pip not in PATH"
Use `python3 -m pip` instead of `pip`.

### CLIP / AlexNet downloads on first run
CLIP (1.26 GB) and AlexNet (233 MB) download to `/root/.cache/` on first run. Cached in the `model-cache` Docker named volume — subsequent runs are fast.

### `transformers` version conflict
Always pin: `transformers==4.40.0`. Higher versions require PyTorch ≥2.2 but the container has 2.1.2.

### `site-packages` volume stale after image rebuild
If the base `figs:latest` image changes, clear the cached packages: `docker compose down -v` in the SINGER directory, then re-run the one-time dependency setup.

### Storage full on `/home/kothari1`
All large outputs must go to `/data/kothari1/singer_figs_data/`. The symlinks (`SINGER/cohorts`, `FiGS-Standalone/3dgs`) ensure this automatically. Never run pipeline scripts that write to `/home/kothari1`.

### Old `video_val{id}.mp4` files in output dir
These are artifacts from earlier pipeline iterations with different naming conventions. Safe to ignore or delete. The correct validation videos are the `video_val_rollout_images_{channel}{id}.mp4` set.

### Blackwell GPUs (GPU 2/3) crash on gsplat load
PyTorch in the container was built for sm_50–sm_90. Blackwell is sm_120. Error: `RuntimeError: CUDA error: no kernel image is available for execution on the device`. Use GPU 1 (L40S, sm_89) only until the image is rebuilt with newer PyTorch.

### Parallel containers crash with `OSError: cannot open shared object file`
ACADOS compiles C solver libraries per call and previously wrote them to a shared path (`/workspace/SINGER/c_generated_code/`). Multiple containers racing on this path caused the error. Fixed in `FiGS-Standalone/src/figs/simulator.py` and `vehicle_rate_mpc.py` — each call now uses an isolated `tempfile.TemporaryDirectory()`. If you see this error, pull the latest FiGS-Standalone submodule commit.

---

## Quick Reference Commands

```bash
# Enter FiGS container
cd FiGS-Standalone && docker compose -f docker-compose.base.yml run --rm figs

# Enter SINGER container
cd SINGER && docker compose run --rm singer

# Run full smoke test pipeline (inside SINGER container)
python3 notebooks/ssv_multi3dgs_campaign.py generate-rollouts --config-file configs/experiment/smoke_test.yml
python3 notebooks/ssv_multi3dgs_campaign.py generate-rollouts --config-file configs/experiment/smoke_test.yml --validation-mode
python3 notebooks/ssv_multi3dgs_campaign.py generate-observations --config-file configs/experiment/smoke_test.yml
python3 notebooks/ssv_multi3dgs_campaign.py train-history --config-file configs/experiment/smoke_test.yml
python3 notebooks/ssv_multi3dgs_campaign.py train-command --config-file configs/experiment/smoke_test.yml

# Check output files
ls /data/kothari1/singer_figs_data/rollouts_singer/smoke_test/rollout_data/flightroom_ssv_exp/

# Train Semantic_HSM (on host, NOT inside container — uses lead_ml conda env)
# Semantic_HSM is not a V-LEAD submodule; clone separately and adjust path below
conda activate /data/kothari1/singer_figs_data/conda_envs/lead_ml
cd <path-to-Semantic_HSM>/scripts
TORCH_HOME=/data/kothari1/singer_figs_data/.torch_hub CUDA_VISIBLE_DEVICES=1 python3 train_03.py

# V-LEAD data gen — dry run (1 traj per object, ~2 min)
cd SINGER && CUDA_VISIBLE_DEVICES=1 docker compose run --rm singer bash -lc \
  "CUDA_VISIBLE_DEVICES=1 python3 notebooks/generate_training_data.py \
   --config-file configs/experiment/vlead_dryrun.yml --validation-mode"

# V-LEAD data gen — full val run, all queries
cd SINGER && CUDA_VISIBLE_DEVICES=1 docker compose run --rm singer bash -lc \
  "CUDA_VISIBLE_DEVICES=1 python3 notebooks/generate_training_data.py \
   --config-file configs/experiment/vlead_flightroom.yml --validation-mode"

# Check V-LEAD output
ls /project/kothari1/vlead_data/rollouts_vlead/vlead_flightroom/rollout_data/
```

---

## Agent Context: V-LEAD Data Generation

> This section is for coding agents and developers. Describes internals, file locations, and design decisions.

### Primary Script

`SINGER/notebooks/generate_training_data.py` — standalone data generation script. Runs entirely inside the SINGER container. Does NOT use `ssv_multi3dgs_campaign.py`.

**Key CLI flags:**
```
--config-file PATH     Experiment YAML (required)
--validation-mode      Full-length trajectories, 1 rep, val filenames (trajectories_val*)
--query-indices STR    Comma-separated ints, e.g. "0,2". Default: all queries.
                       Filters BEFORE CLIP + RRT — each container only processes its subset.
```

**`generate()` call graph:**
```
generate()
  └─ for each flight (scene, course):
       ├─ Simulator(scene_name)           # loads gsplat into GPU memory
       ├─ bd.get_objectives()             # CLIP semantic target detection (filtered by query_indices)
       ├─ th.process_RRT_objectives()     # goal poses and centroids
       ├─ bd.generate_rrt_paths()         # RRT* path planning (filtered by query_indices)
       ├─ th.parameterize_RRT_trajectories()
       └─ for each trajectory branch:
            └─ run_and_save_batch()
                 ├─ simulator.load_frame()       # compiles ACADOS sim solver → temp dir
                 ├─ VehicleRateMPC(tXUd, ...)    # compiles ACADOS OCP solver → temp dir
                 ├─ simulator.simulate()          # renders RGB+depth+semantic at 20 Hz
                 ├─ compute_goal_heading()        # heading_vec (2,N) + dist (N,)
                 └─ _save_batch()                 # .pt files + 3-channel videos
```

### New Fields in Trajectory Dicts

Added to every trajectory in `trajectories(_val){id:05d}.pt`:

```python
traj["goal_xy"]      # np.ndarray (2,)   — 2D XY centroid of target object, Z dropped
traj["heading_vec"]  # np.ndarray (2, N) — unit vector from drone to goal, per timestep
traj["dist"]         # np.ndarray (N,)   — XY scalar distance, per timestep
```

`goal_xy` is the CLIP point-cloud centroid (`obj_centroid`, not `goal_pose`). `heading_vec` is zero-padded when distance < 1e-6.

### ACADOS Parallel Fix {#acados-parallel-fix}

**Problem:** `simulator.load_frame()` and `VehicleRateMPC.__init__()` both compile ACADOS C solvers and delete the generated files afterward. With a shared bind-mount (`/workspace/SINGER/`), parallel containers race on the same `c_generated_code/` path.

**Fix:** Both now use `tempfile.TemporaryDirectory()` — each compilation goes to a unique process-local temp dir. On Linux, `dlopen()` loads the `.so` into memory; the temp dir can be deleted immediately after and the solver remains usable.

**Changed files:**
- `FiGS-Standalone/src/figs/simulator.py` — `load_frame()` method
- `FiGS-Standalone/src/figs/control/vehicle_rate_mpc.py` — `__init__()` method

### Experiment Config Schema (V-LEAD specific)

`vlead_flightroom.yml` / `vlead_dryrun.yml` use a 3-element flight tuple:
```yaml
flights:
  - ["scene_name_for_3dgs", "course_name", "scene_config_name_override"]
```
The third element (optional) lets the scene config YAML differ from the 3DGS scene directory name. Used by `vlead_dryrun.yml` to load `flightroom_ssv_exp_dryrun.yml` while still using the `flightroom_ssv_exp` 3DGS checkpoint.

### Output Directory Layout

```
/project/kothari1/vlead_data/rollouts_vlead/
└── <cohort>/
    └── rollout_data/
        └── <YYYY-MM-DD_HHMMSS>/      ← one per generate() call (run_ts)
            └── <course_name>/
                ├── trajectories(_val){id:05d}.pt
                ├── imgdata(_val){id:05d}.pt
                ├── video_(val_rollout_images_)(rgb|depth|semantic){id:05d}.mp4
                └── ...
```

Multiple parallel runs (different `--query-indices`) create different `run_ts` dirs. Merge them by course when loading for training.

### Image Capture Rate

- Physics integrator: 100 Hz (fixed, `configs/rollout/baseline.json`)
- MPC control + image capture: **20 Hz** (`configs/policy/vrmpc_rrt.json hz: 20`)
- Frame resolution: 640×360 RGB (`configs/frame/carl.json`)
- Training data FPS must match deployment inference rate to avoid distribution shift.

### Adding New Target Objects

Edit `SINGER/configs/scenes/flightroom_ssv_exp.yml` — append one entry to each of:
`queries`, `radii`, `altitudes`, `similarities`, `nbranches` (must stay index-aligned).
`query_indices` uses 0-based position in the `queries` list.
