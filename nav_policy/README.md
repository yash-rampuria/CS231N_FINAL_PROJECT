# nav_policy

Goal-conditioned visual navigation for the FiGS quadrotor simulator. Maps a short
history of onboard RGB frames (+ optional DA2-S depth) and a goal vector to a
receding-horizon sequence of velocity commands `[vx, vy, vz, psi_dot]`, tracked
by FiGS's `VelocityController`.

**Training stack:** behavior cloning (BC) → DAgger → optional **RL fine-tuning**
(PPO default, SAC optional). **No relightable 3DGS** — robustness via 2D
augmentations and observation-latency simulation.

## Architecture (A2 primary)

```text
FiGS RGB (+ live DA2-S depth at deploy)
      |
      v
rolling frame buffer (T=4, optional latency)
      |
      +-- ResNet-18 per-frame encoder --+
      |                                  |
      +-- DA2-S depth CNN encoder -------+--> cross-attention fusion --> GRU
      |
      v
goal vector [hx, hy, d_norm] --> embedding
      |
      v
MLP head -> [vx, vy, vz, psi_dot] x H=10  (first step executed @ 20 Hz)
      |
      v
FiGS VelocityController -> body rates -> ACADOS integrator
```

## Repository layout

```text
nav_policy/
|-- configs/                 BC, DAgger, eval, RL yaml files
|-- data/
|   |-- raw/<run>/           SINGER-format rollouts
|   |-- processed/           cache blobs + manifest (gitignored)
|   |-- weights/             depth_anything_v2_vits.pth
|   |-- checkpoints_*/       BC / RL outputs (gitignored)
|-- src/nav_policy/
|   |-- data/                dataset, augmentations, normalization
|   |-- model/               RGB / RGB+DA2 policies, depth estimator
|   |-- train/               train_bc.py
|   |-- rl/                  PPO / SAC fine-tuning (NEW)
|   |-- deploy/              policy_controller, frame_buffer
|   |-- dagger/              DAgger rollouts + MPC oracle
|   |-- evaluate/            offline + closed-loop eval
|   |-- vendor/              vendored Depth Anything V2 (no PyPI dep)
|-- scripts/                 CLI entry points
|-- modal_train.py           cloud BC on Modal
\-- docker-compose.yml       FiGS container
```

## Full pipeline

### 1. Docker: dataset + depth (from repo root)

```bash
docker compose -f nav_policy/docker-compose.yml run --rm nav_policy bash
cd /workspace/nav_policy

python scripts/build_dataset_flightroom.py --config configs/flightroom.yaml
python scripts/precompute_da2_depth.py --processed-root data/processed_flightroom
```

Place DA2 weights at `data/weights/depth_anything_v2_vits.pth`.

### 2. Modal: BC training (Windows host)

```powershell
modal volume put vlead-data "C:\...\nav_policy\data\processed_flightroom" /processed_flightroom_da2
modal run nav_policy/modal_train.py
modal volume get vlead-data checkpoints_a2_da2_crossattn/bc_best.pt nav_policy/data/checkpoints_a2_da2_crossattn/bc_best.pt
```

Or train locally in Docker:

```bash
python scripts/train_bc.py --config configs/arch_rgb_da2_crossattn.yaml
```

### 3. Docker: DAgger + eval

```bash
python scripts/run_dagger.py --config configs/dagger_round1_mpc.yaml

# Fine-tune on Modal (after uploading manifest + dagger_r1_mpc/ caches):
#   modal run nav_policy/modal_train.py --run-tag dagger_r1_mpc \
#     --resume-from /data/checkpoints_a2_da2_crossattn/bc_best.pt \
#     --checkpoint-dir /data/checkpoints_dagger_r1_mpc
# Best weights: data/checkpoints_dagger_r1_mpc/bc_best.pt
#   (Modal volume: checkpoints_dagger_r1_mpc/bc_best.pt)

# Or fine-tune locally:
python scripts/train_bc.py --config configs/flightroom.yaml \
  --run-tag dagger_r1_mpc \
  --resume-from data/checkpoints_a2_da2_crossattn/bc_best.pt \
  --checkpoint-dir data/checkpoints_dagger_r1_mpc

python scripts/eval_offline.py \
  --config configs/eval_offline_flightroom.yaml \
  --checkpoint data/checkpoints_a2_da2_crossattn/bc_best.pt \
  --output-dir data/eval/flightroom_bc_offline

python scripts/eval_in_figs.py --config configs/eval_closed_loop_flightroom_suite.yaml
python scripts/eval_in_figs.py --config configs/eval_closed_loop_backroom_val.yaml
python scripts/eval_in_figs.py --config configs/eval_closed_loop_ood_suite.yaml
```

### 4. Docker: RL fine-tuning (requires 3DGS scenes)

Warm-start from BC or DAgger checkpoint. **PPO is the default**; switch algorithm
in yaml.

**Smoke test** (2 rollouts, 1 iteration, videos + TensorBoard):

```bash
python scripts/train_rl.py --config configs/train_rl_smoke.yaml --save-videos
```

**Full training:**

```bash
# PPO (default)
python scripts/train_rl.py --config configs/train_rl_flightroom.yaml

# SAC
python scripts/train_rl.py --config configs/train_rl_flightroom_sac.yaml
```

**Logs:** CSV files are written alongside checkpoints (`*_episodes.csv`, `*_log.csv`, `*_summary.json`).

**Videos** (optional): pass `--save-videos` or set `save_videos: true` in yaml.
Files go to `video_dir` (default `data/checkpoints_<tag>/videos/<run_tag>/`).

Key config fields (`configs/train_rl_flightroom.yaml`):

```yaml
rl:
  algorithm: ppo          # or sac
  checkpoint_dir: data/checkpoints_rl_a2
  n_iterations: 20
  rollouts_per_iteration: 4
  rewards:
    progress_weight: 1.0
    collision_penalty: -10.0
    success_bonus: 5.0
```

Outputs: `data/checkpoints_rl_a2/rl_ppo_a2_best.pt` (compatible with
`eval_in_figs.py` via `--checkpoint`).

Resume RL training:

```bash
python scripts/train_rl.py --config configs/train_rl_flightroom.yaml \
  --resume-from data/checkpoints_rl_a2/rl_ppo_a2_latest.pt
```

## Augmentations (training)

Configured under `train:` in yaml (`configs/flightroom.yaml`):

| Augmentation | Config keys |
|---|---|
| Color jitter | `color_jitter: true` |
| Gaussian blur | `photometric_aug.blur_prob`, `blur_sigma_range` |
| Brightness / gamma | `photometric_aug.brightness_*`, `gamma_*` |
| Observation latency | `observation_latency` (train + stress eval) |

Implementation: `src/nav_policy/data/augmentations.py`,
`src/nav_policy/data/rgb_horizon_dataset.py`.

## RL design (no relightable renderer)

| Component | Location |
|---|---|
| Gaussian actor + critic | `src/nav_policy/rl/stochastic_policy.py` |
| FiGS rollout collector | `src/nav_policy/rl/rollout.py` |
| Rewards (progress, heading, bbox, success) | `src/nav_policy/rl/rewards.py` |
| PPO | `src/nav_policy/rl/ppo.py` |
| SAC | `src/nav_policy/rl/sac.py` |
| Training loop | `src/nav_policy/rl/train_rl.py` |

RL rollouts are restricted to **flightroom training scenes** (same as DAgger).
Rewards use goal progress, velocity-heading alignment, expert-bbox violation
penalty, and sparse terminal success — **not** relighting.

## Data assumptions

Each `data/raw/<run>/` directory is SINGER validation-rollout format:

- `trajectories_val{NNNNN}.pt` — state/control logs @ 20 Hz
- `video_val_rollout_images_rgb{NNNNN}.mp4` — 20 fps RGB
- `imgdata_val{NNNNN}.pt` — sub-trajectory frame ranges

Goal supervision: expert sub-trajectory endpoint `Xro[0:2,-1]`.

**Splits (flightroom branch):**

| Split | Scenes |
|---|---|
| Train | flightroom 064652, 071718, 071353 |
| Val | flightroom 071733 + backroom |
| OOD test | packardpark (closed-loop only) |

## Dependencies

- PyTorch + torchvision (Docker: Python 3.10)
- FiGS / ACADOS / 3DGS (closed-loop, DAgger, RL)
- DA2-S vendored under `src/nav_policy/vendor/` (no `pip install depth-anything-v2`)

See `docs/CROSS_SCENE_DA2_RL_PLAN.md` for the full experiment plan.
