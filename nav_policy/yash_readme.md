# V-LEAD Nav Policy — Yash's Setup & Training Notes

## System
- **Machine**: maven server
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM), driver 580.159.03, CUDA 12.0
- **OS**: Linux 6.17.0-29-generic
- **Python**: 3.10.20

---

## Environment Setup

```bash
cd /home/yashr/CS231N/V-LEAD

# Create venv
python3.10 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CUDA 12.8 build)
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install nav_policy and its dependencies
pip install -e nav_policy/src/[rl]

# Install FiGS (editable)
pip install -e FiGS-Standalone/

# Install remaining dependencies
pip install -r nav_policy/requirements.txt
```

---

## Key Paths

| What | Path |
|------|------|
| Nav policy source | `nav_policy/src/nav_policy/` |
| FiGS simulator | `FiGS-Standalone/` |
| BC checkpoint (starting point) | `/home/yashr/CS231N/checkpoints/bc_best_balanced_dagger_r12_new.pt` |
| Training rollout data | `/home/yashr/CS231N/FiGS_data/vlead_flightroom/rollout_data/` |
| RL checkpoints & logs | `nav_policy/data/checkpoints_rl_local/` |
| Eval output | `nav_policy/data/eval/` |

---

## Running RL Training (PPO)

```bash
cd /home/yashr/CS231N/V-LEAD/nav_policy
source ../.venv/bin/activate

python -m nav_policy.rl.train_rl \
  --config configs/train_rl_local.yaml \
  --save-videos
```

Override output dir without editing the config:
```bash
python -m nav_policy.rl.train_rl \
  --config configs/train_rl_local.yaml \
  --checkpoint-dir /path/to/output \
  --run-tag my_run \
  --save-videos
```

Quick smoke test (1 iter, 2 rollouts):
```bash
python -m nav_policy.rl.train_rl \
  --config configs/train_rl_local.yaml \
  --n-iterations 1 \
  --rollouts-per-iteration 2
```

### Training outputs (under `checkpoint_dir/`)
- `<tag>_latest.pt` — saved every iteration
- `<tag>_best.pt` — best checkpoint (by eval success rate if eval enabled, else by return)
- `<tag>_log.csv` — per-iteration stats (return, loss, entropy, KL, success rate)
- `<tag>_episodes.csv` — per-episode stats
- `tensorboard/<tag>/` — TensorBoard logs
- `videos/` — rollout mp4s (only with `--save-videos`)

### Monitor training
```bash
tensorboard --logdir nav_policy/data/checkpoints_rl_local/tensorboard
```

---

## Running Closed-Loop Eval

```bash
cd /home/yashr/CS231N/V-LEAD/nav_policy

python scripts/eval_in_figs.py \
  --config configs/eval_closed_loop_flightroom.yaml \
  --checkpoint data/checkpoints_rl_local/rl_ppo_local_latest.pt \
  --output-dir data/eval/my_eval \
  --run-tag my_eval
```

Eval outputs land in `--output-dir`:
- `summary.json` — aggregate metrics (success rate, RMSE, latency, etc.)
- `per_rollout.csv` — one row per rollout
- `rollout_<name>/video.mp4` — policy-driven video
- `rollout_<name>/metrics.json` — per-rollout metrics

---

## Key Config: `configs/train_rl_local.yaml`

| Setting | Value | Notes |
|---------|-------|-------|
| `algorithm` | `ppo` | PPO or SAC |
| `n_iterations` | 20 | Total training iterations |
| `rollouts_per_iteration` | 4 | Episodes collected per iter before PPO update |
| `init_log_std` | -0.5 | Exploration noise (std ≈ 0.61); lower = less jitter |
| `max_rollout_s` | 40.0 | Max episode length in seconds |
| `settle_steps` | 3 | Consecutive settled steps needed for success |
| `action_smooth_weight` | 0.15 | Reward penalty for jittery commands |
| `success_bonus` | 15.0 | Reward for reaching goal |
| `eval_every_episodes` | 8 | Run held-out eval every N episodes |
| `eval_subset_size` | 10 | Number of held-out queries per intermediate eval |
| `eval_config` | `configs/eval_closed_loop_flightroom.yaml` | 110-rollout held-out set (071733) |

### Training rollout pool
- 40 rollouts: `071718` q00–q19 and `071353` q00–q19
- 4 sampled randomly per iteration

### Held-out eval set
- 110 rollouts from `071733_trajs-110` (never used in training)

---

## Success Criteria (during both training and eval)

The drone is counted as successful when ALL of the following hold for `settle_steps` consecutive control steps:

| Condition | Threshold |
|-----------|-----------|
| 3D distance to goal | < 0.5 m |
| Yaw error to goal | < 0.5 rad (~29°) |
| Speed | ≤ 0.3 m/s |
| Commanded body rate | ≤ 0.35 rad/s |
| Commanded yaw rate | ≤ 0.30 rad/s |

---

## Notes

- Training uses **stochastic** actions (Gaussian noise on top of policy mean) — this causes visible jitter in training videos. Eval uses the **deterministic** mean — hence smooth eval videos.
- The `reuse_simulator` flag keeps the FiGS scene loaded across rollouts — big speedup for both training and eval.
- The `torch.load` monkey-patch at the top of `train_rl.py` and `closed_loop.py` is needed because PyTorch 2.6 changed the default to `weights_only=True`, which breaks nerfstudio/gsplat checkpoints.
- Videos are saved at 640×360 with `macro_block_size=1` to avoid ffmpeg padding warnings.
