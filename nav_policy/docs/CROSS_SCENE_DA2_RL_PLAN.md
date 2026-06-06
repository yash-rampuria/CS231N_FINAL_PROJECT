# V-LEAD Plan: DA2-Small + ResNet Cross-Attention (Flightroom Train, Cross-Scene Test)

**Branch:** `feature/da2-crossattn-rl-generalization` (from `dev-rahul`)  
**Scope:** Implementation plan for this branch. No relightable 3DGS. RL fine-tuning (PPO/SAC) implemented in `nav_policy/rl/`.  
**Report:** Update `report/main.tex` (and figures/scripts as needed) on this branch to reflect the architecture, training split, evaluation protocol, and ablations below.

**Policy I/O (unchanged):**  
- **Inputs:** RGB frame history `T=4`, goal heading `[h_x, h_y]`, normalized distance-to-goal `d̃`  
- **Outputs:** velocity horizon `[vx, vy, vz, ψ̇]` × `H=10`; runtime executes first command only  
- **Goal:** expert sub-trajectory endpoint `Xro[0:2, -1]` everywhere (train, val, closed-loop, DAgger)

---

## 1. Objective

Train a goal-conditioned visuomotor policy **only on flightroom** (best-quality V-LEAD data), using **ResNet-18 + Depth Anything V2 Small (DA2-S)** with **cross-attention fusion and LayerNorm**. Use **2D augmentations and latency simulation** for robustness — **not** relightable 3DGS.

Evaluate:

1. **In-domain:** held-out flightroom trajectories (offline + closed-loop)  
2. **Cross-scene val + OOD test:** backroom in offline/closed-loop **val**; packardpark **test only** (never train/val/DAgger)  

Backroom and packardpark use legacy SINGER-format rollouts; **collision / intersection geometry in those scenes may be less reliable** than flightroom. Treat OOD metrics as exploratory; do not use them for checkpoint selection.

**RL fine-tuning:** optional PPO (default) or SAC after BC + DAgger. See §9.

---

## 2. Data split

| Split | Source | Used for |
|-------|--------|----------|
| **Train** | flightroom train runs (`064652`, `071718`, `071353`) | BC, DAgger collection, DAgger fine-tune |
| **Val** | flightroom `071733` + **backroom** | Offline MSE, early stopping, cross-scene val closed-loop |
| **Test (OOD)** | packardpark only | Closed-loop zero-shot — never processed for training |

Train on flightroom only. Backroom is **not** in train or DAgger, but **is** processed into the val split for checkpoint selection and cross-scene offline metrics. Packardpark stays fully held out.

Pick **diverse flightroom val rollouts** (multiple `course` labels — clock, drill, ladder, etc.), not ladder-only file indices.

---

## 3. Network architecture

### 3.1 DA2-Small + ResNet-18 + cross-attention

Use **Depth Anything V2 Small (ViT-S)** — frozen at inference/training for depth extraction; small enough for reasonable closed-loop latency when batched or run every frame on GPU.

Per frame `t` in history window `T=4`:

```
RGB I_t  ──► ResNet-18 (stem+layer1 frozen) ──► f_rgb  (512-D)
              └──► LayerNorm on f_rgb

I_t      ──► DA2-S (frozen) ──► depth D_t ──► DepthEncoder (small CNN) ──► f_dep  (256-D)
              └──► LayerNorm on f_dep

CrossAttention( Q = f_rgb_norm, K = f_dep_norm, V = f_dep_norm )
              └──► LayerNorm on attention output  ──►  f_fused  (512-D)

[f_fused_{t-T+1}, …, f_fused_t] ──► GRU ──► LayerNorm(h_t)

g_t = [h_x, h_y, d̃] ──► GoalEmbed ──► concat(h_t_norm, e_g) ──► MLP head ──► [H, 4]
```

**Normalization at fusion (required):**

- `LayerNorm` on RGB and depth features **before** cross-attention so scales match (512-D vs 256-D branches).  
- `LayerNorm` on cross-attention output before stacking into the GRU.  
- Keep existing `LayerNorm` on GRU output before goal concat (same as current RGB policy).

Do **not** rely on raw concatenation without normalization.

### 3.2 Depth pipeline

| Stage | Approach |
|-------|----------|
| **Cache build** | Precompute DA2-S depth offline from expert RGB frames; store in cache blobs alongside RGB |
| **Closed-loop deploy** | Run DA2-S on live sim RGB each step (same model/weights as training) |
| **Depth format** | Single-channel, resize to 224×224; normalize consistently (e.g. per-frame or fixed scale to `[0,1]`) |

Log **DA2 + policy** latency in `Tsol`; target p95 model forward `< 25 ms` where possible (may require DA2-S + AMP).

### 3.3 Config flag

Add `model.arch: rgb_da2_crossattn_v1` (or similar) in yaml; checkpoint must record arch + augment flags for deploy.

---

## 4. Training augmentations (no relightable)

Apply during **BC and DAgger training** on RGB (depth computed from augmented RGB at cache-build time, or re-run DA2 on augmented frames — pick one pipeline and stay consistent).

| Augmentation | Setting | Notes |
|--------------|---------|-------|
| **Color jitter** | Keep / extend existing | Brightness, contrast, saturation, hue (small hue range) |
| **Gaussian blur** | σ ∈ [0.5, 1.5], p ≈ 0.3 | Motion / defocus robustness |
| **Brightness / gamma** | Multiplicative exposure ±30–40% | SOUS VIDE-style lighting stress without relighting |
| **Injected observation latency** | 0–2 frame delay on RGB buffer | Simulate render + inference delay; also use at **closed-loop eval** as stress test |

**Not in scope:** relightable 3DGS, HDR autoencoder, FiGS-native relighting.

**Dynamics DR:** keep existing flightroom training-mode domain randomization (mass, disturbances) in expert data where already present.

---

## 5. Training pipeline (current scope)

### Step 0 — Prerequisites

- [x] Rebuild flightroom processed caches with **expert-endpoint goals** (`build_dataset_flightroom.py`)
- [ ] Precompute **DA2-S depth** for all train + val caches (`scripts/precompute_da2_depth.py`)

**DA2 weights:** download `depth_anything_v2_vits.pth` from the [Depth Anything V2 repo](https://github.com/DepthAnything/Depth-Anything-V2) into `nav_policy/data/weights/`, or set `DEPTH_ANYTHING_V2_VITS_WEIGHTS`.

### Step 1 — Behavior cloning

- Train `rgb_da2_crossattn_v1` on flightroom train runs with augmentations (§4).  
- Val on `071733`; select checkpoint on offline val MSE.  
- Baseline: retrain **ResNet RGB-only** on same caches/goals for ablation A0.

### Step 2 — DAgger (1–2 rounds)

- Rollouts **flightroom train scenes only** (`dagger_round*_mpc.yaml`).  
- MPC oracle relabel; write caches with RGB + DA2 depth + goal + labels.  
- Fine-tune from BC best; early stop on flightroom val.  

### Step 3 — Evaluation

- Offline val on `071733`.  
- In-domain closed-loop: stratified flightroom val suite (§6).  
- OOD closed-loop: backroom + packardpark test suite (§6) — **report separately**, no checkpoint tuning.

### Step 4 — Interpretability

- Grad-CAM on ResNet `layer4` (or fused token) for selected success/failure rollouts.  
- Save overlays under `data/eval/<run>/saliency/`.

---

## 6. Robust evaluation protocol

Replace ad-hoc “3 trajectories” with fixed suites. Report **mean ± std** and **95% bootstrap CI** on success rate where N permits.

### 6.1 In-domain (flightroom val) — **primary metrics**

Minimum **≥10 rollouts**, stratified by semantic `course` where available:

- Mix of durations (short / medium / full ~13 s)  
- Diverse start regions  
- Used for checkpoint comparison and ablations  

### 6.2 Cross-scene OOD (test only)

Minimum **≥10 rollouts per scene** (backroom, packardpark):

- Never seen in train, val, or DAgger  
- **Caveat:** collision geometry in these scenes may be less trustworthy than flightroom; report **bbox proxy violations** and qualitative videos alongside success rate  
- Do **not** claim strong collision guarantees on OOD scenes until scene assets are audited  

### 6.3 Metrics (all suites)

| Metric | Notes |
|--------|-------|
| **Goal success** | Final XY error to expert endpoint `< 0.5 m` (and tracking RMSE `< 1.0 m` if keeping current definition) |
| **Final goal error** | Distance to expert endpoint (m) |
| **Tracking RMSE** | vs expert path |
| **BBox violation rate** | Current safety proxy |
| **Latency p95** | DA2 + policy forward (ms) |

### 6.4 Stress tests (flightroom val only)

Report degradation vs clean val — **not** used for model selection:

- Brightness 0.4× / 1.6×  
- Gaussian blur σ = 1.5  
- Injected latency 1–2 frames  

---

## 7. Ablation studies (important only)

Run on flightroom val closed-loop suite (+ OOD for the full model row only).

| ID | Variant | Purpose |
|----|---------|---------|
| **A0** | ResNet RGB-only (no depth) | Baseline |
| **A1** | ResNet + DA2-S, concat fusion (no cross-attention, no pre-fusion LayerNorm) | Is cross-attention + norm needed? |
| **A2** | **Full model:** ResNet + DA2-S + cross-attention + LayerNorm fusion | Primary system |
| **A3** | Full model, **no photometric aug** (jitter/blur/brightness off) | Value of 2D DR |
| **A4** | Full model, **goal dim 2** (no distance scalar) | Value of distance channel |
| **A5** | BC only vs BC + DAgger R1 (vs R2 if run) | DAgger gain |

Skip for now: relightable, multi-scene train, RL, FiGS depth vs DA2, MPC vs reference oracle (unless DAgger round already varies this).

---

## 8. Implementation checklist (for agent)

| Component | Action |
|-----------|--------|
| `scripts/precompute_da2_depth.py` | Batch DA2-S over cache RGB or raw videos |
| `build_dataset_flightroom.py` | Store depth tensor in cache blobs |
| `rgb_horizon_dataset.py` | Load RGB + depth; apply RGB augs; latency buffer option |
| `model/rgb_da2_policy.py` | DA2 fusion policy with LayerNorm + cross-attention |
| `train_bc.py` / configs | `arch_rgb_da2_crossattn.yaml`, `flightroom_modal.yaml` updates |
| `deploy/policy_controller.py` | DA2-S wrapper, fused forward, latency timing |
| `dagger/run_dagger.py` | Depth on rollout frames before cache write |
| `configs/eval_*.yaml` | Stratified flightroom val + OOD test manifests (≥10 each) |
| `scripts/gradcam_saliency.py` | Saliency overlays |
| `scripts/train_rl.py` | PPO / SAC RL fine-tuning from BC/DAgger checkpoint |
| `nav_policy/rl/` | Stochastic policy, rewards, FiGS rollouts, PPO, SAC |

**Not in current scope:** relightable renderer changes.

**Implemented:** `nav_policy/rl/` — PPO (default) and SAC fine-tuning from BC/DAgger checkpoints.
See `configs/train_rl_flightroom.yaml` and `README.md`.

---

## 9. RL fine-tuning (implemented)

After BC + DAgger produces a strong initializer:

- **Algorithm:** PPO default (`rl.algorithm: ppo`); SAC optional (`rl.algorithm: sac`).  
- **Warm-start:** Load BC/DAgger best weights via `checkpoint:` in `configs/train_rl_flightroom.yaml`.  
- **Rewards:** progress to goal, heading alignment, bbox violation penalty, sparse success (see `rl/rewards.py`).  
- **Sim DR:** observation latency + 2D augmentations via shared frame buffer — **no relightable 3DGS**.  
- **Train domain:** flightroom training rollouts only; backroom/packardpark remain eval-only.  
- **Eval:** same §6 closed-loop suites; pass RL checkpoint with `--checkpoint`.

```bash
python scripts/train_rl.py --config configs/train_rl_flightroom.yaml          # PPO
python scripts/train_rl.py --config configs/train_rl_flightroom_sac.yaml      # SAC
```

---

## 10. Related work (brief)

| Work | Relevance to this plan |
|------|------------------------|
| **SOUS VIDE** | FiGS + large-scale BC + dynamics DR + sim→real (same scene). We use same simulator family, explicit goal vector, velocity interface, smaller data scale. |
| **Forest / Relightable 3DGS** | Photometric DR via relighting + RL. We adopt **latency + 2D augmentations** instead of relighting; defer RL. |
| **Depth Anything V2** | Monocular depth branch (Small variant for speed). |
| **DAgger** | Already in pipeline; flightroom-only collection. |

**Our advantages:** explicit goal conditioning; debuggable velocity + inner controller; modular `nav_policy` train/eval/Modal pipeline; depth fusion without changing control I/O.

---

## 11. Report updates (this branch)

Update `report/main.tex` to reflect:

1. **Architecture:** ResNet-18 + DA2-S + LayerNorm cross-attention fusion (diagram + forward pass).  
2. **Training:** flightroom-only; expert-endpoint goals; augmentations (§4) — explicitly **no** relightable 3DGS.  
3. **Evaluation:** stratified flightroom val (≥10) + OOD backroom/packardpark (≥10 each) with collision caveats.  
4. **Ablations:** A0–A5 table.  
5. **Results placeholders:** BC, DAgger R1/R2, full DA2 model; Grad-CAM figure(s).  
6. **Future work:** ~~RL fine-tuning (§9)~~ → **implemented**; populate RL rows in results tables after runs.  
7. **Related work:** shorten SOUS VIDE / Forest comparison; cite DA2.

Regenerate figures via `make_report_figures.py` / `collect_ablations.py` once eval artifacts exist.

---

## 12. Phased order

```
1. Rebuild flightroom caches (expert-end goals) + DA2-S depth precompute
2. Implement rgb_da2_crossattn_v1 + augmentations + latency option
3. BC train (A2) + RGB baseline (A0)
4. DAgger R1 (→ R2 if needed) on flightroom
5. Run ablations A0–A5
6. Robust eval: flightroom val suite + OOD backroom/packardpark suite
7. Grad-CAM + report updates on this branch
8. RL fine-tune (PPO default / SAC) from BC or DAgger checkpoint — `scripts/train_rl.py`
```

### DAgger dilution fix (2026-05-23)

After R1 showed no closed-loop gain over BC, training was updated to counter ~1% DAgger dilution:

| Knob | Config | Effect |
|------|--------|--------|
| `train.dagger_sampling: balanced` | `flightroom.yaml`, `flightroom_modal.yaml` | 50/50 expert vs `round>=1` windows per epoch |
| `train.dagger_sampling: oversample` | same + `dagger_oversample_factor: 15` | ~15× repeat of DAgger windows (~10–20× effective weight) |
| `extended_horizon: true` + `collection:` | `dagger_round*_mpc.yaml` | DAgger rollouts run until goal/collision (`run_until_terminal`, 180s cap), DA2 stride 3 |
| Val-hard collect | `configs/dagger_val_hard_mpc.yaml` | Round 3 on 071733 goal failures (q01,q04,q06,q08,q10,q11) |

**Recommended re-train after R2 collect:**

```powershell
modal run nav_policy/modal_train.py `
  --run-tag dagger_r2_balanced `
  --resume-from /data/checkpoints_dagger_r1_mpc/bc_best.pt `
  --checkpoint-dir /data/checkpoints_dagger_r2_balanced
```

Then optionally collect val-hard (`python scripts/run_dagger.py --config configs/dagger_val_hard_mpc.yaml`) and re-fine-tune again.

---

*Document version: 2026-05-23 (revised). Branch: `feature/da2-crossattn-rl-generalization`.*
