"""
Modal training app for V-LEAD nav_policy.

HOW IT WORKS (Modal 1.x API):
  Code is bundled into the Image using image.add_local_dir() instead of the
  deprecated modal.Mount.  The image is rebuilt when code changes (fast --
  only the add_local_dir layer is invalidated, not the pip-install layers).

SETUP (one-time, on your local Windows machine, NOT inside Docker):
  pip install modal
  modal setup                    # opens browser to authenticate
  modal volume create vlead-data

DATA UPLOAD (one-time):
  modal volume put vlead-data "C:\\path\\to\\nav_policy\\data\\processed_flightroom" /processed_flightroom_da2

RUNNING TRAINING:
  modal run nav_policy/modal_train.py
  modal run nav_policy/modal_train.py --run-tag a2_da2_crossattn

DOWNLOADING THE CHECKPOINT:
  modal volume get vlead-data checkpoints_a2_da2_crossattn/bc_best.pt nav_policy/data/checkpoints_a2_da2_crossattn/bc_best.pt

MONITORING:
  modal.com/apps  ->  live logs, GPU utilization, cost per second
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# ── 1. App ─────────────────────────────────────────────────────────────────────
# Groups all Modal objects for this project.  Name appears in the dashboard.
app = modal.App("vlead-bc-training")

NAV_POLICY_DIR = Path(__file__).parent   # the nav_policy/ directory

# ── 2. Container Image ──────────────────────────────────────────────────────────
# Modal builds this once and caches each layer.  Only layers that change are
# rebuilt on subsequent runs.
#
# Layer order matters for caching:
#   1. Base image  (never changes)
#   2. apt_install (rarely changes)
#   3. pip_install (changes only when dependencies change)
#   4. add_local_dir (changes whenever your Python code changes -- fast layer)
#
# add_local_dir(copy=False) is the default: files are injected at container
# startup without baking them into the image layer, which is fastest for
# iterative development.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
        add_python="3.10",
    )
    .apt_install(["libgl1", "libglib2.0-0", "ffmpeg"])
    .pip_install(
        "torchvision==0.19.0",
        "imageio[ffmpeg]>=2.34",
        "scipy>=1.13",
        "tqdm",
        "pyyaml",
        "numpy",
        "opencv-python-headless",
        "Pillow",
    )
    # Inject the nav_policy source code into the container.
    # Excludes the data/ directory (that lives in the Volume).
    .add_local_dir(
        str(NAV_POLICY_DIR),
        remote_path="/workspace/nav_policy",
        ignore=["data/", "__pycache__", "*.egg-info", ".git", "*.pyc"],
    )
)

# ── 3. Persistent Volume ────────────────────────────────────────────────────────
# Cloud disk that survives between runs.  Holds the processed dataset +
# checkpoints written during training.
DATA_VOLUME_NAME = "vlead-data"
VOLUME_MOUNT_PATH = "/data"

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

# ── 4. Training Function ────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A100",               # 40 GB VRAM, ~80 GB RAM, 40 vCPUs
    volumes={VOLUME_MOUNT_PATH: data_volume},
    timeout=60 * 60 * 8,     # 8-hour limit (~30 min/epoch on A100 with DA2 data → 10 epochs)
)
def train_bc(
    run_tag: str = "a2_da2_crossattn",
    resume_from: str = "",
    checkpoint_dir: str = "",
) -> str:
    """
    Run one complete BC training job in the cloud.

    The function body runs INSIDE the Modal container, not on your machine.
    It calls the exact same train_bc.py script as the local workflow --
    the only difference is the config file, which points paths to /data/.

    Returns the Volume path where bc_best.pt was saved.
    """
    import os
    import subprocess

    # Add the nav_policy src/ to PYTHONPATH so `import nav_policy` works.
    # This is instant -- no pip install needed since the code was injected
    # by add_local_dir above.
    env = {
        **os.environ,
        "PYTHONPATH": "/workspace/nav_policy/src",
    }

    # Confirm the GPU is visible (shows up in the dashboard logs).
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"[modal] GPU: {result.stdout.strip()}", flush=True)

    # Run training.
    cmd = [
        sys.executable, "scripts/train_bc.py",
        "--config", "configs/flightroom_modal.yaml",
        "--run-tag", run_tag,
    ]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    if checkpoint_dir:
        cmd += ["--checkpoint-dir", checkpoint_dir]
    print(f"[modal] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/workspace/nav_policy", env=env)

    # Volumes need an explicit commit() to guarantee writes are persisted
    # before the container exits.
    data_volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"Training exited with code {proc.returncode}")

    out_dir = checkpoint_dir if checkpoint_dir else f"{VOLUME_MOUNT_PATH}/checkpoints_a2_da2_crossattn"
    ckpt_path = f"{out_dir}/bc_best.pt"
    print(f"[modal] Done. Checkpoint at {ckpt_path}", flush=True)
    return ckpt_path


# ── 5. Local Entrypoint ─────────────────────────────────────────────────────────
# Runs on YOUR machine when you type `modal run modal_train.py`.
# It calls train_bc.remote() which submits the job to the cloud.
@app.local_entrypoint()
def main(
    run_tag: str = "a2_da2_crossattn",
    resume_from: str = "",
    checkpoint_dir: str = "",
):
    """
    Trigger remote training and print the checkpoint location.

    Usage:
        modal run nav_policy/modal_train.py
        modal run nav_policy/modal_train.py --run-tag my_run_v2
    """
    print(f"[local] Submitting training job  run_tag='{run_tag}' ...")
    ckpt = train_bc.remote(run_tag=run_tag, resume_from=resume_from, checkpoint_dir=checkpoint_dir)
    print(f"[local] Training complete.  Checkpoint: {ckpt}")
    print()
    print("Download with:")
    print(
        f"  modal volume get {DATA_VOLUME_NAME} "
        f"checkpoints_a2_da2_crossattn/bc_best.pt "
        f"nav_policy/data/checkpoints_a2_da2_crossattn/bc_best.pt"
    )
