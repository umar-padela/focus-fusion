"""
Shared Modal infrastructure for FocusFusion.

Imported by modal_smoke.py, modal_train.py, and modal_eval.py.

Layout
------
- ``app``               — groups this project's Modal functions (dashboard: focus-fusion).
- ``image``             — Debian + CUDA env (torch 2.1.0+cu121, DINOv2, nuScenes deps).
- ``volume``            — single persistent disk for both data and experiments (volume: focus-fusion-experiments).
                          nuScenes data lives at /experiments/data/; checkpoints at /experiments/.
- ``smoke``             — infra health check: GPU + torch + DINOv2 forward pass.
- ``train``             — remote training on GPU_TRAIN (default A100-40GB).
- ``evaluate``          — remote eval on GPU_EVAL (default A10G).

Local vs remote
---------------
Your laptop only runs ``modal run ...`` (see modal_train.py, modal_eval.py).
Functions decorated with ``@app.function`` run inside ``image`` on Modal's GPUs.

Entrypoints use ``.spawn()`` so jobs keep running if the local client disconnects;
use ``--detach`` before the script path to return immediately without waiting.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App & container paths
# ---------------------------------------------------------------------------

app = modal.App("focus-fusion")

REPO_ROOT = "/root/focus-fusion"
EXPERIMENTS_MOUNT = "/experiments"               # focus-fusion-experiments volume root
DATA_MOUNT = f"{EXPERIMENTS_MOUNT}/data"         # nuScenes mini lives here inside the volume

# ---------------------------------------------------------------------------
# Image — mirrors cs224r conda env (torch 2.1.0+cu121) + project deps
# ---------------------------------------------------------------------------
# Layer order matters: pip installs are cached; add_local_dir is last so
# code changes don't invalidate the heavy torch/nuscenes layers above it.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "curl",
        "libglib2.0-0",      # OpenCV runtime
        "libgl1-mesa-glx",   # OpenCV headless
        "libgomp1",          # OpenMP (numpy, scikit-learn)
    )
    .pip_install(
        # torch 2.1.0 + CUDA 12.1 — matches cs224r env
        "torch==2.1.0",
        "torchvision==0.16.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy<2.0",
        "tqdm",
        "einops",           # used by DINOv2 internals
        "omegaconf",        # used by DINOv2 internals + config loading
        "pyyaml",
        "scikit-learn",     # mIoU computation in eval
        "wandb",
        "nuscenes-devkit",  # nuScenes mini dataset loading
        "opencv-python-headless",
        "matplotlib",
        "Pillow",
    )
    # Pre-download DINOv2 ViT-S/14 weights into the image so containers start
    # without a network download. Uses fbaipublicfiles hub; same weights as our
    # local submodule (third_party/dinov2) with source='local'.
    .run_commands(
        "python -c \""
        "import torch; "
        "torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)"
        "\""
    )
    # TODO(Person 1): add ptv3 CUDA ops installation once build steps are confirmed.
    # Expected step (example, verify with Person 1):
    #   .run_commands(
    #       f"cd {REPO_ROOT}/third_party/ptv3 && pip install -e . --no-build-isolation"
    #   )
    .env(
        {
            # Make focus_fusion and the dinov2 submodule importable without install.
            "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT}/third_party/dinov2",
        }
    )
    .add_local_dir(
        ".",
        remote_path=REPO_ROOT,
        ignore=[
            "experiments/**",
            "checkpoints/**",
            ".git/**",
            "**/__pycache__/**",
            "**/.pytest_cache/**",
            "data/**",
        ],
    )
)

# ---------------------------------------------------------------------------
# Volumes — persistent across runs (created on first use)
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name("focus-fusion-experiments", create_if_missing=True)

# ---------------------------------------------------------------------------
# GPU knobs — override without editing code:
#   Windows: $env:MODAL_GPU_TRAIN="A100-80GB"; modal run modal/modal_train.py
#   Unix:    MODAL_GPU_TRAIN=A100-80GB modal run modal/modal_train.py
# ---------------------------------------------------------------------------

GPU_EVAL = os.environ.get("MODAL_GPU_EVAL", "A10G")
GPU_TRAIN = os.environ.get("MODAL_GPU_TRAIN", "A100-40GB")

# ---------------------------------------------------------------------------
# W&B secret — create once: modal secret create wandb WANDB_API_KEY=<key>
# ---------------------------------------------------------------------------

_wandb_secret = modal.Secret.from_name("wandb")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def spawn_modal_function(modal_function, *, label: str, wait: bool = True, **kwargs):
    """
    Start a Modal function with .spawn() so work survives terminal disconnect.

    wait=True  → .spawn().get(): blocks until done; logs stream live.
    wait=False → returns FunctionCall immediately; monitor in Modal dashboard.
    Combine with ``modal run --detach`` to survive Ctrl-C without cancelling.
    """
    call = modal_function.spawn(**kwargs)
    print(f"Spawned {label} (object_id={call.object_id})")
    if not wait:
        print("Detached — job continues; check Modal dashboard for logs.")
        return call
    return call.get()


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU_EVAL,
    timeout=600,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def smoke() -> dict:
    """
    Infrastructure smoke test: GPU + torch + DINOv2 forward pass + volume write.

    Runs a single DINOv2 ViT-S/14 forward pass:
        (B=1, 6 cameras, 3, 448, 448) → (1, 6, 1024, 384)
    to verify the full image → patch embedding pipeline works on GPU.
    """
    import torch
    from focus_fusion.models.backbones.dinov2 import DINOv2Backbone

    cuda_ok = torch.cuda.is_available()
    device = "cuda:0" if cuda_ok else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "cpu"

    backbone = DINOv2Backbone().to(device)
    images = torch.randn(1, 6, 3, 448, 448, device=device)
    patches = backbone(images)
    assert patches.shape == (1, 6, 1024, 384), f"Unexpected patch shape: {patches.shape}"

    out = {
        "torch": str(torch.__version__),
        "cuda_available": cuda_ok,
        "cuda_device": device_name,
        "dinov2_output_shape": list(patches.shape),
    }

    marker = Path(EXPERIMENTS_MOUNT) / "smoke_ok.txt"
    marker.write_text(
        f"torch={out['torch']} cuda={cuda_ok} dinov2={tuple(out['dinov2_output_shape'])}\n",
        encoding="utf-8",
    )
    volume.commit()

    print("FocusFusion smoke OK:", out)
    return out


@app.function(
    image=image,
    gpu=GPU_TRAIN,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def train(
    experiment: str = "e1",
    config: str = "configs/default.yaml",
    extra_args: list[str] | None = None,
) -> None:
    """
    Remote training wrapper.

    Calls:
        python -m focus_fusion.train.trainer
            --config <config>
            --experiment <experiment>
            --data-root /data
            --output-dir /experiments
            [extra_args...]

    Experiments:
        e1 — FocusFusion T=1 (single frame; hparams frozen from Person 3)
        e2 — FocusFusion T=6 (3s temporal history; Person 2 owns this run)

    Checkpoints written to /experiments/<experiment>/ on the volume.
    """
    cmd = [
        "python", "-m", "focus_fusion.train.trainer",
        "--config", config,
        "--experiment", experiment,
        "--data-root", DATA_MOUNT,
        "--output-dir", EXPERIMENTS_MOUNT,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    volume.commit()


@app.function(
    image=image,
    gpu=GPU_EVAL,
    timeout=7200,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def evaluate(
    experiment: str = "e1",
    config: str = "configs/default.yaml",
    checkpoint: str | None = None,
    split: str = "mini_val",
    extra_args: list[str] | None = None,
) -> dict:
    """
    Remote eval wrapper.

    Calls:
        python -m focus_fusion.eval.metrics
            --config <config>
            --experiment <experiment>
            --checkpoint <checkpoint>
            --split <split>
            --data-root /data
            --output-dir /experiments
            [extra_args...]

    Default checkpoint: /experiments/<experiment>/best.pt
    Returns {"experiment", "checkpoint", "split"} on success.
    """
    if checkpoint is None:
        checkpoint = str(Path(EXPERIMENTS_MOUNT) / experiment / "best.pt")

    cmd = [
        "python", "-m", "focus_fusion.eval.metrics",
        "--config", config,
        "--experiment", experiment,
        "--checkpoint", checkpoint,
        "--split", split,
        "--data-root", DATA_MOUNT,
        "--output-dir", EXPERIMENTS_MOUNT,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    volume.commit()

    return {
        "experiment": experiment,
        "checkpoint": checkpoint,
        "split": split,
    }
