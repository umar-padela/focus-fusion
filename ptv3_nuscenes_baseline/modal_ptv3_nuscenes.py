import modal

app = modal.App("ptv3-nuscenes-mini")

# Paths inside Modal container.
NUSCENES_DIR = "/data/nuscenes"
CKPT_DIR = "/checkpoints"
OUT_DIR = "/outputs"
POINTCEPT_DIR = "/root/Pointcept"

# Modal volumes.
nuscenes_vol = modal.Volume.from_name("nuscenes-data")
ckpt_vol = modal.Volume.from_name("ptv3-checkpoints")
out_vol = modal.Volume.from_name("ptv3-outputs")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
        "clang",
        "gcc",
        "g++",
        "libgl1",
        "libglib2.0-0",
        "ninja-build",
    )
    .pip_install(
        # Core torch stack.
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",

        # General scientific/data deps.
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "h5py",
        "pyyaml",
        "tqdm",

        # nuScenes.
        "nuscenes-devkit",
        "pyquaternion",

        # Pointcept common deps.
        "addict",
        "yapf",
        "tensorboard",
        "tensorboardX",
        "termcolor",
        "sharedarray",
        "einops",
        "plyfile",
        "timm",
        "wandb",

        # Newer Pointcept optional imports.
        "transformers==4.41.2",
        "peft==0.11.1",
        "huggingface_hub==0.23.4",
        "accelerate==0.31.0",

        # Sparse conv backend.
        "spconv-cu120",
    )
    .run_commands(
        "python -m pip install "
        "torch-scatter torch-cluster torch-sparse torch-spline-conv torch-geometric "
        "-f https://data.pyg.org/whl/torch-2.1.0+cu121.html"
    )
    .run_commands(
        f"git clone https://github.com/Pointcept/Pointcept.git {POINTCEPT_DIR}",
    )
    .run_commands(
        # Build Pointcept CUDA extension.
        f"cd {POINTCEPT_DIR}/libs/pointops && "
        "TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9' "
        "CUDA_HOME=/usr/local/cuda "
        "python setup.py install"
    )
    .env(
        {
            "PYTHONPATH": f"{POINTCEPT_DIR}:/root/project",
            "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9",
            "CUDA_HOME": "/usr/local/cuda",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
        }
    )
    .run_commands(
        # Build-time smoke test.
        "python - <<'PY'\n"
        "import sys\n"
        f"sys.path.insert(0, '{POINTCEPT_DIR}')\n"
        "import torch\n"
        "import torch_scatter\n"
        "import torch_cluster\n"
        "import torch_sparse\n"
        "import torch_geometric\n"
        "import spconv.pytorch as spconv\n"
        "import wandb\n"
        "import peft\n"
        "import pointops\n"
        "from pointcept.models import build_model\n"
        "print('Pointcept dependency smoke test passed')\n"
        "PY"
    )
    # IMPORTANT: add_local_dir must stay last.
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            ".git",
            "__pycache__",
            "*.pyc",
            ".DS_Store",
            "datasets",
            "runs",
            "outputs",
            "checkpoints",
        ],
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={
        "/data": nuscenes_vol,
        "/outputs": out_vol,
    },
)
def check_data(split: str = "mini_val"):
    import subprocess
    from pathlib import Path

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "/root/project/check_nuscenes_lidarseg.py",
        "--dataroot",
        NUSCENES_DIR,
        "--version",
        "v1.0-mini",
        "--split",
        split,
    ]

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    out_vol.commit()


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes={
        "/data": nuscenes_vol,
        "/checkpoints": ckpt_vol,
        "/outputs": out_vol,
    },
)
def eval_ptv3_mini(
    split: str = "mini_val",
    checkpoint_name: str = "model_best.pth",
):
    import subprocess
    from pathlib import Path

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    config = f"{POINTCEPT_DIR}/configs/nuscenes/semseg-pt-v3m1-0-base.py"
    checkpoint = f"{CKPT_DIR}/{checkpoint_name}"

    cmd = [
        "python",
        "/root/project/eval_pointcept_ptv3.py",
        "--dataroot",
        NUSCENES_DIR,
        "--version",
        "v1.0-mini",
        "--split",
        split,
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--device",
        "cuda",
        "--disable-flash",
        "--patch-size",
        "128",
        "--save-json",
        f"{OUT_DIR}/ptv3_{split}_metrics.json",
        "--save-predictions",
        f"{OUT_DIR}/ptv3_{split}_predictions",
    ]

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    out_vol.commit()
    print(f"Saved metrics to {OUT_DIR}/ptv3_{split}_metrics.json")


@app.local_entrypoint()
def main(
    mode: str = "check",
    split: str = "mini_val",
    checkpoint_name: str = "model_best.pth",
):
    if mode == "check":
        check_data.remote(split)
    elif mode == "eval":
        eval_ptv3_mini.remote(split, checkpoint_name)
    else:
        raise ValueError("mode must be 'check' or 'eval'")