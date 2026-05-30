"""
One-time volume setup for focus-fusion-experiments.

Handles two things:
  1. nuScenes data — upload archives locally, extract server-side.
  2. LitePT checkpoint — pulled from HuggingFace directly into the volume.

Usage
-----
  # Step 1: upload archives from your machine (run once, ~4 GB):
  #   modal volume put focus-fusion-experiments <local>.tgz     data/archives/v1.0-mini.tgz
  #   modal volume put focus-fusion-experiments <local>.tar.bz2 data/archives/nuScenes-lidarseg-mini-v1.0.tar.bz2

  # Step 2: extract archives on the server (fast, runs in Modal):
  modal run modal/modal_setup.py --mode extract-data

  # Step 3: download LitePT checkpoint from HuggingFace into the volume:
  modal run modal/modal_setup.py --mode download-litept

  # Verify everything is ready:
  modal run modal/modal_setup.py --mode check

  # Re-download LitePT even if it already exists:
  modal run modal/modal_setup.py --mode download-litept --force
"""

import modal

# Inline these rather than importing from modal_config — the setup image is
# lightweight (no repo files) so modal_config isn't on the container's path.
app = modal.App("focus-fusion")
EXPERIMENTS_MOUNT = "/experiments"
volume = modal.Volume.from_name("focus-fusion-experiments", create_if_missing=True)

# Lightweight image — no torch/CUDA needed for setup tasks.
_setup_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.23")
)

HF_REPO_ID = "prs-eth/LitePT"
HF_FILENAME = "nuscenes-semseg-litept-small-v1m1/model/model_best.pth"
VOLUME_CKPT_PATH = "checkpoints/litept/model_best.pth"

ARCHIVE_MINI      = "data/archives/v1.0-mini.tgz"
ARCHIVE_LIDARSEG  = "data/archives/nuScenes-lidarseg-mini-v1.0.tar.bz2"
DATA_DIR          = "data"


# ---------------------------------------------------------------------------
# nuScenes extraction
# ---------------------------------------------------------------------------

@app.function(
    image=_setup_image,
    timeout=1800,   # extraction can take a while for the 4 GB mini split
    volumes={EXPERIMENTS_MOUNT: volume},
)
def extract_nuscenes() -> None:
    """Extract nuScenes archives that were uploaded to data/archives/ on the volume."""
    import subprocess
    from pathlib import Path

    root = Path(EXPERIMENTS_MOUNT)
    mini_archive     = root / ARCHIVE_MINI
    lidarseg_archive = root / ARCHIVE_LIDARSEG
    data_dir         = root / DATA_DIR

    if not mini_archive.exists():
        raise FileNotFoundError(
            f"Mini archive not found on volume: {mini_archive}\n"
            "Upload it first:\n"
            "  modal volume put focus-fusion-experiments "
            r"C:\Users\umarp\Downloads\v1.0-mini.tgz data/archives/v1.0-mini.tgz"
        )
    if not lidarseg_archive.exists():
        raise FileNotFoundError(
            f"Lidarseg archive not found on volume: {lidarseg_archive}\n"
            "Upload it first:\n"
            "  modal volume put focus-fusion-experiments "
            r"C:\Users\umarp\Downloads\nuScenes-lidarseg-mini-v1.0.tar.bz2 "
            "data/archives/nuScenes-lidarseg-mini-v1.0.tar.bz2"
        )

    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Extracting {mini_archive.name} ({mini_archive.stat().st_size / 1e9:.2f} GB) ...")
    subprocess.run(["tar", "-xzf", str(mini_archive), "-C", str(data_dir)], check=True)

    print(f"[setup] Extracting {lidarseg_archive.name} ...")
    subprocess.run(["tar", "-xjf", str(lidarseg_archive), "-C", str(data_dir)], check=True)

    # Sanity check
    required = ["samples/LIDAR_TOP", "sweeps/LIDAR_TOP", "v1.0-mini", "lidarseg/v1.0-mini"]
    missing = [r for r in required if not (data_dir / r).exists()]
    if missing:
        raise RuntimeError(f"[setup] Extraction finished but these paths are missing: {missing}")

    volume.commit()
    print("[setup] nuScenes data extracted and committed to volume.")
    print(f"[setup] Layout under {data_dir}:")
    for p in sorted(data_dir.iterdir()):
        print(f"  {p.name}/")


# ---------------------------------------------------------------------------
# LitePT checkpoint
# ---------------------------------------------------------------------------

@app.function(
    image=_setup_image,
    timeout=600,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def download_litept(force: bool = False) -> str:
    """Download LitePT checkpoint from HuggingFace into the Modal volume."""
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    import shutil

    dest = Path(EXPERIMENTS_MOUNT) / VOLUME_CKPT_PATH
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1_000_000
        print(f"[setup] Checkpoint already exists ({size_mb:.1f} MB): {dest}")
        print("[setup] Pass --force to re-download.")
        return str(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Downloading {HF_REPO_ID}/{HF_FILENAME} ...")
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir="/tmp/litept_download",
    )

    shutil.copy2(local_path, dest)
    volume.commit()

    size_mb = dest.stat().st_size / 1_000_000
    print(f"[setup] Saved to volume: {dest} ({size_mb:.1f} MB)")
    return str(dest)


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

@app.function(
    image=_setup_image,
    timeout=60,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def check_volume() -> dict:
    """Print volume contents and whether all training prerequisites are met."""
    from pathlib import Path

    root = Path(EXPERIMENTS_MOUNT)

    def _status(rel: str) -> str:
        p = root / rel
        if not p.exists():
            return "MISSING"
        if p.is_file():
            return f"{p.stat().st_size / 1_000_000:.1f} MB"
        # directory — count immediate children as a quick sanity figure
        children = list(p.iterdir())
        return f"OK ({len(children)} items)"

    checks = {
        "litept_checkpoint":   VOLUME_CKPT_PATH,
        "nuscenes_v1.0-mini":  "data/v1.0-mini",
        "nuscenes_lidarseg":   "data/lidarseg",
        "nuscenes_lidar_top":  "data/samples/LIDAR_TOP",
    }

    results = {}
    all_ok = True
    for label, rel in checks.items():
        s = _status(rel)
        results[label] = s
        ok = s != "MISSING"
        all_ok = all_ok and ok
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {label}: {s}")

    results["ready_to_train"] = all_ok
    icon = "✓" if all_ok else "✗"
    print(f"  {icon}  ready_to_train: {all_ok}")
    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(mode: str = "check", force: bool = False) -> None:
    """
    Parameters
    ----------
    mode
        'extract-data'    — extract nuScenes archives already on the volume.
        'download-litept' — pull LitePT checkpoint from HuggingFace.
        'check'           — print volume contents and readiness (default).
    force
        Re-download LitePT even if checkpoint already exists.
    """
    if mode == "extract-data":
        extract_nuscenes.remote()
    elif mode == "download-litept":
        download_litept.remote(force=force)
    elif mode == "check":
        check_volume.remote()
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'extract-data', 'download-litept', or 'check'.")
