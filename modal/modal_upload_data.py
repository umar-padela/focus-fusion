"""Upload and extract nuScenes dataset archives to the Modal volume.

WORKFLOW
--------
Step 1 — upload archives from your local machine (run in your terminal, not via modal run):

    # Metadata (required — ~250 MB):
    modal volume put focus-fusion-experiments ./v1.0-trainval_meta.tgz /experiments/uploads/v1.0-trainval_meta.tgz

    # LiDAR + camera blobs (10 files, ~40 GB each — upload only what you need):
    modal volume put focus-fusion-experiments ./v1.0-trainval01_blobs.tgz /experiments/uploads/v1.0-trainval01_blobs.tgz
    modal volume put focus-fusion-experiments ./v1.0-trainval02_blobs.tgz /experiments/uploads/v1.0-trainval02_blobs.tgz
    ...

    # LidarSeg labels (required — ~500 MB):
    modal volume put focus-fusion-experiments ./v1.0-trainval_lidarseg.tgz /experiments/uploads/v1.0-trainval_lidarseg.tgz

Step 2 — extract all uploaded archives on Modal (fast — runs server-side):

    modal run modal/modal_upload_data.py

    # Or extract a single archive:
    modal run modal/modal_upload_data.py --archive v1.0-trainval_meta.tgz

Data lands at /experiments/data/ on the volume — the same path the trainer expects.
After extraction you can delete the archive from the volume to reclaim space:
    modal volume rm focus-fusion-experiments /experiments/uploads/

NOTES ON DATASET SIZE
---------------------
- v1.0-trainval full: ~350 GB (10 blobs + meta + lidarseg)
- For 10% fraction training (70 scenes): blobs 01–01 + meta + lidarseg (~90 GB) is enough
- For 30% fraction: blobs 01–03 (~170 GB)
- All 10 blobs needed only for 100% fraction
"""

import modal

app = modal.App("focus-fusion-upload")
volume = modal.Volume.from_name("focus-fusion-experiments", create_if_missing=True)

EXPERIMENTS_MOUNT = "/experiments"
# volume root `/uploads/` → mounted at `/experiments/uploads/` in the container
UPLOADS_DIR = f"{EXPERIMENTS_MOUNT}/uploads"
# handle files already uploaded to the wrong path (volume root `/experiments/uploads/`)
UPLOADS_DIR_LEGACY = f"{EXPERIMENTS_MOUNT}/experiments/uploads"
DATA_DIR = f"{EXPERIMENTS_MOUNT}/data"

# Minimal image — only needs stdlib tar/zip, no ML deps
_image = modal.Image.debian_slim().apt_install("tar", "bzip2", "gzip", "unzip")


@app.function(
    image=_image,
    volumes={EXPERIMENTS_MOUNT: volume},
    timeout=7200,
)
def extract(archive: str | None = None) -> None:
    """Extract one or all archives from /experiments/uploads/ to /experiments/data/.

    Args:
        archive: filename inside uploads/ to extract (e.g. "v1.0-trainval_meta.tgz").
                 None → extract every archive in uploads/.
    """
    import os
    import tarfile
    import zipfile
    from pathlib import Path

    volume.reload()  # pick up files uploaded via `modal volume put`

    data = Path(DATA_DIR)
    data.mkdir(parents=True, exist_ok=True)

    # Support both correct path (/uploads/) and legacy path (/experiments/uploads/)
    uploads = Path(UPLOADS_DIR)
    if not uploads.exists():
        uploads = Path(UPLOADS_DIR_LEGACY)
    if not uploads.exists():
        raise FileNotFoundError(
            f"No uploads directory found. Upload files first:\n"
            f"  modal volume put focus-fusion-experiments <local.tgz> /uploads/<file.tgz>"
        )

    print(f"Reading archives from: {uploads}")

    if archive:
        candidates = [uploads / archive]
    else:
        all_archives = [
            p for p in uploads.iterdir()
            if p.suffix in (".tgz", ".gz", ".bz2", ".zip", ".tar") or ".tar." in p.name
        ]
        # lidarseg must extract last — its category.json must overwrite the meta one
        candidates = sorted(a for a in all_archives if "lidarseg" not in a.name)
        candidates += sorted(a for a in all_archives if "lidarseg" in a.name)

    if not candidates:
        print("No archives found to extract.")
        return

    for path in candidates:
        if not path.exists():
            print(f"[skip] {path.name} — not found")
            continue

        size_gb = path.stat().st_size / 1e9
        print(f"Extracting {path.name} ({size_gb:.1f} GB) → {DATA_DIR} ...")

        if path.suffix == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(data)
        else:
            with tarfile.open(path, "r:*") as tf:
                tf.extractall(data)

        print(f"  done: {path.name}")

    volume.commit()
    print(f"\nAll done. Contents of {DATA_DIR}:")
    for item in sorted(data.iterdir()):
        print(f"  {item.name}")


@app.function(
    image=_image,
    volumes={EXPERIMENTS_MOUNT: volume},
    timeout=600,
)
def fix_category_json() -> None:
    """Extract just category.json from the lidarseg archive and place it correctly.

    nuScenes-lidarseg requires a specific category.json that differs from the
    one in v1.0-trainval_meta.tgz. This function finds and overwrites it.
    """
    import tarfile
    from pathlib import Path

    volume.reload()

    # Find the lidarseg archive (check both upload paths)
    lidarseg_archive = None
    for uploads_dir in (Path(UPLOADS_DIR), Path(UPLOADS_DIR_LEGACY)):
        if uploads_dir.exists():
            for p in uploads_dir.iterdir():
                if "lidarseg" in p.name:
                    lidarseg_archive = p
                    break
        if lidarseg_archive:
            break

    if not lidarseg_archive:
        raise FileNotFoundError("Could not find lidarseg archive in uploads directory.")

    print(f"Using: {lidarseg_archive}")

    # Find category.json inside the archive and extract it
    with tarfile.open(lidarseg_archive, "r:*") as tf:
        members = tf.getnames()
        cat_members = [m for m in members if m.endswith("category.json")]
        print(f"Found category.json entries: {cat_members}")

        for member in cat_members:
            f = tf.extractfile(member)
            if f is None:
                continue
            content = f.read()
            # Write to all matching version dirs under /experiments/data/
            data = Path(DATA_DIR)
            for version_dir in data.iterdir():
                if version_dir.is_dir() and version_dir.name.startswith("v1.0"):
                    dest = version_dir / "category.json"
                    dest.write_bytes(content)
                    print(f"  wrote {len(content)} bytes → {dest}")

    volume.commit()
    print("Done — category.json patched.")


@app.local_entrypoint()
def main(archive: str = "", fix_cat: bool = False) -> None:
    if fix_cat:
        fix_category_json.remote()
    else:
        extract.remote(archive=archive or None)
