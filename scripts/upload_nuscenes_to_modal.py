"""Upload nuScenes archives from a local folder to Modal, then extract.

USAGE
-----
1. Download your nuScenes archives and put them all in data/nuscenes_archives/:

       data/nuscenes_archives/
           nuScenes-lidarseg-all-v1.0.tar.bz2
           v1.0-trainval_meta.tgz
           v1.0-trainval01_blobs.tgz
           v1.0-trainval02_blobs.tgz
           ...

   You don't need all 10 blob files at once — add more later and re-run.

2. Run this script from the repo root:

       python scripts/upload_nuscenes_to_modal.py

   It uploads every archive in the folder then extracts them all on Modal.
   Re-running is safe — Modal volume put overwrites and extraction is idempotent.

OPTIONS
-------
  --src PATH          Folder of archives (default: data/nuscenes_archives)
  --extract-only      Skip upload; just re-run extraction on already-uploaded files
  --upload-only       Upload archives but skip extraction step
  --dry-run           Print what would happen without doing anything
"""

import argparse
import subprocess
import sys
from pathlib import Path

VOLUME_NAME = "focus-fusion-experiments"
REMOTE_UPLOADS = "/uploads"  # volume-root-relative; mounts at /experiments/uploads/ in container

ARCHIVE_EXTENSIONS = {".tgz", ".gz", ".bz2", ".zip", ".tar"}


def is_archive(path: Path) -> bool:
    return path.suffix in ARCHIVE_EXTENSIONS or ".tar." in path.name


def run(cmd: list[str], dry_run: bool) -> None:
    print(f"  $ {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload nuScenes archives to Modal and extract them."
    )
    parser.add_argument(
        "--src",
        default=None,
        help="Folder of nuScenes archive files (default: data/nuscenes_archives)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="One or more specific archive file paths to upload (alternative to --src)",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Skip upload; just run extraction on files already on the volume",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Upload archives but skip extraction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ upload
    if not args.extract_only:
        if args.files:
            archives = [Path(f) for f in args.files]
            missing = [a for a in archives if not a.exists()]
            if missing:
                for m in missing:
                    print(f"Error: file not found: {m}")
                sys.exit(1)
        else:
            src = Path(args.src) if args.src else Path("data/nuscenes_archives")
            if not src.exists():
                print(f"Error: folder not found: {src}")
                print(f"Either create that folder and put archives in it, or use --files:")
                print(f'  python scripts/upload_nuscenes_to_modal.py --files "C:\\Users\\umarp\\Downloads\\v1.0-trainval01_blobs.tgz"')
                sys.exit(1)
            archives = sorted(p for p in src.iterdir() if p.is_file() and is_archive(p))
            if not archives:
                print(f"No archive files found in {src}/")
                sys.exit(1)

        total_gb = sum(p.stat().st_size for p in archives) / 1e9
        print(f"Found {len(archives)} archive(s) ({total_gb:.1f} GB total):")
        for a in archives:
            print(f"  {a.name:50s}  {a.stat().st_size / 1e9:.1f} GB")
        print()

        for archive in archives:
            remote = f"{REMOTE_UPLOADS}/{archive.name}"
            print(f"Uploading: {archive.name} → modal:{remote}")
            if args.dry_run:
                print(f"  $ modal volume put {VOLUME_NAME} {archive} {remote}")
            else:
                result = subprocess.run(
                    ["modal", "volume", "put", VOLUME_NAME, str(archive), remote],
                )
                if result.returncode == 2:
                    print(f"  [skipped] already on volume — delete it first with:")
                    print(f"  modal volume rm {VOLUME_NAME} {remote}")
                elif result.returncode != 0:
                    print(f"  [error] upload failed (exit {result.returncode})")
                    sys.exit(result.returncode)
            print()

    # --------------------------------------------------------------- extract
    if not args.upload_only:
        print("Extracting archives on Modal (runs server-side) ...")
        run(
            ["modal", "run", "modal/modal_upload_data.py"],
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("\n[dry-run] No commands were executed.")
    else:
        print("\nDone. Data is now at /experiments/data on the Modal volume.")
        print("Update configs/default.yaml to use v1.0-trainval and run training.")


if __name__ == "__main__":
    main()
