#!/usr/bin/env python3
"""Evaluate saved official-format nuScenes-lidarseg prediction .bin files.

Useful when predictions came from another inference script, Pointcept's tester,
or eval_pointcept_ptv3.py --save-predictions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from labels import CLASS_NAMES, official_to_internal_ids
from metrics import SegmentationMeter
from nuscenes_lidarseg_dataset import NuScenesLidarSegDataset, collate_single_scan


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--split", default="mini_val", choices=["mini_train", "mini_val", "train", "val"])
    p.add_argument("--pred-dir", required=True, help="Folder with {lidar_token}_lidarseg.bin files containing official ids 1..16.")
    p.add_argument("--voxel-size", type=float, default=0.05, help="Only used to instantiate dataset; predictions are full-res.")
    p.add_argument("--max-scans", type=int, default=None)
    p.add_argument("--save-json", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    ds = NuScenesLidarSegDataset(
        args.dataroot,
        version=args.version,
        split=args.split,
        voxel_size=args.voxel_size,
        max_scans=args.max_scans,
        verbose=True,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_single_scan, num_workers=0)
    meter = SegmentationMeter(num_classes=len(CLASS_NAMES))
    missing = []
    bad_shape = []

    for batch in tqdm(dl, desc="Evaluating saved predictions"):
        path = pred_dir / f"{batch['lidar_token']}_lidarseg.bin"
        if not path.exists():
            missing.append(str(path))
            continue
        pred_official = np.fromfile(path, dtype=np.uint8)
        gt = batch["origin_segment"].numpy()
        if pred_official.shape[0] != gt.shape[0]:
            bad_shape.append((str(path), int(pred_official.shape[0]), int(gt.shape[0])))
            continue
        pred_internal = official_to_internal_ids(pred_official)
        meter.update(pred_internal, gt)

    if missing:
        print(f"[eval_saved] Missing {len(missing)} prediction files. First few: {missing[:5]}")
    if bad_shape:
        print(f"[eval_saved] Bad shape for {len(bad_shape)} files. First few: {bad_shape[:5]}")
    if missing or bad_shape:
        raise SystemExit("Prediction folder is incomplete or has wrong-length .bin files.")

    result = meter.compute()
    print("\n" + meter.format_summary(result))
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(meter.to_dict(result), f, indent=2)
        print(f"[eval_saved] wrote {args.save_json}")


if __name__ == "__main__":
    main()
