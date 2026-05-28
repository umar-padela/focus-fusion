#!/usr/bin/env python3
"""A tiny non-neural sanity baseline for slides: predict the train-set majority class everywhere."""

from __future__ import annotations

import argparse
import json

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from labels import CLASS_NAMES, IGNORE_INDEX
from metrics import SegmentationMeter
from nuscenes_lidarseg_dataset import NuScenesLidarSegDataset, collate_single_scan


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--train-split", default="mini_train", choices=["mini_train", "train"])
    p.add_argument("--eval-split", default="mini_val", choices=["mini_train", "mini_val", "train", "val"])
    p.add_argument("--voxel-size", type=float, default=0.05)
    p.add_argument("--max-train-scans", type=int, default=None)
    p.add_argument("--max-eval-scans", type=int, default=None)
    p.add_argument("--save-json", default="")
    return p.parse_args()


def histogram(ds):
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_single_scan, num_workers=0)
    hist = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for batch in tqdm(dl, desc="train histogram"):
        y = batch["origin_segment"].numpy()
        valid = (y >= 0) & (y < len(CLASS_NAMES))
        hist += np.bincount(y[valid], minlength=len(CLASS_NAMES))
    return hist


def main():
    args = parse_args()
    train_ds = NuScenesLidarSegDataset(
        args.dataroot,
        version=args.version,
        split=args.train_split,
        voxel_size=args.voxel_size,
        max_scans=args.max_train_scans,
        verbose=True,
    )
    eval_ds = NuScenesLidarSegDataset(
        args.dataroot,
        version=args.version,
        split=args.eval_split,
        voxel_size=args.voxel_size,
        max_scans=args.max_eval_scans,
        verbose=True,
    )
    hist = histogram(train_ds)
    majority = int(hist.argmax())
    print(f"Majority class from {args.train_split}: {CLASS_NAMES[majority]} ({hist[majority]:,} points)")

    meter = SegmentationMeter(num_classes=len(CLASS_NAMES))
    dl = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_single_scan, num_workers=0)
    for batch in tqdm(dl, desc="eval majority"):
        y = batch["origin_segment"].numpy()
        pred = np.full_like(y, fill_value=majority)
        meter.update(pred, y)
    result = meter.compute()
    print("\n" + meter.format_summary(result))
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(meter.to_dict(result), f, indent=2)
        print(f"wrote {args.save_json}")


if __name__ == "__main__":
    main()
