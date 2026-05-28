#!/usr/bin/env python3
"""Small nuScenes-lidarseg smoke/preprocess entrypoint.

For now this validates the raw mini dataset and writes a compact split summary.
It intentionally does not materialize large processed tensors yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from focus_fusion.datasets.nuscenes import CLASS_NAMES, IGNORE_INDEX, NuScenesLidarSegDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="mini_val", choices=["mini_train", "mini_val", "train", "val"])
    parser.add_argument("--output", default="data/processed/nuscenes/summary.json")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = NuScenesLidarSegDataset(
        dataroot=args.dataroot,
        version=args.version,
        split=args.split,
        verbose=True,
    )
    hist = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    ignored = 0
    original_points = 0
    voxel_points = 0
    for item in dataset:
        labels = item["origin_segment"].numpy()
        valid = labels != IGNORE_INDEX
        hist += np.bincount(labels[valid], minlength=len(CLASS_NAMES))
        ignored += int((~valid).sum())
        original_points += int(item["num_original_points"])
        voxel_points += int(item["num_voxel_points"])

    summary = {
        "version": args.version,
        "split": args.split,
        "num_scans": len(dataset),
        "num_original_points": original_points,
        "num_voxel_points": voxel_points,
        "ignored_points": ignored,
        "class_histogram": dict(zip(CLASS_NAMES, hist.tolist())),
    }
    print(json.dumps(summary, indent=2))
    if not args.summary_only:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
