#!/usr/bin/env python3
"""Smoke-test the nuScenes-lidarseg Dataset/DataLoader and print label stats."""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from labels import CLASS_NAMES, IGNORE_INDEX
from nuscenes_lidarseg_dataset import NuScenesLidarSegDataset, collate_single_scan


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", required=True, help="nuScenes root containing samples/, lidarseg/, v1.0-mini/.")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--split", default="mini_val", choices=["mini_train", "mini_val", "train", "val", "test"])
    p.add_argument("--voxel-size", type=float, default=0.05)
    p.add_argument("--max-scans", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = NuScenesLidarSegDataset(
        dataroot=args.dataroot,
        version=args.version,
        split=args.split,
        voxel_size=args.voxel_size,
        max_scans=args.max_scans,
        verbose=True,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_single_scan, num_workers=0)

    raw_points = 0
    voxel_points = 0
    hist = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    ignored = 0
    scenes = Counter()
    first = None

    for batch in tqdm(dl, desc="Scanning labels"):
        if first is None:
            first = batch
        raw_points += int(batch["num_original_points"])
        voxel_points += int(batch["num_voxel_points"])
        target = batch["origin_segment"].numpy()
        ignored += int((target == IGNORE_INDEX).sum())
        valid = (target >= 0) & (target < len(CLASS_NAMES))
        hist += np.bincount(target[valid], minlength=len(CLASS_NAMES))
        scenes[batch["scene_name"]] += 1

    print("\nDataset summary")
    print("---------------")
    print(f"split/version:     {args.split} / {args.version}")
    print(f"num scans:         {len(ds)}")
    print(f"num scenes:        {len(scenes)} -> {dict(scenes)}")
    print(f"raw points:        {raw_points:,}")
    print(f"voxel points:      {voxel_points:,}  (voxel_size={args.voxel_size})")
    print(f"ignored raw pts:   {ignored:,}")
    print("\nFirst scan")
    print("----------")
    print(f"lidar_token:       {first['lidar_token']}")
    print(f"scene:             {first['scene_name']}")
    print(f"coord shape:       {tuple(first['coord'].shape)}")
    print(f"feat shape:        {tuple(first['feat'].shape)}")
    print(f"grid_coord shape:  {tuple(first['grid_coord'].shape)}")
    print(f"inverse shape:     {tuple(first['inverse'].shape)}")

    print("\nClass histogram over original points")
    print("------------------------------------")
    total_valid = hist.sum()
    for name, count in zip(CLASS_NAMES, hist):
        pct = 100.0 * count / max(total_valid, 1)
        print(f"{name:<24} {count:>12,}  {pct:6.2f}%")


if __name__ == "__main__":
    main()
