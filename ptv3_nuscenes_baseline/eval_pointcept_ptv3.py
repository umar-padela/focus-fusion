#!/usr/bin/env python3
"""Evaluate a Pointcept PTv3 semantic-segmentation checkpoint on nuScenes-lidarseg.

This script uses the raw nuScenes mini/train/val files directly, not Pointcept's
preprocessed info files.  It builds the Pointcept PTv3 model from a config,
loads an optional checkpoint, runs one scan at a time, maps voxel predictions
back to original points, and reports mIoU/mAcc/fwIoU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from labels import CLASS_NAMES, internal_to_official_ids
from metrics import SegmentationMeter
from nuscenes_lidarseg_dataset import NuScenesLidarSegDataset, collate_single_scan
from pointcept_utils import (
    build_pointcept_model,
    get_logits,
    load_checkpoint_flexible,
    load_pointcept_config,
    patch_ptv3_config,
    to_device_input,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataroot", required=True, help="nuScenes root with samples/, lidarseg/, v1.0-mini/.")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--split", default="mini_val", choices=["mini_train", "mini_val", "train", "val"])
    p.add_argument("--config", required=True, help="Pointcept PTv3 config, e.g. configs/nuscenes/semseg-pt-v3m1-0-base.py")
    p.add_argument("--checkpoint", default="", help="Optional Pointcept checkpoint/model_best.pth. If omitted, evaluates random weights.")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    p.add_argument("--voxel-size", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-scans", type=int, default=None, help="Debug on the first K scans.")
    p.add_argument("--disable-flash", action="store_true", help="Set PTv3 enable_flash=False in the loaded config.")
    p.add_argument("--patch-size", type=int, default=None, help="Override enc/dec patch size, e.g. 128 when flash attention is unavailable.")
    p.add_argument("--ignore-head", action="store_true", help="Do not load checkpoint seg_head/classifier weights.")
    p.add_argument("--save-predictions", default="", help="Optional output folder for official-format *_lidarseg.bin predictions.")
    p.add_argument("--save-json", default="", help="Optional path to save metrics JSON.")
    return p.parse_args()


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, save_predictions: str = "") -> SegmentationMeter:
    model.eval()
    meter = SegmentationMeter(num_classes=len(CLASS_NAMES))
    pred_dir = Path(save_predictions) if save_predictions else None
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="Evaluating PTv3"):
        model_input = to_device_input(batch, device, include_segment=False)
        output = model(model_input)
        logits = get_logits(output)
        if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
            raise RuntimeError(
                f"Expected logits [N, {len(CLASS_NAMES)}], got {tuple(logits.shape)}. "
                "Check that the config has num_classes=16 and backbone_out_channels match the checkpoint."
            )
        pred_voxel = logits.argmax(dim=1).detach().cpu()
        pred_full = pred_voxel[batch["inverse"]].numpy().astype(np.int64)
        target_full = batch["origin_segment"].numpy().astype(np.int64)
        meter.update(pred_full, target_full)

        if pred_dir is not None:
            official = internal_to_official_ids(pred_full)
            out_path = pred_dir / f"{batch['lidar_token']}_lidarseg.bin"
            official.tofile(out_path)

    return meter


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[eval] CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dataset = NuScenesLidarSegDataset(
        dataroot=args.dataroot,
        version=args.version,
        split=args.split,
        voxel_size=args.voxel_size,
        max_scans=args.max_scans,
        verbose=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_single_scan,
    )

    cfg = load_pointcept_config(args.config)
    cfg = patch_ptv3_config(
        cfg,
        disable_flash=args.disable_flash,
        patch_size=args.patch_size,
        freeze_backbone=False,
    )
    model = build_pointcept_model(cfg, device)

    if args.checkpoint:
        ignore = ("seg_head", "cls_head", "classifier") if args.ignore_head else ()
        missing, unexpected, dropped = load_checkpoint_flexible(
            model, args.checkpoint, device=device, ignore_substrings=ignore
        )
        print(f"[eval] Loaded checkpoint: {args.checkpoint}")
        print(f"[eval] missing_keys={len(missing)} unexpected_keys={len(unexpected)} dropped_keys={len(dropped)}")
        if missing[:8]:
            print(f"[eval] first missing keys: {missing[:8]}")
        if dropped[:8]:
            print(f"[eval] first dropped keys: {dropped[:8]}")
    else:
        print("[eval] WARNING: no checkpoint supplied; evaluating random PTv3 weights.")

    meter = evaluate(model, loader, device, save_predictions=args.save_predictions)
    result = meter.compute()
    print("\n" + meter.format_summary(result))

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(meter.to_dict(result), f, indent=2)
        print(f"[eval] wrote metrics JSON to {args.save_json}")


if __name__ == "__main__":
    main()
