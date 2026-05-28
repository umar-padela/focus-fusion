#!/usr/bin/env python3
"""Mini-set training/evaluation loop for a PTv3 LiDAR-only baseline.

Use this when you need a concrete baseline run for Milestone 3.  The best
scientific baseline is to load a nuScenes-pretrained PTv3 checkpoint and evaluate
on mini_val.  This script also lets you fine-tune/linear-probe on mini_train so
that you can verify the full data->model->metrics path end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from labels import CLASS_NAMES
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
    p.add_argument("--dataroot", required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--train-split", default="mini_train", choices=["mini_train", "train"])
    p.add_argument("--val-split", default="mini_val", choices=["mini_val", "val"])
    p.add_argument("--config", required=True, help="Pointcept PTv3 nuScenes config.")
    p.add_argument("--checkpoint", default="", help="Optional init checkpoint.")
    p.add_argument("--output", default="runs/ptv3_mini", help="Output dir for checkpoints/metrics.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=5e-3)
    p.add_argument("--voxel-size", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-scans", type=int, default=None)
    p.add_argument("--max-val-scans", type=int, default=None)
    p.add_argument("--freeze-backbone", action="store_true", help="Train only the segmentation head. Best used with a pretrained checkpoint.")
    p.add_argument("--ignore-head", action="store_true", help="Drop seg_head/classifier keys when loading checkpoint.")
    p.add_argument("--disable-flash", action="store_true")
    p.add_argument("--patch-size", type=int, default=None, help="e.g. 128 if flash attention unavailable.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1)
    return p.parse_args()


@torch.inference_mode()
def evaluate(model, loader, device: torch.device) -> SegmentationMeter:
    model.eval()
    meter = SegmentationMeter(num_classes=len(CLASS_NAMES))
    for batch in tqdm(loader, desc="val", leave=False):
        model_input = to_device_input(batch, device, include_segment=False)
        logits = get_logits(model(model_input))
        pred_voxel = logits.argmax(dim=1).detach().cpu()
        pred_full = pred_voxel[batch["inverse"]].numpy()
        target_full = batch["origin_segment"].numpy()
        meter.update(pred_full, target_full)
    return meter


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    train_ds = NuScenesLidarSegDataset(
        args.dataroot,
        version=args.version,
        split=args.train_split,
        voxel_size=args.voxel_size,
        max_scans=args.max_train_scans,
        verbose=True,
    )
    val_ds = NuScenesLidarSegDataset(
        args.dataroot,
        version=args.version,
        split=args.val_split,
        voxel_size=args.voxel_size,
        max_scans=args.max_val_scans,
        verbose=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_single_scan,
    )
    val_loader = DataLoader(
        val_ds,
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
        freeze_backbone=args.freeze_backbone,
    )
    model = build_pointcept_model(cfg, device)

    if args.checkpoint:
        ignore = ("seg_head", "cls_head", "classifier") if args.ignore_head else ()
        missing, unexpected, dropped = load_checkpoint_flexible(
            model, args.checkpoint, device=device, ignore_substrings=ignore
        )
        print(f"[train] Loaded checkpoint: {args.checkpoint}")
        print(f"[train] missing_keys={len(missing)} unexpected_keys={len(unexpected)} dropped_keys={len(dropped)}")
    elif args.freeze_backbone:
        print("[train] WARNING: --freeze-backbone without --checkpoint trains only a random head on a random backbone.")

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    best_miou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_steps = 0
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}")
        for batch in pbar:
            optim.zero_grad(set_to_none=True)
            # Pointcept DefaultSegmentorV2 computes its configured criteria when
            # training and a segment tensor is present.  The default nuScenes PTv3
            # config uses CE + Lovasz; that is fine for this baseline.
            model_input = to_device_input(batch, device, include_segment=True)
            out = model(model_input)
            if "loss" not in out:
                raise RuntimeError("Model did not return loss in training mode; check Pointcept DefaultSegmentorV2 config.")
            loss = out["loss"]
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optim.step()
            running_loss += float(loss.detach().cpu())
            num_steps += 1
            pbar.set_postfix(loss=running_loss / max(num_steps, 1))

        epoch_record = {"epoch": epoch, "train_loss": running_loss / max(num_steps, 1)}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            meter = evaluate(model, val_loader, device)
            result = meter.compute()
            epoch_record.update(meter.to_dict(result))
            print("\n" + meter.format_summary(result))
            if result.miou > best_miou:
                best_miou = result.miou
                ckpt_path = out_dir / "model_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optim.state_dict(),
                        "best_mIoU": best_miou,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"[train] saved new best checkpoint to {ckpt_path}")

        history.append(epoch_record)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, out_dir / "model_last.pth")
    print(f"[train] done. best mini-val mIoU={best_miou:.4f}. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
