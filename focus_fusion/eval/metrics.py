"""Evaluation metrics for FocusFusion.

Computes per-class IoU and mean IoU (mIoU) over a DataLoader.

Called by:
    trainer._val_epoch  → evaluate_model(model, loader, device, num_classes, ignore_index)
    modal/modal_eval.py → python -m focus_fusion.eval.metrics  (CLI stub below)
"""
from __future__ import annotations

import argparse
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    num_classes: int = 32,
    ignore_index: int = -1,
) -> Dict[str, float]:
    """Run inference over *loader* and return mIoU metrics.

    Args:
        model: FocusFusion (or any module with forward(batch) → {"logits": (B,N,C)}).
        loader: DataLoader yielding batches with "labels" (B, N).
        device: torch device.
        num_classes: number of semantic classes.
        ignore_index: label value excluded from metric computation.

    Returns:
        dict with "mIoU" (float 0-1) and "iou_per_class" (list of floats).
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()

    intersection = torch.zeros(num_classes, dtype=torch.long)
    union = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(batch)
            logits = out["logits"]          # (B, N, C)
            labels = batch["labels"]        # (B, N)

            preds = logits.argmax(dim=-1)   # (B, N)

            for c in range(num_classes):
                pred_c = preds == c
                gt_c = labels == c
                valid = labels != ignore_index

                inter = (pred_c & gt_c & valid).sum().item()
                uni = ((pred_c | gt_c) & valid).sum().item()

                intersection[c] += inter
                union[c] += uni

    iou_per_class = []
    for c in range(num_classes):
        if union[c] == 0:
            iou_per_class.append(float("nan"))
        else:
            iou_per_class.append(float(intersection[c]) / float(union[c]))

    valid_ious = [x for x in iou_per_class if not (x != x)]  # filter NaN
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    return {"mIoU": miou, "iou_per_class": iou_per_class}


# ---------------------------------------------------------------------------
# CLI — python -m focus_fusion.eval.metrics  (called by modal_eval.py)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FocusFusion checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default="e1")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="mini_val")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import json
    import yaml
    from pathlib import Path

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from focus_fusion.models.focus_fusion import FocusFusion
    model = FocusFusion(config)
    model.to(args.device)

    checkpoint = args.checkpoint or str(
        Path(args.output_dir) / args.experiment / "best.pt"
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    print(f"[eval] Loaded checkpoint: {checkpoint}")

    from focus_fusion.datasets.nuscenes import NuScenesLidarSegDataset
    from torch.utils.data import DataLoader

    dc = config.get("data", {})
    ds = NuScenesLidarSegDataset(
        dataroot=args.data_root,
        split=args.split,
        num_points=config.get("model", {}).get("N_points", 16384),
        img_size=config.get("model", {}).get("img_size", 448),
        T=config.get("model", {}).get("T", 1),
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=4)

    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=args.device,
        num_classes=config.get("loss", {}).get("num_classes", 32),
        ignore_index=config.get("loss", {}).get("ignore_index", -1),
    )

    out_path = Path(args.output_dir) / args.experiment / f"eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[eval] mIoU = {metrics['mIoU']:.4f}")
    print(f"[eval] Results saved to {out_path}")


if __name__ == "__main__":
    main()
