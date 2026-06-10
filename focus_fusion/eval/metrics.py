import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from focus_fusion.models.focus_fusion import FocusFusion
from focus_fusion.datasets.nuscenes import collate_focusfusion

from focus_fusion.datasets.nuscenes import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NuScenesLidarSegDataset,
    collate_single_scan,
    official_to_internal_ids,
)

@dataclass
class MetricResult:
    miou: float
    macc: float
    fw_iou: float
    all_acc: float
    per_class_iou: np.ndarray
    per_class_acc: np.ndarray
    support: np.ndarray
    confusion: np.ndarray


class SegmentationMeter:
    """Accumulates predictions and computes mIoU, mAcc, fwIoU via confusion matrix."""

    def __init__(
        self,
        num_classes: int = len(CLASS_NAMES),
        ignore_index: int = IGNORE_INDEX,
        class_names: Sequence[str] = CLASS_NAMES,
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.class_names = tuple(class_names)
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target) -> None:
        pred = np.asarray(pred, dtype=np.int64).reshape(-1)
        target = np.asarray(target, dtype=np.int64).reshape(-1)
        valid = (
            (target != self.ignore_index)
            & (target >= 0) & (target < self.num_classes)
            & (pred >= 0) & (pred < self.num_classes)
        )
        if np.any(valid):
            encoded = self.num_classes * target[valid] + pred[valid]
            hist = np.bincount(encoded, minlength=self.num_classes ** 2)
            self.confusion += hist.reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        self.confusion[:] = 0

    def compute(self) -> MetricResult:
        hist = self.confusion.astype(np.float64)
        tp = np.diag(hist)
        gt = hist.sum(axis=1)
        pred = hist.sum(axis=0)
        union = gt + pred - tp
        iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
        acc = np.divide(tp, gt, out=np.full_like(tp, np.nan), where=gt > 0)
        total = gt.sum()
        freq = np.divide(gt, total, out=np.zeros_like(gt), where=total > 0)
        return MetricResult(
            miou=float(np.nanmean(iou)),
            macc=float(np.nanmean(acc)),
            fw_iou=float(np.nansum(freq * iou)),
            all_acc=float(tp.sum() / total) if total > 0 else float("nan"),
            per_class_iou=iou,
            per_class_acc=acc,
            support=gt.astype(np.int64),
            confusion=self.confusion.copy(),
        )

    def format_summary(self, result: MetricResult | None = None) -> str:
        result = result or self.compute()
        lines = [
            f"mIoU={result.miou:.4f}  mAcc={result.macc:.4f}  "
            f"fwIoU={result.fw_iou:.4f}  allAcc={result.all_acc:.4f}",
            "",
            f"{'class':<24} {'IoU':>8} {'Acc':>8} {'support':>12}",
            "-" * 56,
        ]
        for name, iou, acc, support in zip(
            self.class_names, result.per_class_iou, result.per_class_acc, result.support
        ):
            lines.append(
                f"{name:<24} "
                f"{'nan' if np.isnan(iou) else f'{iou:.4f}':>8} "
                f"{'nan' if np.isnan(acc) else f'{acc:.4f}':>8} "
                f"{int(support):>12}"
            )
        return "\n".join(lines)

    def to_dict(self, result: MetricResult | None = None) -> Dict[str, object]:
        result = result or self.compute()
        return {
            "mIoU": result.miou,
            "mAcc": result.macc,
            "fwIoU": result.fw_iou,
            "allAcc": result.all_acc,
            "classes": list(self.class_names),
            "per_class_iou": result.per_class_iou.tolist(),
            "per_class_acc": result.per_class_acc.tolist(),
            "support": result.support.tolist(),
            "confusion": result.confusion.tolist(),
        }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    num_classes: int = len(CLASS_NAMES),
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, object]:
    """Run model inference over loader and return metrics dict.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()
    meter = SegmentationMeter(
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_names=list(CLASS_NAMES)[:num_classes],
    )

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(batch)
            preds = out["logits"].argmax(dim=-1).cpu().numpy()   # (B, N)
            labels = batch["labels"].cpu().numpy()               # (B, N)
            for b in range(preds.shape[0]):
                meter.update(preds[b], labels[b])

    return meter.to_dict()

def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str,
    dataroot: str,
    split: str,
    out_path: str,
) -> None:
    """Load a FocusFusion checkpoint and run live inference eval."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FocusFusion(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    mc = config.get("model", {})
    dataset = NuScenesLidarSegDataset(
        dataroot=dataroot,
        version="v1.0-mini",  
        split=split,
        num_points=mc.get("N_points", 16384),
        img_size=mc.get("img_size", 448),
        T=mc.get("T", 1),
        verbose=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_focusfusion,
    )

    lc = config.get("loss", {})
    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        num_classes=int(lc.get("num_classes", 16)),
        ignore_index=int(lc.get("ignore_index", -1)),
    )

    meter = SegmentationMeter(num_classes=int(lc.get("num_classes", 16)))
    result = MetricResult(
        miou=metrics["mIoU"], macc=metrics["mAcc"],
        fw_iou=metrics["fwIoU"], all_acc=metrics["allAcc"],
        per_class_iou=np.array(metrics["per_class_iou"]),
        per_class_acc=np.array(metrics["per_class_acc"]),
        support=np.array(metrics["support"]),
        confusion=np.array(metrics["confusion"]),
    )
    print(meter.format_summary(result))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {out_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved lidarseg predictions")
    parser.add_argument("--experiment", default="e0")
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="mini_val")
    parser.add_argument("--pred-dir", default="",
                        help="Folder with *_lidarseg.bin official-id predictions (LitePT output).")
    parser.add_argument("--out", default="",
                        help="Output metrics JSON path.")
    args = parser.parse_args()

    if not args.pred_dir:
        raise SystemExit("--pred-dir is missing.")

    out_path = args.out or f"experiments/logs/{args.experiment}/metrics.json"
    dataset = NuScenesLidarSegDataset(args.dataroot, version=args.version, split=args.split)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_single_scan)
    pred_dir = Path(args.pred_dir)
    meter = SegmentationMeter()

    for batch in tqdm(loader, desc="Evaluating"):
        pred_path = pred_dir / f"{batch['lidar_token']}_lidarseg.bin"
        official = np.fromfile(str(pred_path), dtype=np.uint8)
        pred = official_to_internal_ids(official)
        meter.update(pred, batch["origin_segment"].numpy())

    result = meter.compute()
    print(meter.format_summary(result))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meter.to_dict(result), indent=2))
    print(f"Metrics saved to {out}")


if __name__ == "__main__":
    main()
