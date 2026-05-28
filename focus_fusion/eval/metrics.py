"""Semantic-segmentation metrics and saved-prediction evaluation CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

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
            & (target >= 0)
            & (target < self.num_classes)
            & (pred >= 0)
            & (pred < self.num_classes)
        )
        if np.any(valid):
            encoded = self.num_classes * target[valid] + pred[valid]
            hist = np.bincount(encoded, minlength=self.num_classes**2)
            self.confusion += hist.reshape(self.num_classes, self.num_classes)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="e0", choices=["e0", "e1", "e2"])
    parser.add_argument("--checkpoint", default="", help="Reserved for trained FocusFusion checkpoints.")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="mini_val")
    parser.add_argument("--pred-dir", default="", help="Folder with *_lidarseg.bin official-id predictions.")
    parser.add_argument("--out", default="", help="Metrics JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pred_dir:
        raise SystemExit(
            "--pred-dir is required for saved-prediction evaluation. "
            "For LitePT E0, generate predictions with the Modal/Pointcept eval first."
        )
    out_path = args.out or f"experiments/logs/{args.experiment}/metrics.json"
    dataset = NuScenesLidarSegDataset(args.dataroot, version=args.version, split=args.split)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_single_scan)
    pred_dir = Path(args.pred_dir)
    meter = SegmentationMeter()
    for batch in tqdm(loader, desc="Evaluating saved predictions"):
        pred_path = pred_dir / f"{batch['lidar_token']}_lidarseg.bin"
        official = np.fromfile(pred_path, dtype=np.uint8)
        pred = official_to_internal_ids(official)
        meter.update(pred, batch["origin_segment"].numpy())
    result = meter.compute()
    print(meter.format_summary(result))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meter.to_dict(result), indent=2))


if __name__ == "__main__":
    main()
