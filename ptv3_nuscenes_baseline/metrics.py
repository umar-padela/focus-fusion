"""Semantic segmentation metrics for point-wise nuScenes-lidarseg evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from labels import CLASS_NAMES, IGNORE_INDEX


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

    def reset(self) -> None:
        self.confusion.fill(0)

    def update(self, pred, target) -> None:
        """Update confusion matrix with 0..C-1 predictions and labels.

        Args:
            pred: array-like [N] predicted internal ids 0..C-1.
            target: array-like [N] target internal ids 0..C-1 or ignore_index.
        """
        pred = np.asarray(pred, dtype=np.int64).reshape(-1)
        target = np.asarray(target, dtype=np.int64).reshape(-1)
        if pred.shape[0] != target.shape[0]:
            raise ValueError(f"pred and target length mismatch: {pred.shape} vs {target.shape}")
        valid = (
            (target != self.ignore_index)
            & (target >= 0)
            & (target < self.num_classes)
            & (pred >= 0)
            & (pred < self.num_classes)
        )
        if not np.any(valid):
            return
        encoded = self.num_classes * target[valid] + pred[valid]
        hist = np.bincount(encoded, minlength=self.num_classes ** 2)
        self.confusion += hist.reshape(self.num_classes, self.num_classes)

    def compute(self) -> MetricResult:
        hist = self.confusion.astype(np.float64)
        tp = np.diag(hist)
        gt = hist.sum(axis=1)
        pred = hist.sum(axis=0)
        union = gt + pred - tp

        iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
        acc = np.divide(tp, gt, out=np.full_like(tp, np.nan), where=gt > 0)
        miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else float("nan")
        macc = float(np.nanmean(acc)) if np.any(~np.isnan(acc)) else float("nan")
        total = gt.sum()
        all_acc = float(tp.sum() / total) if total > 0 else float("nan")
        freq = np.divide(gt, total, out=np.zeros_like(gt), where=total > 0)
        fw_iou = float(np.nansum(freq * iou))
        return MetricResult(
            miou=miou,
            macc=macc,
            fw_iou=fw_iou,
            all_acc=all_acc,
            per_class_iou=iou,
            per_class_acc=acc,
            support=gt.astype(np.int64),
            confusion=self.confusion.copy(),
        )

    def format_summary(self, result: MetricResult | None = None) -> str:
        if result is None:
            result = self.compute()
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
            iou_s = "nan" if np.isnan(iou) else f"{iou:.4f}"
            acc_s = "nan" if np.isnan(acc) else f"{acc:.4f}"
            lines.append(f"{name:<24} {iou_s:>8} {acc_s:>8} {int(support):>12}")
        return "\n".join(lines)

    def to_dict(self, result: MetricResult | None = None) -> Dict[str, object]:
        if result is None:
            result = self.compute()
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
