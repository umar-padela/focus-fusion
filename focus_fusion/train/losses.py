"""Per-point segmentation losses.

Primary: cross-entropy on point labels.
Optional: Lovász-Softmax (weighted addition), which directly optimises a
surrogate of the per-class IoU and tends to improve mIoU convergence speed.
Reference: Berman et al., "The Lovász-Softmax Loss" (CVPR 2018).

Interface expected by the trainer:
    loss, loss_dict = criterion(output, batch)
    output: {"logits": Tensor (B, N, C)}
    batch:  {"labels": Tensor (B, N)}   — integer class indices; -1 = ignore
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SegmentationLoss(nn.Module):
    """Cross-entropy + optional Lovász-Softmax for per-point segmentation.

    Args:
        ignore_index: label value to skip in loss computation.
            Use -1 for padded/invalid points (default).
            nuScenes lidarseg class 0 ("noise") may also want ignoring —
            set ignore_index=0 if the dataloader preserves raw class indices.
        lovasz_weight: weight for the Lovász-Softmax term.
            0.0 (default) = CE only. 1.0 = equal mix. Paper used CE + Lovász
            unweighted sum; start at 0 and enable once baseline is stable.
        class_weights: optional (C,) float tensor for class-frequency reweighting.
    """

    def __init__(
        self,
        ignore_index: int = -1,
        lovasz_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.lovasz_weight = lovasz_weight

    def forward(self, output: dict, batch: dict) -> tuple[Tensor, dict]:
        """
        Args:
            output: model output dict — must contain "logits" (B, N, C)
            batch:  batch dict — must contain "labels" (B, N)
        Returns:
            loss:      scalar total loss
            loss_dict: {"loss": float, "ce": float, "lovasz": float}
        """
        logits: Tensor = output["logits"]   # (B, N, C)
        labels: Tensor = batch["labels"]    # (B, N)

        B, N, C = logits.shape
        logits_flat = logits.reshape(B * N, C)   # (B*N, C)
        labels_flat = labels.reshape(B * N)       # (B*N,)

        ce = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
        )

        loss = ce
        lovasz_val = torch.zeros(1, device=logits.device)

        if self.lovasz_weight > 0.0:
            probs = logits_flat.softmax(dim=-1)   # (B*N, C)
            lovasz_val = _lovasz_softmax_flat(probs, labels_flat, self.ignore_index)
            loss = ce + self.lovasz_weight * lovasz_val

        loss_dict = {
            "loss": loss.item(),
            "ce": ce.item(),
            "lovasz": lovasz_val.item(),
        }
        return loss, loss_dict


# ---------------------------------------------------------------------------
# Lovász-Softmax implementation
# ---------------------------------------------------------------------------

def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
    """Lovász extension gradient for a sorted binary ground-truth vector."""
    p = gt_sorted.shape[0]
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_softmax_flat(
    probs: Tensor,
    labels: Tensor,
    ignore_index: int = -1,
) -> Tensor:
    """Lovász-Softmax loss on flattened (P, C) probabilities and (P,) labels.

    Averages the per-class Lovász hinge loss over all present classes.
    Classes absent from the batch contribute 0 (not averaged over).
    """
    if ignore_index >= 0:
        valid = labels != ignore_index
    else:
        valid = labels != ignore_index  # handles negative ignore_index too
    probs = probs[valid]
    labels = labels[valid]

    if probs.numel() == 0:
        return probs.sum() * 0.0

    C = probs.shape[1]
    losses = []
    for c in range(C):
        fg = (labels == c).float()                        # (P,) foreground mask
        if fg.numel() == 0:
            continue
        errors = (fg - probs[:, c]).abs()                 # (P,) per-point error
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))

    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()
