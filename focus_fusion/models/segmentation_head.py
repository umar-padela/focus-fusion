"""
Input:  fused per-point features (B, N, D_f)
Output: class logits (B, N, C)
"""

from __future__ import annotations

import torch.nn as nn


class SegmentationHead(nn.Module):
    """Two-layer MLP per-point segmentation head.

    Args:
        d_in: Input feature dim, must match CrossAttentionFusion output (d_f).
        num_classes: Number of semantic classes ]
        hidden_dim: Hidden layer width. None means single layer
        dropout: Dropout between hidden and output layers.
    """

    def __init__(
        self,
        d_in: int = 256,
        num_classes: int = 32,
        hidden_dim: int | None = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim is not None:
            self.layers = nn.Sequential(
                nn.Linear(d_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.layers = nn.Linear(d_in, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, N, D_f)
        Returns:
            logits: (B, N, C)
        """
        return self.layers(x)
