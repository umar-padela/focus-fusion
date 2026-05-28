"""
Wires:
  ptv3 backbone (frozen)   →  per-point Q features    (B, N, D_l)
  dinov2 backbone (frozen)  →  patch embeddings        (B, T*6*P, D_v)
  CrossAttentionFusion      →  fused per-point features (B, N, D_model)
  SegmentationHead          →  per-point class logits   (B, N, C)

Images arrive from the dataloader as (B, T, 6, 3, H, W). DINOv2 is run
once over all T*6 images; the output is flattened into K/V tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.components.cross_attention_fusion import CrossAttentionFusion
from models.components.segmentation_head import SegmentationHead


@dataclass
class FocusFusionConfig:
    d_lidar: int = 256        # ptv3 per-point feature dim
    d_vision: int = 768       # DINOv2 ViT-B patch dim (1024 for ViT-L)
    d_model: int = 256        # cross-attention hidden dim
    n_heads: int = 8
    attn_dropout: float = 0.1
    num_classes: int = 32     # nuScenes lidarseg
    head_hidden_dim: int = 128
    head_dropout: float = 0.1
    return_attn_weights: bool = False  # enable for eval / visualisation


class FocusFusion(nn.Module):
    """Multi-modal 3D semantic segmentation via cross-attention fusion.

    Backbone wrappers must expose:
        ptv3_backbone(batch)  -> (B, N, D_l)
        dinov2_backbone(imgs) -> (B*, P, D_v)  where imgs is (B*, 3, H, W)
    """

    def __init__(
        self,
        config: FocusFusionConfig,
        ptv3_backbone: nn.Module,
        dinov2_backbone: nn.Module,
    ) -> None:
        super().__init__()
        self.config = config
        self.ptv3 = ptv3_backbone
        self.dinov2 = dinov2_backbone

        self.fusion = CrossAttentionFusion(
            d_lidar=config.d_lidar,
            d_vision=config.d_vision,
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.attn_dropout,
            return_attn_weights=config.return_attn_weights,
        )
        self.head = SegmentationHead(
            d_in=config.d_model,
            num_classes=config.num_classes,
            hidden_dim=config.head_hidden_dim,
            dropout=config.head_dropout,
        )
        self._freeze_backbones()

    def set_return_attn_weights(self, flag: bool) -> None:
        self.config.return_attn_weights = flag
        self.fusion.return_attn_weights = flag

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch:
                "points" — fed directly to ptv3
                "images" — (B, T, 6, 3, H, W)

        Returns:
                "logits" — (B, N, C)
                "attn_weights" — (B, N, L) or None
        """
        images = batch["images"]
        B, T, num_cams, C, H, W = images.shape

        with torch.no_grad():
            lidar_feats = self.ptv3(batch)                           # (B, N, D_l)

            imgs_flat = images.view(B * T * num_cams, C, H, W)
            patches_flat = self.dinov2(imgs_flat)                    # (B*T*6, P, D_v)
            P, D_v = patches_flat.shape[1], patches_flat.shape[2]
            vision_tokens = patches_flat.view(B, T * num_cams * P, D_v)  # (B, T*6*P, D_v)

        fused, attn_weights = self.fusion(lidar_feats, vision_tokens)
        logits = self.head(fused)

        return {"logits": logits, "attn_weights": attn_weights}

    def _freeze_backbones(self) -> None:
        for m in [self.ptv3, self.dinov2]:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def trainable_parameters(self):
        return list(self.fusion.parameters()) + list(self.head.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())