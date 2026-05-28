"""
Wires:
  ptv3 backbone (frozen) -> per-point Q features
  dinov2 backbone (frozen) -> patch embeddings pushed into MemoryBank
  MemoryBank  -> K/V tensor (T*P tokens)
  CrossAttentionFusion -> fused per-point features
  SegmentationHead -> per-point class logits

Both E1 (T=1) and E2 (T=6) use this same module; only memory_bank.T differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.components.memory_bank import MemoryBank
from models.components.cross_attention_fusion import CrossAttentionFusion
from models.components.segmentation_head import SegmentationHead


# ---------------------------------------------------------------------------
# Config dataclass (mirrors configs/default.yaml sections)
# ---------------------------------------------------------------------------

@dataclass
class FocusFusionConfig:
    # Backbone output dims — must match whatever ptv3 / dinov2 actually produce
    d_lidar: int = 256  # ptv3 per-point feature dim
    d_vision: int = 768 # DINOv2 ViT-B patch dim (1024 for ViT-L)

    # Fusion
    d_model: int = 256 # cross-attention hidden dim
    n_heads: int = 8
    attn_dropout: float = 0.1

    # Memory bank
    T: int = 1 # stack depth (1 -> E1, 6 -> E2)
    learnable_pos_emb: bool = False

    # Head
    num_classes: int = 32 # nuScenes lidarseg
    head_hidden_dim: int = 128
    head_dropout: float = 0.1

    # Misc
    return_attn_weights: bool = False # enable for eval / visualisation


# ---------------------------------------------------------------------------
# FocusFusion
# ---------------------------------------------------------------------------

class FocusFusion(nn.Module):
    """Multi-modal 3D semantic segmentation via cross-attention fusion.

    The model expects *backbone wrappers* to be passed in at construction time
    so that this file stays decoupled from third_party submodule paths.  Each
    wrapper must expose a simple callable interface:

        ptv3_backbone(batch)  -> Tensor (B, N, D_l)
        dinov2_backbone(imgs) -> Tensor (B, num_cams, P, D_v)
                                   OR (B, P_total, D_v) if already flattened

    Args:
        config: FocusFusionConfig instance.
        ptv3_backbone: Frozen ptv3 wrapper (nn.Module).
        dinov2_backbone: Frozen DINOv2 wrapper (nn.Module).
    """

    def __init__(
        self,
        config: FocusFusionConfig,
        ptv3_backbone: nn.Module,
        dinov2_backbone: nn.Module,
    ) -> None:
        super().__init__()

        self.config = config

        # Frozen backbones — owned externally, but stored as sub-modules so
        # .to(device) / .parameters() work correctly.
        self.ptv3 = ptv3_backbone
        self.dinov2 = dinov2_backbone

        # Learned components
        self.memory_bank = MemoryBank(
            T=config.T,
            d_vision=config.d_vision,
            learnable_pos_emb=config.learnable_pos_emb,
        )

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

        # Freeze backbones (guard — wrappers may already be frozen)
        self._freeze_backbones()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset_memory(self) -> None:
        """Clear the memory bank — call at the start of every new scene."""
        self.memory_bank.reset()

    def set_return_attn_weights(self, flag: bool) -> None:
        self.config.return_attn_weights = flag
        self.fusion.return_attn_weights = flag

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with keys:
                "points" — (B, N, 3+)  LiDAR xyz (+ optional features)
                "images" — (B, num_cams, C, H, W)  camera frames for *this* timestep
                            (the dataset/dataloader should have already pre-loaded the
                            correct window of frames if T>1; only *one* timestep of
                            images is passed per forward call — the bank accumulates
                            history across push() calls driven by the trainer).

        Returns:
            dict with keys:
                "logits" — (B, N, C)  per-point class logits
                "attn_weights" — (B, N, L) or None
        """
        images = batch["images"]  # (B, num_cams, C, H, W)

        # --- LiDAR branch ---
        with torch.no_grad():
            lidar_feats = self.ptv3(batch) # (B, N, D_l)

        # --- Vision branch ---
        with torch.no_grad():
            # Run DINOv2 per camera; handle both (B, num_cams, ...) and batched loops
            patch_embs = self._encode_images(images) # (B, num_cams, P, D_v)

        # Push this timestep into the bank; bank accumulates across calls
        self.memory_bank.push(patch_embs)

        # Get stacked K/V: (B, T_actual * P_total, D_v)
        vision_tokens = self.memory_bank.get_kv()

        # check - if bank is empty (should not happen after push, but for debugging for now)
        if vision_tokens.numel() == 0:
            raise RuntimeError("MemoryBank is empty after push — check push() call.")

        # --- Fusion ---
        fused, attn_weights = self.fusion(lidar_feats, vision_tokens)

        # --- Head ---
        logits = self.head(fused) # (B, N, C)

        return {"logits": logits, "attn_weights": attn_weights}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Run DINOv2 on each camera image and return patch embeddings.
        Args:
            images: (B, num_cams, C, H, W)

        Returns:
            patch_embs: (B, num_cams, P, D_v)
        """
        B, num_cams, C, H, W = images.shape
        # Flatten batch and cameras for a single forward pass
        imgs_flat = images.view(B * num_cams, C, H, W)
        patches_flat = self.dinov2(imgs_flat)          # (B*num_cams, P, D_v)
        P = patches_flat.shape[1]
        D = patches_flat.shape[2]
        return patches_flat.view(B, num_cams, P, D)

    def _freeze_backbones(self) -> None:
        for m in [self.ptv3, self.dinov2]:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def trainable_parameters(self):
        """Returns only the parameters that should be optimised."""
        return (
            list(self.memory_bank.parameters())
            + list(self.fusion.parameters())
            + list(self.head.parameters())
        )

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def extra_repr(self) -> str:
        return (
            f"T={self.config.T}, d_lidar={self.config.d_lidar}, "
            f"d_vision={self.config.d_vision}, d_model={self.config.d_model}, "
            f"num_classes={self.config.num_classes}, "
            f"trainable_params={self.num_trainable_params():,}"
        )