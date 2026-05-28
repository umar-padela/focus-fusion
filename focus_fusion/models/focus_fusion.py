"""FocusFusion — end-to-end model.

Data flow:
  batch["points"]      → ptv3 (frozen) → Q (B, N, D_l)
  batch["images"]      → dinov2 (frozen) → patches (B, 1, 6, P, D_v)   [E1, T=1]
  batch["images_seq"]  → dinov2 (frozen) → patches (B, T, 6, P, D_v)   [E2, T=6]
  patches → MemoryBank.forward_preloaded → K/V (B, T*6*P, D_v)
  Q, K/V → CrossAttentionFusion → fused (B, N, D_f)
  fused → SegmentationHead → logits (B, N, C)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from focus_fusion.models.fusion.cross_attention import CrossAttentionFusion
from focus_fusion.models.temporal.memory_bank import MemoryBank
from focus_fusion.models.segmentation_head import SegmentationHead

if TYPE_CHECKING:
    from focus_fusion.models.backbones.dinov2 import DINOv2Backbone


class FocusFusion(nn.Module):
    """Full FocusFusion model.

    Constructor accepts a plain dict or any object with attribute access (e.g. OmegaConf).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()

        m = cfg.get("model", cfg) if isinstance(cfg, dict) else cfg.model
        ckpt = cfg.get("checkpoints", {}) if isinstance(cfg, dict) else getattr(cfg, "checkpoints", {})

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        self.d_l = _get(m, "d_l", 512)
        self.d_v = _get(m, "d_v", 384)
        self.d_f = _get(m, "d_f", 256)
        self.T = _get(m, "T", 1)
        self.img_size = _get(m, "img_size", 448)
        self.num_classes = _get(m, "num_classes", 32)
        self.eval_mode = _get(m, "eval_mode", False)

        # Vision backbone (lazy import to avoid hard dependency when submodule absent)
        from focus_fusion.models.backbones.dinov2 import DINOv2Backbone
        self.dinov2 = DINOv2Backbone(
            img_size=self.img_size,
            normalize_input=_get(m, "normalize_input", True),
        )

        # ptv3 backbone loaded separately (Person 1's module)
        # Set via set_ptv3() after construction, or loaded inside __init__ if path given.
        self.ptv3: nn.Module | None = None
        ptv3_path = _get(ckpt, "ptv3", None)
        if ptv3_path:
            self._load_ptv3(ptv3_path)

        # Memory bank
        self.memory_bank = MemoryBank(
            T=self.T,
            d_v=self.d_v,
            use_temporal_pe=_get(m, "use_temporal_pe", False),
            use_camera_pe=_get(m, "use_camera_pe", False),
        )

        # Cross-attention fusion
        self.cross_attn = CrossAttentionFusion(
            d_l=self.d_l,
            d_v=self.d_v,
            d_f=self.d_f,
            num_heads=_get(m, "num_heads", 8),
            dropout=_get(m, "dropout", 0.1),
            return_attn_weights=self.eval_mode,
        )

        # Segmentation head
        self.seg_head = SegmentationHead(
            d_in=self.d_f,
            num_classes=self.num_classes,
            hidden_dim=_get(m, "head_hidden_dim", 128),
            dropout=_get(m, "dropout", 0.1),
        )

    # ------------------------------------------------------------------
    # ptv3 integration
    # ------------------------------------------------------------------

    def set_ptv3(self, ptv3_module: nn.Module) -> None:
        """Attach Person 1's ptv3 wrapper after construction."""
        ptv3_module.requires_grad_(False)
        self.ptv3 = ptv3_module

    def _load_ptv3(self, path: str) -> None:
        from focus_fusion.models.backbones.ptv3 import PTV3Backbone  # type: ignore[import]
        self.ptv3 = PTV3Backbone(checkpoint_path=path)
        self.ptv3.requires_grad_(False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: dict with at minimum:
                "points"     : (B, N, 3)
                "images"     : (B, 6, 3, H, W)          — E1
                "images_seq" : (B, T, 6, 3, H, W)        — E2 (optional)
        Returns:
            dict with:
                "logits"      : (B, N, C)
                "attn_weights": (B, H, N, S) or None
        """
        points: Tensor = batch["points"]         # (B, N, 3)
        B, N, _ = points.shape

        # 1. LiDAR features (frozen ptv3 or stub)
        with torch.no_grad():
            if self.ptv3 is not None:
                q_feats: Tensor = self.ptv3(points)   # (B, N, D_l)
            else:
                # Stub for smoke tests without ptv3 installed
                q_feats = torch.zeros(B, N, self.d_l, device=points.device, dtype=points.dtype)

        assert q_feats.shape[-1] == self.d_l, (
            f"ptv3 output dim {q_feats.shape[-1]} ≠ cfg.model.d_l={self.d_l}. "
            "Update configs/default.yaml to match Person 1's ptv3 wrapper."
        )

        # 2. Vision features (frozen dinov2)
        use_temporal = "images_seq" in batch and self.T > 1
        with torch.no_grad():
            if use_temporal:
                imgs_seq: Tensor = batch["images_seq"]  # (B, T, 6, 3, H, W)
                T_actual = imgs_seq.shape[1]
                C = imgs_seq.shape[2]
                imgs_flat = imgs_seq.reshape(B * T_actual, C, *imgs_seq.shape[3:])
                patches_flat = self.dinov2(imgs_flat)   # (B*T, 6, P, D_v)
                P = patches_flat.shape[2]
                patches = patches_flat.view(B, T_actual, C, P, self.d_v)
            else:
                images: Tensor = batch["images"]        # (B, 6, 3, H, W)
                patches_cur = self.dinov2(images)       # (B, 6, P, D_v)
                patches = patches_cur.unsqueeze(1)      # (B, 1, 6, P, D_v)

        # 3. Memory bank → K/V tokens
        kv_tokens = self.memory_bank(patches)  # (B, T*6*P, D_v)

        # 4. Cross-attention fusion
        fused, attn_w = self.cross_attn(q_feats, kv_tokens)     # (B, N, D_f)

        # 5. Segmentation head
        logits = self.seg_head(fused)                            # (B, N, C)

        return {"logits": logits, "attn_weights": attn_w}

    # ------------------------------------------------------------------
    # Trainer helpers
    # ------------------------------------------------------------------

    def reset_memory(self) -> None:
        """No-op for preloaded-window training (MemoryBank is stateless).

        Called by the trainer at scene boundaries. In our design the dataloader
        assembles the T-frame window upfront so there is no state to clear.
        """

    def trainable_parameters(self) -> list:
        """Parameters that should be optimised (fusion + seg head, not frozen backbones)."""
        return list(self.cross_attn.parameters()) + list(self.seg_head.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # ------------------------------------------------------------------
    # Attention export utility (Week 2)
    # ------------------------------------------------------------------

    @staticmethod
    def save_attn_sample(
        attn_weights: Tensor,
        batch: dict,
        save_path: str,
    ) -> None:
        """Save one attention sample for visualisation.

        Args:
            attn_weights: (B, H, N, S) — from forward() with eval_mode=True
            batch: the batch dict (for sample_token)
            save_path: where to write the .pt file
        """
        payload = {"attn": attn_weights[0].cpu()}
        if "sample_token" in batch:
            payload["token"] = batch["sample_token"][0]
        torch.save(payload, save_path)


# ---------------------------------------------------------------------------
# Smoke test — run as: python -m focus_fusion.models.focus_fusion
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    B, N, C_cam, H, W = 2, 64, 6, 448, 448
    num_classes = 32

    cfg = {
        "model": {
            "d_l": 512,
            "d_v": 384,
            "d_f": 256,
            "num_heads": 8,
            "num_classes": num_classes,
            "T": 1,
            "img_size": H,
            "eval_mode": False,
            "dropout": 0.1,
            "use_temporal_pe": False,
            "use_camera_pe": False,
            "normalize_input": True,
        }
    }

    print("Building FocusFusion (no ptv3, no real dinov2)...")

    # Patch dinov2 with a stub for the smoke test
    import unittest.mock as mock

    def _fake_dinov2_forward(self_inner, images):
        bC, _, _, _ = images.shape
        P = (cfg["model"]["img_size"] // 14) ** 2
        return torch.randn(bC, 6, P, cfg["model"]["d_v"])

    from focus_fusion.models import focus_fusion as ff_mod
    with mock.patch.object(
        ff_mod.DINOv2Backbone,
        "forward",
        _fake_dinov2_forward,
    ):
        model = FocusFusion(cfg)
        model.eval()

        fake_batch = {
            "points": torch.randn(B, N, 3),
            "images": torch.randn(B, 6, 3, H, W),
        }

        out = model(fake_batch)
        assert out["logits"].shape == (B, N, num_classes), f"Bad logits shape: {out['logits'].shape}"
        assert out["attn_weights"] is None
        print(f"E1 smoke test passed: logits {tuple(out['logits'].shape)}")

        # E2 smoke test
        cfg["model"]["T"] = 6
        model_e2 = FocusFusion(cfg)
        model_e2.eval()
        fake_batch_e2 = {
            "points": torch.randn(B, N, 3),
            "images_seq": torch.randn(B, 6, 6, 3, H, W),
        }
        out_e2 = model_e2(fake_batch_e2)
        assert out_e2["logits"].shape == (B, N, num_classes), f"Bad E2 logits shape: {out_e2['logits'].shape}"
        print(f"E2 smoke test passed: logits {tuple(out_e2['logits'].shape)}")

    print("All smoke tests passed.")
