import torch
import pytest
from unittest.mock import patch

from focus_fusion.models.backbones.dinov2 import DINOv2Backbone

IMG_SIZE = 56   # smallest valid: 56 / 14 = 4 → P = 16
D_V = 384       # ViT-S hidden dim
B, N_CAMS = 2, 6
P = (IMG_SIZE // 14) ** 2  # 16


class _FakeDINO(torch.nn.Module):
    """Minimal stand-in for the real DINOv2 model.

    Returns zero tensors of the correct shape so shape/dtype/freeze
    tests work without downloading weights.
    """
    def __init__(self, d_v: int = D_V) -> None:
        super().__init__()
        self.d_v = d_v
        self._dummy = torch.nn.Linear(1, 1)  # gives parameters() a real entry

    def get_intermediate_layers(self, x, n=1, return_class_token=False):
        B_flat = x.shape[0]
        h, w = x.shape[-2], x.shape[-1]
        P_local = (h // 14) * (w // 14)
        return [torch.zeros(B_flat, P_local, self.d_v, dtype=x.dtype, device=x.device)] * n


def _make_backbone(img_size: int = IMG_SIZE, **kwargs) -> DINOv2Backbone:
    with patch("torch.hub.load", return_value=_FakeDINO()):
        return DINOv2Backbone(img_size=img_size, **kwargs)


# ── Shape ────────────────────────────────────────────────────────────────────

def test_output_shape():
    backbone = _make_backbone()
    images = torch.rand(B, N_CAMS, 3, IMG_SIZE, IMG_SIZE)
    out = backbone(images)
    assert out.shape == (B, N_CAMS, P, D_V)


def test_num_patches_attribute():
    backbone = _make_backbone(img_size=112)
    assert backbone.num_patches == (112 // 14) ** 2  # 64


def test_output_dtype_is_float32():
    backbone = _make_backbone()
    images = torch.rand(B, N_CAMS, 3, IMG_SIZE, IMG_SIZE)
    out = backbone(images)
    assert out.dtype == torch.float32


# ── Normalization ─────────────────────────────────────────────────────────────

def test_normalize_buffers_registered():
    backbone = _make_backbone(normalize_input=True)
    assert hasattr(backbone, "mean") and hasattr(backbone, "std")
    assert backbone.mean.shape == (1, 3, 1, 1)
    assert backbone.std.shape == (1, 3, 1, 1)


def test_no_normalize_buffers_absent():
    backbone = _make_backbone(normalize_input=False)
    assert not hasattr(backbone, "mean")
    assert not hasattr(backbone, "std")


# ── Frozen parameters ─────────────────────────────────────────────────────────

def test_backbone_frozen_by_default():
    backbone = _make_backbone(freeze=True)
    for p in backbone.model.parameters():
        assert not p.requires_grad


def test_backbone_unfrozen_when_requested():
    backbone = _make_backbone(freeze=False)
    assert any(p.requires_grad for p in backbone.model.parameters())


# ── Input validation ──────────────────────────────────────────────────────────

def test_invalid_img_size_raises():
    with pytest.raises(AssertionError, match="divisible by 14"):
        with patch("torch.hub.load", return_value=_FakeDINO()):
            DINOv2Backbone(img_size=100)  # 100 % 14 != 0


# ── Numerical ────────────────────────────────────────────────────────────────

def test_no_nan_inf():
    backbone = _make_backbone()
    images = torch.rand(B, N_CAMS, 3, IMG_SIZE, IMG_SIZE)
    out = backbone(images)
    assert not out.isnan().any()
    assert not out.isinf().any()


# ── Integration (slow; downloads/uses real weights) ───────────────────────────

@pytest.mark.slow
def test_real_dinov2_shape():
    backbone = DINOv2Backbone(img_size=IMG_SIZE)  # downloads ~100 MB on first run
    images = torch.rand(1, N_CAMS, 3, IMG_SIZE, IMG_SIZE)
    out = backbone(images)
    assert out.shape == (1, N_CAMS, P, D_V)


@pytest.mark.slow
def test_real_dinov2_no_nan_inf():
    backbone = DINOv2Backbone(img_size=IMG_SIZE)
    images = torch.rand(1, N_CAMS, 3, IMG_SIZE, IMG_SIZE)
    out = backbone(images)
    assert not out.isnan().any()
    assert not out.isinf().any()
