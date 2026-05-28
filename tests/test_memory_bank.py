import torch
import pytest

from focus_fusion.models.temporal.memory_bank import MemoryBank


B, C, P, Dv = 2, 6, 1024, 384


def _bank(T=1, **kwargs) -> MemoryBank:
    return MemoryBank(T=T, num_cameras=C, d_v=Dv, **kwargs)


# ── Shape tests ────────────────────────────────────────────────────────────

def test_t1_shape():
    bank = _bank(T=1)
    patch_seq = torch.randn(B, 1, C, P, Dv)
    out = bank(patch_seq)
    assert out.shape == (B, 1 * C * P, Dv)


def test_t6_shape():
    bank = _bank(T=6)
    patch_seq = torch.randn(B, 6, C, P, Dv)
    out = bank(patch_seq)
    assert out.shape == (B, 6 * C * P, Dv)


# ── Value preservation ─────────────────────────────────────────────────────

def test_value_preservation():
    """Specific patch values must survive the reshape exactly."""
    bank = _bank(T=2)
    patch_seq = torch.zeros(B, 2, C, P, Dv)
    patch_seq[0, 0, 0, 0, 0] = 42.0   # batch 0, frame 0, cam 0, patch 0, dim 0
    patch_seq[0, 1, 2, 3, 5] = 99.0   # batch 0, frame 1, cam 2, patch 3, dim 5
    out = bank(patch_seq)

    # Frame 0, cam 0, patch 0 is at index 0 in the sequence dim
    assert out[0, 0, 0].item() == pytest.approx(42.0)
    # Frame 1, cam 2, patch 3 → index = 1*C*P + 2*P + 3
    idx = 1 * C * P + 2 * P + 3
    assert out[0, idx, 5].item() == pytest.approx(99.0)


# ── Positional embeddings ──────────────────────────────────────────────────

def test_temporal_pe_changes_output():
    bank_no_pe = _bank(T=2, use_temporal_pe=False)
    bank_pe = _bank(T=2, use_temporal_pe=True)
    patch_seq = torch.randn(B, 2, C, P, Dv)
    out_no_pe = bank_no_pe(patch_seq)
    out_pe = bank_pe(patch_seq)
    assert not torch.allclose(out_no_pe, out_pe), "Temporal PE should change the output"
    assert out_pe.shape == out_no_pe.shape


def test_camera_pe_changes_output():
    bank_no_pe = _bank(T=1, use_camera_pe=False)
    bank_pe = _bank(T=1, use_camera_pe=True)
    patch_seq = torch.randn(B, 1, C, P, Dv)
    out_no_pe = bank_no_pe(patch_seq)
    out_pe = bank_pe(patch_seq)
    assert not torch.allclose(out_no_pe, out_pe), "Camera PE should change the output"
    assert out_pe.shape == out_no_pe.shape


def test_combined_pe_shape():
    bank = _bank(T=3, use_temporal_pe=True, use_camera_pe=True)
    patch_seq = torch.randn(B, 3, C, P, Dv)
    out = bank(patch_seq)
    assert out.shape == (B, 3 * C * P, Dv)
    assert not out.isnan().any()
    assert not out.isinf().any()
