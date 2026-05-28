import torch
import pytest

from focus_fusion.models.fusion.cross_attention import CrossAttentionFusion

B, N, S = 2, 64, 256
D_l, D_v, D_f, H = 512, 384, 256, 8


def _model(**kwargs) -> CrossAttentionFusion:
    return CrossAttentionFusion(d_l=D_l, d_v=D_v, d_f=D_f, num_heads=H, **kwargs)


def _inputs(device="cpu"):
    q = torch.randn(B, N, D_l, device=device)
    kv = torch.randn(B, S, D_v, device=device)
    return q, kv


# ── Shape tests ─────────────────────────────────────────────────────────────

def test_output_shape_no_attn_weights():
    model = _model()
    q, kv = _inputs()
    out, attn = model(q, kv)
    assert out.shape == (B, N, D_f)
    assert attn is None


def test_output_shape_with_attn_weights():
    model = _model(return_attn_weights=True)
    q, kv = _inputs()
    out, attn = model(q, kv)
    assert out.shape == (B, N, D_f)
    assert attn is not None
    assert attn.shape == (B, H, N, S)


# ── Attention correctness ───────────────────────────────────────────────────

def test_attn_weights_sum_to_one():
    model = _model(return_attn_weights=True).eval()  # dropout off in eval mode
    q, kv = _inputs()
    _, attn = model(q, kv)
    row_sums = attn.sum(dim=-1)  # (B, H, N)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


# ── Gradient flow ───────────────────────────────────────────────────────────

def test_gradient_flows_to_q_proj():
    model = _model()
    q = torch.randn(B, N, D_l, requires_grad=True)
    kv = torch.randn(B, S, D_v)
    out, _ = model(q, kv)
    out.sum().backward()
    assert q.grad is not None
    assert model.q_proj.weight.grad is not None


# ── Numerical stability ─────────────────────────────────────────────────────

def test_no_nan_inf_large_scale():
    """T=6 scale: S=36864 tokens (6 frames × 6 cams × 1024 patches)."""
    model = _model()
    q = torch.randn(1, 128, D_l)
    kv = torch.randn(1, 6 * 6 * 1024, D_v)
    out, _ = model(q, kv)
    assert not out.isnan().any()
    assert not out.isinf().any()


# ── Sensitivity test ────────────────────────────────────────────────────────

def test_different_kv_produce_different_output():
    model = _model()
    q, kv1 = _inputs()
    kv2 = torch.randn_like(kv1)
    out1, _ = model(q, kv1)
    out2, _ = model(q, kv2)
    assert not torch.allclose(out1, out2)
