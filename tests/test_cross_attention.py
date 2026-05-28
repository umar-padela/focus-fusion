import torch
import pytest

from focus_fusion.models.fusion.cross_attention import CrossAttentionFusion

B, N, S = 2, 64, 256
D_l, D_v, D_f, H = 512, 768, 256, 8


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
    model = _model(return_attn_weights=True)
    q, kv = _inputs()
    _, attn = model(q, kv)
    row_sums = attn.sum(dim=-1)  # (B, H, N)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
        f"Attention rows should sum to 1; max deviation: {(row_sums - 1).abs().max().item():.2e}"


# ── Gradient flow ───────────────────────────────────────────────────────────

def test_gradient_flows_through_projections():
    model = _model()
    q = torch.randn(B, N, D_l, requires_grad=True)
    kv = torch.randn(B, S, D_v)   # no grad on K/V (frozen backbone)
    out, _ = model(q, kv)
    out.sum().backward()
    assert q.grad is not None, "Gradient must reach q_feats"
    assert model.q_proj.weight.grad is not None, "q_proj.weight must have grad"
    assert model.k_proj.weight.grad is not None, "k_proj.weight must have grad (through K)"
    assert model.v_proj.weight.grad is not None, "v_proj.weight must have grad (through V)"


def test_frozen_backbone_params_have_no_grad():
    """Simulates frozen ptv3/dinov2: parameters themselves should have no grad."""
    model = _model()
    q = torch.randn(B, N, D_l)
    kv = torch.randn(B, S, D_v)
    # Fusion params should be trainable
    assert model.q_proj.weight.requires_grad
    # Backbone params (simulated): no requires_grad on backbone tensors
    q_detached = q.detach()
    kv_detached = kv.detach()
    out, _ = model(q_detached, kv_detached)
    # No backward error expected even with detached inputs
    out.sum().backward()


# ── Numerical stability ─────────────────────────────────────────────────────

def test_no_nan_inf_large_scale():
    """T=6 scale: S=36864 tokens (T=6, 6 cams, P=1024). Run on CPU in float32."""
    S_large = 6 * 6 * 1024
    model = _model()
    q = torch.randn(1, 128, D_l)
    kv = torch.randn(1, S_large, D_v)
    out, _ = model(q, kv)
    assert not out.isnan().any(), "NaN in output at T=6 scale"
    assert not out.isinf().any(), "Inf in output at T=6 scale"


# ── Sensitivity test ────────────────────────────────────────────────────────

def test_different_kv_produce_different_output():
    model = _model()
    q, kv1 = _inputs()
    kv2 = torch.randn_like(kv1)
    out1, _ = model(q, kv1)
    out2, _ = model(q, kv2)
    assert not torch.allclose(out1, out2), "Different K/V should produce different fused features"


# ── Constructor validation ──────────────────────────────────────────────────

def test_invalid_num_heads_raises():
    with pytest.raises(ValueError):
        CrossAttentionFusion(d_l=D_l, d_v=D_v, d_f=256, num_heads=7)  # 256 % 7 ≠ 0
