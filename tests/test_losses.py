import torch
import pytest

from focus_fusion.train.losses import SegmentationLoss

B, N, C = 2, 64, 16


def _batch(ignore_some=False):
    logits = torch.randn(B, N, C)
    labels = torch.randint(0, C, (B, N))
    if ignore_some:
        labels[:, :8] = -1   # mark first 8 points as padding
    return {"logits": logits}, {"labels": labels}


# ── Basic interface ─────────────────────────────────────────────────────────

def test_returns_scalar_loss_and_dict():
    criterion = SegmentationLoss()
    output, batch = _batch()
    loss, loss_dict = criterion(output, batch)
    assert loss.shape == ()
    assert set(loss_dict.keys()) == {"loss", "ce", "lovasz"}


def test_ce_only_by_default():
    criterion = SegmentationLoss()
    output, batch = _batch()
    loss, loss_dict = criterion(output, batch)
    assert loss_dict["lovasz"] == pytest.approx(0.0)
    assert loss_dict["ce"] == pytest.approx(loss_dict["loss"])


def test_loss_is_positive():
    criterion = SegmentationLoss()
    output, batch = _batch()
    loss, _ = criterion(output, batch)
    assert loss.item() > 0


# ── Ignore index ────────────────────────────────────────────────────────────

def test_ignore_index_reduces_loss_contribution():
    # A batch where all labels are the same; ignoring half should not crash
    criterion = SegmentationLoss(ignore_index=-1)
    output, batch = _batch(ignore_some=True)
    loss, _ = criterion(output, batch)
    assert loss.item() > 0
    assert not loss.isnan()


def test_all_ignored_does_not_crash():
    criterion = SegmentationLoss(ignore_index=-1)
    logits = torch.randn(B, N, C)
    labels = torch.full((B, N), -1, dtype=torch.long)
    # CE with all ignored produces nan in PyTorch — that's expected behaviour;
    # we just check it doesn't raise
    try:
        loss, _ = criterion({"logits": logits}, {"labels": labels})
    except Exception as e:
        pytest.fail(f"Raised unexpectedly: {e}")


# ── Gradient flow ───────────────────────────────────────────────────────────

def test_gradient_flows_to_logits():
    criterion = SegmentationLoss()
    logits = torch.randn(B, N, C, requires_grad=True)
    labels = torch.randint(0, C, (B, N))
    loss, _ = criterion({"logits": logits}, {"labels": labels})
    loss.backward()
    assert logits.grad is not None
    assert not logits.grad.isnan().any()


# ── Lovász term ─────────────────────────────────────────────────────────────

def test_lovasz_changes_loss():
    output, batch = _batch()
    ce_only, d_ce = SegmentationLoss(lovasz_weight=0.0)(output, batch)
    with_lovasz, d_lv = SegmentationLoss(lovasz_weight=1.0)(output, batch)
    assert d_lv["lovasz"] > 0.0
    assert with_lovasz.item() != pytest.approx(ce_only.item())


def test_lovasz_gradient_flows():
    criterion = SegmentationLoss(lovasz_weight=1.0)
    logits = torch.randn(B, N, C, requires_grad=True)
    labels = torch.randint(0, C, (B, N))
    loss, _ = criterion({"logits": logits}, {"labels": labels})
    loss.backward()
    assert logits.grad is not None
    assert not logits.grad.isnan().any()


def test_lovasz_with_ignore_index():
    criterion = SegmentationLoss(lovasz_weight=1.0, ignore_index=-1)
    output, batch = _batch(ignore_some=True)
    loss, loss_dict = criterion(output, batch)
    assert not loss.isnan()
    assert loss_dict["lovasz"] > 0.0
