import numpy as np
import pytest
import torch

from focus_fusion.datasets.nuscenes import subsample_points
from focus_fusion.datasets.fake import FakeLidarSegDataset, build_fake_dataloader

N, C = 16384, 32


# ── subsample_points ────────────────────────────────────────────────────────

def _make_pc(n: int):
    pts = np.random.randn(n, 3).astype(np.float32)
    lbls = np.random.randint(0, C, n).astype(np.int64)
    return pts, lbls


def test_subsample_exact():
    pts, lbls = _make_pc(N)
    p, l = subsample_points(pts, lbls, N)
    assert p.shape == (N, 3)
    assert l.shape == (N,)


def test_subsample_above():
    pts, lbls = _make_pc(40000)
    p, l = subsample_points(pts, lbls, N)
    assert p.shape == (N, 3)
    assert l.shape == (N,)
    # All labels come from the valid set (no -1 introduced)
    assert (l >= 0).all()


def test_subsample_below_pads_with_ignore():
    pts, lbls = _make_pc(500)
    p, l = subsample_points(pts, lbls, N)
    assert p.shape == (N, 3)
    assert l.shape == (N,)
    # First 500 entries kept as-is; the rest padded with -1
    assert (l[:500] >= 0).all()
    assert (l[500:] == -1).all()


def test_subsample_returns_random_subset():
    pts, lbls = _make_pc(40000)
    p1, _ = subsample_points(pts, lbls, N)
    p2, _ = subsample_points(pts, lbls, N)
    # Two random subsamples should differ (fails with p≈1e-96)
    assert not np.array_equal(p1, p2)


# ── FakeLidarSegDataset — E1 ─────────────────────────────────────────────────

def test_fake_e1_item_shapes():
    ds = FakeLidarSegDataset(length=4, num_points=N, img_size=64, T=1)
    item = ds[0]
    assert item["points"].shape == (N, 3)
    assert item["images"].shape == (6, 3, 64, 64)
    assert item["labels"].shape == (N,)
    assert "images_seq" not in item


def test_fake_e1_label_range():
    ds = FakeLidarSegDataset(length=4, num_points=N, num_classes=C, T=1)
    item = ds[0]
    assert item["labels"].min() >= 0
    assert item["labels"].max() < C


def test_fake_e1_image_range():
    ds = FakeLidarSegDataset(length=4, img_size=64, T=1)
    item = ds[0]
    assert item["images"].min() >= 0.0
    assert item["images"].max() <= 1.0


# ── FakeLidarSegDataset — E2 ─────────────────────────────────────────────────

def test_fake_e2_item_shapes():
    ds = FakeLidarSegDataset(length=4, num_points=N, img_size=64, T=6)
    item = ds[0]
    assert item["points"].shape == (N, 3)
    assert item["images_seq"].shape == (6, 6, 3, 64, 64)
    assert item["labels"].shape == (N,)
    assert "images" not in item


# ── Batched DataLoader ───────────────────────────────────────────────────────

def test_fake_dataloader_e1_batch_shapes():
    loader = build_fake_dataloader(length=8, num_points=N, img_size=64, T=1, batch_size=2)
    batch = next(iter(loader))
    assert batch["points"].shape == (2, N, 3)
    assert batch["images"].shape == (2, 6, 3, 64, 64)
    assert batch["labels"].shape == (2, N)


def test_fake_dataloader_e2_batch_shapes():
    loader = build_fake_dataloader(length=8, num_points=N, img_size=64, T=6, batch_size=2)
    batch = next(iter(loader))
    assert batch["points"].shape == (2, N, 3)
    assert batch["images_seq"].shape == (2, 6, 6, 3, 64, 64)
    assert batch["labels"].shape == (2, N)


def test_fake_dataloader_dtypes():
    loader = build_fake_dataloader(length=4, num_points=N, img_size=64, T=1, batch_size=2)
    batch = next(iter(loader))
    assert batch["points"].dtype == torch.float32
    assert batch["images"].dtype == torch.float32
    assert batch["labels"].dtype == torch.int64
