"""Synthetic dataset for testing and smoke runs without real nuScenes data.

FakeLidarSegDataset returns random tensors with the exact shapes the real
NuScenesLidarSegDataset would produce. Use it in:
  - Unit tests (no data download required)
  - Modal smoke runs before the data volume is populated
  - Integration tests for the training loop
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from focus_fusion.datasets.nuscenes import _voxelize, collate_focusfusion


class FakeLidarSegDataset(Dataset):
    """Drop-in replacement for NuScenesLidarSegDataset using random tensors.

    Args:
        length:      number of synthetic samples
        num_points:  LiDAR points per sample (default: 16384)
        num_classes: number of segmentation classes (default: 32)
        img_size:    camera image spatial size (default: 448)
        T:           temporal window depth (default: 1)
        num_cameras: number of cameras (default: 6)
    """

    def __init__(
        self,
        length: int = 64,
        num_points: int = 16384,
        num_classes: int = 32,
        img_size: int = 448,
        T: int = 1,
        num_cameras: int = 6,
    ) -> None:
        self.length = length
        self.num_points = num_points
        self.num_classes = num_classes
        self.img_size = img_size
        self.T = T
        self.num_cameras = num_cameras

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        xyz = torch.randn(self.num_points, 3)
        labels = torch.randint(0, self.num_classes, (self.num_points,))

        # Voxelise to match NuScenesLidarSegDataset's batch contract
        grid_idx, inverse, vox_xyz = _voxelize(xyz.numpy().astype(np.float32))
        n_vox = len(vox_xyz)
        vox_feat = np.concatenate(
            [vox_xyz, np.zeros((n_vox, 1), np.float32)], axis=1
        )
        vox_keys = {
            "vox_coord":      torch.from_numpy(vox_xyz),
            "vox_feat":       torch.from_numpy(vox_feat),
            "vox_grid_coord": torch.from_numpy(grid_idx),
            "inverse":        torch.from_numpy(inverse),
        }

        base = {"points": xyz, "labels": labels, "sample_token": f"fake_{idx:05d}", **vox_keys}
        if self.T == 1:
            base["images"] = torch.rand(self.num_cameras, 3, self.img_size, self.img_size)
        else:
            base["images_seq"] = torch.rand(
                self.T, self.num_cameras, 3, self.img_size, self.img_size
            )
        return base


def build_fake_dataloader(
    split: str = "mini_train",
    *,
    length: int = 16,
    num_points: int = 16384,
    num_classes: int = 32,
    img_size: int = 64,   # small default so tests are fast
    T: int = 1,
    batch_size: int = 2,
) -> DataLoader:
    """Build a DataLoader backed by FakeLidarSegDataset."""
    dataset = FakeLidarSegDataset(
        length=length,
        num_points=num_points,
        num_classes=num_classes,
        img_size=img_size,
        T=T,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "mini_train"),
        num_workers=0,
        collate_fn=collate_focusfusion,
    )
