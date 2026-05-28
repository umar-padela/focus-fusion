"""PyTorch Dataset/DataLoader for nuScenes-lidarseg mini/train/val splits.

This loader reads raw nuScenes LIDAR_TOP keyframes and their lidarseg labels,
remaps the raw 32 lidarseg ids to the official 16 challenge classes, and returns
Pointcept/PTv3-style tensors:

    feat:       [M, 4] = xyz + lidar strength scaled to [0, 1]
    coord:      [M, 3] original xyz coordinates
    grid_coord: [M, 3] integer voxel coordinates
    offset:     [1] cumulative point count for a batch of one scan

It also returns ``inverse`` so voxel-level predictions can be mapped back to all
original points for full-resolution metric computation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from labels import IGNORE_INDEX, build_learning_map, remap_raw_labels


@dataclass(frozen=True)
class LidarSegSample:
    sample_token: str
    lidar_token: str
    scene_name: str
    lidar_path: str
    label_path: str
    timestamp: int


def _load_lidar_bin(path: str) -> np.ndarray:
    """Load a nuScenes LIDAR_TOP .bin file as [N, 5] float32."""
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % 5 != 0:
        raise ValueError(f"Expected lidar file size to be divisible by 5 floats: {path}")
    return arr.reshape(-1, 5)


def _voxelize_first_point(coords: np.ndarray, voxel_size: float):
    """Voxelize points and keep the first point encountered in each voxel.

    Returns:
        grid_all: [N, 3] int64 grid coords for every original point.
        unique_indices: [M] indices into the original point array.
        inverse: [N] maps each original point to its voxel representative row.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be [N, 3], got {coords.shape}")
    min_coord = coords.min(axis=0, keepdims=True)
    grid_all = np.floor((coords - min_coord) / voxel_size).astype(np.int64)
    # np.unique returns unique rows in lexicographic order. unique_indices and
    # inverse both refer to that order, so pred_voxel[inverse] is valid.
    _, unique_indices, inverse = np.unique(
        grid_all, axis=0, return_index=True, return_inverse=True
    )
    return grid_all, unique_indices.astype(np.int64), inverse.astype(np.int64)


class NuScenesLidarSegDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        split: str = "mini_val",
        voxel_size: float = 0.05,
        ignore_index: int = IGNORE_INDEX,
        verbose: bool = False,
        max_scans: Optional[int] = None,
    ) -> None:
        """Create a raw nuScenes-lidarseg dataset.

        Args:
            dataroot: nuScenes root containing samples/, sweeps/, lidarseg/, and
                v1.0-mini/ or v1.0-trainval/.
            version: nuScenes metadata version. Use v1.0-mini for the mini set.
            split: One of mini_train, mini_val, train, val, or test. For this
                baseline you typically want mini_train/mini_val.
            voxel_size: grid size in meters. 0.05 matches the Pointcept PTv3
                nuScenes config for val/eval.
            ignore_index: Label for void/ignored classes after remapping.
            verbose: Print devkit loading messages and sample count.
            max_scans: Optional cap for quick smoke tests.
        """
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes

        self.dataroot = os.path.abspath(dataroot)
        self.version = version
        self.split = split
        self.voxel_size = float(voxel_size)
        self.ignore_index = int(ignore_index)

        self.nusc = NuScenes(version=version, dataroot=self.dataroot, verbose=verbose)
        self.learning_map = build_learning_map(self.nusc, ignore_index=self.ignore_index)

        split_to_scenes = create_splits_scenes()
        if split not in split_to_scenes:
            raise ValueError(
                f"Unknown split '{split}'. Expected one of: {sorted(split_to_scenes)}"
            )
        allowed_scenes = set(split_to_scenes[split])
        scene_token_to_name = {scene["token"]: scene["name"] for scene in self.nusc.scene}

        samples: List[LidarSegSample] = []
        for sample in self.nusc.sample:
            scene_name = scene_token_to_name[sample["scene_token"]]
            if scene_name not in allowed_scenes:
                continue
            lidar_token = sample["data"]["LIDAR_TOP"]
            sd_record = self.nusc.get("sample_data", lidar_token)
            if not sd_record.get("is_key_frame", False):
                continue
            try:
                lidarseg_record = self.nusc.get("lidarseg", lidar_token)
            except Exception:
                # Test set labels are not public.  This baseline evaluates train/val.
                continue
            samples.append(
                LidarSegSample(
                    sample_token=sample["token"],
                    lidar_token=lidar_token,
                    scene_name=scene_name,
                    lidar_path=os.path.join(self.dataroot, sd_record["filename"]),
                    label_path=os.path.join(self.dataroot, lidarseg_record["filename"]),
                    timestamp=int(sample.get("timestamp", 0)),
                )
            )

        samples.sort(key=lambda x: (x.scene_name, x.timestamp, x.lidar_token))
        if max_scans is not None:
            samples = samples[: int(max_scans)]
        if not samples:
            raise RuntimeError(
                f"No labeled LIDAR_TOP keyframes found for split={split}, version={version}, "
                f"dataroot={self.dataroot}. Check that the nuScenes mini data and "
                "lidarseg expansion are both extracted into the same root."
            )
        self.samples = samples
        if verbose:
            print(f"[NuScenesLidarSegDataset] Loaded {len(samples)} scans for {split}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        points = _load_lidar_bin(item.lidar_path)
        raw_labels = np.fromfile(item.label_path, dtype=np.uint8)
        if raw_labels.shape[0] != points.shape[0]:
            raise ValueError(
                f"Point/label count mismatch for {item.lidar_token}: "
                f"{points.shape[0]} points vs {raw_labels.shape[0]} labels"
            )

        coords = points[:, :3].astype(np.float32, copy=False)
        # Match Pointcept's NuScenesDataset preprocessing: raw lidar intensity
        # is used as "strength" after scaling to [0, 1].
        strength = points[:, 3:4].astype(np.float32, copy=False) / 255.0
        labels = remap_raw_labels(raw_labels, self.learning_map, ignore_index=self.ignore_index)

        grid_all, unique_indices, inverse = _voxelize_first_point(coords, self.voxel_size)
        coord = coords[unique_indices]
        grid_coord = grid_all[unique_indices]
        feat = np.concatenate([coord, strength[unique_indices]], axis=1).astype(np.float32)
        segment = labels[unique_indices]

        # PTv3/Pointcept expects offset to be cumulative counts.  This collate
        # supports batch size 1; batching multiple scans is possible but not needed
        # for the mini milestone baseline.
        num_vox = int(coord.shape[0])
        return {
            "coord": torch.from_numpy(coord).float(),
            "grid_coord": torch.from_numpy(grid_coord).int(),
            "feat": torch.from_numpy(feat).float(),
            "offset": torch.tensor([num_vox], dtype=torch.long),
            "segment": torch.from_numpy(segment).long(),
            "origin_segment": torch.from_numpy(labels).long(),
            "inverse": torch.from_numpy(inverse).long(),
            "lidar_token": item.lidar_token,
            "sample_token": item.sample_token,
            "scene_name": item.scene_name,
            "lidar_path": item.lidar_path,
            "label_path": item.label_path,
            "num_original_points": int(points.shape[0]),
            "num_voxel_points": num_vox,
        }


def collate_single_scan(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for batch_size=1.

    PTv3 on outdoor point clouds can be memory-heavy; for the mini milestone,
    evaluating one scan at a time is the least surprising path.  If you later
    want multi-scan batches, concatenate coord/feat/grid_coord/segment and set
    offset to the cumulative point counts.
    """
    if len(batch) != 1:
        raise ValueError("collate_single_scan expects DataLoader(batch_size=1).")
    return batch[0]
