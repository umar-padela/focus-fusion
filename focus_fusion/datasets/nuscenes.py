"""nuScenes-lidarseg mini dataset utilities for LitePT experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -1

CLASS_NAMES: Sequence[str] = (
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
)

OFFICIAL_CHALLENGE_ID_BY_GENERAL_NAME: Mapping[str, int] = {
    "noise": 0,
    "animal": 0,
    "human.pedestrian.personal_mobility": 0,
    "human.pedestrian.stroller": 0,
    "human.pedestrian.wheelchair": 0,
    "movable_object.debris": 0,
    "movable_object.pushable_pullable": 0,
    "static_object.bicycle_rack": 0,
    "vehicle.emergency.ambulance": 0,
    "vehicle.emergency.police": 0,
    "static.other": 0,
    "vehicle.ego": 0,
    "movable_object.barrier": 1,
    "vehicle.bicycle": 2,
    "vehicle.bus.bendy": 3,
    "vehicle.bus.rigid": 3,
    "vehicle.car": 4,
    "vehicle.construction": 5,
    "vehicle.motorcycle": 6,
    "human.pedestrian.adult": 7,
    "human.pedestrian.child": 7,
    "human.pedestrian.construction_worker": 7,
    "human.pedestrian.police_officer": 7,
    "movable_object.trafficcone": 8,
    "vehicle.trailer": 9,
    "vehicle.truck": 10,
    "flat.driveable_surface": 11,
    "flat.other": 12,
    "flat.sidewalk": 13,
    "flat.terrain": 14,
    "static.manmade": 15,
    "static.vegetation": 16,
}


@dataclass(frozen=True)
class LidarSegSample:
    sample_token: str
    lidar_token: str
    scene_name: str
    lidar_path: str
    label_path: str
    timestamp: int


def _load_lidar_bin(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % 5 != 0:
        raise ValueError(f"Expected lidar file size to be divisible by 5 floats: {path}")
    return arr.reshape(-1, 5)


def _voxelize_first_point(coords: np.ndarray, voxel_size: float):
    min_coord = coords.min(axis=0, keepdims=True)
    grid_all = np.floor((coords - min_coord) / voxel_size).astype(np.int64)
    _, unique_indices, inverse = np.unique(
        grid_all, axis=0, return_index=True, return_inverse=True
    )
    return grid_all, unique_indices.astype(np.int64), inverse.astype(np.int64)


def _idx_name_mapping_from_nusc(nusc) -> Dict[int, str]:
    if hasattr(nusc, "lidarseg_idx2name_mapping"):
        return {int(k): str(v) for k, v in nusc.lidarseg_idx2name_mapping.items()}
    if hasattr(nusc, "lidarseg_name2idx_mapping"):
        return {int(v): str(k) for k, v in nusc.lidarseg_name2idx_mapping.items()}
    raise RuntimeError("NuScenes object does not expose lidarseg id/name mapping.")


def build_learning_map(nusc, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    idx_to_name = _idx_name_mapping_from_nusc(nusc)
    lut = np.full(max(256, max(idx_to_name) + 1), ignore_index, dtype=np.int64)
    for raw_idx, name in idx_to_name.items():
        official_id = OFFICIAL_CHALLENGE_ID_BY_GENERAL_NAME.get(name, 0)
        lut[raw_idx] = ignore_index if official_id == 0 else official_id - 1
    return lut


def remap_raw_labels(raw_labels: np.ndarray, learning_map: np.ndarray) -> np.ndarray:
    raw = raw_labels.astype(np.int64, copy=False)
    out = np.full(raw.shape, IGNORE_INDEX, dtype=np.int64)
    valid = (raw >= 0) & (raw < len(learning_map))
    out[valid] = learning_map[raw[valid]]
    return out


def internal_to_official_ids(pred_internal: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_internal, dtype=np.int64)
    pred = np.clip(pred, 0, len(CLASS_NAMES) - 1)
    return (pred + 1).astype(np.uint8)


def official_to_internal_ids(pred_official: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_official, dtype=np.int64)
    out = np.full(pred.shape, IGNORE_INDEX, dtype=np.int64)
    valid = (pred >= 1) & (pred <= len(CLASS_NAMES))
    out[valid] = pred[valid] - 1
    return out


class NuScenesLidarSegDataset(Dataset):
    """Single-keyframe nuScenes-lidarseg loader matching Pointcept feature scale."""

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
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes

        self.dataroot = os.path.abspath(dataroot)
        self.version = version
        self.split = split
        self.voxel_size = float(voxel_size)
        self.ignore_index = int(ignore_index)

        self.nusc = NuScenes(version=version, dataroot=self.dataroot, verbose=verbose)
        self.learning_map = build_learning_map(self.nusc, ignore_index=self.ignore_index)

        allowed_scenes = set(create_splits_scenes()[split])
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
            lidarseg_record = self.nusc.get("lidarseg", lidar_token)
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
        self.samples = samples[: int(max_scans)] if max_scans is not None else samples
        if verbose:
            print(f"[NuScenesLidarSegDataset] Loaded {len(self.samples)} scans for {split}.")

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
        strength = points[:, 3:4].astype(np.float32, copy=False) / 255.0
        labels = remap_raw_labels(raw_labels, self.learning_map)

        grid_all, unique_indices, inverse = _voxelize_first_point(coords, self.voxel_size)
        coord = coords[unique_indices]
        feat = np.concatenate([coord, strength[unique_indices]], axis=1).astype(np.float32)
        num_vox = int(coord.shape[0])
        return {
            "coord": torch.from_numpy(coord).float(),
            "grid_coord": torch.from_numpy(grid_all[unique_indices]).int(),
            "feat": torch.from_numpy(feat).float(),
            "offset": torch.tensor([num_vox], dtype=torch.long),
            "segment": torch.from_numpy(labels[unique_indices]).long(),
            "origin_segment": torch.from_numpy(labels).long(),
            "inverse": torch.from_numpy(inverse).long(),
            "lidar_token": item.lidar_token,
            "sample_token": item.sample_token,
            "scene_name": item.scene_name,
            "num_original_points": int(points.shape[0]),
            "num_voxel_points": num_vox,
        }


def collate_single_scan(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("collate_single_scan expects DataLoader(batch_size=1).")
    return batch[0]
