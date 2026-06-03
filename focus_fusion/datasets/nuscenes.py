"""nuScenes mini lidarseg dataset for FocusFusion.

Merges Person 1's 16-class label remapping with Person 2's camera loading.

Batch dict contract (used by FocusFusion.forward()):

    E1 (T=1):
        points       (N, 3)           float32  LiDAR xyz, current frame
        images       (6, 3, H, W)     float32  [0,1]; DINOv2Backbone normalizes internally
        labels       (N,)             int64    0–15 challenge classes; -1 = ignore
        sample_token str
        scene_name   str              for trainer scene-boundary detection

    E2 (T>1):
        points       (N, 3)
        images_seq   (T, 6, 3, H, W)  float32  chronological, oldest → current
        labels       (N,)
        sample_token str
        scene_name   str
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Label constants (Person 1's 16 challenge-class mapping)
# ---------------------------------------------------------------------------

IGNORE_INDEX: int = -1

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

# Raw nuScenes general category name → official challenge id (0 = void/ignore, 1–16 = valid)
_CHALLENGE_ID: Mapping[str, int] = {
    "noise": 0, "animal": 0,
    "human.pedestrian.personal_mobility": 0, "human.pedestrian.stroller": 0,
    "human.pedestrian.wheelchair": 0, "movable_object.debris": 0,
    "movable_object.pushable_pullable": 0, "static_object.bicycle_rack": 0,
    "vehicle.emergency.ambulance": 0, "vehicle.emergency.police": 0,
    "static.other": 0, "vehicle.ego": 0,
    "movable_object.barrier": 1, "vehicle.bicycle": 2,
    "vehicle.bus.bendy": 3, "vehicle.bus.rigid": 3,
    "vehicle.car": 4, "vehicle.construction": 5,
    "vehicle.motorcycle": 6,
    "human.pedestrian.adult": 7, "human.pedestrian.child": 7,
    "human.pedestrian.construction_worker": 7, "human.pedestrian.police_officer": 7,
    "movable_object.trafficcone": 8, "vehicle.trailer": 9, "vehicle.truck": 10,
    "flat.driveable_surface": 11, "flat.other": 12, "flat.sidewalk": 13,
    "flat.terrain": 14, "static.manmade": 15, "static.vegetation": 16,
}

CAMERAS: List[str] = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def _voxelize(
    xyz: np.ndarray,
    grid_size: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid-sample a point cloud.

    Args:
        xyz:       (N, 3) float32 point coordinates
        grid_size: voxel edge length in metres
    Returns:
        grid_idx:  (n_vox, 3) int64  — integer grid coords of each unique voxel
        inverse:   (N,)       int64  — maps each point to its voxel in [0, n_vox)
        vox_xyz:   (n_vox, 3) float32 — mean coordinate within each voxel
    """
    xyz = xyz.astype(np.float32)
    g = np.floor((xyz - xyz.min(axis=0)) / grid_size).astype(np.int64)
    struct_dt = np.dtype((np.void, g.itemsize * 3))
    _, uniq_pos, inverse = np.unique(
        g.view(struct_dt).ravel(), return_index=True, return_inverse=True
    )
    uniq_g = g[uniq_pos]
    n = len(uniq_pos)
    vox_xyz = np.zeros((n, 3), np.float32)
    np.add.at(vox_xyz, inverse, xyz)
    vox_xyz /= np.bincount(inverse, minlength=n).reshape(-1, 1).astype(np.float32)
    return uniq_g, inverse, vox_xyz


def collate_focusfusion(samples: list) -> dict:
    """Custom collate for FocusFusion batches.

    Dense keys (points, labels, images, inverse) are stacked.
    Sparse voxel keys (coord, feat, grid_coord) are concatenated with an
    offset tensor tracking each batch item's voxel count — Pointcept's
    standard sparse-batch format.
    """
    out: dict = {}

    # Dense stackable tensors
    for key in ("points", "labels", "images", "images_seq", "inverse"):
        if key in samples[0]:
            out[key] = torch.stack([s[key] for s in samples])

    # String fields
    for key in ("sample_token", "scene_name"):
        if key in samples[0]:
            out[key] = [s[key] for s in samples]

    # Sparse voxel tensors — concatenate across batch items
    if "vox_coord" in samples[0]:
        out["coord"]      = torch.cat([s["vox_coord"]       for s in samples])
        out["feat"]       = torch.cat([s["vox_feat"]        for s in samples])
        out["grid_coord"] = torch.cat([s["vox_grid_coord"]  for s in samples])
        counts = [len(s["vox_coord"]) for s in samples]
        out["offset"] = torch.tensor(
            [sum(counts[: i + 1]) for i in range(len(counts))], dtype=torch.long
        )

    return out


def build_learning_map(nusc, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Build a lookup table: raw_label_id → internal 0-based class id (or ignore_index)."""
    if hasattr(nusc, "lidarseg_idx2name_mapping"):
        idx_to_name = {int(k): str(v) for k, v in nusc.lidarseg_idx2name_mapping.items()}
    elif hasattr(nusc, "lidarseg_name2idx_mapping"):
        idx_to_name = {int(v): str(k) for k, v in nusc.lidarseg_name2idx_mapping.items()}
    else:
        raise RuntimeError("NuScenes object does not expose lidarseg id/name mapping.")

    lut = np.full(max(256, max(idx_to_name) + 1), ignore_index, dtype=np.int64)
    for raw_idx, name in idx_to_name.items():
        official_id = _CHALLENGE_ID.get(name, 0)
        lut[raw_idx] = ignore_index if official_id == 0 else official_id - 1  # 0-based
    return lut


def remap_raw_labels(raw_labels: np.ndarray, learning_map: np.ndarray) -> np.ndarray:
    raw = raw_labels.astype(np.int64, copy=False)
    out = np.full(raw.shape, IGNORE_INDEX, dtype=np.int64)
    valid = (raw >= 0) & (raw < len(learning_map))
    out[valid] = learning_map[raw[valid]]
    return out


def internal_to_official_ids(pred_internal: np.ndarray) -> np.ndarray:
    pred = np.clip(np.asarray(pred_internal, dtype=np.int64), 0, len(CLASS_NAMES) - 1)
    return (pred + 1).astype(np.uint8)


def official_to_internal_ids(pred_official: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_official, dtype=np.int64)
    out = np.full(pred.shape, IGNORE_INDEX, dtype=np.int64)
    valid = (pred >= 1) & (pred <= len(CLASS_NAMES))
    out[valid] = pred[valid] - 1
    return out


# ---------------------------------------------------------------------------
# Point-cloud helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LidarSegSample:
    sample_token: str
    lidar_token: str
    scene_name: str
    lidar_path: str
    label_path: str
    timestamp: int


def _load_lidar_bin(path: str) -> np.ndarray:
    """Load a nuScenes LIDAR_TOP .bin file as (N, 5) float32 [x,y,z,intensity,ring]."""
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % 5 != 0:
        raise ValueError(f"Lidar bin size not divisible by 5: {path}")
    return arr.reshape(-1, 5)


def subsample_points(
    points: np.ndarray,
    labels: np.ndarray,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly subsample or pad to exactly num_points.

    Padding (rare — nuScenes ~34k pts >> 16384): repeated real points get label=-1
    so SegmentationLoss ignores them.
    """
    N = points.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
        return points[idx], labels[idx]
    pad = num_points - N
    pad_idx = np.random.choice(N, pad, replace=True)
    pad_labels = np.full(pad, IGNORE_INDEX, dtype=np.int64)
    return (
        np.concatenate([points, points[pad_idx]], axis=0),
        np.concatenate([labels, pad_labels], axis=0),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NuScenesLidarSegDataset(Dataset):
    """Per-keyframe LiDAR + camera dataset for nuScenes mini lidarseg.

    LiDAR labels are remapped to the 16 official challenge classes (Person 1's mapping).
    Camera images are loaded for all 6 cameras (Person 2's DINOv2 K/V pipeline).

    Args:
        dataroot:    path containing v1.0-mini/
        version:     nuScenes version string
        split:       'mini_train' or 'mini_val'
        num_points:  fixed LiDAR points per sample (subsampled/padded)
        img_size:    camera resize target (must be divisible by 14 for DINOv2)
        T:           temporal window depth — 1 for E1, 6 for E2
        ignore_index: label value for void / padded points
        verbose:     print dataset size on load
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        split: str = "mini_train",
        num_points: int = 16384,
        img_size: int = 448,
        T: int = 1,
        ignore_index: int = IGNORE_INDEX,
        verbose: bool = False,
        max_scans: Optional[int] = None,
        fraction: float = 1.0,
        seed: int = 231,
    ) -> None:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes

        self.dataroot = os.path.abspath(dataroot)
        self.num_points = num_points
        self.img_size = img_size
        self.T = T

        self.nusc = NuScenes(version=version, dataroot=self.dataroot, verbose=verbose)
        self.learning_map = build_learning_map(self.nusc, ignore_index=ignore_index)

        allowed_scenes = set(create_splits_scenes()[split])
        scene_token_to_name = {s["token"]: s["name"] for s in self.nusc.scene}

        samples: List[LidarSegSample] = []
        for sample in self.nusc.sample:
            scene_name = scene_token_to_name[sample["scene_token"]]
            if scene_name not in allowed_scenes:
                continue
            lidar_token = sample["data"]["LIDAR_TOP"]
            sd = self.nusc.get("sample_data", lidar_token)
            if not sd.get("is_key_frame", False):
                continue
            ls = self.nusc.get("lidarseg", lidar_token)
            samples.append(LidarSegSample(
                sample_token=sample["token"],
                lidar_token=lidar_token,
                scene_name=scene_name,
                lidar_path=os.path.join(self.dataroot, sd["filename"]),
                label_path=os.path.join(self.dataroot, ls["filename"]),
                timestamp=int(sample.get("timestamp", 0)),
            ))

        samples.sort(key=lambda x: (x.scene_name, x.timestamp))

        # Filter to files that are actually present on disk first, so that
        # fraction applies to uploaded data rather than all of NuScenes.
        present = [s for s in samples if os.path.exists(s.lidar_path) and os.path.exists(s.label_path)]
        n_dropped = len(samples) - len(present)
        if n_dropped:
            n_present_scenes = len(set(s.scene_name for s in present))
            print(
                f"[NuScenesLidarSegDataset] {n_present_scenes} scenes / {len(present)} scans available "
                f"({n_dropped} skipped — blobs not uploaded)"
            )
        samples = present

        if fraction < 1.0:
            all_scenes = sorted(set(s.scene_name for s in samples))
            rng = np.random.default_rng(seed)
            n_scenes = max(1, round(len(all_scenes) * fraction))
            chosen = set(rng.choice(all_scenes, n_scenes, replace=False).tolist())
            samples = [s for s in samples if s.scene_name in chosen]

        self.samples = samples[:int(max_scans)] if max_scans is not None else samples

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # HWC uint8 → CHW float32 [0, 1]
        ])

        if verbose:
            n_scenes = len(set(s.scene_name for s in self.samples))
            print(f"[NuScenesLidarSegDataset] {len(self.samples)} scans / {n_scenes} scenes ({split}, fraction={fraction:.2f})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        points, labels, vox_keys = self._load_lidar(item)

        base = {
            "points":       points,
            "labels":       labels,
            "sample_token": item.sample_token,
            "scene_name":   item.scene_name,
            **vox_keys,
        }
        if self.T == 1:
            base["images"] = self._load_images(item.sample_token)
        else:
            base["images_seq"] = self._load_images_seq(item.sample_token)
        return base

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------

    def _load_lidar(self, item: LidarSegSample) -> tuple[Tensor, Tensor, dict]:
        points = _load_lidar_bin(item.lidar_path)          # (N_raw, 5): x,y,z,intensity,ring
        xyz = points[:, :3].astype(np.float32)             # (N_raw, 3)
        intensity = points[:, 3:4].astype(np.float32)      # (N_raw, 1)
        raw_labels = np.fromfile(item.label_path, dtype=np.uint8)
        labels = remap_raw_labels(raw_labels, self.learning_map)
        xyzi = np.concatenate([xyz, intensity], axis=1)    # (N_raw, 4)
        xyzi, labels = subsample_points(xyzi, labels, self.num_points)
        xyz = xyzi[:, :3]

        # Voxelise for LitePT (Pointcept sparse format)
        grid_idx, inverse, vox_xyz = _voxelize(xyz)
        n_vox = len(vox_xyz)
        # Mean intensity per voxel
        vox_intensity = np.zeros((n_vox, 1), np.float32)
        np.add.at(vox_intensity[:, 0], inverse, xyzi[:, 3])
        vox_intensity /= np.bincount(inverse, minlength=n_vox).reshape(-1, 1).astype(np.float32)
        vox_feat = np.concatenate([vox_xyz, vox_intensity], axis=1)
        vox_keys = {
            "vox_coord":      torch.from_numpy(vox_xyz),
            "vox_feat":       torch.from_numpy(vox_feat),
            "vox_grid_coord": torch.from_numpy(grid_idx),
            "inverse":        torch.from_numpy(inverse),   # (N,) local voxel indices
        }
        return torch.from_numpy(xyz), torch.from_numpy(labels), vox_keys

    # ------------------------------------------------------------------
    # Images (Person 2's DINOv2 pipeline)
    # ------------------------------------------------------------------

    def _load_images(self, sample_token: str) -> Tensor:
        """Load all 6 cameras for one keyframe → (6, 3, H, W)."""
        sample = self.nusc.get("sample", sample_token)
        imgs = []
        for cam in CAMERAS:
            cam_sd = self.nusc.get("sample_data", sample["data"][cam])
            img = Image.open(Path(self.dataroot) / cam_sd["filename"]).convert("RGB")
            imgs.append(self.img_transform(img))
        return torch.stack(imgs)

    def _load_images_seq(self, sample_token: str) -> Tensor:
        """Load T-frame camera sequence → (T, 6, 3, H, W), chronological."""
        tokens: list[str] = []
        token = sample_token
        for _ in range(self.T):
            tokens.append(token)
            prev = self.nusc.get("sample", token)["prev"]
            token = prev if prev else token  # hold at scene start
        tokens.reverse()
        return torch.stack([self._load_images(t) for t in tokens])


# ---------------------------------------------------------------------------
# Collate (Person 1's single-scan eval helper)
# ---------------------------------------------------------------------------

def collate_single_scan(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """collate_fn for DataLoader(batch_size=1) — returns the single dict unwrapped."""
    if len(batch) != 1:
        raise ValueError("collate_single_scan expects batch_size=1.")
    return batch[0]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataroot: str,
    split: str,
    *,
    version: str = "v1.0-mini",
    num_points: int = 16384,
    img_size: int = 448,
    T: int = 1,
    batch_size: int = 2,
    num_workers: int = 4,
) -> DataLoader:
    dataset = NuScenesLidarSegDataset(
        dataroot=dataroot,
        version=version,
        split=split,
        num_points=num_points,
        img_size=img_size,
        T=T,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "mini_train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "mini_train"),
    )
