"""nuScenes mini lidarseg dataset for FocusFusion.

Returns batch dicts matching the contract used by FocusFusion.forward():

    E1 (T=1):
        points       (N, 3)              float32  LiDAR xyz, current frame
        images       (6, 3, H, W)        float32  [0, 1]; DINOv2Backbone normalizes internally
        labels       (N,)                int64    class 0–31; -1 = ignore (padding / noise)
        sample_token str

    E2 (T>1):
        points       (N, 3)              same as above
        images_seq   (T, 6, 3, H, W)    float32  chronological, oldest → current
        labels       (N,)                same as above
        sample_token str

Subsampling: random per __getitem__ (training augmentation).
Padding (rare — nuScenes ~34k pts >> 16384): padded points are set to label -1
so SegmentationLoss ignores them.

TODO (Person 1 coordination):
  - The _load_lidar() method currently returns a simple (N, 3) xyz tensor.
    Person 1's ptv3 branch uses a voxelized sparse format with keys:
      coord (M,3), grid_coord (M,3), feat (M,4 xyz+intensity), offset (1,), segment (M,)
    Once Person 1 finalises their dataset class, either:
      (a) subclass their NuScenesLidarSegDataset and add _load_images / _load_images_seq,
      (b) or call their voxelization util from _load_lidar() here.
  - Person 1 uses 16 challenge classes (remapped from raw 32). Align num_classes
    and the label remapping table (labels.py in the ptv3 branch) before training.
  - Confirm whether 'scene_token' needs to be added to the batch dict so the
    trainer can detect scene boundaries and reset the memory bank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.splits import create_splits_scenes
except ImportError:
    NuScenes = LidarPointCloud = create_splits_scenes = None  # type: ignore[assignment,misc]

CAMERAS: list[str] = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def subsample_points(
    points: np.ndarray,
    labels: np.ndarray,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly subsample or pad (points, labels) to exactly num_points.

    Subsampling (N >= num_points): random without replacement.
    Padding (N < num_points, rare): repeat-sample from existing points;
    padded entries get label -1 so the loss ignores them.
    """
    N = points.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
        return points[idx], labels[idx]

    # N < num_points: pad by repeating real points, ignore-label the extras
    pad = num_points - N
    pad_idx = np.random.choice(N, pad, replace=True)
    pad_labels = np.full(pad, -1, dtype=np.int64)
    return (
        np.concatenate([points, points[pad_idx]], axis=0),
        np.concatenate([labels, pad_labels], axis=0),
    )


class NuScenesLidarSegDataset(Dataset):
    """Per-keyframe LiDAR + camera dataset for nuScenes mini lidarseg.

    Args:
        dataroot:    path to nuScenes dataset root (the dir that contains v1.0-mini/)
        version:     nuScenes version string (default: 'v1.0-mini')
        split:       'mini_train' or 'mini_val'
        num_points:  fixed LiDAR points per sample (default: 16384)
        img_size:    camera image resize target in pixels (default: 448)
        T:           temporal window depth — 1 for E1, 6 for E2 (default: 1)
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        split: str = "mini_train",
        num_points: int = 16384,
        img_size: int = 448,
        T: int = 1,
    ) -> None:
        if NuScenes is None:
            raise ImportError("nuscenes-devkit is required: pip install nuscenes-devkit")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.num_points = num_points
        self.img_size = img_size
        self.T = T
        self.samples = self._collect_samples(split)

        # Images returned as float32 [0, 1].
        # DINOv2Backbone applies ImageNet normalization internally (normalize_input=True),
        # so we do NOT apply it here.
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # HWC uint8 → CHW float32 [0, 1]
        ])

    # ------------------------------------------------------------------
    # Sample collection
    # ------------------------------------------------------------------

    def _collect_samples(self, split: str) -> list[str]:
        """Walk every scene in the split and collect sample tokens in order."""
        scene_names = set(create_splits_scenes()[split])
        tokens: list[str] = []
        for scene in self.nusc.scene:
            if scene["name"] not in scene_names:
                continue
            token = scene["first_sample_token"]
            while token:
                tokens.append(token)
                token = self.nusc.get("sample", token)["next"]
        return tokens

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_token = self.samples[idx]
        points, labels = self._load_lidar(sample_token)

        if self.T == 1:
            return {
                "points": points,                              # (N, 3)
                "images": self._load_images(sample_token),    # (6, 3, H, W)
                "labels": labels,                             # (N,)
                "sample_token": sample_token,
            }
        return {
            "points": points,
            "images_seq": self._load_images_seq(sample_token),  # (T, 6, 3, H, W)
            "labels": labels,
            "sample_token": sample_token,
        }

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------

    def _load_lidar(self, sample_token: str) -> tuple[Tensor, Tensor]:
        sample = self.nusc.get("sample", sample_token)
        lidar_sd_token = sample["data"]["LIDAR_TOP"]

        lidar_sd = self.nusc.get("sample_data", lidar_sd_token)
        pc = LidarPointCloud.from_file(
            str(Path(self.nusc.dataroot) / lidar_sd["filename"])
        )
        points_xyz = pc.points[:3, :].T.astype(np.float32)  # (N_raw, 3)

        lidarseg_entry = self.nusc.get("lidarseg", lidar_sd_token)
        labels_raw = np.fromfile(
            str(Path(self.nusc.dataroot) / lidarseg_entry["filename"]),
            dtype=np.uint8,
        ).astype(np.int64)  # (N_raw,) values in [0, 31]

        points_xyz, labels_raw = subsample_points(
            points_xyz, labels_raw, self.num_points
        )
        return torch.from_numpy(points_xyz), torch.from_numpy(labels_raw)

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def _load_images(self, sample_token: str) -> Tensor:
        """Load all 6 cameras for one keyframe → (6, 3, H, W)."""
        sample = self.nusc.get("sample", sample_token)
        imgs = []
        for cam in CAMERAS:
            cam_sd = self.nusc.get("sample_data", sample["data"][cam])
            img = Image.open(
                Path(self.nusc.dataroot) / cam_sd["filename"]
            ).convert("RGB")
            imgs.append(self.img_transform(img))  # (3, H, W)
        return torch.stack(imgs)  # (6, 3, H, W)

    def _load_images_seq(self, sample_token: str) -> Tensor:
        """Load a T-frame camera sequence → (T, 6, 3, H, W), chronological.

        Walks backward T-1 steps through the nuScenes sample chain.
        At scene boundaries (fewer than T previous frames available), the
        earliest frame in the scene is repeated to fill the window.
        """
        tokens: list[str] = []
        token = sample_token
        for _ in range(self.T):
            tokens.append(token)
            prev = self.nusc.get("sample", token)["prev"]
            token = prev if prev else token  # hold at scene start
        tokens.reverse()  # oldest → current

        return torch.stack([self._load_images(t) for t in tokens])  # (T, 6, 3, H, W)


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
    """Build a DataLoader for the given nuScenes split."""
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
        drop_last=(split == "mini_train"),  # keep val batches complete for metrics
    )
