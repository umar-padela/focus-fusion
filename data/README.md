# Data — nuScenes mini

Raw data is **not committed to git**. This directory is gitignored.

FocusFusion trains and evaluates on **nuScenes mini** (`v1.0-mini`):
10 scenes, ~400 keyframes, 32-beam LiDAR, 6 cameras, 2 Hz keyframe rate.

---

## Step 1 — Register

Go to [https://www.nuscenes.org/download](https://www.nuscenes.org/download), create an account, and accept the terms of use. This is required — the dataset is not public.

---

## Step 2 — Download

After logging in, download these two packages to this `data/` directory:

| Package | Size | Notes |
|---------|------|-------|
| **nuScenes-mini** | ~4 GB | Main package (metadata, images, LiDAR) |
| **nuScenes-lidarseg-all-v1.0** | ~90 MB | Per-point semantic labels — required for training |

Both are available under the **"Mini"** and **"Lidarseg"** sections of the download page.

---

## Step 3 — Extract

```powershell
# From this data/ directory
tar -xzf nuScenes-mini.tar.gz
tar -xzf nuScenes-lidarseg-all-v1.0.tar.bz2
```

After extraction the structure should look like:

```
data/
  v1.0-mini/          ← metadata JSONs (scene, sample, calibration, etc.)
  samples/
    CAM_BACK/
    CAM_BACK_LEFT/
    CAM_BACK_RIGHT/
    CAM_FRONT/
    CAM_FRONT_LEFT/
    CAM_FRONT_RIGHT/
    LIDAR_TOP/        ← .bin point cloud files
  maps/
  lidarseg/
    v1.0-mini/        ← per-point label .bin files
```

> **Skip `sweeps/`** if offered — we only use keyframes (2 Hz samples), not inter-frame sweeps.
> Skipping saves ~3 GB.

---

## Step 4 — Upload to Modal volume

Training runs on Modal. Data lives inside the `focus-fusion-experiments` volume
under a `data/` subdirectory (same volume as checkpoints and logs).

```powershell
conda activate cs224r

modal volume put focus-fusion-experiments data\v1.0-mini  data/v1.0-mini
modal volume put focus-fusion-experiments data\samples    data/samples
modal volume put focus-fusion-experiments data\maps       data/maps
modal volume put focus-fusion-experiments data\lidarseg   data/lidarseg
```

Verify the upload:

```powershell
modal volume ls focus-fusion-experiments data
# Expected output:
#   data/v1.0-mini/
#   data/samples/
#   data/maps/
#   data/lidarseg/
```

Inside Modal containers the volume is mounted at `/experiments`, so the data
is at `/experiments/data/` — which is the `dataroot` our `NuScenesLidarSegDataset`
and `build_dataloader` expect.

---

## Step 5 — Local smoke test (optional)

To test the dataloader locally before uploading:

```powershell
conda activate cs224r
python -c "
from focus_fusion.datasets.nuscenes import NuScenesLidarSegDataset
ds = NuScenesLidarSegDataset('data', split='mini_train')
item = ds[0]
print('points:', item['points'].shape)
print('images:', item['images'].shape)
print('labels:', item['labels'].shape)
print(f'Dataset size: {len(ds)} samples')
"
```

Expected output:
```
points: torch.Size([16384, 3])
images: torch.Size([6, 3, 448, 448])
labels: torch.Size([16384])
Dataset size: 323 samples
```

---

## Split sizes (nuScenes mini)

| Split | Scenes | Approx. samples |
|-------|--------|-----------------|
| `mini_train` | 8 | ~323 keyframes |
| `mini_val` | 2 | ~81 keyframes |

---

## Lidarseg class map

nuScenes lidarseg has 32 raw label IDs. **16 are used for evaluation** (the
"challenge classes"). The mapping lives in Person 1's `labels.py` in the ptv3
branch — coordinate with Person 1 before setting `num_classes` in
`configs/default.yaml`.
