# Modal — remote training and eval

GPU work runs on [Modal](https://modal.com/). Everything below runs from the repo root.

---

## First-time setup

### 1. Install Modal and authenticate

```powershell
pip install modal
modal token new   # opens browser — log in once
```

### 2. Create the W&B secret

Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize), then:

```powershell
modal secret create wandb WANDB_API_KEY=<your_key>
```

This creates a Modal secret named `wandb` that the training container reads automatically.
You only do this once — it persists across all your Modal runs.

**Project isolation:** training always logs to the `focus-fusion` W&B project
(set in `configs/default.yaml` under `wandb.project`). This is completely separate
from any of your other W&B projects — you'll find it at
`wandb.ai/<your-username>/focus-fusion`. To change the project name, edit `wandb.project`
in the config.

To disable W&B for a run (e.g. a quick smoke check):
```powershell
modal run modal/modal_train.py --experiment e1 --extra-args "--no-wandb"
```

### 3. Upload nuScenes data to the volume

Download the two archives from the [nuScenes download page](https://www.nuscenes.org/nuscenes#download):
- **Mini** — `v1.0-mini.tgz` (~4 GB sensor data + images)
- **LiDAR segmentation** — `nuScenes-lidarseg-mini-v1.0.tar.bz2` (~50 MB per-point class labels)

You need both. The mini archive has point cloud XYZ; the lidarseg archive has the semantic
class label for every point. Without lidarseg there is nothing to train against.

Upload the compressed archives to the volume (uploading compressed is faster than uploading
the extracted tree). Replace the paths below with wherever you saved the archives locally:

**Windows:**
```powershell
modal volume put focus-fusion-experiments "C:\path\to\v1.0-mini.tgz" data/archives/v1.0-mini.tgz
modal volume put focus-fusion-experiments "C:\path\to\nuScenes-lidarseg-mini-v1.0.tar.bz2" data/archives/nuScenes-lidarseg-mini-v1.0.tar.bz2
```

**Mac / Linux:**
```bash
modal volume put focus-fusion-experiments /path/to/v1.0-mini.tgz data/archives/v1.0-mini.tgz
modal volume put focus-fusion-experiments /path/to/nuScenes-lidarseg-mini-v1.0.tar.bz2 data/archives/nuScenes-lidarseg-mini-v1.0.tar.bz2
```

Then extract them server-side (fast — runs on Modal, not your laptop):

```powershell
modal run modal/modal_setup.py --mode extract-data
```

After extraction the volume will have:
```
data/
  samples/LIDAR_TOP/   ← raw point clouds (.bin files)
  sweeps/LIDAR_TOP/
  lidarseg/v1.0-mini/  ← per-point class labels (.bin files)
  v1.0-mini/           ← scene/sample/calibration metadata
  maps/
  archives/            ← original compressed archives (kept for reference)
```

### 4. Download the LitePT checkpoint

This pulls [`prs-eth/LitePT`](https://huggingface.co/prs-eth/LitePT) directly from
HuggingFace into the volume — no local download needed:

```powershell
modal run modal/modal_setup.py --mode download-litept
```

### 5. Verify everything is ready

```powershell
modal run modal/modal_setup.py --mode check
```

Expected output:
```
  ✓  litept_checkpoint: 123.4 MB
  ✓  nuscenes_v1.0-mini: OK (x items)
  ✓  nuscenes_lidarseg: OK (x items)
  ✓  nuscenes_lidar_top: OK (x items)
  ✓  ready_to_train: True
```

### 6. Run the smoke test

```powershell
modal run modal/modal_smoke.py
```

Checks GPU availability, DINOv2 forward pass, and volume write. Should complete in ~30 s.

---

## Training

| Command | What it does |
|---------|-------------|
| `modal run --detach modal/modal_train.py --experiment e1` | E1: T=1 single-frame (10 epochs default) |
| `modal run --detach modal/modal_train.py --experiment e2` | E2: T=6 temporal (Person 2) |
| `modal run --detach modal/modal_train.py --experiment e1 --epochs 50` | Override epoch count |

`--detach` before the script path keeps the job alive after Ctrl-C. The entrypoint
blocks showing live logs; the remote job continues independently if you close the terminal.

Override the GPU without editing code:
```powershell
$env:MODAL_GPU_TRAIN="A100-80GB"; modal run --detach modal/modal_train.py --experiment e1
```

## Eval

```powershell
modal run modal/modal_eval.py --experiment e1
modal run modal/modal_eval.py --experiment e1 --split mini_train
modal run modal/modal_eval.py --experiment e1 --checkpoint /experiments/e1/checkpoints/epoch_5.pt
```

---

## Volume layout

One volume (`focus-fusion-experiments`) holds everything:

| Path on volume | Contents |
|----------------|----------|
| `data/` | nuScenes mini (samples, sweeps, v1.0-mini, lidarseg) |
| `checkpoints/litept/model_best.pth` | LitePT pretrained weights |
| `e1/checkpoints/` | E1 training checkpoints (`latest.pt`, `best.pt`) |
| `e2/checkpoints/` | E2 training checkpoints |
| `e1/train_log.csv` | Per-epoch loss log |
| `e1/val_epoch*.json` | Per-val-epoch metrics |

Browse and download:
```powershell
modal volume ls  focus-fusion-experiments
modal volume ls  focus-fusion-experiments e1/checkpoints
modal volume get focus-fusion-experiments e1/checkpoints/best.pt best.pt
```

---

## GPU defaults

| Function | Default GPU | Override env var |
|----------|-------------|-----------------|
| `smoke`, `evaluate` | A10G | `$env:MODAL_GPU_EVAL` |
| `train` | A100-80GB | `$env:MODAL_GPU_TRAIN` |

---

## Files

| File | Role |
|------|------|
| `modal_config.py` | App, Image, Volume, `smoke` / `train` / `evaluate` functions |
| `modal_setup.py` | One-time setup: download LitePT checkpoint, check volume readiness |
| `modal_smoke.py` | CLI → `smoke` (infra health check) |
| `modal_train.py` | CLI → `train` (E1 or E2) |
| `modal_eval.py` | CLI → `evaluate` |
