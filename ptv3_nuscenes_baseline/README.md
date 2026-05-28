# PTv3 nuScenes-lidarseg mini baseline

This package gives you a concrete LiDAR-only baseline for CS231N Milestone 3:

1. a raw `nuScenes-lidarseg` Dataset/DataLoader for `v1.0-mini`,
2. remapping from the 32 raw lidarseg ids to the official 16 lidarseg classes,
3. mIoU / mAcc / fwIoU computation,
4. Pointcept/PTv3 checkpoint evaluation, and
5. an optional mini-set train/fine-tune loop.

The scripts use batch size 1 by default because PTv3 outdoor scans are memory-heavy and mini-set evaluation is small.

## Folder layout

```text
ptv3_nuscenes_baseline/
  labels.py                       # raw 32 -> official 16 label mapping
  nuscenes_lidarseg_dataset.py     # Dataset/DataLoader + voxelization
  metrics.py                      # confusion-matrix metrics
  pointcept_utils.py              # Pointcept config/model/checkpoint helpers
  check_nuscenes_lidarseg.py       # smoke test + class histogram
  eval_pointcept_ptv3.py           # evaluate PTv3 checkpoint on mini_val
  train_ptv3_mini.py              # optional mini_train fine-tuning/linear probe
  eval_saved_predictions.py        # evaluate saved *_lidarseg.bin predictions
  eval_majority_baseline.py        # non-neural sanity baseline for comparisons
  requirements.txt
```

## Data expected

Your nuScenes root should contain the mini sensor data and the lidarseg expansion in the same directory:

```text
/data/sets/nuscenes/
  maps/
  samples/
  sweeps/
  lidarseg/
  v1.0-mini/
```

Use the nuScenes download page to get the `v1.0-mini` archive and the lidarseg annotations, then extract the `lidarseg/` folder and the lidarseg-provided `v1.0-*` metadata into the same root.

## Environment

The loader and metrics only need PyTorch, NumPy, tqdm, and `nuscenes-devkit`. PTv3 itself needs Pointcept and its CUDA dependencies.

A typical setup is:

```bash
conda create -n ptv3-nusc python=3.10 -y
conda activate ptv3-nusc

# Install PyTorch matching your CUDA version first.
# Example for CUDA 12.4; adjust for your machine.
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

pip install -r requirements.txt

git clone https://github.com/Pointcept/Pointcept.git
export POINTCEPT_DIR=$PWD/Pointcept
export PYTHONPATH=$POINTCEPT_DIR:$PYTHONPATH
```

Pointcept installation is the fragile part. If `spconv`, `torch-scatter`, or `flash_attn` fail, use Pointcept's Docker/environment file. On machines without FlashAttention, run the evaluation/training scripts with `--disable-flash --patch-size 128`.

## 1. Smoke-test the mini dataloader

```bash
python check_nuscenes_lidarseg.py \
  --dataroot /data/sets/nuscenes \
  --version v1.0-mini \
  --split mini_val
```

You should see two mini-val scenes, point/voxel counts, and a class histogram.

## 2. Evaluate PTv3 on mini-val

Use Pointcept's nuScenes PTv3 config and a nuScenes PTv3 checkpoint if you have one:

```bash
python eval_pointcept_ptv3.py \
  --dataroot /data/sets/nuscenes \
  --version v1.0-mini \
  --split mini_val \
  --config $POINTCEPT_DIR/configs/nuscenes/semseg-pt-v3m1-0-base.py \
  --checkpoint /path/to/model_best.pth \
  --device cuda \
  --save-json runs/ptv3_mini_val_metrics.json \
  --save-predictions runs/ptv3_mini_val_preds
```

If FlashAttention is unavailable:

```bash
python eval_pointcept_ptv3.py \
  --dataroot /data/sets/nuscenes \
  --version v1.0-mini \
  --split mini_val \
  --config $POINTCEPT_DIR/configs/nuscenes/semseg-pt-v3m1-0-base.py \
  --checkpoint /path/to/model_best.pth \
  --device cuda \
  --disable-flash \
  --patch-size 128
```

The output gives mIoU, mAcc, fwIoU, all-accuracy, and per-class IoU/accuracy/support. It evaluates at original-point resolution by using the voxel `inverse` mapping.

## 3. Optional: train/fine-tune on mini_train

For a quick milestone run, start from a pretrained PTv3 checkpoint if possible. To train only the classifier head on mini_train:

```bash
python train_ptv3_mini.py \
  --dataroot /data/sets/nuscenes \
  --version v1.0-mini \
  --config $POINTCEPT_DIR/configs/nuscenes/semseg-pt-v3m1-0-base.py \
  --checkpoint /path/to/model_best.pth \
  --freeze-backbone \
  --ignore-head \
  --epochs 5 \
  --lr 2e-4 \
  --output runs/ptv3_mini_linear_probe
```

To fine-tune the full model on mini_train:

```bash
python train_ptv3_mini.py \
  --dataroot /data/sets/nuscenes \
  --version v1.0-mini \
  --config $POINTCEPT_DIR/configs/nuscenes/semseg-pt-v3m1-0-base.py \
  --checkpoint /path/to/model_best.pth \
  --epochs 3 \
  --lr 1e-5 \
  --output runs/ptv3_mini_finetune
```

Training from scratch on only mini is useful as a pipeline sanity check, but it is not a meaningful baseline for your slides.


## Optional: majority-class sanity baseline

This gives you a lower-bound row for your milestone comparison table:

```bash
python eval_majority_baseline.py \
  --dataroot /data/sets/nuscenes \
  --version v1.0-mini \
  --train-split mini_train \
  --eval-split mini_val
```

## 4. Evaluate saved predictions

If you have official-format prediction files named `{lidar_token}_lidarseg.bin` with uint8 ids 1..16:

```bash
python eval_saved_predictions.py \
  --dataroot /data/sets/nuscenes \
  --version v1.0-mini \
  --split mini_val \
  --pred-dir runs/ptv3_mini_val_preds
```

## What to put on the milestone slide

Minimum quantitative table:

```text
Method                 Split       mIoU   mAcc   fwIoU   Notes
PTv3 LiDAR-only         mini_val    ...    ...    ...     voxel_size=0.05, original-point eval
Majority/debug baseline mini_val    ...    ...    ...     optional sanity baseline
```

Then add a per-class IoU bar/table. Rare classes like bicycle, motorcycle, construction vehicle, and traffic cone will likely be noisy on `v1.0-mini`; discuss that as a limitation rather than over-interpreting it.
