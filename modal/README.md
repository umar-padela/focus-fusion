# Modal (remote train/eval)

GPU work runs on [Modal](https://modal.com/), not locally.

## One-time setup

```powershell
conda activate cs224r
pip install modal
modal token new   # browser opens, log in once
```

Create the W&B secret (used for logging):
```powershell
modal secret create wandb WANDB_API_KEY=<your_key>
```

Upload nuScenes mini to the data volume (one-time, ~4 GB):
```powershell
# Download nuScenes mini locally first, then:
modal volume put focus-fusion-data data/nuscenes/ nuscenes/
```

## Commands (run from repo root)

| Command | Purpose |
|---------|---------|
| `modal run modal/modal_smoke.py` | Infra smoke test: GPU + DINOv2 forward |
| `modal run modal/modal_train.py --experiment e1` | Train E1 (T=1 single frame) |
| `modal run modal/modal_train.py --experiment e2` | Train E2 (T=6 temporal, Person 2) |
| `modal run modal/modal_eval.py --experiment e1` | Eval E1 on mini_val |
| `modal run modal/modal_eval.py --experiment e2 --split mini_val` | Eval E2 |
| `modal run modal/modal_eval.py --experiment e1 --checkpoint /experiments/e1/epoch_40.pt` | Eval specific checkpoint |

Add `--detach` **before** the script path to keep the job running after Ctrl-C:
```powershell
modal run --detach modal/modal_train.py --experiment e2
```

## GPUs

| Function | Default | Override (Windows) |
|----------|---------|-------------------|
| `smoke`, `evaluate` | A10G | `$env:MODAL_GPU_EVAL="A10G"` |
| `train` | A100-40GB | `$env:MODAL_GPU_TRAIN="A100-80GB"` |

## Volume

One volume holds everything:

| Volume | Mount | Contents |
|--------|-------|---------|
| `focus-fusion-experiments` | `/experiments` | `data/` — nuScenes mini; `e1/`, `e2/` — checkpoints & logs |

Browse / download (paths relative to volume root, no leading `/experiments`):
```powershell
modal volume ls  focus-fusion-experiments
modal volume ls  focus-fusion-experiments data
modal volume ls  focus-fusion-experiments e1
modal volume get focus-fusion-experiments e1/best.pt best.pt
```

## Files

| File | Role |
|------|------|
| `modal_config.py` | App, Image, Volumes, `smoke` / `train` / `evaluate` functions |
| `modal_smoke.py` | CLI → `smoke` (infra health check) |
| `modal_train.py` | CLI → `train` (E1 or E2) |
| `modal_eval.py` | CLI → `evaluate` (mini_val or mini_train) |

## Pending (block on Person 1)

The `train` and `evaluate` functions call `focus_fusion.train.trainer` and
`focus_fusion.eval.metrics` — these modules are not yet implemented (Week 2).

The Modal image includes a `TODO` comment for ptv3 CUDA ops installation.
Once Person 1 confirms their build steps, add a `.run_commands()` layer to the
image in `modal_config.py` before `add_local_dir`.
