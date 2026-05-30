# Hyperparameter Sweep Proposal — E1 (T=1)

Sweeps apply to E1 only. The best config found here gets locked in and used
unchanged for E2 (T=6) — the only difference for E2 is `model.T: 6`.

Baseline for all comparisons is the initial E1 run with `configs/default.yaml`.

---

## Baseline (E1 initial)

| Group | Parameter | Baseline value |
|-------|-----------|---------------|
| Training | lr | 1e-4 |
| Training | weight_decay | 1e-4 |
| Training | lr warmup | none |
| Training | dropout | 0.1 |
| Loss | lovasz_weight | 0.0 (CE only) |
| Loss | class_weights | none (uniform) |
| Architecture | d_f | 256 |
| Architecture | num_heads | 8 |
| Architecture | head_hidden_dim | 128 |
| Input | N_points | 16384 |

---

## Phase 1 — Learning rate (run first, everything else depends on it)

Most impactful single parameter. Run these 3 before anything else.

| Run | lr | weight_decay | Notes |
|-----|----|-------------|-------|
| E1-lr-low | 3e-5 | 1e-4 | conservative |
| E1-lr-base | **1e-4** | 1e-4 | baseline |
| E1-lr-high | 3e-4 | 1e-4 | aggressive |

Pick the run with best val mIoU at epoch 10. Use that lr for all subsequent phases.

---

## Phase 2 — Regularization

Run with best lr from Phase 1.

| Run | dropout | weight_decay | Notes |
|-----|---------|-------------|-------|
| E1-reg-none | 0.0 | 0.0 | no regularization |
| E1-reg-base | **0.1** | **1e-4** | baseline |
| E1-reg-mid | 0.2 | 1e-3 | stronger — nuScenes mini is tiny, may help |
| E1-reg-wd-only | 0.0 | 1e-3 | WD without dropout |

---

## Phase 3 — Loss function

Run with best lr + regularization from Phases 1–2.

| Run | lovasz_weight | class_weights | Notes |
|-----|--------------|--------------|-------|
| E1-loss-ce | **0.0** | none | baseline (CE only) |
| E1-loss-lovasz | 0.5 | none | CE + Lovász mix |
| E1-loss-weighted | 0.0 | inverse_freq | reweight by class frequency |
| E1-loss-both | 0.5 | inverse_freq | both together |

Class weights (inverse_freq): computed from mini_train label distribution.
Useful because nuScenes is heavily dominated by `driveable_surface` and `vegetation` —
rare classes like `bicycle`, `motorcycle`, and `traffic_cone` likely get poor IoU otherwise.

---

## Phase 4 — Architecture

Run with best config from Phases 1–3. Lower priority — architecture is harder to
justify changing post-hoc without more runs.

| Run | d_f | num_heads | head_hidden_dim | head_dim | Notes |
|-----|-----|-----------|----------------|----------|-------|
| E1-arch-small | 128 | 4 | 64 | 32 | leaner |
| E1-arch-base | **256** | **8** | **128** | 32 | baseline |
| E1-arch-wide | 512 | 8 | 256 | 64 | larger head dim |

---

## Phase 5 — LR schedule (run alongside Phase 1 or after)

With only 10 epochs, cosine annealing decays 100× (1e-4 → 1e-6). Worth comparing
against a flat schedule where every epoch gets the same gradient signal.

| Run | schedule | lr | eta_min | Notes |
|-----|----------|----|---------|-------|
| E1-sched-cosine | **cosine** | **1e-4** | **1e-6** | baseline |
| E1-sched-cosine-soft | cosine | 1e-4 | 1e-5 | less aggressive decay |
| E1-sched-constant | constant | 1e-4 | — | flat LR, simplest |

---

## Phase 6 — LR warmup (optional, if Phase 1 shows instability early)

Transformers often benefit from warmup. Only run this if the Phase 1 loss curves
show instability in the first 1–2 epochs.

| Run | warmup_epochs | Notes |
|-----|--------------|-------|
| E1-warm-0 | **0** | baseline |
| E1-warm-1 | 1 | 1-epoch linear warmup |
| E1-warm-2 | 2 | 2-epoch linear warmup |

> **Note:** warmup requires adding a `LinearLR` warmup scheduler to `trainer.py`
> before these runs — not yet implemented.

---

## Decision rule

Pick the config that maximises **val mIoU** at the final epoch across all runs in
a phase. If two runs are within 0.5 mIoU, prefer the simpler / more regularized one
(less likely to be a fluke on the tiny val set).

---

## E2 config (locked after sweeps)

Once the best E1 config is identified:
1. Copy `configs/default.yaml` → `configs/e2.yaml`
2. Change only `model.T: 6`
3. Run E2 with `modal run --detach modal/modal_train.py --experiment e2 --config configs/e2.yaml`

No further hyperparameter changes for E2 — keeping everything fixed isolates the
effect of temporal context.

---

## Running sweeps

Each phase needs additional CLI override support in the trainer. Currently only
`--epochs` is exposed. Before sweeping, add a `--set` flag to `trainer.py` that
accepts `key=value` pairs applied on top of the config:

```powershell
# Example (once --set is implemented):
modal run --detach modal/modal_train.py --experiment e1-lr-high `
  --extra-args "--set train.lr=3e-4 --set wandb.run_name=e1-lr-high"
```

Until then, duplicate `configs/default.yaml` per run and pass `--config`.

W&B dashboard at `wandb.ai/<username>/focus-fusion` — compare all E1-* runs
on the `val/mIoU` metric.
