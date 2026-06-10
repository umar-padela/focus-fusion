"""FocusFusion training loop.

CLI:
    python -m focus_fusion.train.trainer --config configs/default.yaml --experiment e1
    python -m focus_fusion.train.trainer --config configs/default.yaml --experiment e2

Called remotely via modal/modal_train.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from focus_fusion.train.losses import SegmentationLoss
from focus_fusion.models.focus_fusion import FocusFusion

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> Dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_output_dir(base: str) -> Path:
    out = Path(base)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    return out


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def _load_checkpoint(path: str, model: nn.Module, optimizer, scheduler) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"[trainer] Resumed from epoch {ckpt['epoch']} ({path})")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# CSVLogger
# ---------------------------------------------------------------------------

class CSVLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._writer = None

    def _ensure_open(self, fieldnames):
        if self._writer is None:
            self._file = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            if self.path.stat().st_size == 0:
                self._writer.writeheader()

    def log(self, row: Dict) -> None:
        self._ensure_open(list(row.keys()))
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Trains FocusFusion fusion + head; backbones stay frozen.

    Args:
        model: FocusFusion instance.
        train_loader: DataLoader — scene-sorted (scene_token used for memory resets).
        val_loader: DataLoader for mini_val.
        config: Parsed YAML config dict.
        out_dir: Output root (checkpoints + logs written here).
        device: torch device string.
        resume: Path to checkpoint to resume from.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        out_dir: str = "experiments/logs",
        device: str = "cuda",
        resume: Optional[str] = None,
        experiment: str = "e1",
        use_wandb: bool = True,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.out_dir = _make_output_dir(out_dir)

        self.model.to(self.device)

        trainable = model.trainable_parameters()
        print(f"[trainer] Trainable params: {sum(p.numel() for p in trainable):,}")

        tc = config.get("train", {})
        self.optimizer = AdamW(
            trainable,
            lr=float(tc.get("lr", 1e-4)),
            weight_decay=float(tc.get("weight_decay", 1e-4)),
        )

        self.num_epochs = int(tc.get("epochs", 50))
        self.val_every = int(tc.get("val_every_epochs", 1))

        lr_schedule = tc.get("lr_schedule", "constant")
        warmup_epochs = int(tc.get("warmup_epochs", 0))
        if lr_schedule == "cosine":
            cosine_epochs = max(1, self.num_epochs - warmup_epochs)
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
                eta_min=float(tc.get("lr_min", 1e-6)),
            )
            if warmup_epochs > 0:
                warmup = LinearLR(
                    self.optimizer,
                    start_factor=1.0 / warmup_epochs,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = cosine
        else:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)

        lc = config.get("loss", {})
        self.criterion = SegmentationLoss(
            ignore_index=int(lc.get("ignore_index", -1)),
            lovasz_weight=float(lc.get("lovasz_weight", 0.0)),
        ).to(self.device)

        self.start_epoch = 0
        self.best_miou = 0.0

        if resume:
            self.start_epoch = _load_checkpoint(
                resume, self.model, self.optimizer, self.scheduler
            )

        self.csv_logger = CSVLogger(self.out_dir / "train_log.csv")

        self.wandb = None
        if use_wandb and _WANDB_AVAILABLE:
            wc = config.get("wandb", {})
            run_name = wc.get("run_name") or experiment
            try:
                self.wandb = _wandb.init(
                    project=wc.get("project", "focus-fusion"),
                    entity=wc.get("entity") or None,
                    name=run_name,
                    config=config,
                    resume="allow",
                    dir=str(self.out_dir),
                )
                print(f"[trainer] W&B run: {self.wandb.url}")
            except Exception as e:
                print(f"[trainer] W&B init failed ({e}) — continuing without W&B logging")
                self.wandb = None
        elif use_wandb and not _WANDB_AVAILABLE:
            print("[trainer] wandb not installed — skipping W&B logging")

    # ------------------------------------------------------------------

    def train(self) -> None:
        print(f"[trainer] Starting {self.num_epochs} epochs → {self.out_dir}")

        for epoch in range(self.start_epoch, self.num_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            elapsed = time.time() - t0

            current_lr = self.scheduler.get_last_lr()[0]
            # Use the last global step of this epoch so epoch-level logs are on the
            # same axis as per-batch logs (step must be monotonically increasing in wandb).
            epoch_step = (epoch + 1) * len(self.train_loader)

            log_row: Dict = {
                "epoch": epoch,
                "split": "train",
                "loss": train_metrics["loss"],
                "elapsed_s": f"{elapsed:.1f}",
            }

            wandb_log: Dict = {
                "train/loss": train_metrics["loss"],
                "train/lr": current_lr,
                "epoch": epoch,
            }

            if (epoch + 1) % self.val_every == 0:
                val_metrics = self._val_epoch(epoch)
                log_row.update({f"val_{k}": v for k, v in val_metrics.items()})
                # W&B only accepts scalars — skip arrays (per_class_iou, confusion, etc.)
                wandb_log.update({
                    f"val/{k}": v for k, v in val_metrics.items()
                    if isinstance(v, (int, float))
                })

                if val_metrics.get("mIoU", 0) > self.best_miou:
                    self.best_miou = val_metrics["mIoU"]
                    _save_checkpoint(
                        self.out_dir / "checkpoints" / "best.pt",
                        self.model, self.optimizer, self.scheduler,
                        epoch, val_metrics,
                    )
                    print(f"  New best mIoU: {self.best_miou:.4f} (epoch {epoch})")
                    if self.wandb:
                        self.wandb.summary["best_mIoU"] = self.best_miou
                        self.wandb.summary["best_epoch"] = epoch

            _save_checkpoint(
                self.out_dir / "checkpoints" / "latest.pt",
                self.model, self.optimizer, self.scheduler,
                epoch, train_metrics,
            )

            self.csv_logger.log(log_row)
            if self.wandb:
                self.wandb.log(wandb_log, step=epoch_step)

            print(
                f"Epoch {epoch:3d}/{self.num_epochs-1} | "
                f"loss={train_metrics['loss']:.4f} | lr={current_lr:.2e} | {elapsed:.1f}s"
            )

        self.csv_logger.close()
        if self.wandb:
            self.wandb.finish()
        print(f"[trainer] Done. Best mIoU = {self.best_miou:.4f}")

    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        # Keep frozen backbones in eval mode (affects BN / dropout)
        self.model.dinov2.eval()
        if self.model.litept is not None:
            self.model.litept.eval()

        total_loss = 0.0
        num_batches = 0
        prev_scene = None
        global_step = epoch * len(self.train_loader)

        lc = self.config.get("loss", {})
        num_classes = int(lc.get("num_classes", 16))
        ignore_index = int(lc.get("ignore_index", -1))
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", unit="batch", dynamic_ncols=True)

        for batch in pbar:
            # Scene boundary → reset stateful memory (no-op for preloaded approach)
            scene_token = batch.get("scene_name")
            if scene_token is not None:
                current = scene_token[0] if isinstance(scene_token, list) else scene_token
                if current != prev_scene:
                    self.model.reset_memory()
                    prev_scene = current

            batch = self._to_device(batch)

            self.optimizer.zero_grad()
            out = self.model(batch)

            loss, _ = self.criterion(out, batch)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.trainable_parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate confusion matrix from this batch's predictions (free — reuses logits)
            with torch.no_grad():
                preds = out["logits"].argmax(dim=-1).reshape(-1).cpu()
                gt = batch["labels"].reshape(-1).cpu()
                mask = gt != ignore_index
                preds, gt = preds[mask], gt[mask]
                idx = gt * num_classes + preds
                confusion.reshape(-1).scatter_add_(0, idx, torch.ones_like(idx))

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Running mIoU from cumulative confusion matrix
            inter = confusion.diag()
            union = confusion.sum(1) + confusion.sum(0) - inter
            valid = union > 0
            running_miou = (inter[valid].float() / union[valid].float()).mean().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", mIoU=f"{running_miou:.3f}")

            if self.wandb:
                self.wandb.log({
                    "train/loss_step": loss.item(),
                    "train/mIoU_step": running_miou,
                }, step=global_step)

        self.scheduler.step()
        return {"loss": total_loss / max(num_batches, 1)}

    # ------------------------------------------------------------------

    def _val_epoch(self, epoch: int) -> Dict:
        from focus_fusion.eval.metrics import evaluate_model

        lc = self.config.get("loss", {})
        metrics = evaluate_model(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            num_classes=int(lc.get("num_classes", 32)),
            ignore_index=int(lc.get("ignore_index", -1)),
        )

        out_path = self.out_dir / f"val_epoch{epoch:04d}.json"
        with open(out_path, "w") as f:
            json.dump({"epoch": epoch, **metrics}, f, indent=2)

        return metrics

    # ------------------------------------------------------------------

    def _to_device(self, batch: Dict) -> Dict:
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


# ---------------------------------------------------------------------------
# Config-driven model + dataloader builders
# ---------------------------------------------------------------------------

def build_model_from_config(config: Dict) -> nn.Module:
    """Construct FocusFusion from a YAML config dict."""
    return FocusFusion(config)


def build_loaders_from_config(config: Dict, experiment: str):
    """Build train and val DataLoaders; falls back to FakeLidarSegDataset."""
    T = config.get("model", {}).get("T", 1)

    try:
        from focus_fusion.datasets.nuscenes import NuScenesLidarSegDataset
        dc = config.get("data", {})
        max_scenes = dc.get("max_scenes", None)
        dataroot = dc.get("dataroot", "data")
        version = dc.get("version", "v1.0-mini")
        num_points = config.get("model", {}).get("N_points", 16384)
        img_size = config.get("model", {}).get("img_size", 448)
        seed = int(config.get("seed", 231))
        fraction = float(dc.get("fraction", 1.0))
        val_fraction = float(dc.get("val_fraction", 1.0))
        train_ds = NuScenesLidarSegDataset(
            dataroot=dataroot,
            version=version,
            split=dc.get("train_split", "mini_train"),
            num_points=num_points,
            img_size=img_size,
            T=T,
            fraction=fraction,
            seed=seed,
            verbose=True,
            max_scenes=max_scenes
        )
        val_ds = NuScenesLidarSegDataset(
            dataroot=dataroot,
            version=version,
            split=dc.get("val_split", "mini_val"),
            num_points=num_points,
            img_size=img_size,
            T=T,
            fraction=val_fraction,
            seed=seed,
            verbose=True,
        )
    except Exception as e:
        print(f"[trainer] NuScenesLidarSegDataset not available ({e}); using FakeLidarSegDataset")
        from focus_fusion.datasets.fake import FakeLidarSegDataset
        mc = config.get("model", {})
        train_ds = FakeLidarSegDataset(
            length=16,
            num_points=mc.get("N_points", 256),   # small for smoke run
            num_classes=config.get("loss", {}).get("num_classes", 32),
            img_size=mc.get("img_size", 56),       # small for smoke run; must be div by 14
            T=T,
        )
        val_ds = FakeLidarSegDataset(
            length=4,
            num_points=mc.get("N_points", 256),
            num_classes=config.get("loss", {}).get("num_classes", 32),
            img_size=mc.get("img_size", 56),
            T=T,
        )

    from focus_fusion.datasets.nuscenes import collate_focusfusion

    tc = config.get("train", {})
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tc.get("batch_size", 2)),
        shuffle=False,   # must be False — scene ordering matters
        num_workers=int(tc.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_focusfusion,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tc.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(tc.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_focusfusion,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train FocusFusion")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default="e1",
                        help="Experiment tag (e1=single-frame, e2=temporal)")
    parser.add_argument("--data-root", default=None,
                        help="Override data.dataroot from config")
    parser.add_argument("--output-dir", default=None,
                        help="Override output dir (default: experiments/<experiment>)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override train.epochs from config")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging (useful for local smoke runs)")
    args = parser.parse_args()

    config = _load_config(args.config)

    # CLI overrides
    if args.data_root:
        config.setdefault("data", {})["dataroot"] = args.data_root
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    out_dir = args.output_dir or f"experiments/{args.experiment}"

    model = build_model_from_config(config)
    train_loader, val_loader = build_loaders_from_config(config, args.experiment)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        out_dir=out_dir,
        device=args.device,
        resume=args.resume,
        experiment=args.experiment,
        use_wandb=not args.no_wandb,
    )
    trainer.train()


if __name__ == "__main__":
    main()
