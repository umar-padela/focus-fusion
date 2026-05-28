from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# from Umar's losses file
from train.losses import SegmentationLoss
from models.focus_fusion import FocusFusion, FocusFusionConfig
from models.backbones.ptv3 import PTv3Backbone
from models.backbones.dinov2 import DINOv2Backbone
from eval.metrics import evaluate_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> Dict:
    """Load a YAML config file into a plain dict."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_output_dir(base: str = "experiments/logs") -> Path:
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
# CSVLogger — lightweight alternative to wandb
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
    """Trains FocusFusion (fusion + head only, backbones are frozen).

    Args:
        model: FocusFusion instance (backbones frozen internally).
        train_loader: DataLoader yielding scene-ordered batches.
                      Each batch must include "scene_token" to detect scene
                      boundaries for memory bank resets.
        val_loader: DataLoader for mini_val (same format).
        config: Parsed config dict.
        out_dir: Root output directory.
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
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.out_dir = _make_output_dir(out_dir)

        # Move model
        self.model.to(self.device)

        # Only optimise trainable params (fusion + head + optional bank pos emb)
        trainable = model.trainable_parameters()
        print(
            f"[trainer] Trainable params: "
            f"{sum(p.numel() for p in trainable):,}"
        )

        tc = config.get("train", {})
        self.optimizer = AdamW(
            trainable,
            lr=float(tc.get("lr", 1e-4)),
            weight_decay=float(tc.get("weight_decay", 1e-4)),
        )

        self.num_epochs = int(tc.get("epochs", 50))
        self.val_every = int(tc.get("val_every_epochs", 5))

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=float(tc.get("lr_min", 1e-6)),
        )

        lc = config.get("loss", {})
        self.criterion = SegmentationLoss(
            num_classes=int(lc.get("num_classes", 32)),
            ignore_index=int(lc.get("ignore_index", 0)),
        ).to(self.device)

        self.start_epoch = 0
        self.best_miou = 0.0

        if resume:
            self.start_epoch = _load_checkpoint(
                resume, self.model, self.optimizer, self.scheduler
            )

        self.csv_logger = CSVLogger(self.out_dir / "train_log.csv")

    # ------------------------------------------------------------------

    def train(self) -> None:
        """Full training loop."""
        print(f"[trainer] Starting training "
              f"{self.num_epochs} epochs → {self.out_dir}")

        for epoch in range(self.start_epoch, self.num_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            elapsed = time.time() - t0

            log_row = {
                "epoch": epoch,
                "split": "train",
                "loss": train_metrics["loss"],
                "elapsed_s": f"{elapsed:.1f}",
            }

            if (epoch + 1) % self.val_every == 0:
                val_metrics = self._val_epoch(epoch)
                log_row.update({f"val_{k}": v for k, v in val_metrics.items()})

                if val_metrics.get("mIoU", 0) > self.best_miou:
                    self.best_miou = val_metrics["mIoU"]
                    _save_checkpoint(
                        self.out_dir / "checkpoints" / "best.pt",
                        self.model, self.optimizer, self.scheduler,
                        epoch, val_metrics,
                    )
                    print(f"New best mIoU: {self.best_miou:.4f} (epoch {epoch})")

            _save_checkpoint(
                self.out_dir / "checkpoints" / "latest.pt",
                self.model, self.optimizer, self.scheduler,
                epoch, train_metrics,
            )

            self.csv_logger.log(log_row)
            print(
                f"Epoch {epoch:3d}/{self.num_epochs-1} | "
                f"loss={train_metrics['loss']:.4f} | "
                f"{elapsed:.1f}s"
            )

        self.csv_logger.close()
        print(f"[trainer] Done. Best mIoU = {self.best_miou:.4f}")

    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        # Keep backbones in eval mode (they're frozen, but BN/dropout matter)
        self.model.ptv3.eval()
        self.model.dinov2.eval()

        total_loss = 0.0
        num_batches = 0
        prev_scene = None

        for batch in self.train_loader:
            # --- Scene boundary: reset memory bank ---
            scene_token = batch.get("scene_token")
            if scene_token is not None:
                # scene_token is a list of strings (one per batch item);
                # reset if the first item changed (assumes scene-sorted loader)
                current_scene = scene_token[0] if isinstance(scene_token, list) else scene_token
                if current_scene != prev_scene:
                    self.model.reset_memory()
                    prev_scene = current_scene

            # --- Move tensors to device ---
            batch = self._to_device(batch)

            # --- Forward ---
            self.optimizer.zero_grad()
            out = self.model(batch)
            logits = out["logits"]           # (B, N, C)
            labels = batch["labels"]         # (B, N)

            loss, _ = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping (helps with attention layers)
            nn.utils.clip_grad_norm_(self.model.trainable_parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self.scheduler.step()
        return {"loss": total_loss / max(num_batches, 1)}

    # ------------------------------------------------------------------

    def _val_epoch(self, epoch: int) -> Dict:
        """Run validation and return metric dict.

        Delegates to eval/metrics.py for the actual metric computation so
        Person 1's eval stack can be reused here.
        """
        # Import here to avoid circular deps and allow Person 1 to land this
        # independently
        metrics = evaluate_model(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            num_classes=self.config.get("loss", {}).get("num_classes", 32),
            ignore_index=self.config.get("loss", {}).get("ignore_index", 0),
        )


        # Save snapshot
        metrics_path = self.out_dir / f"val_epoch{epoch:04d}.json"
        with open(metrics_path, "w") as f:
            json.dump({"epoch": epoch, **metrics}, f, indent=2)

        return metrics

    # ------------------------------------------------------------------

    def _to_device(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_model_from_config(config: Dict) -> nn.Module:
    """Construct FocusFusion from a config dict.

    Imports backbone wrappers — if they're not available yet (week 1),
    falls back to DummyFocusFusion from contracts so training can smoke-test.
    """
    mc = config.get("model", {})
    tc_bank = config.get("memory_bank", {})

    cfg = FocusFusionConfig(
        d_lidar=mc.get("d_lidar", 256),
        d_vision=mc.get("d_vision", 768),
        d_model=mc.get("d_model", 256),
        n_heads=mc.get("n_heads", 8),
        attn_dropout=mc.get("attn_dropout", 0.1),
        T=tc_bank.get("T", 1),
        learnable_pos_emb=tc_bank.get("learnable_pos_emb", False),
        num_classes=config.get("loss", {}).get("num_classes", 32),
    )

    ptv3 = PTv3Backbone(config)
    dinov2 = DINOv2Backbone(config)
    return FocusFusion(cfg, ptv3, dinov2)


def build_loaders_from_config(config: Dict, experiment: str):
    """Build train and val DataLoaders.

    Falls back to FakeDataset if real dataset isn't wired up yet.
    """
    try:
        from focus_fusion.datasets.nuscenes import NuScenesDataset
        dc = config.get("data", {})
        train_ds = NuScenesDataset(dc, split="mini_train", T=config["memory_bank"]["T"])
        val_ds = NuScenesDataset(dc, split="mini_val", T=config["memory_bank"]["T"])
    except (ImportError, Exception) as e:
        print(f"[trainer] NuScenesDataset not available ({e}); using FakeDataset")
        from focus_fusion.contracts.dataset import FakeDataset
        mc = config.get("model", {})
        train_ds = FakeDataset(
            num_samples=32,
            N=mc.get("N_points", 8192),
            num_cams=6,
            num_classes=config.get("loss", {}).get("num_classes", 32),
        )
        val_ds = FakeDataset(
            num_samples=8,
            N=mc.get("N_points", 8192),
            num_cams=6,
            num_classes=config.get("loss", {}).get("num_classes", 32),
        )

    tc = config.get("train", {})
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tc.get("batch_size", 2)),
        shuffle=False,   # Must be False — scene ordering matters for memory bank
        num_workers=int(tc.get("num_workers", 4)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tc.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(tc.get("num_workers", 4)),
        pin_memory=True,
    )
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train FocusFusion")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out_dir", default="experiments/logs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    config = _load_config(args.config)

    model = build_model_from_config(config)
    train_loader, val_loader = build_loaders_from_config(config, args.experiment)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        out_dir=args.out_dir,
        device=args.device,
        resume=args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()