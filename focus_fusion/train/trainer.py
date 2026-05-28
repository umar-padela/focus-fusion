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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from focus_fusion.train.losses import SegmentationLoss
from focus_fusion.models.focus_fusion import FocusFusion


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
        self.val_every = int(tc.get("val_every_epochs", 5))

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=float(tc.get("lr_min", 1e-6)),
        )

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

    # ------------------------------------------------------------------

    def train(self) -> None:
        print(f"[trainer] Starting {self.num_epochs} epochs → {self.out_dir}")

        for epoch in range(self.start_epoch, self.num_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            elapsed = time.time() - t0

            log_row: Dict = {
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
                    print(f"  New best mIoU: {self.best_miou:.4f} (epoch {epoch})")

            _save_checkpoint(
                self.out_dir / "checkpoints" / "latest.pt",
                self.model, self.optimizer, self.scheduler,
                epoch, train_metrics,
            )

            self.csv_logger.log(log_row)
            print(
                f"Epoch {epoch:3d}/{self.num_epochs-1} | "
                f"loss={train_metrics['loss']:.4f} | {elapsed:.1f}s"
            )

        self.csv_logger.close()
        print(f"[trainer] Done. Best mIoU = {self.best_miou:.4f}")

    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        # Keep frozen backbones in eval mode (affects BN / dropout)
        self.model.dinov2.eval()
        if self.model.ptv3 is not None:
            self.model.ptv3.eval()

        total_loss = 0.0
        num_batches = 0
        prev_scene = None

        for batch in self.train_loader:
            # Scene boundary → reset stateful memory (no-op for preloaded approach)
            scene_token = batch.get("scene_token")
            if scene_token is not None:
                current = scene_token[0] if isinstance(scene_token, list) else scene_token
                if current != prev_scene:
                    self.model.reset_memory()
                    prev_scene = current

            batch = self._to_device(batch)

            self.optimizer.zero_grad()
            out = self.model(batch)

            # SegmentationLoss expects (output_dict, batch_dict)
            loss, _ = self.criterion(out, batch)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.trainable_parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

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
        dataroot = dc.get("dataroot", "data")
        num_points = config.get("model", {}).get("N_points", 16384)
        img_size = config.get("model", {}).get("img_size", 448)
        train_ds = NuScenesLidarSegDataset(
            dataroot=dataroot, split="mini_train",
            num_points=num_points, img_size=img_size, T=T,
        )
        val_ds = NuScenesLidarSegDataset(
            dataroot=dataroot, split="mini_val",
            num_points=num_points, img_size=img_size, T=T,
        )
    except Exception as e:
        print(f"[trainer] NuScenesLidarSegDataset not available ({e}); using FakeLidarSegDataset")
        from focus_fusion.datasets.fake import FakeLidarSegDataset
        mc = config.get("model", {})
        train_ds = FakeLidarSegDataset(
            num_samples=32,
            num_points=mc.get("N_points", 8192),
            num_classes=config.get("loss", {}).get("num_classes", 32),
            T=T,
        )
        val_ds = FakeLidarSegDataset(
            num_samples=8,
            num_points=mc.get("N_points", 8192),
            num_classes=config.get("loss", {}).get("num_classes", 32),
            T=T,
        )

    tc = config.get("train", {})
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tc.get("batch_size", 2)),
        shuffle=False,   # must be False — scene ordering matters
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
    args = parser.parse_args()

    config = _load_config(args.config)

    # CLI overrides
    if args.data_root:
        config.setdefault("data", {})["dataroot"] = args.data_root
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
    )
    trainer.train()


if __name__ == "__main__":
    main()
