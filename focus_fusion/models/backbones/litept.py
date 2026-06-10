"""LitePT backbone wrapper for FocusFusion.

Wraps Pointcept's DefaultSegmentorV2 to extract per-point encoder features.

Voxelisation is handled by the dataloader.

    """

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch


class LitePTBackbone(torch.nn.Module):
    """Frozen LitePT encoder — returns per-point backbone features (B, N, 72).

    Expects the batch to contain pre-voxelised Pointcept keys produced by
    collate_focusfusion(): coord, feat, grid_coord, offset, inverse.

    Args:
        config_path:     Path to the Pointcept model config (.py file).
        checkpoint_path: Optional path to model_best.pth weights.
        disable_flash:   Set enable_flash=False in the config (avoids flash_attn dep).
        grid_size:       Voxel edge length used during dataset voxelisation (metres).
    """

    BACKBONE_OUT_CHANNELS: int = 72

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str | None = None,
        disable_flash: bool = True,
        grid_size: float = 0.05,
    ) -> None:
        super().__init__()
        from pointcept.models import build_model
        from pointcept.utils.config import Config

        self.config_path = str(config_path)
        self.grid_size = grid_size

        cfg = Config.fromfile(self.config_path)
        if disable_flash:
            _disable_flash(cfg.model)
        self.model = build_model(cfg.model)

        if checkpoint_path:
            missing, unexpected, dropped = self.load_checkpoint(checkpoint_path)
            print(f"[LitePTBackbone] Loaded {checkpoint_path}")
            if missing:
                print(f"  missing keys : {len(missing)}")
            if dropped:
                print(f"  dropped keys : {len(dropped)} (shape mismatch)")

        self.requires_grad_(False)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract per-point LitePT backbone features.

        Args:
            batch: FocusFusion batch — must contain Pointcept sparse keys
                   (coord, feat, grid_coord, offset, inverse) produced by
                   collate_focusfusion() in the dataloader.
        Returns:
            (B, N, BACKBONE_OUT_CHANNELS=72) float32
        """
        Point = _import_point_class()

        point = Point({
            "coord":      batch["coord"],
            "feat":       batch["feat"],
            "grid_coord": batch["grid_coord"],
            "offset":     batch["offset"],
            "grid_size":  self.grid_size,
        })

        # Run backbone only, skip seg_head to get encoder features
        point = self.model.backbone(point)
        vox_feats = point.feat          # (total_vox, 72)

        # Expand voxel features back to per-point using precomputed inverse (B, N)
        B, N = batch["inverse"].shape
        offsets = batch["offset"]
        vox_starts = torch.cat([
            torch.zeros(1, dtype=torch.long, device=offsets.device),
            offsets[:-1],
        ])

        result_parts = []
        for b in range(B):
            s = int(vox_starts[b])
            e = int(offsets[b])
            inv = batch["inverse"][b]               # (N,) local voxel indices
            result_parts.append(vox_feats[s:e][inv])

        return torch.stack(result_parts)            # (B, N, 72)

    def load_checkpoint(
        self, checkpoint_path: str
    ) -> Tuple[Sequence[str], Sequence[str], Sequence[str]]:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state = extract_state_dict(ckpt)
        model_state = self.model.state_dict()
        filtered = OrderedDict()
        dropped = []
        for raw_key, value in state.items():
            key = strip_common_prefixes(raw_key)
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
                filtered[key] = value
            else:
                dropped.append(raw_key)
        load_info = self.model.load_state_dict(filtered, strict=False)
        return list(load_info.missing_keys), list(load_info.unexpected_keys), dropped

    @classmethod
    def from_default_layout(
        cls,
        pointcept_root: str = "third_party/pointcept",
        checkpoint_root: str = "checkpoints",
    ) -> "LitePTBackbone":
        config = Path(pointcept_root) / "configs/nuscenes/semseg-litept-v1m1-0-small.py"
        checkpoint = (
            Path(checkpoint_root)
            / "LitePT/nuscenes-semseg-litept-small-v1m1/model/model_best.pth"
        )
        return cls(str(config), str(checkpoint))


def _import_point_class():
    """Import Pointcept's Point class — handles different repo layouts."""
    for module_path in (
        "pointcept.models.utils.structure",  # current Pointcept layout
        "pointcept.utils.structure",
        "pointcept.utils.point",
        "pointcept.models.utils",
    ):
        try:
            import importlib
            mod = importlib.import_module(module_path)
            return mod.Point
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "Could not import Point from Pointcept. "
        "Ensure Pointcept is on PYTHONPATH: "
        "export PYTHONPATH=/path/to/Pointcept:$PYTHONPATH"
    )


def _disable_flash(obj) -> None:
    """Recursively set enable_flash=False in a Pointcept config object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "enable_flash":
                obj[k] = False
            else:
                _disable_flash(v)
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if k == "enable_flash":
                setattr(obj, k, False)
            else:
                _disable_flash(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _disable_flash(item)


def strip_common_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def extract_state_dict(checkpoint_obj) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for field in ("state_dict", "model_state_dict", "model", "net"):
            if field in checkpoint_obj and isinstance(checkpoint_obj[field], dict):
                return checkpoint_obj[field]
        if all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError("Could not extract a model state_dict from checkpoint.")
