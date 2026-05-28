"""Thin Pointcept/LitePT wrapper for LiDAR-only segmentation experiments."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch


class LitePTBackbone(torch.nn.Module):
    """Build and run a Pointcept LitePT semantic segmentor.

    The wrapper assumes Pointcept is importable, e.g. by setting
    `PYTHONPATH=/path/to/Pointcept:$PYTHONPATH`.
    """

    def __init__(self, config_path: str, checkpoint_path: str | None = None) -> None:
        super().__init__()
        from pointcept.models import build_model
        from pointcept.utils.config import Config

        self.config_path = str(config_path)
        cfg = Config.fromfile(self.config_path)
        self.model = build_model(cfg.model)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.model(batch)

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Sequence[str], Sequence[str], Sequence[str]]:
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


def strip_common_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
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
