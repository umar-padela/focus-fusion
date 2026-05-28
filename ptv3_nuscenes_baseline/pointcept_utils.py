"""Small helpers for using Pointcept/PTv3 from standalone scripts."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def load_pointcept_config(config_file: str):
    try:
        from pointcept.utils.config import Config
    except Exception as exc:  # pragma: no cover - only runs in user's env
        raise ImportError(
            "Could not import Pointcept. Clone Pointcept and export PYTHONPATH=/path/to/Pointcept:$PYTHONPATH."
        ) from exc
    return Config.fromfile(config_file)


def patch_ptv3_config(
    cfg,
    *,
    disable_flash: bool = False,
    patch_size: Optional[int] = None,
    freeze_backbone: bool = False,
):
    """Patch a Pointcept config object for laptop/single-GPU runs."""
    if hasattr(cfg, "model"):
        model_cfg = cfg.model
    else:
        model_cfg = cfg["model"]

    if freeze_backbone:
        model_cfg["freeze_backbone"] = True

    backbone = model_cfg.get("backbone", None)
    if backbone is not None:
        if disable_flash:
            backbone["enable_flash"] = False
        if patch_size is not None:
            patch_size = int(patch_size)
            # These exist in PTv3 configs; use current tuple lengths if present.
            if "enc_patch_size" in backbone:
                backbone["enc_patch_size"] = tuple([patch_size] * len(backbone["enc_patch_size"]))
            if "dec_patch_size" in backbone:
                backbone["dec_patch_size"] = tuple([patch_size] * len(backbone["dec_patch_size"]))
    return cfg


def build_pointcept_model(cfg, device: torch.device):
    try:
        from pointcept.models import build_model
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Could not import pointcept.models.build_model. Ensure Pointcept is on PYTHONPATH and its dependencies are installed."
        ) from exc
    model = build_model(cfg.model)
    return model.to(device)


def _strip_common_prefixes(key: str) -> str:
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
        # It may already be a raw state dict.
        if all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError(
        "Could not extract a model state_dict. Expected a dict with one of "
        "state_dict/model_state_dict/model/net, or a raw state_dict."
    )


def load_checkpoint_flexible(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    device: torch.device,
    ignore_substrings: Sequence[str] = (),
) -> Tuple[List[str], List[str], List[str]]:
    """Load a checkpoint, stripping common DDP prefixes and skipping mismatched shapes.

    Returns:
        missing_keys, unexpected_keys, dropped_keys
    """
    if not checkpoint_path:
        return [], [], []
    ckpt = torch.load(checkpoint_path, map_location=device)
    raw_state = extract_state_dict(ckpt)
    model_state = model.state_dict()
    filtered = OrderedDict()
    dropped: List[str] = []

    for raw_key, value in raw_state.items():
        key = _strip_common_prefixes(raw_key)
        if any(s in key for s in ignore_substrings):
            dropped.append(raw_key)
            continue
        if key not in model_state:
            dropped.append(raw_key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            dropped.append(raw_key)
            continue
        filtered[key] = value

    load_info = model.load_state_dict(filtered, strict=False)
    missing = list(load_info.missing_keys)
    unexpected = list(load_info.unexpected_keys)
    return missing, unexpected, dropped


def to_device_input(batch: Dict[str, object], device: torch.device, include_segment: bool = False):
    keys = ["coord", "grid_coord", "feat", "offset"]
    if include_segment:
        keys.append("segment")
    out = {}
    for key in keys:
        value = batch[key]
        if hasattr(value, "to"):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def get_logits(output):
    if isinstance(output, dict):
        for key in ("seg_logits", "logits", "prediction"):
            if key in output:
                return output[key]
    if torch.is_tensor(output):
        return output
    raise RuntimeError(f"Could not find segmentation logits in model output of type {type(output)}")
