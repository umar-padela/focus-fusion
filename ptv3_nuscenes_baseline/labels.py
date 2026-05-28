"""nuScenes-lidarseg label utilities.

The raw lidarseg files store the 32 general nuScenes lidarseg category ids.
For semantic-segmentation baselines we evaluate on the official 16 lidarseg
challenge classes and ignore the void/rare classes.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

IGNORE_INDEX = -1

# Internal training/eval ids are 0..15.  Official lidarseg challenge ids are
# 1..16, with 0 reserved for void/ignore.  This list is internal order.
CLASS_NAMES: Sequence[str] = (
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
)

# Map raw nuScenes-lidarseg *general* class names to official challenge ids.
# Challenge id 0 means void/ignore.  Non-void classes map to 1..16.
OFFICIAL_CHALLENGE_ID_BY_GENERAL_NAME: Mapping[str, int] = {
    # void / ignore
    "noise": 0,
    "animal": 0,
    "human.pedestrian.personal_mobility": 0,
    "human.pedestrian.stroller": 0,
    "human.pedestrian.wheelchair": 0,
    "movable_object.debris": 0,
    "movable_object.pushable_pullable": 0,
    "static_object.bicycle_rack": 0,
    "vehicle.emergency.ambulance": 0,
    "vehicle.emergency.police": 0,
    "static.other": 0,
    "vehicle.ego": 0,
    # 16 challenge classes
    "movable_object.barrier": 1,
    "vehicle.bicycle": 2,
    "vehicle.bus.bendy": 3,
    "vehicle.bus.rigid": 3,
    "vehicle.car": 4,
    "vehicle.construction": 5,
    "vehicle.motorcycle": 6,
    "human.pedestrian.adult": 7,
    "human.pedestrian.child": 7,
    "human.pedestrian.construction_worker": 7,
    "human.pedestrian.police_officer": 7,
    "movable_object.trafficcone": 8,
    "vehicle.trailer": 9,
    "vehicle.truck": 10,
    "flat.driveable_surface": 11,
    "flat.other": 12,
    "flat.sidewalk": 13,
    "flat.terrain": 14,
    "static.manmade": 15,
    "static.vegetation": 16,
}


def _idx_name_mapping_from_nusc(nusc) -> Dict[int, str]:
    """Read the devkit's raw lidarseg id -> class-name mapping robustly."""
    if hasattr(nusc, "lidarseg_idx2name_mapping"):
        return {int(k): str(v) for k, v in nusc.lidarseg_idx2name_mapping.items()}
    if hasattr(nusc, "lidarseg_name2idx_mapping"):
        return {int(v): str(k) for k, v in nusc.lidarseg_name2idx_mapping.items()}

    # Fallback for older/custom devkit objects.  The official devkit exposes the
    # mappings above, but this keeps the loader easy to debug if APIs differ.
    mapping: Dict[int, str] = {}
    for category in getattr(nusc, "category", []):
        name = category.get("name")
        idx = category.get("index", category.get("id"))
        if name is not None and idx is not None:
            mapping[int(idx)] = str(name)
    if not mapping:
        raise RuntimeError(
            "Could not find a raw lidarseg id -> class-name mapping on the NuScenes object. "
            "Expected nusc.lidarseg_idx2name_mapping or nusc.lidarseg_name2idx_mapping."
        )
    return mapping


def build_learning_map(nusc, ignore_index: int = IGNORE_INDEX, lut_size: int = 256) -> np.ndarray:
    """Build a vectorized lookup table from raw lidarseg ids to internal ids.

    Returns:
        np.ndarray of shape [lut_size], dtype int64. Raw labels that map to void
        or are unknown become ``ignore_index``. Non-void labels become 0..15.
    """
    idx_to_name = _idx_name_mapping_from_nusc(nusc)
    max_idx = max(idx_to_name) if idx_to_name else 0
    lut = np.full(max(lut_size, max_idx + 1), ignore_index, dtype=np.int64)

    missing_names = []
    for raw_idx, name in idx_to_name.items():
        if name not in OFFICIAL_CHALLENGE_ID_BY_GENERAL_NAME:
            # Leave unknown classes ignored, but report them to make mistakes visible.
            missing_names.append(name)
            continue
        official_id = OFFICIAL_CHALLENGE_ID_BY_GENERAL_NAME[name]
        lut[raw_idx] = ignore_index if official_id == 0 else official_id - 1

    if missing_names:
        print(
            "[labels] Warning: these raw lidarseg classes were not in the official mapping "
            f"and will be ignored: {sorted(set(missing_names))}"
        )
    return lut


def remap_raw_labels(raw_labels: np.ndarray, learning_map: np.ndarray, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Vectorized remap from raw uint8 lidarseg labels to internal 0..15/-1 labels."""
    raw = raw_labels.astype(np.int64, copy=False)
    out = np.full(raw.shape, ignore_index, dtype=np.int64)
    valid = (raw >= 0) & (raw < len(learning_map))
    out[valid] = learning_map[raw[valid]]
    return out


def internal_to_official_ids(pred_internal: np.ndarray) -> np.ndarray:
    """Convert internal class ids 0..15 to official challenge ids 1..16.

    The official result format should not contain the void class.  Any invalid
    internal prediction is clipped to class id 1 rather than producing 0.
    """
    pred = np.asarray(pred_internal, dtype=np.int64)
    pred = np.clip(pred, 0, len(CLASS_NAMES) - 1)
    return (pred + 1).astype(np.uint8)


def official_to_internal_ids(pred_official: np.ndarray, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Convert official ids 1..16 to internal ids 0..15; official 0 -> ignore."""
    pred = np.asarray(pred_official, dtype=np.int64)
    out = np.full(pred.shape, ignore_index, dtype=np.int64)
    valid = (pred >= 1) & (pred <= len(CLASS_NAMES))
    out[valid] = pred[valid] - 1
    return out
