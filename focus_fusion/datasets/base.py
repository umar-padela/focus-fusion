"""Dataset protocol shared by training and evaluation code."""

from __future__ import annotations

from typing import Protocol


class SegmentationDataset(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int):
        ...
