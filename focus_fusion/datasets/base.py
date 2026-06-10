from typing import Protocol


class SegmentationDataset(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int):
        ...
