"""
FIFO stack of DINOv2 patch embeddings.

Holds the last T timesteps of multi-camera patch embeddings and returns a
flat (B, T*P, D_v) tensor for use as K and V in cross-attention.

T=1  -> current frame only (E1)
T=6  -> 3 s of history at nuScenes 2 Hz keyframe rate (E2)
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn


class MemoryBank(nn.Module):
    """FIFO stack of DINOv2 patch embeddings across T timesteps.

    Usage:

        bank = MemoryBank(T=6, d_vision=768)

        # Each call to push() adds one timestep's worth of patch embeddings.
        # patches: (B, num_cams, P, D_v)  — one keyframe, all cameras
        bank.push(patches)

        # Retrieve stacked K/V tokens: (B, T*P_total, D_v)
        kv = bank.get_kv()

    The bank persists across push() calls within a scene.
    Call reset() when starting a new scene / sequence.

    Args:
        T: Stack depth (number of timesteps to keep).
        d_vision: Patch embedding dimension (must match DINOv2 output).
        learnable_pos_emb: If True, add learned position embeddings indexed by
                           (timestep_offset, camera, patch) before returning KV.
                           Defaults to False to keep the interface simple.
    """

    def __init__(
        self,
        T: int = 6,
        d_vision: int = 768,
        learnable_pos_emb: bool = False,
    ) -> None:
        super().__init__()

        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")

        self.T = T
        self.d_vision = d_vision
        self.learnable_pos_emb = learnable_pos_emb

        # Each slot: (B, P_total, D_v)  where P_total = num_cams * P_per_cam
        self._buffer: deque[torch.Tensor] = deque(maxlen=T)

        # Optional learned temporal offset embedding (T offsets)
        if learnable_pos_emb:
            # Index 0 = oldest frame still in buffer, T-1 = most recent.
            self.temporal_emb = nn.Embedding(T, d_vision)
        else:
            self.temporal_emb = None

    # Public API -----------------------------------------------------------------

    def reset(self) -> None:
        """Clear the buffer (call at the start of every new scene)."""
        self._buffer.clear()

    def push(self, patches: torch.Tensor) -> None:
        """Add one timestep of patch embeddings to the bank.

        Args:
            patches: Tensor of shape (B, num_cams, P, D_v) **or**
                     (B, P_total, D_v) if cameras were already flattened.
                     The bank flattens cameras internally.
        """
        if patches.dim() == 4:
            # (B, num_cams, P, D_v) -> (B, num_cams*P, D_v)
            B, num_cams, P, D = patches.shape
            patches = patches.reshape(B, num_cams * P, D)

        if patches.shape[-1] != self.d_vision:
            raise ValueError(
                f"Expected d_vision={self.d_vision}, got {patches.shape[-1]}"
            )

        # Detach before storing — don't backprop through frozen DINOv2
        self._buffer.append(patches.detach())

    def get_kv(self) -> torch.Tensor:
        """Return all buffered patches concatenated along the token dimension.

        Returns:
            Tensor of shape (B, L, D_v) where L = len(buffer) * P_total.
            If the buffer is empty, returns a zero tensor of shape (B, 0, D_v);
            callers should handle this edge case (e.g. skip fusion).

        Raises:
            RuntimeError: If tensors in the buffer have inconsistent batch sizes.
        """
        if len(self._buffer) == 0:
            # Return an empty placeholder — caller decides what to do
            return torch.zeros(0, 0, self.d_vision)

        frames = list(self._buffer)  # oldest first

        if self.temporal_emb is not None:
            # Align embedding indices: oldest = 0, newest = len-1
            offset = self.T - len(frames)
            frames = [
                f + self.temporal_emb.weight[offset + i]  # broadcast over B, P
                for i, f in enumerate(frames)
            ]

        # Concatenate along token dim: (B, T*P_total, D_v)
        kv = torch.cat(frames, dim=1)
        return kv

    # Convenience -----------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Current number of frames stored (≤ T)."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.T

    def __repr__(self) -> str:
        return (
            f"MemoryBank(T={self.T}, d_vision={self.d_vision}, "
            f"depth={self.depth}, learnable_pos_emb={self.learnable_pos_emb})"
        )