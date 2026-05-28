import math
from collections import deque

import torch
import torch.nn as nn
from torch import Tensor


class MemoryBank(nn.Module):
    """FIFO stack of DINOv2 patch embeddings across T timesteps.

    Two usage modes:
    - Training / offline: call forward_preloaded(patch_seq) with a (B,T,6,P,Dv) tensor
      produced by the dataloader. No hidden state; safe with shuffled batches.
    - Streaming inference: push(patches) per timestep, then get_stacked() to retrieve K/V.
    """

    def __init__(
        self,
        T: int = 1,
        num_cameras: int = 6,
        d_v: int = 768,
        use_temporal_pe: bool = False,
        use_camera_pe: bool = False,
    ) -> None:
        super().__init__()
        self.T = T
        self.num_cameras = num_cameras
        self.d_v = d_v
        self.use_temporal_pe = use_temporal_pe
        self.use_camera_pe = use_camera_pe

        if use_camera_pe:
            self.camera_embed = nn.Embedding(num_cameras, d_v)

        # Streaming state (not used during training)
        self._deque: deque[Tensor] = deque(maxlen=T)

    def forward_preloaded(self, patch_seq: Tensor) -> Tensor:
        """Concatenate a full temporal window into K/V tokens.

        Args:
            patch_seq: (B, T, C, P, D_v)  where C = num_cameras
        Returns:
            kv_tokens: (B, T*C*P, D_v)
        """
        B, T, C, P, Dv = patch_seq.shape
        out = patch_seq  # (B, T, C, P, Dv)

        if self.use_camera_pe:
            # camera_embed: (C, Dv) → broadcast to (1, 1, C, 1, Dv)
            cam_idx = torch.arange(C, device=patch_seq.device)
            cam_pe = self.camera_embed(cam_idx).view(1, 1, C, 1, Dv)
            out = out + cam_pe

        if self.use_temporal_pe:
            frame_pe = self._sinusoidal_pe(T, Dv, device=patch_seq.device)
            # frame_pe: (T, Dv) → (1, T, 1, 1, Dv)
            out = out + frame_pe.view(1, T, 1, 1, Dv)

        return out.reshape(B, T * C * P, Dv)

    # ------------------------------------------------------------------
    # Streaming inference API
    # ------------------------------------------------------------------

    def push(self, patches: Tensor) -> None:
        """Push one timestep of patches into the deque.

        Args:
            patches: (B, C, P, D_v)
        """
        self._deque.append(patches.detach())

    def get_stacked(self) -> Tensor:
        """Return stacked K/V from current deque contents.

        Returns:
            kv_tokens: (B, T_actual*C*P, D_v)  — T_actual = len(deque) ≤ T
        """
        if not self._deque:
            raise RuntimeError("MemoryBank deque is empty; call push() first.")
        stacked = torch.stack(list(self._deque), dim=1)  # (B, T_actual, C, P, Dv)
        return self.forward_preloaded(stacked)

    def reset(self) -> None:
        self._deque.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sinusoidal_pe(length: int, d_model: int, device: torch.device) -> Tensor:
        """Standard sinusoidal positional encoding over the frame axis."""
        pe = torch.zeros(length, d_model, device=device)
        position = torch.arange(length, device=device).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe
