import math

import torch
import torch.nn as nn
from torch import Tensor


class MemoryBank(nn.Module):
    """Stacks DINOv2 patch embeddings from T timesteps into K/V tokens for cross-attention.

    Takes a preloaded temporal window (B, T, 6, P, D_v) from the dataloader
    and flattens it to (B, T*6*P, D_v).
    """

    def __init__(
        self,
        T: int = 1,
        num_cameras: int = 6,
        d_v: int = 384,
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

    def forward(self, patch_seq: Tensor) -> Tensor:
        """Flatten a temporal window of patch embeddings into K/V tokens.

        Args:
            patch_seq: (B, T, C, P, D_v)  where C = num_cameras
        Returns:
            kv_tokens: (B, T*C*P, D_v)
        """
        B, T, C, P, Dv = patch_seq.shape
        out = patch_seq

        if self.use_camera_pe:
            cam_idx = torch.arange(C, device=patch_seq.device)
            cam_pe = self.camera_embed(cam_idx).view(1, 1, C, 1, Dv)
            out = out + cam_pe

        if self.use_temporal_pe:
            frame_pe = self._sinusoidal_pe(T, Dv, device=patch_seq.device)
            out = out + frame_pe.view(1, T, 1, 1, Dv)

        return out.reshape(B, T * C * P, Dv)

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
