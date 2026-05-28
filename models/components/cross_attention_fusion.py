"""

Q  = per-point features from ptv3: (B, N, D_l)
K,V = stacked DINOv2 patch tokens: (B, L, D_v)   L = T * P_total

Uses F.scaled_dot_product_attention which dispatches to FlashAttention on
CUDA with fp16/bf16 — no extra dependencies required (PyTorch 2.0+).

When return_attn_weights=True (eval / visualisation only) we fall back to
manual matmul because Flash does not expose attention weights.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention fusing LiDAR point features with vision tokens.

    Training path — F.scaled_dot_product_attention → FlashAttention on CUDA+fp16/bf16.
    Eval/viz path — manual matmul when return_attn_weights=True (weights not available
                     from Flash); only used for a handful of visualisation samples.

    Args:
        d_lidar: ptv3 per-point feature dim (Q input).
        d_vision: DINOv2 patch embedding dim (K/V input).
        d_model: Hidden / output dim. Must be divisible by n_heads.
        n_heads: Number of attention heads.
        dropout: Attention dropout (training only; ignored by Flash in eval).
        return_attn_weights: If True forward() returns (B, N, L) weights. Forces
                             the slower manual path — only enabling for visualisation.
    """

    def __init__(
        self,
        d_lidar: int = 256,
        d_vision: int = 768,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        return_attn_weights: bool = False,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.return_attn_weights = return_attn_weights

        self.q_proj = nn.Linear(d_lidar, d_model, bias=False)
        self.k_proj = nn.Linear(d_vision, d_model, bias=False)
        self.v_proj = nn.Linear(d_vision, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.residual_proj = nn.Linear(d_lidar, d_model, bias=False) if d_lidar != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)

    def _split_heads(self, x: torch.Tensor, seq: int) -> torch.Tensor:
        """(B, seq, D) -> (B, H, seq, d_head)"""
        B = x.shape[0]
        return x.view(B, seq, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        lidar_feats: torch.Tensor,
        vision_tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            lidar_feats: (B, N, D_l)
            vision_tokens: (B, L, D_v)
            key_padding_mask: (B, L) bool — True = ignore that token.

        Returns:
            fused: (B, N, D_model)
            attn_weights: (B, N, L) averaged over heads, or None.
        """
        B, N, _ = lidar_feats.shape
        L = vision_tokens.shape[1]

        Q = self._split_heads(self.q_proj(lidar_feats), N)   # (B, H, N, d_head)
        K = self._split_heads(self.k_proj(vision_tokens), L) # (B, H, L, d_head)
        V = self._split_heads(self.v_proj(vision_tokens), L) # (B, H, L, d_head)

        if self.return_attn_weights:
            # Manual path (eval/visualisation)
            scale = math.sqrt(self.d_head)
            attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / scale, dim=-1)
            if key_padding_mask is not None:
                attn = attn.masked_fill(key_padding_mask[:, None, None, :], 0.0)
            attended = torch.matmul(attn, V)
            weights = attn.mean(dim=1)  # (B, N, L)
        else:
            # Flash path (training + standard eval)
            # Convert bool padding mask -> additive float mask expected by sdpa
            attn_mask = None
            if key_padding_mask is not None:
                # (B, 1, 1, L) — broadcast over heads and query positions
                attn_mask = key_padding_mask[:, None, None, :].expand(B, self.n_heads, N, L)
                attn_mask = attn_mask.to(dtype=Q.dtype) * torch.finfo(Q.dtype).min

            attended = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
            weights = None

        # (B, H, N, d_head) -> (B, N, D)
        attended = attended.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.norm(self.out_proj(attended) + self.residual_proj(lidar_feats))
        return out, weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_head={self.d_head}, return_attn_weights={self.return_attn_weights}"
        )