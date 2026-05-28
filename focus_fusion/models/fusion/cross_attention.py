import torch
import torch.nn as nn
from torch import Tensor


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion: Q from ptv3 points, K/V from DINOv2 patch tokens.

    Uses nn.MultiheadAttention with kdim/vdim for the D_l != D_v case.
    Training path (need_weights=False) uses PyTorch's fused FlashAttention kernel.
    Eval path (return_attn_weights=True) returns per-head weights for visualisation.
    """

    def __init__(
        self,
        d_l: int,
        d_v: int = 384,
        d_f: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        return_attn_weights: bool = False,
    ) -> None:
        super().__init__()
        self.return_attn_weights = return_attn_weights

        # Project Q from D_l to D_f before attention (D_l != D_f in general)
        self.q_proj = nn.Linear(d_l, d_f)
        self.q_norm = nn.LayerNorm(d_f)

        # MHA handles K/V projections internally via kdim/vdim
        self.attn = nn.MultiheadAttention(
            embed_dim=d_f,
            num_heads=num_heads,
            kdim=d_v,
            vdim=d_v,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, q_feats: Tensor, kv_tokens: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            q_feats:   (B, N, D_l) — per-point LiDAR features from ptv3
            kv_tokens: (B, S, D_v) — stacked patch tokens from MemoryBank
        Returns:
            fused:       (B, N, D_f)
            attn_weights:(B, num_heads, N, S) or None
        """
        Q = self.q_norm(self.q_proj(q_feats))  # (B, N, D_f)

        if self.return_attn_weights:
            # need_weights=True disables the fused kernel but returns per-head weights
            out, attn_w = self.attn(
                Q, kv_tokens, kv_tokens,
                need_weights=True,
                average_attn_weights=False,  # keep per-head: (B, num_heads, N, S)
            )
        else:
            # need_weights=False → PyTorch uses fused FlashAttention kernel
            out, _ = self.attn(Q, kv_tokens, kv_tokens, need_weights=False)
            attn_w = None

        return out, attn_w
