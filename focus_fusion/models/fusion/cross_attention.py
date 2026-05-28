import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttentionFusion(nn.Module):
    """Global learned cross-attention between LiDAR point features and vision patch tokens.

    Q = ptv3 per-point features (B, N, D_l)
    K = V = stacked DINOv2 patch tokens from MemoryBank (B, S, D_v), S = T*6*P

    Two forward paths:
    - Training (return_attn_weights=False): uses F.scaled_dot_product_attention
      (FlashAttention-2 kernel on PyTorch >= 2.0) — never materialises the full
      N×S attention matrix, required for T=6 scale.
    - Eval (return_attn_weights=True): manual softmax path that returns the full
      (B, H, N, S) weight tensor for attention visualisation.
    """

    def __init__(
        self,
        d_l: int,
        d_v: int = 768,
        d_f: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        return_attn_weights: bool = False,
    ) -> None:
        super().__init__()
        if d_f % num_heads != 0:
            raise ValueError(f"d_f={d_f} must be divisible by num_heads={num_heads}")

        self.d_f = d_f
        self.num_heads = num_heads
        self.d_head = d_f // num_heads
        self.scale = self.d_head ** -0.5
        self.return_attn_weights = return_attn_weights

        self.q_proj = nn.Linear(d_l, d_f)
        self.k_proj = nn.Linear(d_v, d_f)
        self.v_proj = nn.Linear(d_v, d_f)
        self.out_proj = nn.Linear(d_f, d_f)

        self.q_norm = nn.LayerNorm(d_f)
        self.k_norm = nn.LayerNorm(d_f)

        self.attn_drop = nn.Dropout(dropout)

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
        B, N, _ = q_feats.shape
        S = kv_tokens.shape[1]

        Q = self.q_norm(self.q_proj(q_feats))    # (B, N, D_f)
        K = self.k_norm(self.k_proj(kv_tokens))  # (B, S, D_f)
        V = self.v_proj(kv_tokens)               # (B, S, D_f)

        # Reshape to multi-head: (B, H, seq, d_head)
        Q = Q.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.d_head).transpose(1, 2)

        if self.return_attn_weights:
            # Manual path — materialises the full attention matrix for export
            attn_logits = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N, S)
            attn_w = attn_logits.softmax(dim=-1)
            out = self.attn_drop(attn_w) @ V                       # (B, H, N, d_head)
        else:
            # FlashAttention-2 path — required for T=6 scale
            dropout_p = self.attn_drop.p if self.training else 0.0
            out = F.scaled_dot_product_attention(Q, K, V, dropout_p=dropout_p)
            attn_w = None

        out = out.transpose(1, 2).contiguous().view(B, N, self.d_f)
        return self.out_proj(out), attn_w
