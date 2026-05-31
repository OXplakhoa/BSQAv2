"""
Temporal Multi-Head Self-Attention

Learns which frames are most important for the stroke classification.
Output attention weights are visualizable for explainability.

Architecture:
    (B, T, dim=256) → MultiHeadAttention(Q,K,V) → (B, T, 256)
                     → Residual + LayerNorm
                     → Global mean pool over T → (B, 256)
"""
import torch
import torch.nn as nn
from typing import Tuple
from ..config import ATTENTION_HEADS, ATTENTION_DIM, DROPOUT


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over the temporal dimension.
    Highlights critical frames (e.g., impact moment) with high attention weights.

    Input:  (B, T, dim)  — BiLSTM output
    Output: (B, dim)      — pooled context vector
    """

    def __init__(
        self,
        dim: int = ATTENTION_DIM,
        heads: int = ATTENTION_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.dim = dim

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = x.mean(dim=1)
        return x, attn_weights


if __name__ == "__main__":
    model = TemporalAttention(dim=256, heads=4)
    x = torch.randn(4, 64, 256)
    context, weights = model(x)
    print(f"Input:           {x.shape}")
    print(f"Context:         {context.shape}")
    print(f"Attn weights:    {weights.shape}")
    assert context.shape == (4, 256), f"Expected (4, 256), got {context.shape}"
    assert weights.shape == (4, 64, 64), f"Expected (4, 64, 64), got {weights.shape}"
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
    print("TemporalAttention test passed!")
