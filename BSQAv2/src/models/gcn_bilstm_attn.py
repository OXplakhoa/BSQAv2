"""
GCN + BiLSTM + Temporal Attention — Full Proposed Model

Combines spatial graph convolution, bidirectional LSTM temporal modeling,
and temporal self-attention for stroke classification and quality scoring.

Architecture:
    (B, T, 17, 2) → SpatialGCN → (B, T, 128)
                   → TemporalBiLSTM → (B, T, 256)
                   → TemporalAttention → context (B, 256)
                   ──┬── Classifier → (B, 5)
                     └── Quality Head → (B, 1)
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from .gcn import SpatialGCN
from .bilstm import TemporalBiLSTM
from .attention import TemporalAttention
from ..config import (
    GCN_HIDDEN_DIM, GCN_NUM_LAYERS, COORD_DIM_VELOCITY,
    BILSTM_HIDDEN_DIM, BILSTM_NUM_LAYERS, BILSTM_DROPOUT,
    ATTENTION_HEADS, ATTENTION_DIM,
    NUM_CLASSES, DROPOUT,
)
from ..data.preprocessing import add_velocity_torch


class GCNBiLSTMAttention(nn.Module):
    """
    Proposed model: GCN + BiLSTM + Multi-Head Temporal Attention.
    Dual output: stroke classification + quality score.
    Exposes attention weights for visualization.
    """

    def __init__(
        self,
        gcn_hidden_dim: int = GCN_HIDDEN_DIM,
        gcn_num_layers: int = GCN_NUM_LAYERS,
        bilstm_hidden_dim: int = BILSTM_HIDDEN_DIM,
        bilstm_num_layers: int = BILSTM_NUM_LAYERS,
        attention_heads: int = ATTENTION_HEADS,
        attention_dim: int = ATTENTION_DIM,
        dropout: float = BILSTM_DROPOUT,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.gcn = SpatialGCN(
            in_channels=COORD_DIM_VELOCITY,
            hidden_dim=gcn_hidden_dim,
            num_layers=gcn_num_layers,
            dropout=dropout,
        )

        self.bilstm = TemporalBiLSTM(
            input_dim=gcn_hidden_dim,
            hidden_dim=bilstm_hidden_dim,
            num_layers=bilstm_num_layers,
            dropout=dropout,
        )

        self.attention = TemporalAttention(
            dim=attention_dim,
            heads=attention_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim // 2, num_classes),
        )

        self.quality_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = add_velocity_torch(x)
        x = self.gcn(x)            # (B, T, 128)
        x = self.bilstm(x)         # (B, T, 256)
        context, attn_weights = self.attention(x)  # (B, 256), (B, T, T)

        class_logits = self.classifier(context)
        quality_score = self.quality_head(context).squeeze(-1)

        if return_attention:
            return class_logits, quality_score, attn_weights
        return class_logits, quality_score, None


if __name__ == "__main__":
    model = GCNBiLSTMAttention()
    x = torch.randn(4, 64, 17, 2)
    class_out, quality_out, attn = model(x, return_attention=True)
    print(f"Input:         {x.shape}")
    print(f"Class logits:  {class_out.shape}")
    print(f"Quality score: {quality_out.shape}")
    print(f"Attn weights:  {attn.shape}")
    assert class_out.shape == (4, 5), f"Expected (4, 5), got {class_out.shape}"
    assert quality_out.shape == (4,), f"Expected (4,), got {quality_out.shape}"
    assert attn.shape == (4, 64, 64), f"Expected (4, 64, 64), got {attn.shape}"
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
    print("GCNBiLSTMAttention test passed!")
