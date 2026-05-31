"""
GCN + (Bi)LSTM Ablation Model (no attention)

Combines spatial GCN with (bidirectional) LSTM for temporal modeling.
Used in the ablation study to isolate the contribution of bidirectionality
and attention.

Architecture:
    (B, T, 17, 2) → SpatialGCN → (B, T, 128)
                   → Temporal(Bi)LSTM → (B, T, output_dim)
                   → Global mean pool → (B, output_dim)
                   → Classifier → (B, 5)
"""
import torch
import torch.nn as nn
from .gcn import SpatialGCN
from .bilstm import TemporalBiLSTM
from ..config import (
    GCN_HIDDEN_DIM, GCN_NUM_LAYERS, COORD_DIM_VELOCITY,
    BILSTM_HIDDEN_DIM, BILSTM_NUM_LAYERS, BILSTM_DROPOUT,
    NUM_CLASSES, DROPOUT,
)
from ..data.preprocessing import add_velocity_torch


class GCNBiLSTM(nn.Module):
    """
    Ablation model: GCN encoder + (Bi)LSTM temporal model.

    Set bidirectional=False for GCN+LSTM (unidirectional) ablation.
    """

    def __init__(
        self,
        gcn_hidden_dim: int = GCN_HIDDEN_DIM,
        gcn_num_layers: int = GCN_NUM_LAYERS,
        bilstm_hidden_dim: int = BILSTM_HIDDEN_DIM,
        bilstm_num_layers: int = BILSTM_NUM_LAYERS,
        dropout: float = BILSTM_DROPOUT,
        num_classes: int = NUM_CLASSES,
        bidirectional: bool = True,
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
            bidirectional=bidirectional,
        )

        lstm_output_dim = self.bilstm.output_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = add_velocity_torch(x)
        x = self.gcn(x)          # (B, T, 128)
        x = self.bilstm(x)       # (B, T, output_dim)
        x = x.mean(dim=1)        # (B, output_dim)
        return self.classifier(x)


if __name__ == "__main__":
    model = GCNBiLSTM()
    x = torch.randn(4, 64, 17, 2)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
    print("GCNBiLSTM test passed!")
