"""
BiLSTM Temporal Module

Models temporal dynamics across the frame sequence after spatial
features have been extracted by the GCN.

Architecture:
    (B, T, input_dim=128) → BiLSTM₁ → (B, T, 256)
                          → Dropout
                          → BiLSTM₂ → (B, T, 256)

Output = 256 (128 hidden × 2 directions)
"""
import torch
import torch.nn as nn
from ..config import BILSTM_HIDDEN_DIM, BILSTM_NUM_LAYERS, BILSTM_DROPOUT


class TemporalBiLSTM(nn.Module):
    """
    2-layer bidirectional LSTM for temporal feature extraction.

    Input:  (B, T, input_dim)  — GCN spatial features per frame
    Output: (B, T, hidden_dim * 2)  — temporally enriched features
    """

    def __init__(
        self,
        input_dim: int = BILSTM_HIDDEN_DIM,  # matches GCN hidden_dim
        hidden_dim: int = BILSTM_HIDDEN_DIM,
        num_layers: int = BILSTM_NUM_LAYERS,
        dropout: float = BILSTM_DROPOUT,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


if __name__ == "__main__":
    model = TemporalBiLSTM(input_dim=128, hidden_dim=128, num_layers=2)
    x = torch.randn(4, 64, 128)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (4, 64, 256), f"Expected (4, 64, 256), got {out.shape}"
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
    print("TemporalBiLSTM test passed!")
