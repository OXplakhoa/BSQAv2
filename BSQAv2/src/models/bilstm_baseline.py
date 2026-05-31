"""
BiLSTM Baseline Model for Ablation Study

Identical architecture to LSTMBaseline but with bidirectional LSTM.
Now with velocity features (pos + vel = 4 channels).

Architecture:
    (B, T, 17, 2) → +velocity → (B, T, 17, 4) → flatten → (B, T, 68)
        → BiLSTM x2 (hidden=128, bidirectional) → (B, T, 256)
        → take last hidden → (B, 256)
        → FC classifier → (B, 5)
"""
import torch
import torch.nn as nn
from ..config import INPUT_DIM_VELOCITY, BILSTM_HIDDEN_DIM, BILSTM_NUM_LAYERS, NUM_CLASSES, BILSTM_DROPOUT, NUM_KEYPOINTS
from ..data.preprocessing import add_velocity_torch


class BiLSTMBaseline(nn.Module):
    """2-layer bidirectional LSTM for stroke classification."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM_VELOCITY,
        hidden_dim: int = BILSTM_HIDDEN_DIM,
        num_layers: int = BILSTM_NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = BILSTM_DROPOUT,
    ):
        super().__init__()
        self.num_keypoints = NUM_KEYPOINTS

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        output_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = add_velocity_torch(x)
        B, T, N, C = x.shape
        x = x.reshape(B, T, N * C)
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :])


if __name__ == "__main__":
    model = BiLSTMBaseline()
    x = torch.randn(4, 64, 17, 2)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"
    print("BiLSTMBaseline test passed!")
