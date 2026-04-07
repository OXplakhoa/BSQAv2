"""
LSTM Baseline Model for Stroke Classification
"""
import torch
import torch.nn as nn
from ..config import INPUT_DIM, HIDDEN_DIM, NUM_LSTM_LAYERS, NUM_CLASSES, DROPOUT


class LSTMBaseline(nn.Module):
    """
    Simple 2-layer LSTM for stroke classification.
    
    Architecture:
        (batch, seq, 34) -> LSTM -> (batch, hidden) -> FC -> (batch, 3)
    """
    
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LSTM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) tensor
        
        Returns:
            (batch, num_classes) logits
        """
        # LSTM output: (batch, seq, hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        
        # Classify
        logits = self.classifier(last_hidden)
        
        return logits


if __name__ == "__main__":
    # Quick test
    model = LSTMBaseline()
    x = torch.randn(4, 64, 34)  # batch=4, seq=64, features=34
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (4, 3), f"Expected (4, 3), got {out.shape}"
    print("✓ Model test passed!")
