"""
Hand-Rolled Spatial Graph Convolutional Network
No torch-geometric dependency — pure adjacency matrix math.

Graph convolution per frame:
    H_out = ReLU( A_norm @ H_in @ W )

Where A_norm = D̃⁻¹/² × (A + I) × D̃⁻¹/² is precomputed from skeleton.py.

Architecture:
    (B, T, 17, 2) → flatten frames → (B*T, 17, 2)
        → GCN₁ (17, 2) → (17, 64) + LayerNorm + ReLU + Dropout
        → GCN₂ (17, 64) → (17, 128) + LayerNorm + ReLU + Dropout
        → GCN₃ (17, 128) → (17, 128) + LayerNorm + ReLU + Dropout
    → reshape → (B, T, 17, 128)
    → mean pool over joints → (B, T, 128)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..config import GCN_HIDDEN_DIM, GCN_NUM_LAYERS, GCN_POOL, DROPOUT, NUM_KEYPOINTS, COORD_DIM_VELOCITY
from ..data.skeleton import build_normalized_adjacency
from ..data.preprocessing import add_velocity_torch


class GCNLayer(nn.Module):
    """
    Single graph convolution: H_out = σ(A_norm @ H @ W)
    Followed by LayerNorm, ReLU, Dropout.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.kaiming_uniform_(self.W, nonlinearity='relu')
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        x = A_norm @ x @ self.W
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class SpatialGCN(nn.Module):
    """
    Per-frame spatial GCN encoder.
    Processes each frame independently through stacked GCN layers,
    then mean-pools over the 17 joint nodes.

    Output: (B, T, hidden_dim) — one spatial feature vector per frame.
    """

    def __init__(
        self,
        in_channels: int = COORD_DIM_VELOCITY,
        hidden_dim: int = GCN_HIDDEN_DIM,
        num_layers: int = GCN_NUM_LAYERS,
        dropout: float = DROPOUT,
        num_nodes: int = NUM_KEYPOINTS,
        adjacency: Optional[torch.Tensor] = None,
        pool: str = GCN_POOL,
    ):
        super().__init__()

        if adjacency is None:
            adjacency = build_normalized_adjacency(num_nodes)

        self.register_buffer("A_norm", adjacency)  # (N, N)
        self.num_nodes = num_nodes
        self.pool = pool
        self.hidden_dim = hidden_dim

        dims = [in_channels] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(GCNLayer(dims[i], dims[i + 1], dropout))

        if pool == "joint_attn":
            self.joint_attn_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        assert N == self.num_nodes, f"Expected {self.num_nodes} nodes, got {N}"

        x = x.reshape(B * T, N, C)

        for layer in self.layers:
            x = layer(x, self.A_norm)

        x = x.reshape(B, T, N, self.hidden_dim)

        if self.pool == "mean":
            x = x.mean(dim=2)  # (B, T, hidden_dim)
        elif self.pool == "max":
            x = x.max(dim=2).values
        elif self.pool == "flatten":
            x = x.reshape(B, T, N * self.hidden_dim)  # (B, T, N*hidden_dim)
        elif self.pool == "joint_attn":
            # Learnable attention over joints per frame
            # (B*T, N, H) → attention weights → (B*T, H)
            x_bt = x.reshape(B * T, N, self.hidden_dim)
            attn_scores = self.joint_attn_proj(x_bt)  # (B*T, N, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            x = (x_bt * attn_weights).sum(dim=1)  # (B*T, H)
            x = x.reshape(B, T, self.hidden_dim)
        else:
            raise ValueError(f"Unknown pool strategy: {self.pool}")

        return x


if __name__ == "__main__":
    spat = SpatialGCN(in_channels=2, hidden_dim=128, num_layers=3)
    x = torch.randn(4, 64, 17, 2)
    out = spat(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (4, 64, 128), f"Expected (4, 64, 128), got {out.shape}"
    params = sum(p.numel() for p in spat.parameters())
    print(f"Params: {params:,}")
    print("SpatialGCN test passed!")
