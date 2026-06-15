"""Simple spatiotemporal building blocks.

These are lightweight, CPU-friendly layers that keep the architecture runnable in the
shared project environment while preserving the GAT + TCN design intent.
"""
import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.dropout = nn.Dropout(p=0.2)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )
        # residual projection when channel dims differ
        if in_channels != out_channels:
            self.res_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.net(x))
        res = self.res_proj(x) if self.res_proj is not None else x
        return out + res


class GraphAttentionBlock(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) / max(x.shape[-1] ** 0.5, 1.0)
        if adjacency is not None:
            if adjacency.dim() == 2:
                adjacency = adjacency.unsqueeze(0)
            scores = scores + adjacency
        weights = torch.softmax(scores, dim=-1)
        out = self.dropout(torch.matmul(weights, v))
        # residual connection
        try:
            return out + x
        except Exception:
            return out

