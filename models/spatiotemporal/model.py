"""Spatiotemporal forecasting model scaffold (GAT + TCN)."""
from typing import Any
import numpy as np
import torch
from torch import nn
from models.spatiotemporal.layers import GraphAttentionBlock, TemporalBlock


class SpatioTemporalModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.graph_block = GraphAttentionBlock(input_dim)
        self.temporal_block = TemporalBlock(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.25)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean = mean.detach().clone().float()
        self.input_std = torch.clamp(std.detach().clone().float(), min=1e-6)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        x = (x - self.input_mean.view(1, 1, -1)) / self.input_std.view(1, 1, -1)
        graph_out = self.graph_block(x, adjacency)
        temporal_in = graph_out.transpose(1, 2)
        temporal_out = self.temporal_block(temporal_in)
        pooled = temporal_out.mean(dim=-1)
        return self.head(self.dropout(self.norm(pooled)))

    def predict(self, features: Any) -> np.ndarray:
        if isinstance(features, dict):
            array = np.asarray([features[key] for key in sorted(features.keys())], dtype=float)
        else:
            array = np.asarray(features, dtype=float)
        tensor = torch.tensor(array, dtype=torch.float32)
        with torch.no_grad():
            output = self.forward(tensor)
        return output.detach().cpu().numpy()

