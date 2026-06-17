"""Pickle-friendly predictors for all baseline and STGNN models."""
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import numpy as np

from analytics.flink_jobs.geo import haversine_m, north_east_offsets_m

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

try:
    from torch_geometric.nn import GATConv  # type: ignore[import-not-found]
    from torch_geometric.data import Data  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    GATConv = None
    Data = None

DEFAULT_STATION_COORDS = {
    "station_0": (43.6532, -79.3832),
    "station_1": (43.7000, -79.4000),
    "station_2": (43.6200, -79.3500),
}
NUM_STATIONS = len(DEFAULT_STATION_COORDS)


@dataclass
class ConstantPredictor:
    """Historical-average style constant predictor."""

    value: float

    def predict(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.value, dtype=np.float32)

    def predict_from_payload(
        self,
        features: Mapping[str, Any] | Sequence[Any],
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> float:
        del features, history
        return float(self.value)


@dataclass
class TabularPredictor:
    """Sklearn-style tabular model + scaler."""

    estimator: Any
    scaler: Any
    feature_columns: list[str]

    def _row(self, features: Mapping[str, Any] | Sequence[Any]) -> np.ndarray:
        if isinstance(features, Mapping):
            row = [float(features.get(col, 0.0) or 0.0) for col in self.feature_columns]
        else:
            row = [float(v) for v in features]
        return np.asarray(row, dtype=np.float32).reshape(1, -1)

    def predict(self, X: Any) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_from_payload(
        self,
        features: Mapping[str, Any] | Sequence[Any],
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> float:
        del history
        scaled = self.scaler.transform(self._row(features))
        return float(np.asarray(self.estimator.predict(scaled)).flatten()[0])


if nn is not None:

    class TorchLSTMRegressor(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.l1 = nn.LSTM(in_dim, 64, batch_first=True)
            self.d1 = nn.Dropout(0.2)
            self.l2 = nn.LSTM(64, 32, batch_first=True)
            self.d2 = nn.Dropout(0.2)
            self.f1 = nn.Linear(32, 16)
            self.f2 = nn.Linear(16, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.l1(x)
            out = self.d1(out)
            out, _ = self.l2(out)
            out = self.d2(out[:, -1, :])
            out = torch.relu(self.f1(out))
            return self.f2(out)


    @dataclass
    class LSTMPredictor:
        model: TorchLSTMRegressor
        scaler: Any
        feature_columns: list[str]
        lookback: int = 12

        def _history_matrix(self, history: Sequence[Mapping[str, Any]]) -> np.ndarray:
            rows = []
            for step in history[-self.lookback :]:
                rows.append([float(step.get(col, 0.0) or 0.0) for col in self.feature_columns])
            while len(rows) < self.lookback:
                rows.insert(0, rows[0] if rows else [0.0] * len(self.feature_columns))
            return np.asarray(rows, dtype=np.float32)

        def predict_from_payload(
            self,
            features: Mapping[str, Any] | Sequence[Any],
            *,
            history: Sequence[Mapping[str, Any]] | None = None,
        ) -> float:
            hist = list(history or [])
            if not hist and isinstance(features, Mapping):
                hist = [dict(features)] * self.lookback
            matrix = self._history_matrix(hist)
            scaled = self.scaler.transform(matrix)
            tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                out = self.model(tensor).detach().cpu().numpy().flatten()
            return float(out[0])


    class AirQualitySTGNN(nn.Module):
        def __init__(self, num_features: int, num_timesteps_input: int):
            super().__init__()
            self.gat1 = GATConv(num_features, 32, heads=2, concat=True)
            self.gat2 = GATConv(32 * 2, 32, heads=1, concat=False)
            self.tcn1 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
            self.tcn2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
            self.skip_project = nn.Linear(num_features * num_timesteps_input, 16 * num_timesteps_input)
            self.regression_head = nn.Linear(16 * num_timesteps_input, 1)

        def forward(self, data: Any) -> torch.Tensor:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            num_nodes = x.size(0)
            x_skip = x.reshape(num_nodes, -1)
            x_skip = self.skip_project(x_skip)
            spatial_slices = []
            for t in range(x.size(2)):
                h_s = F.relu(self.gat1(x[:, :, t], edge_index, edge_attr=edge_attr))
                h_s = self.gat2(h_s, edge_index, edge_attr=edge_attr)
                spatial_slices.append(h_s)
            x_space = torch.stack(spatial_slices, dim=-1)
            x_time = F.relu(self.tcn1(x_space))
            x_time = F.relu(self.tcn2(x_time))
            x_flat = x_time.reshape(num_nodes, -1) + x_skip
            return self.regression_head(x_flat).squeeze(-1)


    def _compute_dynamic_graph_edges(
        wind_u: float,
        wind_v: float,
        station_coords: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = station_coords or DEFAULT_STATION_COORDS
        sources, targets, weights = [], [], []
        w_vec = np.array([wind_u, wind_v], dtype=float)
        for i, s_idx in enumerate(coords.keys()):
            for j, t_idx in enumerate(coords.keys()):
                if i == j:
                    continue
                c_i, c_j = coords[s_idx], coords[t_idx]
                dist = haversine_m(c_i[0], c_i[1], c_j[0], c_j[1])
                if dist == 0:
                    continue
                north, east = north_east_offsets_m(c_i[0], c_i[1], c_j[0], c_j[1])
                d_ij_norm = np.array([north, east], dtype=float) / math.hypot(north, east)
                alignment = float(np.dot(d_ij_norm, w_vec))
                base_weight = 1.0 / (dist + 1e-5)
                weight = base_weight * (1.0 + alignment) if alignment > 0 else base_weight * np.exp(alignment)
                sources.append(i)
                targets.append(j)
                weights.append(weight)
        return (
            torch.tensor([sources, targets], dtype=torch.long),
            torch.tensor(weights, dtype=torch.float).unsqueeze(-1),
        )


    @dataclass
    class STGNNPredictor:
        model: AirQualitySTGNN
        scaler: Any
        feature_columns: list[str]
        lookback: int = 12
        station_coords: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_STATION_COORDS))

        def _history_matrix(self, history: Sequence[Mapping[str, Any]]) -> np.ndarray:
            rows = []
            for step in history[-self.lookback :]:
                rows.append([float(step.get(col, 0.0) or 0.0) for col in self.feature_columns])
            while len(rows) < self.lookback:
                rows.insert(0, rows[0] if rows else [0.0] * len(self.feature_columns))
            return np.asarray(rows, dtype=np.float32)

        def _build_graph(self, scaled_window: np.ndarray, wind_u: float, wind_v: float) -> Any:
            node_feats = [scaled_window * (1.0 + (s * 0.05)) for s in range(NUM_STATIONS)]
            node_feats_t = torch.tensor(np.array(node_feats), dtype=torch.float).transpose(1, 2)
            edge_index, edge_attr = _compute_dynamic_graph_edges(wind_u, wind_v, self.station_coords)
            return Data(x=node_feats_t, edge_index=edge_index, edge_attr=edge_attr)

        def predict_from_payload(
            self,
            features: Mapping[str, Any] | Sequence[Any],
            *,
            history: Sequence[Mapping[str, Any]] | None = None,
        ) -> float:
            hist = list(history or [])
            if not hist and isinstance(features, Mapping):
                hist = [dict(features)] * self.lookback
            matrix = self._history_matrix(hist)
            scaled = self.scaler.transform(matrix)
            last = hist[-1] if hist else (features if isinstance(features, Mapping) else {})
            wind_u = float(last.get("wind_u", 0.0) or 0.0)
            wind_v = float(last.get("wind_v", 0.0) or 0.0)
            graph = self._build_graph(scaled, wind_u, wind_v)
            self.model.eval()
            with torch.no_grad():
                out = self.model(graph).detach().cpu().numpy().flatten()
            return float(np.mean(out))

else:  # pragma: no cover
    TorchLSTMRegressor = None
    LSTMPredictor = None
    AirQualitySTGNN = None
    STGNNPredictor = None
