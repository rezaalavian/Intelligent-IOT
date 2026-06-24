"""Pickle-friendly predictors for all baseline and STGNN models."""
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import numpy as np

from analytics.features.geo import haversine_m, north_east_offsets_m

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

from infrastructure.kafka.station_registry import STATIONS as _STATIONS
DEFAULT_STATION_COORDS = {s.id: (s.lat, s.lon) for s in _STATIONS.values()}
NUM_STATIONS = len(DEFAULT_STATION_COORDS)


@dataclass
class ConstantPredictor:
    """Historical-average style constant predictor."""

    value: float
    log_target: bool = False

    def predict(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        raw_pred = np.full(n, self.value, dtype=np.float32)
        if self.log_target:
            raw_pred = np.expm1(raw_pred)
        return np.clip(raw_pred, 0.0, None)

    def predict_from_payload(
        self,
        features: Mapping[str, Any] | Sequence[Any],
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> float:
        del features, history
        raw_pred = float(self.value)
        if self.log_target:
            raw_pred = np.expm1(raw_pred)
        return float(max(0.0, raw_pred))


@dataclass
class TabularPredictor:
    """Sklearn-style tabular model + scaler."""

    estimator: Any
    scaler: Any
    feature_columns: list[str]
    log_target: bool = False

    def _row(self, features: Mapping[str, Any] | Sequence[Any], history: Sequence[Mapping[str, Any]] | None = None) -> np.ndarray:
        if isinstance(features, Mapping):
            import datetime
            now = datetime.datetime.now(datetime.timezone.utc)
            hour = now.hour
            dayofweek = now.weekday()
            
            # Apply log-transform to skewed features in payload
            LOG_SKEWED_COLS = ["pm25", "upwind_pm25", "transport_potential", "no", "no2", "nox"]
            logged_features = dict(features)
            for col in LOG_SKEWED_COLS:
                if col in logged_features:
                    logged_features[col] = np.log1p(max(0.0, float(logged_features[col] or 0.0)))
            
            # Log-transform history as well
            logged_history = []
            if history:
                for step in history:
                    logged_step = dict(step)
                    for col in LOG_SKEWED_COLS:
                        if col in logged_step:
                            logged_step[col] = np.log1p(max(0.0, float(logged_step[col] or 0.0)))
                    logged_history.append(logged_step)
            
            row = []
            for col in self.feature_columns:
                if col in logged_features:
                    val = logged_features[col]
                elif col == "hour_sin":
                    val = np.sin(2 * np.pi * hour / 24)
                elif col == "hour_cos":
                    val = np.cos(2 * np.pi * hour / 24)
                elif col == "dow_sin":
                    val = np.sin(2 * np.pi * dayofweek / 7)
                elif col == "dow_cos":
                    val = np.cos(2 * np.pi * dayofweek / 7)
                elif "_lag" in col:
                    parts = col.split("_lag")
                    base_col = parts[0]
                    lag_idx = int(parts[1])
                    if logged_history and len(logged_history) >= lag_idx:
                        val = logged_history[-lag_idx].get(base_col, logged_features.get(base_col, 0.0))
                    else:
                        val = logged_features.get(base_col, 0.0)
                elif "_roll" in col and "_mean" in col:
                    parts = col.split("_roll")
                    base_col = parts[0]
                    window = int(parts[1].split("_")[0])
                    vals = [logged_features.get(base_col, 0.0)]
                    for idx in range(1, window):
                        if logged_history and len(logged_history) >= idx:
                            vals.append(logged_history[-idx].get(base_col, logged_features.get(base_col, 0.0)))
                        else:
                            vals.append(vals[-1])
                    val = np.mean([float(v or 0.0) for v in vals])
                else:
                    val = 0.0
                row.append(float(val or 0.0))
        else:
            row = [float(v) for v in features]
        return np.asarray(row, dtype=np.float32).reshape(1, -1)

    def predict(self, X: Any) -> np.ndarray:
        raw_pred = self.estimator.predict(X)
        if self.log_target:
            raw_pred = np.expm1(raw_pred)
        return np.clip(raw_pred, 0.0, None)

    def predict_from_payload(
        self,
        features: Mapping[str, Any] | Sequence[Any],
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> float:
        scaled = self.scaler.transform(self._row(features, history))
        raw_pred = float(np.asarray(self.estimator.predict(scaled)).flatten()[0])
        if self.log_target:
            raw_pred = np.expm1(raw_pred)
        return float(max(0.0, raw_pred))


if nn is not None:

    class TorchLSTMRegressor(nn.Module):
        def __init__(self, in_dim: int, hidden_dim1: int = 32, hidden_dim2: int = 16, dropout: float = 0.2):
            super().__init__()
            self.l1 = nn.LSTM(in_dim, hidden_dim1, batch_first=True)
            self.d1 = nn.Dropout(dropout)
            self.l2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
            self.d2 = nn.Dropout(dropout)
            self.f1 = nn.Linear(hidden_dim2, 16)
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
        lookback: int = 24
        log_target: bool = False

        def _enrich_history_step(self, step: Mapping[str, Any], steps_ago: int) -> dict[str, Any]:
            import datetime
            enriched = dict(step)
            now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=steps_ago)
            hour = now.hour
            dayofweek = now.weekday()
            enriched["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            enriched["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            enriched["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
            enriched["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)
            
            # Apply log-transform to skewed features
            LOG_SKEWED_COLS = ["pm25", "upwind_pm25", "transport_potential", "no", "no2", "nox"]
            for col in LOG_SKEWED_COLS:
                if col in enriched:
                    enriched[col] = np.log1p(max(0.0, float(enriched[col] or 0.0)))
            return enriched

        def _history_matrix(self, history: Sequence[Mapping[str, Any]]) -> np.ndarray:
            import datetime
            import pandas as pd
            needed = self.lookback + 12
            hist_slice = list(history[-needed:])
            if not hist_slice:
                hist_slice = [{}]
            while len(hist_slice) < needed:
                hist_slice.insert(0, dict(hist_slice[0]))
            df = pd.DataFrame(hist_slice)
            LOG_SKEWED_COLS = ["pm25", "upwind_pm25", "transport_potential", "no", "no2", "nox"]
            for col in LOG_SKEWED_COLS:
                if col in df.columns:
                    df[col] = np.log1p(np.clip(pd.to_numeric(df[col], errors='coerce').fillna(0.0), 0.0, None))
                else:
                    df[col] = 0.0
            now = datetime.datetime.now(datetime.timezone.utc)
            hours = []
            dows = []
            for i in range(len(df)):
                steps_ago = len(df) - 1 - i
                dt = now - datetime.timedelta(hours=steps_ago)
                hours.append(dt.hour)
                dows.append(dt.weekday())
            df["hour_sin"] = np.sin(2 * np.pi * np.array(hours) / 24)
            df["hour_cos"] = np.cos(2 * np.pi * np.array(hours) / 24)
            df["dow_sin"] = np.sin(2 * np.pi * np.array(dows) / 7)
            df["dow_cos"] = np.cos(2 * np.pi * np.array(dows) / 7)
            for col in LOG_SKEWED_COLS:
                for lag in [1, 2, 3, 6, 12]:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)
                df[f"{col}_roll3_mean"] = df[col].rolling(window=3, min_periods=1).mean()
                df[f"{col}_roll6_mean"] = df[col].rolling(window=6, min_periods=1).mean()
            df = df.ffill().bfill().fillna(0.0)
            df_last = df.iloc[-self.lookback:].copy()
            for col in self.feature_columns:
                if col not in df_last.columns:
                    df_last[col] = 0.0
            matrix = df_last[self.feature_columns].to_numpy(dtype=np.float32)
            return matrix

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
            tensor = torch.tensor(scaled, dtype=np.float32).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                out = self.model(tensor).detach().cpu().numpy().flatten()
            raw_pred = float(out[0])
            if self.log_target:
                raw_pred = np.expm1(raw_pred)
            return float(max(0.0, raw_pred))


    class AirQualitySTGNN(nn.Module):
        def __init__(self, num_features: int, num_timesteps_input: int, dropout: float = 0.1):
            super().__init__()
            self.gat1 = GATConv(num_features, 32, heads=2, concat=True)
            self.gat2 = GATConv(32 * 2, 32, heads=1, concat=False)
            self.tcn1 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
            self.tcn2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
            self.skip_project = nn.Linear(num_features * num_timesteps_input, 16 * num_timesteps_input)
            self.regression_head = nn.Linear(16 * num_timesteps_input, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, data: Any) -> torch.Tensor:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            num_nodes = x.size(0)
            x_skip = x.reshape(num_nodes, -1)
            x_skip = self.skip_project(x_skip)
            x_skip = self.dropout(x_skip)
            spatial_slices = []
            for t in range(x.size(2)):
                h_s = F.relu(self.gat1(x[:, :, t], edge_index, edge_attr=edge_attr))
                h_s = self.dropout(h_s)
                h_s = self.gat2(h_s, edge_index, edge_attr=edge_attr)
                h_s = self.dropout(h_s)
                spatial_slices.append(h_s)
            x_space = torch.stack(spatial_slices, dim=-1)
            x_time = F.relu(self.tcn1(x_space))
            x_time = self.dropout(x_time)
            x_time = F.relu(self.tcn2(x_time))
            x_time = self.dropout(x_time)
            x_flat = x_time.reshape(num_nodes, -1) + x_skip
            return self.regression_head(x_flat).squeeze(-1)


    def _compute_dynamic_graph_edges(
        wind_u: float,
        wind_v: float,
        station_coords: dict[int, tuple[float, float]] | None = None,
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
        lookback: int = 24
        station_coords: dict[int, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_STATION_COORDS))
        log_target: bool = False

        def _enrich_history_step(self, step: Mapping[str, Any], steps_ago: int) -> dict[str, Any]:
            import datetime
            enriched = dict(step)
            now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=steps_ago)
            hour = now.hour
            dayofweek = now.weekday()
            enriched["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            enriched["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            enriched["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
            enriched["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)
            
            # Apply log-transform to skewed features
            LOG_SKEWED_COLS = ["pm25", "upwind_pm25", "transport_potential", "no", "no2", "nox"]
            for col in LOG_SKEWED_COLS:
                if col in enriched:
                    enriched[col] = np.log1p(max(0.0, float(enriched[col] or 0.0)))
            return enriched

        def _history_matrix(self, history: Sequence[Mapping[str, Any]]) -> np.ndarray:
            import datetime
            import pandas as pd
            needed = self.lookback + 12
            hist_slice = list(history[-needed:])
            if not hist_slice:
                hist_slice = [{}]
            while len(hist_slice) < needed:
                hist_slice.insert(0, dict(hist_slice[0]))
            df = pd.DataFrame(hist_slice)
            LOG_SKEWED_COLS = ["pm25", "upwind_pm25", "transport_potential", "no", "no2", "nox"]
            for col in LOG_SKEWED_COLS:
                if col in df.columns:
                    df[col] = np.log1p(np.clip(pd.to_numeric(df[col], errors='coerce').fillna(0.0), 0.0, None))
                else:
                    df[col] = 0.0
            now = datetime.datetime.now(datetime.timezone.utc)
            hours = []
            dows = []
            for i in range(len(df)):
                steps_ago = len(df) - 1 - i
                dt = now - datetime.timedelta(hours=steps_ago)
                hours.append(dt.hour)
                dows.append(dt.weekday())
            df["hour_sin"] = np.sin(2 * np.pi * np.array(hours) / 24)
            df["hour_cos"] = np.cos(2 * np.pi * np.array(hours) / 24)
            df["dow_sin"] = np.sin(2 * np.pi * np.array(dows) / 7)
            df["dow_cos"] = np.cos(2 * np.pi * np.array(dows) / 7)
            for col in LOG_SKEWED_COLS:
                for lag in [1, 2, 3, 6, 12]:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)
                df[f"{col}_roll3_mean"] = df[col].rolling(window=3, min_periods=1).mean()
                df[f"{col}_roll6_mean"] = df[col].rolling(window=6, min_periods=1).mean()
            df = df.ffill().bfill().fillna(0.0)
            df_last = df.iloc[-self.lookback:].copy()
            for col in self.feature_columns:
                if col not in df_last.columns:
                    df_last[col] = 0.0
            matrix = df_last[self.feature_columns].to_numpy(dtype=np.float32)
            return matrix

        def _build_graph(self, scaled_window: np.ndarray, wind_u: float, wind_v: float, station_windows=None) -> Any:
            if station_windows is not None:
                node_feats = list(station_windows)          # real per-station windows
            else:
                node_feats = [scaled_window for _ in range(NUM_STATIONS)]   # fallback: target only
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
            raw_pred = float(out[0])
            if self.log_target:
                raw_pred = np.expm1(raw_pred)
            return float(max(0.0, raw_pred))

else:  # pragma: no cover
    TorchLSTMRegressor = None
    LSTMPredictor = None
    AirQualitySTGNN = None
    STGNNPredictor = None
