"""Feature engineering utilities for hourly raw data.

The project now separates feature introduction from feature transformations.
Raw features are introduced first, then optional lag/rolling/graph transforms can be
applied per model and horizon.
"""
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Sequence
import numpy as np
import pandas as pd
import torch
from torch import nn

from .geo import haversine_m, north_east_offsets_m

try:  # pragma: no cover - optional dependency
    from torch_geometric.data import Data  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    Data = None

try:  # pragma: no cover - optional dependency
    from torch_geometric.nn import GATConv, global_mean_pool  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    GATConv = None
    global_mean_pool = None


FEATURE_COLUMNS = [
    "Temp Definition °C",
    "Dew Point Definition °C",
    "Rel Hum Definition %",
    "Precip. Amount Definition mm",
    "Wind Dir Definition 10's deg",
    "Wind Spd Definition km/h",
    "Visibility Definition km",
    "Stn Press Definition kPa",
    "Hmdx Definition",
    "Wind Chill Definition",
    "no",
    "no2",
    "nox",
    "o3",
    "pm2",
    "co",
    "so2",
]

HORIZONS = (1, 2, 3)
MODELS = ("ha", "lr", "rf", "lstm", "stgnn")
MODEL_FAMILIES = ("ha", "lr", "rf", "lstm", "stgnn")

LOWER_TO_CANONICAL_COLUMNS = {
    "temp definition °c": "Temp Definition °C",
    "dew point definition °c": "Dew Point Definition °C",
    "rel hum definition %": "Rel Hum Definition %",
    "precip. amount definition mm": "Precip. Amount Definition mm",
    "wind dir definition 10's deg": "Wind Dir Definition 10's deg",
    "wind spd definition km/h": "Wind Spd Definition km/h",
    "visibility definition km": "Visibility Definition km",
    "stn press definition kpa": "Stn Press Definition kPa",
    "hmdx definition": "Hmdx Definition",
    "wind chill definition": "Wind Chill Definition",
    "weather definition": "Weather Definition",
    "pm25": "pm2",
}


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    rename_map = {
        column: LOWER_TO_CANONICAL_COLUMNS[column.strip().lower()]
        for column in renamed.columns
        if column.strip().lower() in LOWER_TO_CANONICAL_COLUMNS
    }
    if rename_map:
        renamed = renamed.rename(columns=rename_map)
    return renamed


def _build_timestamp(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"], errors="coerce").dt.floor("h")

    year_column = next((column for column in frame.columns if column.strip().lower() == "year"), None)
    month_column = next((column for column in frame.columns if column.strip().lower() == "month"), None)
    day_column = next((column for column in frame.columns if column.strip().lower() == "day"), None)
    hour_column = next((column for column in frame.columns if column.strip().lower() in {"time lst", "time", "hour", "clean_hour"}), None)

    if year_column and month_column and day_column:
        year = pd.to_numeric(frame[year_column], errors="coerce").fillna(2026).astype(int)
        month = pd.to_numeric(frame[month_column], errors="coerce").fillna(1).astype(int)
        day = pd.to_numeric(frame[day_column], errors="coerce").fillna(1).astype(int)
        if hour_column is not None:
            hour = pd.to_numeric(frame[hour_column].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(0).astype(int)
        else:
            hour = pd.Series(0, index=frame.index)
        return pd.to_datetime(
            {
                "year": year,
                "month": month,
                "day": day,
                "hour": hour,
            },
            errors="coerce",
        ).dt.floor("h")

    return pd.date_range(start="2026-01-01", periods=len(frame), freq="h")


def _prepare_notebook_frame(frame: pd.DataFrame) -> pd.DataFrame:
    features = _canonicalize_columns(frame).copy()
    features["timestamp"] = _build_timestamp(features)
    features = features.dropna(subset=["timestamp"]).sort_values([col for col in ["city_name", "timestamp"] if col in features.columns])

    numeric_columns = [column for column in FEATURE_COLUMNS if column in features.columns]
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    if "pm2" in features.columns:
        features["pm2"] = pd.to_numeric(features["pm2"], errors="coerce")

    # Avoid using future values when imputing: forward-fill only, then fallback to zeros
    features = features.ffill().fillna(0.0)

    if "Wind Spd Definition km/h" in features.columns and "Wind Dir Definition 10's deg" in features.columns:
        wind_speed = pd.to_numeric(features["Wind Spd Definition km/h"], errors="coerce")
        wind_direction = np.deg2rad(pd.to_numeric(features["Wind Dir Definition 10's deg"], errors="coerce") * 10.0)
        features["wind_u"] = wind_speed * np.cos(wind_direction)
        features["wind_v"] = wind_speed * np.sin(wind_direction)

    features["hour"] = features["timestamp"].dt.hour
    features["dayofweek"] = features["timestamp"].dt.dayofweek
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["dow_sin"] = np.sin(2 * np.pi * features["dayofweek"] / 7)
    features["dow_cos"] = np.cos(2 * np.pi * features["dayofweek"] / 7)

    if "city_name" in features.columns:
        features["city_code"] = features["city_name"].astype("category").cat.codes.astype(float)

    return features


def introduce_raw_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and expose the raw hourly feature table without engineered transforms."""

    features = frame.copy()
    if "timestamp" not in features.columns:
        raise ValueError("timestamp column is required")

    features["timestamp"] = pd.to_datetime(features["timestamp"], errors="coerce").dt.floor("h")
    features = features.sort_values([col for col in ["city_name", "timestamp"] if col in features.columns])

    for column in FEATURE_COLUMNS:
        if column in features.columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")

    return features


def compute_rolling_features(frame: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Create rolling means and lag features from the hourly dataset."""
    features = introduce_raw_features(frame)

    numeric_cols = [column for column in FEATURE_COLUMNS if column in features.columns]
    max_lag = 6

    # prepare container for new columns to avoid DataFrame fragmentation
    new_cols: Dict[str, pd.Series] = {}

    # group index mapping (None if no city grouping)
    has_group = "city_name" in features.columns
    gb = features.groupby("city_name", sort=False) if has_group else None

    for column in numeric_cols:
        base = pd.to_numeric(features[column], errors="coerce")

        # 1) imputation: group-aware forward-only interpolation to avoid leakage
        if has_group:
            # forward-direction interpolation within groups, then forward-fill, then fallback to group median
            def _impute_forward(s: pd.Series) -> pd.Series:
                return s.interpolate(limit_direction="forward").ffill().fillna(s.median())

            cleaned = gb[column].transform(lambda s: _impute_forward(s))
        else:
            cleaned = base.interpolate(limit_direction="forward").ffill().fillna(base.median())

        # 2) outlier clipping: compute group quantiles once and clip
        if has_group:
            qlow = gb[column].transform(lambda s: s.quantile(0.01))
            qhigh = gb[column].transform(lambda s: s.quantile(0.99))
        else:
            qlow = cleaned.quantile(0.01)
            qhigh = cleaned.quantile(0.99)
        clipped = cleaned.clip(lower=qlow, upper=qhigh)

        # write back cleaned column
        features[column] = clipped

        # lag features (group-aware)
        for lag in range(1, max_lag + 1):
            if has_group:
                # shift within groups using groupby.transform for index alignment
                new_cols[f"{column}_lag{lag}"] = gb[column].transform(lambda s, l=lag: s.shift(l)).reindex(features.index)
            else:
                new_cols[f"{column}_lag{lag}"] = clipped.shift(lag)

        # rolling statistics computed within groups to avoid leakage
        if has_group:
            new_cols[f"{column}_roll{window}_mean"] = gb[column].transform(lambda s: s.rolling(window=window, min_periods=1).mean()).reindex(features.index)
            new_cols[f"{column}_roll{window}"] = new_cols[f"{column}_roll{window}_mean"]
            new_cols[f"{column}_roll{window}_std"] = gb[column].transform(lambda s: s.rolling(window=window, min_periods=1).std()).reindex(features.index)
            new_cols[f"{column}_roll{window}_min"] = gb[column].transform(lambda s: s.rolling(window=window, min_periods=1).min()).reindex(features.index)
            new_cols[f"{column}_roll{window}_max"] = gb[column].transform(lambda s: s.rolling(window=window, min_periods=1).max()).reindex(features.index)
        else:
            new_cols[f"{column}_roll{window}_mean"] = clipped.rolling(window=window, min_periods=1).mean()
            new_cols[f"{column}_roll{window}"] = new_cols[f"{column}_roll{window}_mean"]
            new_cols[f"{column}_roll{window}_std"] = clipped.rolling(window=window, min_periods=1).std()
            new_cols[f"{column}_roll{window}_min"] = clipped.rolling(window=window, min_periods=1).min()
            new_cols[f"{column}_roll{window}_max"] = clipped.rolling(window=window, min_periods=1).max()

    # build resulting DataFrame by concatenating original (stable small set) and new columns
    result = features.copy()
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=features.index)
        result = pd.concat([result, new_df], axis=1)

    # time cyclic features
    result["hour"] = result["timestamp"].dt.hour
    result["dayofweek"] = result["timestamp"].dt.dayofweek
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["dayofweek"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["dayofweek"] / 7)

    return result


def _prepare_base_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return introduce_raw_features(frame)


def add_wind_components(frame: pd.DataFrame, speed_column: str = "Wind Spd Definition km/h", direction_column: str = "Wind Dir Definition 10's deg") -> pd.DataFrame:
    if speed_column not in frame.columns or direction_column not in frame.columns:
        return frame.copy()
    wind_speed = pd.to_numeric(frame[speed_column], errors="coerce")
    wind_direction = np.deg2rad(pd.to_numeric(frame[direction_column], errors="coerce") * 10.0)
    enriched = frame.copy()
    enriched["wind_u"] = wind_speed * np.cos(wind_direction)
    enriched["wind_v"] = wind_speed * np.sin(wind_direction)
    return enriched


def compute_dynamic_graph_edges(wind_u: float, wind_v: float, station_coords: dict[str, tuple[float, float]]) -> tuple[torch.Tensor, torch.Tensor]:
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    wind_vector = np.array([wind_u, wind_v], dtype=float)
    for source_index, source_key in enumerate(station_coords.keys()):
        for target_index, target_key in enumerate(station_coords.keys()):
            if source_index == target_index:
                continue
            source_lat, source_lon = station_coords[source_key]
            target_lat, target_lon = station_coords[target_key]
            distance = haversine_m(source_lat, source_lon, target_lat, target_lon)
            if distance == 0.0:
                continue
            north, east = north_east_offsets_m(source_lat, source_lon, target_lat, target_lon)
            delta_norm = np.array([north, east], dtype=float) / math.hypot(north, east)
            alignment = float(np.dot(delta_norm, wind_vector))
            base_weight = 1.0 / (distance + 1e-5)
            weight = base_weight * (1.0 + alignment) if alignment > 0 else base_weight * np.exp(alignment)
            sources.append(source_index)
            targets.append(target_index)
            weights.append(weight)
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(-1)
    return edge_index, edge_attr


def build_graph_sequences(scaled_data: np.ndarray, raw_df: pd.DataFrame, lookback: int = 12, horizon: int = 1) -> list[Any]:
    if Data is None:
        raise RuntimeError("torch_geometric is required to build graph sequences")
    X_graphs: list[Any] = []
    target_series = raw_df["target_shifted"] if "target_shifted" in raw_df.columns else raw_df["pm2"]
    for t in range(len(scaled_data) - lookback - horizon + 1):
        target_val = float(target_series.iloc[t + lookback - 1])
        y_tensor = torch.tensor([target_val], dtype=torch.float)
        node_feats = torch.tensor(np.asarray(scaled_data[t : t + lookback], dtype=np.float32), dtype=torch.float)
        if node_feats.dim() == 2:
            node_feats = node_feats.unsqueeze(0)
        if "wind_u" in raw_df.columns and "wind_v" in raw_df.columns:
            wind_u = float(raw_df["wind_u"].iloc[t + lookback - 1])
            wind_v = float(raw_df["wind_v"].iloc[t + lookback - 1])
        else:
            wind_u = 0.0
            wind_v = 0.0
        station_coords = {
            "station_0": (43.6532, -79.3832),
            "station_1": (43.7000, -79.4000),
            "station_2": (43.6200, -79.3500),
        }
        edge_index, edge_attr = compute_dynamic_graph_edges(wind_u, wind_v, station_coords)
        X_graphs.append(Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor))
    return X_graphs


def build_lstm_sequences(X_data: np.ndarray, y_data: np.ndarray, lookback_steps: int = 12) -> tuple[np.ndarray, np.ndarray]:
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for index in range(len(X_data) - lookback_steps):
        Xs.append(X_data[index : index + lookback_steps])
        ys.append(y_data[index + lookback_steps])
    return np.asarray(Xs), np.asarray(ys)


def compute_dynamic_edge_index(
    station_frame: pd.DataFrame,
    station_id_column: str = "station_id",
    x_column: str = "longitude",
    y_column: str = "latitude",
    k_neighbors: int = 3,
) -> torch.Tensor:
    unique_stations = station_frame.dropna(subset=[station_id_column]).drop_duplicates(subset=[station_id_column])
    if unique_stations.empty:
        return torch.empty((2, 0), dtype=torch.long)
    coordinates = unique_stations.loc[:, [x_column, y_column]].to_numpy(dtype=float, copy=True)
    coordinates = np.nan_to_num(coordinates, nan=0.0)
    count = len(coordinates)
    pairwise = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(count):
            pairwise[i, j] = haversine_m(coordinates[i, 1], coordinates[i, 0],
                                         coordinates[j, 1], coordinates[j, 0])
    np.fill_diagonal(pairwise, np.inf)
    edges: list[tuple[int, int]] = []
    for source_index in range(len(unique_stations)):
        neighbors = np.argsort(pairwise[source_index])[:k_neighbors]
        edges.extend((source_index, int(target_index)) for target_index in neighbors)
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_sequence_windows(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    history: int = 24,
    horizon: int = 1,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    features = frame.loc[:, [column for column in feature_columns if column in frame.columns]].to_numpy(dtype=np.float32, copy=True)
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=np.float32, copy=True)
    sequences: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    stop_index = len(frame) - horizon
    for index in range(history, stop_index + 1):
        window = features[index - history : index]
        y_value = target[index + horizon - 1]
        if np.isnan(window).any() or np.isnan(y_value):
            continue
        sequences.append(torch.tensor(window, dtype=torch.float32))
        labels.append(torch.tensor(y_value, dtype=torch.float32))
    return sequences, labels


def make_lstm_sequences(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    history: int = 24,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    features = frame.loc[:, [column for column in feature_columns if column in frame.columns]].to_numpy(dtype=np.float32, copy=True)
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=np.float32, copy=True)
    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    stop_index = len(frame) - horizon
    for index in range(history, stop_index + 1):
        window = features[index - history : index]
        y_value = target[index + horizon - 1]
        if np.isnan(window).any() or np.isnan(y_value):
            continue
        x_values.append(window)
        y_values.append(float(y_value))
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


@dataclass(frozen=True)
class SequenceSpec:
    history: int = 24
    horizon: int = 1
    target_column: str = "pm2"
    feature_columns: tuple[str, ...] = ()


class AirQualitySTGNN(nn.Module):
    def __init__(self, input_features: int, hidden_features: int = 64, output_features: int = 1):
        super().__init__()
        self.use_geometric = GATConv is not None and global_mean_pool is not None
        if self.use_geometric:
            self.gat1 = GATConv(input_features, hidden_features, heads=2, concat=False)
            self.gat2 = GATConv(hidden_features, hidden_features, heads=2, concat=False)
            self.head = nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, output_features),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(input_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, output_features),
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None, batch: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_geometric and edge_index is not None:
            x = torch.relu(self.gat1(x, edge_index))
            x = torch.relu(self.gat2(x, edge_index))
            pooled = global_mean_pool(x, batch) if batch is not None else x.mean(dim=0, keepdim=True)
            return self.head(pooled)
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.head(x)


def prepare_ha_h1_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_base_frame(frame)


def prepare_ha_h2_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_base_frame(frame)


def prepare_ha_h3_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_base_frame(frame)


def prepare_lr_h1_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_notebook_frame(frame)


def prepare_lr_h2_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_notebook_frame(frame)


def prepare_lr_h3_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_notebook_frame(frame)


def prepare_rf_h1_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_notebook_frame(frame)


def prepare_rf_h2_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_notebook_frame(frame)


def prepare_rf_h3_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_notebook_frame(frame)


def prepare_lstm_h1_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_notebook_frame(frame)


def prepare_lstm_h2_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_notebook_frame(frame)


def prepare_lstm_h3_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_notebook_frame(frame)


def prepare_stgnn_h1_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_notebook_frame(frame)


def prepare_stgnn_h2_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_notebook_frame(frame)


def prepare_stgnn_h3_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _prepare_notebook_frame(frame)


_FEATURE_BUILDERS: dict[str, dict[int, Callable[[pd.DataFrame], pd.DataFrame]]] = {
    "ha": {
        1: prepare_ha_h1_features,
        2: prepare_ha_h2_features,
        3: prepare_ha_h3_features,
    },
    "lr": {
        1: prepare_lr_h1_features,
        2: prepare_lr_h2_features,
        3: prepare_lr_h3_features,
    },
    "rf": {
        1: prepare_rf_h1_features,
        2: prepare_rf_h2_features,
        3: prepare_rf_h3_features,
    },
    "lstm": {
        1: prepare_lstm_h1_features,
        2: prepare_lstm_h2_features,
        3: prepare_lstm_h3_features,
    },
    "stgnn": {
        1: prepare_stgnn_h1_features,
        2: prepare_stgnn_h2_features,
        3: prepare_stgnn_h3_features,
    },
}


def build_feature_frame(model_name: str, horizon: int, frame: pd.DataFrame) -> pd.DataFrame:
    model_key = model_name.lower()
    if model_key not in _FEATURE_BUILDERS:
        raise ValueError(f"Unsupported model name: {model_name}")
    horizon_key = int(horizon)
    try:
        builder = _FEATURE_BUILDERS[model_key][horizon_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported horizon: {horizon}") from exc
    return builder(frame)


def main():
    sample = pd.DataFrame(
        [
            {"timestamp": "2022-01-01 00:00:00", "city_name": "Toronto", "pm2": 10.0, "no2": 3.0, "o3": 1.0, "co": 0.1, "so2": 0.02},
            {"timestamp": "2022-01-01 01:00:00", "city_name": "Toronto", "pm2": 12.0, "no2": 4.0, "o3": 1.2, "co": 0.12, "so2": 0.03},
            {"timestamp": "2022-01-01 02:00:00", "city_name": "Toronto", "pm2": 14.0, "no2": 5.0, "o3": 1.4, "co": 0.14, "so2": 0.04},
        ]
    )
    print(introduce_raw_features(sample).head())
    print(compute_rolling_features(sample).head())


if __name__ == "__main__":
    main()
