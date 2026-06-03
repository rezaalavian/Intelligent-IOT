"""Baseline benchmarking aligned to tested_models.py behavior.

This module keeps the project structure while reproducing the exact tested flow:
- raw CSV loading and lowercase columns
- simple chronological split (70/15/15)
- RobustScaler fit on train only
- horizon-wise shifted target
- HA, LR, LGBM/RF, LSTM, and STGNN baselines
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    RandomForestRegressor = None

mlflow = None

try:
    from lightgbm import LGBMRegressor  # type: ignore[import-not-found]
    USE_LGBM = True
except Exception:
    LGBMRegressor = None
    USE_LGBM = False

try:
    import tensorflow as tf  # type: ignore[import-not-found]
    from tensorflow.keras.models import Sequential  # type: ignore[import-not-found]
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore[import-not-found]
    USE_TF = True
except Exception:
    tf = None
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    USE_TF = False

try:
    from torch_geometric.nn import GATConv  # type: ignore[import-not-found]
    from torch_geometric.data import Data  # type: ignore[import-not-found]
    from torch_geometric.loader import DataLoader as PyGDataLoader  # type: ignore[import-not-found]
    USE_PYG = True
except Exception:
    GATConv = None
    Data = None
    PyGDataLoader = None
    USE_PYG = False


DEFAULT_PATH = Path("data/raw/Raw_Data.csv")
STATION_COORDS = {
    "station_0": (43.6532, -79.3832),
    "station_1": (43.7000, -79.4000),
    "station_2": (43.6200, -79.3500),
}
NUM_STATIONS = len(STATION_COORDS)
LOOKBACK_STEPS = 12


def _resolve_path(path: Path) -> Path:
    if path.exists():
        return path
    for candidate in (
        Path("data/raw/Raw_Data.csv"),
        Path("data/raw/RawData.csv"),
        Path("Raw_Data.csv"),
        Path("RawData.csv"),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find raw data CSV. Tried: {path} and Raw_Data/RawData variants")


def _load_base_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.copy().reset_index(drop=True)
    frame.columns = frame.columns.str.strip().str.lower()

    if "time lst" in frame.columns:
        if frame["time lst"].dtype == "object":
            frame["clean_hour"] = frame["time lst"].astype(str).str.extract(r"(\d+)").astype(float).fillna(0).astype(int)
        else:
            frame["clean_hour"] = pd.to_numeric(frame["time lst"], errors="coerce").fillna(0).astype(int)
    else:
        frame["clean_hour"] = 0

    if "timestamp" in frame.columns:
        frame["datetime_combined"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    else:
        date_strings = (
            frame["year"].astype(str).str.extract(r"(\d+)")[0].fillna("2026")
            + "-"
            + frame["month"].astype(str).str.extract(r"(\d+)")[0].fillna("1")
            + "-"
            + frame["day"].astype(str).str.extract(r"(\d+)")[0].fillna("1")
            + " "
            + frame["clean_hour"].astype(str)
            + ":00:00"
        )
        frame["datetime_combined"] = pd.to_datetime(date_strings, errors="coerce")

    if frame["datetime_combined"].isna().all():
        frame["datetime_combined"] = pd.date_range(start="2026-01-01", periods=len(frame), freq="h")

    frame = frame.dropna(subset=["datetime_combined"]).sort_values("datetime_combined")

    cols_to_numeric = [
        "temp definition °c",
        "dew point definition °c",
        "rel hum definition %",
        "wind spd definition km/h",
        "wind dir definition 10's deg",
        "pm25",
        "pm2",
    ]
    for c in cols_to_numeric:
        if c in frame.columns:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")

    frame = frame.ffill().bfill().fillna(0.0)

    if "wind dir definition 10's deg" in frame.columns and "wind spd definition km/h" in frame.columns:
        rad = np.deg2rad(frame["wind dir definition 10's deg"] * 10)
        frame["wind_u"] = frame["wind spd definition km/h"] * np.cos(rad)
        frame["wind_v"] = frame["wind spd definition km/h"] * np.sin(rad)
    else:
        frame["wind_u"] = 0.0
        frame["wind_v"] = 0.0

    return frame


def compile_metrics(y_true, y_pred):
    y_true_clean = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred_clean = np.asarray(y_pred, dtype=np.float32).flatten()
    mask = np.isfinite(y_true_clean) & np.isfinite(y_pred_clean)
    y_true_clean = y_true_clean[mask]
    y_pred_clean = y_pred_clean[mask]
    if len(y_true_clean) == 0:
        return {"R2": 0.0, "MAE": 0.0, "MSE": 0.0, "RMSE": 0.0}
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    return {
        "R2": float(r2_score(y_true_clean, y_pred_clean)),
        "MAE": float(mean_absolute_error(y_true_clean, y_pred_clean)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
    }


def build_lstm_sequences(X_data, y_data, lookback_steps=12):
    Xs, ys = [], []
    for i in range(len(X_data) - lookback_steps):
        Xs.append(X_data[i : (i + lookback_steps)])
        ys.append(y_data[i + lookback_steps])
    return np.array(Xs), np.array(ys)


def compute_dynamic_graph_edges(wind_u, wind_v, station_coords):
    sources, targets, weights = [], [], []
    w_vec = np.array([wind_u, wind_v])
    for i, s_idx in enumerate(station_coords.keys()):
        for j, t_idx in enumerate(station_coords.keys()):
            if i == j:
                continue
            c_i, c_j = station_coords[s_idx], station_coords[t_idx]
            d_ij = np.array([c_j[0] - c_i[0], c_j[1] - c_i[1]])
            dist = np.linalg.norm(d_ij)
            if dist == 0:
                continue
            d_ij_norm = d_ij / dist
            alignment = np.dot(d_ij_norm, w_vec)
            base_weight = 1.0 / (dist + 1e-5)
            weight = base_weight * (1.0 + alignment) if alignment > 0 else base_weight * np.exp(alignment)
            sources.append(i)
            targets.append(j)
            weights.append(weight)
    return torch.tensor([sources, targets], dtype=torch.long), torch.tensor(weights, dtype=torch.float).unsqueeze(-1)


def build_graph_sequences(scaled_data, raw_df, lookback=12, horizon=1):
    X_graphs = []
    for t in range(len(scaled_data) - lookback - horizon + 1):
        target_val = raw_df["target_shifted"].iloc[t + lookback - 1]
        y_tensor = torch.tensor([target_val] * NUM_STATIONS, dtype=torch.float)
        node_feats = [scaled_data[t : t + lookback] * (1.0 + (s * 0.05)) for s in range(NUM_STATIONS)]
        node_feats = torch.tensor(np.array(node_feats), dtype=torch.float).transpose(1, 2)
        u, v = raw_df["wind_u"].iloc[t + lookback - 1], raw_df["wind_v"].iloc[t + lookback - 1]
        edge_index, edge_attr = compute_dynamic_graph_edges(u, v, STATION_COORDS)
        X_graphs.append(Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor))
    return X_graphs


class AirQualitySTGNN(nn.Module):
    def __init__(self, num_features, num_timesteps_input):
        super().__init__()
        self.gat1 = GATConv(num_features, 32, heads=2, concat=True)
        self.gat2 = GATConv(32 * 2, 32, heads=1, concat=False)
        self.tcn1 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.tcn2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.skip_project = nn.Linear(num_features * num_timesteps_input, 16 * num_timesteps_input)
        self.regression_head = nn.Linear(16 * num_timesteps_input, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        num_nodes = x.size(0)
        x_skip = x.view(num_nodes, -1)
        x_skip = self.skip_project(x_skip)
        spatial_slices = []
        for t in range(x.size(2)):
            h_s = F.relu(self.gat1(x[:, :, t], edge_index, edge_attr=edge_attr))
            h_s = self.gat2(h_s, edge_index, edge_attr=edge_attr)
            spatial_slices.append(h_s)
        x_space = torch.stack(spatial_slices, dim=-1)
        x_time = F.relu(self.tcn1(x_space))
        x_time = F.relu(self.tcn2(x_time))
        x_flat = x_time.view(num_nodes, -1) + x_skip
        return self.regression_head(x_flat).squeeze(-1)


def _log_test_metrics(metrics: dict[str, float], prefix: str):
    if mlflow is None:
        return
    for metric_name, metric_val in metrics.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", float(metric_val))


def _normalize_results(results_master: dict[int, dict]) -> dict[str, dict]:
    # Keep backward compatibility with project consumers that read JSON output.
    normalized: dict[str, dict] = {}
    for horizon, models in results_master.items():
        hkey = f"h{horizon}"
        normalized[hkey] = {}
        for model_name, splits in models.items():
            normalized[hkey][model_name] = {
                "train": {"r2": splits["Train"]["R2"], "mae": splits["Train"]["MAE"], "mse": splits["Train"]["MSE"], "rmse": splits["Train"]["RMSE"]},
                "val": {"r2": splits["Val"]["R2"], "mae": splits["Val"]["MAE"], "mse": splits["Val"]["MSE"], "rmse": splits["Val"]["RMSE"]},
                "test": {"r2": splits["Test"]["R2"], "mae": splits["Test"]["MAE"], "mse": splits["Test"]["MSE"], "rmse": splits["Test"]["RMSE"]},
            }
    return normalized


def train_and_eval(
    model_name: str,
    path: Path,
    epochs: int = 5,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    device: str = "auto",
    rf_backend: str = "sklearn",
    horizon: int | None = None,
    log_to_mlflow: bool = True,
    mlflow_experiment_name: str = "Intelligent-IOT-baselines",
    *,
    seed: int = 42,
    weight_decay: float = 1e-4,
    patience: int = 5,
):
    del lr, hidden_dim, rf_backend, weight_decay, patience

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    resolved_path = _resolve_path(path)
    frame = _load_base_frame(resolved_path)

    if mlflow is not None and log_to_mlflow:
        try:
            mlflow.set_experiment(mlflow_experiment_name)
            print(f"MLflow experiment: {mlflow_experiment_name}")
        except Exception:
            pass

    feature_cols = [
        "temp definition °c",
        "dew point definition °c",
        "rel hum definition %",
        "wind_u",
        "wind_v",
        "pm25" if "pm25" in frame.columns else "pm2",
    ]
    feature_cols = [c for c in feature_cols if c in frame.columns]

    target_col = "pm25" if "pm25" in frame.columns else "pm2"
    horizons = [int(horizon)] if horizon is not None else [1, 2, 3]
    selected = model_name.lower()

    results_master: dict[int, dict] = {1: {}, 2: {}, 3: {}}

    for h in horizons:
        print("\n" + "=" * 85)
        print(f" RUNNING BENCHMARK MATRIX FOR HORIZON: +{h}-HOUR WINDOW")
        print("=" * 85)

        df_h = frame.copy()
        df_h["target_shifted"] = df_h[target_col].shift(-h)
        df_h = df_h.dropna(subset=feature_cols + ["target_shifted"]).copy()

        X = df_h[feature_cols].copy()
        y = df_h["target_shifted"].copy()

        n = len(X)
        train_end, val_end = int(n * 0.70), int(n * 0.85)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        y_train = y_train.to_numpy().astype(np.float32)
        y_val = y_val.to_numpy().astype(np.float32)
        y_test = y_test.to_numpy().astype(np.float32)

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        results_master[h] = {}

        if selected in ("ha", "all"):
            if mlflow is not None and log_to_mlflow:
                run = mlflow.start_run(run_name=f"H{h}_Historical_Average")
            else:
                run = None
            try:
                ha_mean = float(np.nanmean(y_train))
                train_m = compile_metrics(y_train, np.full(shape=y_train.shape, fill_value=ha_mean, dtype=np.float32))
                val_m = compile_metrics(y_val, np.full(shape=y_val.shape, fill_value=ha_mean, dtype=np.float32))
                test_m = compile_metrics(y_test, np.full(shape=y_test.shape, fill_value=ha_mean, dtype=np.float32))
                results_master[h]["Historical Average"] = {"Train": train_m, "Val": val_m, "Test": test_m}
                if run is not None:
                    _log_test_metrics(test_m, "Test")
            finally:
                if run is not None:
                    mlflow.end_run()

        if selected in ("lr", "all"):
            if mlflow is not None and log_to_mlflow:
                run = mlflow.start_run(run_name=f"H{h}_Linear_Regression")
            else:
                run = None
            try:
                lr_model = LinearRegression().fit(X_train_scaled, y_train)
                train_m = compile_metrics(y_train, lr_model.predict(X_train_scaled))
                val_m = compile_metrics(y_val, lr_model.predict(X_val_scaled))
                test_m = compile_metrics(y_test, lr_model.predict(X_test_scaled))
                results_master[h]["Linear Regression"] = {"Train": train_m, "Val": val_m, "Test": test_m}
                if run is not None:
                    _log_test_metrics(test_m, "Test")
            finally:
                if run is not None:
                    mlflow.end_run()

        if selected in ("rf", "all"):
            if mlflow is not None and log_to_mlflow:
                run = mlflow.start_run(run_name=f"H{h}_Tree_Regressor")
            else:
                run = None
            try:
                if USE_LGBM:
                    tree = LGBMRegressor(
                        n_estimators=200,
                        learning_rate=0.04,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=seed,
                        n_jobs=-1,
                        verbose=-1,
                    )
                    tree.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)])
                else:
                    tree = RandomForestRegressor(n_estimators=40, max_depth=12, random_state=seed, n_jobs=-1)
                    tree.fit(X_train_scaled, y_train)
                train_m = compile_metrics(y_train, tree.predict(X_train_scaled))
                val_m = compile_metrics(y_val, tree.predict(X_val_scaled))
                test_m = compile_metrics(y_test, tree.predict(X_test_scaled))
                results_master[h]["Gradient Boosting/RF"] = {"Train": train_m, "Val": val_m, "Test": test_m}
                if run is not None:
                    _log_test_metrics(test_m, "Test")
            finally:
                if run is not None:
                    mlflow.end_run()

        if selected in ("lstm", "all"):
            if mlflow is not None and log_to_mlflow:
                run = mlflow.start_run(run_name=f"H{h}_LSTM_Network")
            else:
                run = None
            try:
                X_tr_3d, y_tr_3d = build_lstm_sequences(X_train_scaled, y_train, LOOKBACK_STEPS)
                X_va_3d, y_va_3d = build_lstm_sequences(X_val_scaled, y_val, LOOKBACK_STEPS)
                X_te_3d, y_te_3d = build_lstm_sequences(X_test_scaled, y_test, LOOKBACK_STEPS)

                if USE_TF:
                    lstm = Sequential(
                        [
                            tf.keras.Input(shape=(X_tr_3d.shape[1], X_tr_3d.shape[2])),
                            LSTM(64, activation="tanh", return_sequences=True),
                            Dropout(0.2),
                            LSTM(32, activation="tanh", return_sequences=False),
                            Dropout(0.2),
                            Dense(16, activation="relu"),
                            Dense(1),
                        ]
                    )
                    lstm.compile(optimizer="adam", loss="mse")
                    lstm.fit(X_tr_3d, y_tr_3d, validation_data=(X_va_3d, y_va_3d), epochs=epochs, batch_size=256, verbose=0)
                    train_pred = lstm.predict(X_tr_3d, verbose=0).flatten()
                    val_pred = lstm.predict(X_va_3d, verbose=0).flatten()
                    test_pred = lstm.predict(X_te_3d, verbose=0).flatten()
                else:
                    class _TorchLSTM(nn.Module):
                        def __init__(self, in_dim: int):
                            super().__init__()
                            self.l1 = nn.LSTM(in_dim, 64, batch_first=True)
                            self.d1 = nn.Dropout(0.2)
                            self.l2 = nn.LSTM(64, 32, batch_first=True)
                            self.d2 = nn.Dropout(0.2)
                            self.f1 = nn.Linear(32, 16)
                            self.f2 = nn.Linear(16, 1)

                        def forward(self, x):
                            out, _ = self.l1(x)
                            out = self.d1(out)
                            out, _ = self.l2(out)
                            out = self.d2(out[:, -1, :])
                            out = torch.relu(self.f1(out))
                            return self.f2(out)

                    tlstm = _TorchLSTM(X_tr_3d.shape[2]).to(device)
                    opt = torch.optim.Adam(tlstm.parameters(), lr=1e-3)
                    loss_fn = nn.MSELoss()
                    xtr = torch.tensor(X_tr_3d, dtype=torch.float32).to(device)
                    ytr = torch.tensor(y_tr_3d, dtype=torch.float32).unsqueeze(1).to(device)
                    for _ in range(epochs):
                        opt.zero_grad()
                        loss = loss_fn(tlstm(xtr), ytr)
                        loss.backward()
                        opt.step()
                    tlstm.eval()
                    with torch.no_grad():
                        train_pred = tlstm(xtr).detach().cpu().numpy().flatten()
                        val_pred = tlstm(torch.tensor(X_va_3d, dtype=torch.float32).to(device)).detach().cpu().numpy().flatten()
                        test_pred = tlstm(torch.tensor(X_te_3d, dtype=torch.float32).to(device)).detach().cpu().numpy().flatten()

                train_m = compile_metrics(y_tr_3d, train_pred)
                val_m = compile_metrics(y_va_3d, val_pred)
                test_m = compile_metrics(y_te_3d, test_pred)
                results_master[h]["LSTM Sequential"] = {"Train": train_m, "Val": val_m, "Test": test_m}
                if run is not None:
                    _log_test_metrics(test_m, "Test")
            finally:
                if run is not None:
                    mlflow.end_run()

        if selected in ("stgnn", "all") and USE_PYG:
            if mlflow is not None and log_to_mlflow:
                run = mlflow.start_run(run_name=f"H{h}_Graph_STGNN")
            else:
                run = None
            try:
                scaled_all = RobustScaler().fit_transform(df_h[feature_cols])
                X_graphs = build_graph_sequences(scaled_all, df_h, lookback=LOOKBACK_STEPS, horizon=h)
                g_train = X_graphs[:train_end]
                g_val = X_graphs[train_end:val_end]
                g_test = X_graphs[val_end:]

                loader_tr = PyGDataLoader(g_train, batch_size=64, shuffle=False)
                loader_va = PyGDataLoader(g_val, batch_size=128, shuffle=False)
                loader_te = PyGDataLoader(g_test, batch_size=128, shuffle=False)

                stgnn = AirQualitySTGNN(num_features=len(feature_cols), num_timesteps_input=LOOKBACK_STEPS)
                opt = torch.optim.AdamW(stgnn.parameters(), lr=0.002, weight_decay=1e-3)
                criterion = nn.MSELoss()

                stgnn.train()
                for _ in range(25):
                    for batch in loader_tr:
                        opt.zero_grad()
                        out = stgnn(batch)
                        loss = criterion(out, batch.y.flatten())
                        loss.backward()
                        opt.step()

                stgnn.eval()

                def _preds(loader):
                    preds, trues = [], []
                    with torch.no_grad():
                        for batch in loader:
                            preds.append(stgnn(batch).detach().cpu().numpy())
                            trues.append(batch.y.flatten().detach().cpu().numpy())
                    return np.concatenate(trues), np.concatenate(preds)

                y_tr_true, tr_p = _preds(loader_tr)
                y_va_true, va_p = _preds(loader_va)
                y_te_true, te_p = _preds(loader_te)
                train_m = compile_metrics(y_tr_true, tr_p)
                val_m = compile_metrics(y_va_true, va_p)
                test_m = compile_metrics(y_te_true, te_p)
                results_master[h]["Spatiotemporal Graph"] = {"Train": train_m, "Val": val_m, "Test": test_m}
                if run is not None:
                    _log_test_metrics(test_m, "Test")
            finally:
                if run is not None:
                    mlflow.end_run()

    print("\n" + "=" * 95)
    print("             MASTER HEAD-TO-HEAD COMPREHENSIVE PERFORMANCE EVALUATION")
    print("=" * 95)
    for h in horizons:
        print(f"\n[TIMEFRAME EVALUATION MATRIX]: +{h}-HOUR FORECASTING WINDOW")
        print("-" * 95)
        rows = []
        for model_name_row, split_dict in results_master[h].items():
            for split_name, metrics in split_dict.items():
                rows.append(
                    {
                        "Model Name": model_name_row,
                        "Data Split": split_name,
                        "R2": metrics["R2"],
                        "MAE": metrics["MAE"],
                        "MSE": metrics["MSE"],
                        "RMSE": metrics["RMSE"],
                    }
                )
        if rows:
            print(pd.DataFrame(rows).set_index(["Model Name", "Data Split"]).round(4))
        else:
            print("No models were executed for this horizon.")
        print("-" * 95)

    out_path = Path("models/saved_models/baseline_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"timestamp": time.time(), "results": _normalize_results(results_master)}, fh, indent=2)

    return results_master


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ha", "lr", "rf", "lstm", "stgnn", "all"], default="all")
    p.add_argument("--path", type=Path, default=DEFAULT_PATH)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--rf-backend", choices=["sklearn", "xgboost"], default="sklearn")
    p.add_argument("--horizon", type=int, choices=[1, 2, 3], default=None)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--mlflow-experiment-name", default="Intelligent-IOT-baselines")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    args = p.parse_args()
    train_and_eval(
        args.model,
        args.path,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        device=args.device,
        rf_backend=args.rf_backend,
        horizon=args.horizon,
        log_to_mlflow=not args.no_mlflow,
        mlflow_experiment_name=args.mlflow_experiment_name,
        seed=args.seed,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )


if __name__ == "__main__":
    cli()
