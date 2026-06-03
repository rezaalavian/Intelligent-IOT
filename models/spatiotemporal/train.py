"""Training routine for the spatiotemporal model scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

try:
    import mlflow  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional runtime dependency
    mlflow = None

# MLflow is intentionally disabled for the cleaned project baseline.
mlflow = None

from analytics.flink_jobs.feature_engineering import introduce_raw_features
from models.model_io import save_model
from models.spatiotemporal.model import SpatioTemporalModel


HORIZONS = (1, 2, 3)
LOOKBACK = 24
TARGET_COLUMN = "pm2"


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_columns: list[str]


@dataclass(frozen=True)
class TrainResult:
    artifact_path: Path
    train_mae: float
    val_mae: float
    test_mae: float
    train_rmse: float
    val_rmse: float
    test_rmse: float
    train_r2: float
    val_r2: float
    test_r2: float
    target: str | None = None
    n_samples: int | None = None
    per_horizon_metrics: dict | None = None


def load_training_frame(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if max_rows is not None:
        frame = frame.head(max_rows)
    frame = introduce_raw_features(frame)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce").dt.floor("h")
    frame = frame.sort_values([column for column in ["city_name", "timestamp"] if column in frame.columns])
    if "city_name" in frame.columns:
        city_dummies = pd.get_dummies(frame["city_name"].astype("string"), prefix="city", dtype=float)
        frame = pd.concat([frame, city_dummies], axis=1)
        frame["city_code"] = frame["city_name"].astype("category").cat.codes.astype(float)
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    # Avoid using future values to impute past values (prevents data leakage).
    # Use forward-direction interpolation + forward-fill, then fallback to zeros.
    if "city_name" in frame.columns:
        def _impute_group(g: pd.DataFrame) -> pd.DataFrame:
            return g.interpolate(limit_direction="forward").ffill().fillna(g.median())

        # Use groupby.transform to produce an aligned DataFrame (avoids MultiIndex from apply)
        imputed = frame.groupby("city_name", sort=False)[numeric_columns].transform(
            lambda g: g.interpolate(limit_direction="forward").ffill().fillna(g.median())
        )
        frame.loc[:, numeric_columns] = imputed.fillna(0.0)
    else:
        frame[numeric_columns] = frame[numeric_columns].interpolate(limit_direction="forward").ffill().fillna(0.0)
    # drop numeric columns that remain entirely NaN (e.g. targets that were missing)
    all_nan = [c for c in numeric_columns if frame[c].isna().all()]
    if all_nan:
        print(f"Dropping all-NaN numeric columns: {all_nan}")
        frame = frame.drop(columns=all_nan)
    return frame


def build_windows(frame: pd.DataFrame, lookback: int = LOOKBACK, horizons: Iterable[int] = HORIZONS, target_column: str = TARGET_COLUMN) -> tuple[np.ndarray, np.ndarray, list[str]]:
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    feature_frame = frame.select_dtypes(include=[np.number]).copy()
    if target_column not in feature_frame.columns:
        raise ValueError(f"{target_column} target column is missing")

    feature_columns = list(feature_frame.columns)
    target_values = feature_frame[target_column].to_numpy(dtype=float)
    feature_values = feature_frame[feature_columns].to_numpy(dtype=float)

    max_horizon = max(horizons)
    samples: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for end_idx in range(lookback, len(feature_frame) - max_horizon):
        start_idx = end_idx - lookback
        window = feature_values[start_idx:end_idx]
        future_targets = np.asarray([target_values[end_idx + horizon] for horizon in horizons], dtype=float)
        samples.append(window)
        targets.append(future_targets)

    if not samples:
        raise ValueError("Not enough rows to build training windows")

    return np.asarray(samples, dtype=float), np.asarray(targets, dtype=float), feature_columns


def build_grouped_windows(
    frame: pd.DataFrame,
    lookback: int = LOOKBACK,
    horizons: Iterable[int] = HORIZONS,
    target_column: str = TARGET_COLUMN,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    if "city_name" not in frame.columns:
        return build_windows(frame, lookback=lookback, horizons=horizons, target_column=target_column)

    samples: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    feature_columns: list[str] | None = None

    for _, city_frame in frame.groupby("city_name", sort=False):
        city_samples, city_targets, city_features = build_windows(
            city_frame,
            lookback=lookback,
            horizons=horizons,
            target_column=target_column,
        )
        if feature_columns is None:
            feature_columns = city_features
        samples.append(city_samples)
        targets.append(city_targets)

    if not samples or feature_columns is None:
        raise ValueError("Not enough rows to build grouped training windows")

    return np.concatenate(samples, axis=0), np.concatenate(targets, axis=0), feature_columns


def grouped_chronological_split(
    frame: pd.DataFrame,
    lookback: int = LOOKBACK,
    horizons: Iterable[int] = HORIZONS,
    target_column: str = TARGET_COLUMN,
) -> tuple[SplitData, list[str]]:
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    if "city_name" not in frame.columns:
        X, y, feature_columns = build_windows(frame, lookback=lookback, horizons=horizons, target_column=target_column)
        split = chronological_split(X, y)
        return split, feature_columns

    train_parts: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    val_targets: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    test_targets: list[np.ndarray] = []
    feature_columns: list[str] | None = None

    for _, city_frame in frame.groupby("city_name", sort=False):
        city_samples, city_targets, city_features = build_windows(
            city_frame,
            lookback=lookback,
            horizons=horizons,
            target_column=target_column,
        )
        if feature_columns is None:
            feature_columns = city_features
        city_split = chronological_split(city_samples, city_targets)
        if len(city_split.X_train):
            train_parts.append(city_split.X_train)
            train_targets.append(city_split.y_train)
        if len(city_split.X_val):
            val_parts.append(city_split.X_val)
            val_targets.append(city_split.y_val)
        if len(city_split.X_test):
            test_parts.append(city_split.X_test)
            test_targets.append(city_split.y_test)

    if feature_columns is None or not train_parts or not val_parts or not test_parts:
        raise ValueError("Not enough grouped data to create validation and test splits")

    split = SplitData(
        X_train=np.concatenate(train_parts, axis=0),
        y_train=np.concatenate(train_targets, axis=0),
        X_val=np.concatenate(val_parts, axis=0),
        y_val=np.concatenate(val_targets, axis=0),
        X_test=np.concatenate(test_parts, axis=0),
        y_test=np.concatenate(test_targets, axis=0),
        feature_columns=feature_columns,
    )
    return split, feature_columns


def chronological_split(X: np.ndarray, y: np.ndarray) -> SplitData:
    total = len(X)
    train_end = max(int(total * 0.7), 1)
    val_end = max(int(total * 0.85), train_end + 1)
    val_end = min(val_end, total - 1) if total > 2 else total

    return SplitData(
        X_train=X[:train_end],
        y_train=y[:train_end],
        X_val=X[train_end:val_end],
        y_val=y[train_end:val_end],
        X_test=X[val_end:],
        y_test=y[val_end:],
        feature_columns=[],
    )


def fit_input_stats(X_train: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = X_train.reshape(-1, X_train.shape[-1])
    mean = torch.tensor(flattened.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(flattened.std(axis=0), dtype=torch.float32)
    std = torch.clamp(std, min=1e-6)
    return mean, std


def evaluate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Remove samples with NaNs in either true or predicted arrays
    import numpy as _np

    y_true = _np.asarray(y_true)
    y_pred = _np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape for evaluation")

    valid_mask = _np.isfinite(y_true).all(axis=1) & _np.isfinite(y_pred).all(axis=1)
    if valid_mask.sum() == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}

    y_true_f = y_true[valid_mask]
    y_pred_f = y_pred[valid_mask]

    # Compute metrics; r2_score can fail on constant arrays, guard it
    mae = float(mean_absolute_error(y_true_f, y_pred_f))
    mse = float(mean_squared_error(y_true_f, y_pred_f))
    rmse = float(np.sqrt(mse))
    try:
        r2 = float(r2_score(y_true_f, y_pred_f))
        if not _np.isfinite(r2):
            r2 = 0.0
    except Exception:
        r2 = 0.0

    return {"mae": mae, "rmse": rmse, "r2": r2}


def train(
    path: str | Path = "data/raw/historical_rawdata.csv",
    output_path: str | Path = "models/saved_models/spatiotemporal_model.pt",
    max_rows: int | None = None,
    target_column: str | None = None,
    *,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 60,
    patience: int = 8,
    log_to_mlflow: bool = True,
    mlflow_experiment_name: str = "Intelligent-IOT-spatiotemporal",
    mlflow_run_name: str | None = None,
    mlflow_tracking_dir: str | Path = "mlruns",
) -> TrainResult:
    # Note: deterministic seed can be provided via environment or callers should set global seeds.
    # make runs deterministic where possible
    try:
        import random as _random

        _random.seed(42)
    except Exception:
        pass
    try:
        np.random.seed(42)
    except Exception:
        pass
    try:
        import torch as _torch

        _torch.manual_seed(42)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(42)
        try:
            _torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass

    frame = load_training_frame(path, max_rows=max_rows)
    # determine selected target (CLI overrides default)
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    selected_target = target_column or TARGET_COLUMN
    if selected_target not in numeric_cols or frame[selected_target].notna().sum() == 0:
        # choose the numeric column with the most non-null values
        counts = {col: int(frame[col].notna().sum()) for col in numeric_cols}
        counts = {k: v for k, v in counts.items() if v > 0}
        if not counts:
            raise ValueError("No numeric target column with non-null values found in frame")
        selected_target = max(counts.items(), key=lambda x: x[1])[0]
        print(f"Warning: target '{TARGET_COLUMN}' is empty; using fallback target '{selected_target}' for this training run")

    split, feature_columns = grouped_chronological_split(frame, target_column=selected_target)

    if len(split.X_val) == 0 or len(split.X_test) == 0:
        raise ValueError("Need more data to create validation and test splits")

    mean, std = fit_input_stats(split.X_train)
    model = SpatioTemporalModel(input_dim=split.X_train.shape[-1], hidden_dim=hidden_dim, output_dim=len(HORIZONS))
    model.set_normalization_stats(mean, std)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss()
    x_train = torch.tensor(split.X_train, dtype=torch.float32)
    y_train = torch.tensor(split.y_train, dtype=torch.float32)
    x_val = torch.tensor(split.X_val, dtype=torch.float32)
    y_val = torch.tensor(split.y_val, dtype=torch.float32)
    x_test = torch.tensor(split.X_test, dtype=torch.float32)
    y_test = torch.tensor(split.y_test, dtype=torch.float32)

    mlflow_run = None
    if log_to_mlflow and mlflow is not None:
        mlflow.set_tracking_uri(Path(mlflow_tracking_dir).resolve().as_uri())
        mlflow.set_experiment(mlflow_experiment_name)
        mlflow_run = mlflow.start_run(run_name=mlflow_run_name)

    try:
        best_state = None
        best_val_loss = float("inf")
        stagnation = 0

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = model(x_train)
            loss = criterion(preds, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(x_val), y_val).item()
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= patience:
                    break
            model.train()

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            train_pred = model(x_train).detach().cpu().numpy()
            val_pred = model(x_val).detach().cpu().numpy()
            test_pred = model(x_test).detach().cpu().numpy()

        artifact_path = save_model(model, output_path)
        train_metrics = evaluate_arrays(split.y_train, train_pred)
        val_metrics = evaluate_arrays(split.y_val, val_pred)
        test_metrics = evaluate_arrays(split.y_test, test_pred)

        # per-horizon metrics
        per_horizon: dict[int, dict[str, float]] = {}
        for i, horizon in enumerate(HORIZONS):
            try:
                per_horizon[horizon] = {
                    "train_mae": float(mean_absolute_error(split.y_train[:, i], train_pred[:, i])) if split.y_train.size else 0.0,
                    "val_mae": float(mean_absolute_error(split.y_val[:, i], val_pred[:, i])) if split.y_val.size else 0.0,
                    "test_mae": float(mean_absolute_error(split.y_test[:, i], test_pred[:, i])) if split.y_test.size else 0.0,
                }
            except Exception:
                per_horizon[horizon] = {"train_mae": 0.0, "val_mae": 0.0, "test_mae": 0.0}

        if mlflow_run is not None:
            mlflow.log_params(
                {
                    "path": str(path),
                    "max_rows": max_rows if max_rows is not None else "all",
                    "target_column": selected_target,
                    "hidden_dim": hidden_dim,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "patience": patience,
                    "lookback": LOOKBACK,
                    "horizons": ",".join(str(h) for h in HORIZONS),
                    "input_dim": split.X_train.shape[-1],
                    "n_samples": len(split.X_train) + len(split.X_val) + len(split.X_test),
                }
            )
            # Log model metadata: class name and total parameter count
            try:
                n_params = int(sum(p.numel() for p in model.parameters()))
            except Exception:
                n_params = -1
            mlflow.log_param("model_class", model.__class__.__name__)
            mlflow.log_param("n_parameters", n_params)
            mlflow.log_metrics(
                {
                    "train_mae": train_metrics["mae"],
                    "val_mae": val_metrics["mae"],
                    "test_mae": test_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "val_rmse": val_metrics["rmse"],
                    "test_rmse": test_metrics["rmse"],
                    "train_r2": train_metrics["r2"],
                    "val_r2": val_metrics["r2"],
                    "test_r2": test_metrics["r2"],
                }
            )
            for horizon, horizon_metrics in per_horizon.items():
                mlflow.log_metrics(
                    {
                        f"h{horizon}_train_mae": horizon_metrics["train_mae"],
                        f"h{horizon}_val_mae": horizon_metrics["val_mae"],
                        f"h{horizon}_test_mae": horizon_metrics["test_mae"],
                    }
                )
            mlflow.log_artifact(str(artifact_path))
            # If available, log the PyTorch model as an MLflow model flavor so it appears under "Models"
            try:
                # mlflow.pytorch may not be available in minimal installs; guard against that
                if hasattr(mlflow, "pytorch"):
                    mlflow.pytorch.log_model(model, artifact_path="pytorch_model")
            except Exception as e:  # pragma: no cover - non-critical logging
                print(f"Warning: failed to log PyTorch model via mlflow.pytorch: {e}")

        return TrainResult(
            artifact_path=artifact_path,
            train_mae=train_metrics["mae"],
            val_mae=val_metrics["mae"],
            test_mae=test_metrics["mae"],
            train_rmse=train_metrics["rmse"],
            val_rmse=val_metrics["rmse"],
            test_rmse=test_metrics["rmse"],
            train_r2=train_metrics["r2"],
            val_r2=val_metrics["r2"],
            test_r2=test_metrics["r2"],
            target=selected_target,
            n_samples=len(split.X_train) + len(split.X_val) + len(split.X_test),
            per_horizon_metrics=per_horizon,
        )
    finally:
        if mlflow_run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    result = train()
    print(result)
