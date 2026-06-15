from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
except Exception:  # pragma: no cover - optional dependency
    RandomForestRegressor = None
    LinearRegression = None


@dataclass(frozen=True)
class HorizonResult:
    horizon: int
    model_name: str
    mae: float
    rmse: float
    r2: float


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2) / total) if total else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _make_model(model_name: str, random_state: int = 42) -> Any:
    if model_name == "lgbm" and LGBMRegressor is not None:
        return LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )
    if model_name in {"rf", "plus"} and RandomForestRegressor is not None:
        return RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    if LinearRegression is not None:
        return LinearRegression()
    raise RuntimeError("No supported regression backend is available")


def _feature_target_split(
    frame: pd.DataFrame,
    target_column: str,
    horizon: int,
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    target_name = f"{target_column}_h{horizon}"
    if target_name not in frame.columns:
        raise KeyError(f"Missing horizon target column: {target_name}")
    if feature_columns is None:
        feature_columns = [column for column in frame.columns if column != target_name]
    features = frame.loc[:, [column for column in feature_columns if column in frame.columns]].copy()
    target = frame[target_name].copy()
    mask = features.notna().all(axis=1) & target.notna()
    return features.loc[mask], target.loc[mask]


def train_horizon_plus_models(
    frame: pd.DataFrame,
    horizons: Iterable[int],
    target_column: str = "pm2_5",
    feature_columns: Sequence[str] | None = None,
    model_name: str = "plus",
    random_state: int = 42,
) -> list[HorizonResult]:
    results: list[HorizonResult] = []
    for horizon in horizons:
        features, target = _feature_target_split(frame, target_column, horizon, feature_columns)
        if features.empty:
            results.append(HorizonResult(horizon=horizon, model_name=model_name, mae=float("nan"), rmse=float("nan"), r2=float("nan")))
            continue
        split_index = max(1, int(len(features) * 0.8))
        x_train = features.iloc[:split_index]
        x_test = features.iloc[split_index:]
        y_train = target.iloc[:split_index]
        y_test = target.iloc[split_index:]
        model = _make_model(model_name, random_state=random_state)
        model.fit(x_train, y_train)
        if x_test.empty:
            predictions = np.asarray(model.predict(x_train))[-len(y_train) :]
            y_test = y_train
        else:
            predictions = np.asarray(model.predict(x_test))
        metrics = _safe_metrics(np.asarray(y_test), predictions)
        results.append(
            HorizonResult(
                horizon=horizon,
                model_name=model_name,
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r2=metrics["r2"],
            )
        )
    return results


def summarize_results(results: Sequence[HorizonResult]) -> pd.DataFrame:
    return pd.DataFrame([result.__dict__ for result in results])