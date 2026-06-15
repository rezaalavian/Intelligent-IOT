"""Evaluation metrics for forecasting and operational latency."""

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    r2: float


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> ForecastMetrics:
    true_values = np.asarray(list(y_true), dtype=float)
    pred_values = np.asarray(list(y_pred), dtype=float)
    mae = mean_absolute_error(true_values, pred_values)
    rmse = float(np.sqrt(mean_squared_error(true_values, pred_values)))
    r2 = r2_score(true_values, pred_values)
    return ForecastMetrics(mae=mae, rmse=rmse, r2=r2)


def timed_call(fn, *args, **kwargs):
    start = perf_counter()
    result = fn(*args, **kwargs)
    elapsed = perf_counter() - start
    return result, elapsed
