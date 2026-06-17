"""Deployment bundle for CPU inference and the Phase 5 API/dashboard."""
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import numpy as np

@dataclass
class ForecastBundle:
    """Serializable artifact: scaler + per-horizon models + feature schema."""

    feature_columns: list[str]
    target_column: str
    horizons: list[int]
    model_type: str
    scaler: Any
    models: dict[int, Any] = field(default_factory=dict)
    scalers: dict[int, Any] | None = None
    feature_columns_by_horizon: dict[int, list[str]] | None = None
    metrics: dict[str, Any] | None = None
    lookback: int = 12

    def _horizon_scaler(self, horizon: int) -> Any:
        if self.scalers and int(horizon) in self.scalers:
            return self.scalers[int(horizon)]
        return self.scaler

    def _feature_row(self, features: Mapping[str, Any] | Sequence[Any], horizon: int | None = None) -> np.ndarray:
        cols = self.feature_columns
        if horizon is not None and self.feature_columns_by_horizon:
            cols = self.feature_columns_by_horizon.get(int(horizon), cols)
        if isinstance(features, Mapping):
            row = [float(features.get(c, 0.0) or 0.0) for c in cols]
        else:
            row = [float(v) for v in features]
        return np.asarray(row, dtype=np.float32).reshape(1, -1)

    def predict_horizon(
        self,
        features: Mapping[str, Any] | Sequence[Any],
        horizon: int,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> float:
        model = self.models.get(int(horizon))
        if model is None:
            raise KeyError(f"No model registered for horizon +{horizon}h")
        if hasattr(model, "predict_from_payload"):
            return float(model.predict_from_payload(features, history=history))
        row = self._feature_row(features, horizon)
        scaled = self._horizon_scaler(horizon).transform(row)
        pred = model.predict(scaled)
        return float(np.asarray(pred).flatten()[0])

    def predict(
        self,
        features: Mapping[str, Any] | Sequence[Any],
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        return {
            f"h{h}": self.predict_horizon(features, h, history=history)
            for h in self.horizons
        }

    def predict_pm25(
        self,
        features: Mapping[str, Any] | Sequence[Any],
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        forecasts = self.predict(features, history=history)
        primary = forecasts.get("h1", next(iter(forecasts.values()), 0.0))
        return {
            "target": self.target_column,
            "horizons": self.horizons,
            "forecasts": forecasts,
            "forecast_pm25": primary,
            "model_type": self.model_type,
        }


def is_forecast_bundle(obj: Any) -> bool:
    return isinstance(obj, ForecastBundle)
