"""Decision logic for proactive response and inference."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from models.forecast_bundle import ForecastBundle, is_forecast_bundle
from models.model_io import load_model
from models.model_registry import (
    ACTIVE_MODEL_KEY,
    MODEL_KEYS,
    REGISTRY_PATH,
    SAVED_MODELS_DIR,
    default_active_horizons,
    load_composite_bundle,
    parse_horizon_map,
)

DEFAULT_MODEL_PATH = SAVED_MODELS_DIR / "active_model.pkl"
METRICS_PATH = SAVED_MODELS_DIR / "baseline_metrics.json"
THRESHOLD_WARNING = 35.5
THRESHOLD_CRITICAL = 125.5


@dataclass
class ForecastController:
    bundle: ForecastBundle | None = None
    model: Any | None = None
    active_model_key: str = ACTIVE_MODEL_KEY
    active_horizons: dict[int, str] = field(default_factory=dict)
    threshold_pm25: float = THRESHOLD_WARNING
    metrics: dict[str, Any] = field(default_factory=dict)
    registry: dict[str, Any] = field(default_factory=dict)

    def is_ready(self) -> bool:
        return self.bundle is not None or self.model is not None

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        features = payload.get("features", payload)
        history = payload.get("history")
        if self.bundle is not None:
            result = self.bundle.predict_pm25(features, history=history)
            return {
                "ready": True,
                "target": result["target"],
                "forecasts": result["forecasts"],
                "forecast_pm25": result["forecast_pm25"],
                "model_type": result["model_type"],
                "active_model": self.active_model_key,
                "active_horizons": {str(h): m for h, m in sorted(self.active_horizons.items())},
                #"feature_columns": self.bundle.feature_columns,
                "lookback": self.bundle.lookback,
            }

        prediction = None
        if self.model is not None:
            if hasattr(self.model, "predict"):
                if isinstance(features, dict):
                    row = [features[key] for key in sorted(features.keys())]
                else:
                    row = list(features)
                prediction = self.model.predict([row])
            elif callable(self.model):
                prediction = self.model(features)
        return {
            "ready": prediction is not None,
            "prediction": _to_serializable(prediction),
            "forecasts": {},
            "forecast_pm25": _first_scalar(prediction),
            "active_model": self.active_model_key,
            "active_horizons": {str(h): m for h, m in sorted(self.active_horizons.items())},
        }

    def evaluate_alerts(self, payload: dict[str, Any]) -> dict[str, Any]:
        current_pm25 = float(
            payload.get("pm25")
            or payload.get("pm2")
            or payload.get("current_pm25")
            or 0.0
        )
        forecast_pm25 = float(payload.get("forecast_pm25") or current_pm25)

        if "features" in payload and self.bundle is not None and "forecast_pm25" not in payload:
            forecast_pm25 = float(self.predict(payload).get("forecast_pm25") or current_pm25)

        alert = forecast_pm25 >= self.threshold_pm25 or current_pm25 >= self.threshold_pm25
        level = "normal"
        if forecast_pm25 >= THRESHOLD_CRITICAL or current_pm25 >= THRESHOLD_CRITICAL:
            level = "critical"
        elif alert:
            level = "warning"

        recommendation = _recommendation(level, current_pm25, forecast_pm25)
        return {
            "alert": alert,
            "level": level,
            "current_pm25": current_pm25,
            "forecast_pm25": forecast_pm25,
            "threshold_warning": self.threshold_pm25,
            "threshold_critical": THRESHOLD_CRITICAL,
            "recommendation": recommendation,
            "active_model": self.active_model_key,
            "active_horizons": {str(h): m for h, m in sorted(self.active_horizons.items())},
        }

    def status(self) -> dict[str, Any]:
        available = list(self.registry.get("families", {}).keys())
        return {
            "model_loaded": self.is_ready(),
            "active_model": self.active_model_key,
            "active_horizons": {str(h): m for h, m in sorted(self.active_horizons.items())},
            "available_models": available,
            "model_type": self.bundle.model_type if self.bundle else type(self.model).__name__ if self.model else None,
            "horizons": self.bundle.horizons if self.bundle else [],
            "feature_columns": self.bundle.feature_columns if self.bundle else [],
            "target_column": self.bundle.target_column if self.bundle else None,
            "lookback": self.bundle.lookback if self.bundle else None,
            "metrics_available": bool(self.metrics),
            "registry_path": str(REGISTRY_PATH) if REGISTRY_PATH.exists() else None,
        }


def load_controller(
    model_path: str | Path | None = None,
    *,
    horizon_models: dict[int, str] | None = None,
) -> ForecastController:
    registry: dict[str, Any] = {}
    if REGISTRY_PATH.exists():
        try:
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            registry = {}

    active_key = os.environ.get("IOT_ACTIVE_MODEL", registry.get("active_model", ACTIVE_MODEL_KEY))
    active_horizons = parse_horizon_map(registry.get("active_horizons"), fallback_key=active_key)

    if horizon_models:
        active_horizons = horizon_models
        bundle = load_composite_bundle(horizon_models)
    else:
        path = Path(model_path) if model_path else Path(registry.get("active_path", DEFAULT_MODEL_PATH))
        bundle = None
        model = None

        if path.exists():
            loaded = load_model(path)
            if is_forecast_bundle(loaded):
                bundle = loaded
            else:
                model = None

        if bundle is None:
            try:
                bundle = load_composite_bundle(active_horizons)
            except FileNotFoundError:
                bundle = None

        if bundle is None:
            for candidate in (
                SAVED_MODELS_DIR / f"{active_key}_bundle.pkl",
                SAVED_MODELS_DIR / "stgnn_bundle.pkl",
            ):
                if candidate.exists():
                    loaded = load_model(candidate)
                    if is_forecast_bundle(loaded):
                        bundle = loaded
                        break

    metrics: dict[str, Any] = {}
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    if horizon_models:
        active_key = "composite" if len(set(horizon_models.values())) > 1 else next(iter(horizon_models.values()))

    return ForecastController(
        bundle=bundle,
        model=None,
        active_model_key=active_key,
        active_horizons=active_horizons,
        metrics=metrics,
        registry=registry,
    )


def _recommendation(level: str, current: float, forecast: float) -> str:
    if level == "critical":
        return "Activate emergency ventilation and restrict outdoor industrial activity immediately."
    if level == "warning":
        return "Increase monitoring frequency and prepare proactive emission controls before the forecast peak."
    if forecast > current + 10:
        return "Trend rising — schedule preventive checks within the next hour."
    return "Conditions normal — continue standard monitoring."


def _first_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        flat = value.tolist()
        if isinstance(flat, list) and flat:
            return float(flat[0] if not isinstance(flat[0], list) else flat[0][0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_serializable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
