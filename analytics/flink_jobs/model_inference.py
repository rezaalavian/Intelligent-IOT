"""Flink-friendly inference helpers for CPU deployment."""
from pathlib import Path
from typing import Any, Mapping, Sequence
from models.forecast_bundle import ForecastBundle, is_forecast_bundle
from models.model_io import load_model


def load_predictor(model_path: str | Path) -> Any:
    """Load the serialized model artifact for repeated CPU inference."""
    return load_model(model_path)


def predict_record(predictor: Any, features: dict[str, Any]) -> Any:
    """Predict a single record using a preloaded model or ForecastBundle."""
    if is_forecast_bundle(predictor):
        return predictor.predict_pm25(features)
    if hasattr(predictor, "predict_pm25"):
        return predictor.predict_pm25(features)
    if hasattr(predictor, "predict"):
        row = _to_feature_row(features)
        return predictor.predict([row])
    if callable(predictor):
        return predictor(features)
    raise TypeError("Unsupported predictor type")


def predict_horizons(predictor: Any, features: Mapping[str, Any]) -> dict[str, float]:
    if isinstance(predictor, ForecastBundle):
        return predictor.predict(features)
    result = predict_record(predictor, dict(features))
    if isinstance(result, dict) and "forecasts" in result:
        return result["forecasts"]
    scalar = float(result[0]) if hasattr(result, "__len__") else float(result)
    return {"h1": scalar}


def _to_feature_row(features: Mapping[str, Any] | Sequence[Any]) -> list[Any]:
    if isinstance(features, Mapping):
        return [features[key] for key in sorted(features.keys())]
    return list(features)
