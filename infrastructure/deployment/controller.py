"""Decision logic for proactive response and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.model_io import load_model


@dataclass
class ForecastController:
    model: Any | None = None
    threshold_pm25: float = 150.0

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        features = payload.get("features", payload)
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
        return {"prediction": _to_serializable(prediction)}

    def evaluate_alerts(self, payload: dict[str, Any]) -> dict[str, Any]:
        current_pm25 = float(payload.get("pm25", 0.0) or 0.0)
        forecast_pm25 = float(payload.get("forecast_pm25", current_pm25) or current_pm25)
        alert = forecast_pm25 >= self.threshold_pm25 or current_pm25 >= self.threshold_pm25
        level = "warning" if alert else "normal"
        if forecast_pm25 >= 250 or current_pm25 >= 250:
            level = "critical"
        return {
            "alert": alert,
            "level": level,
            "current_pm25": current_pm25,
            "forecast_pm25": forecast_pm25,
        }


def load_controller(model_path: str | Path | None = None) -> ForecastController:
    if model_path is None:
        default_model = Path("models/saved_models/demo_model.pkl")
        model = load_model(default_model) if default_model.exists() else None
    else:
        path = Path(model_path)
        model = load_model(path) if path.exists() else None
    return ForecastController(model=model)


def _to_serializable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
