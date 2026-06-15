"""FastAPI application for forecasting and alerts."""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from infrastructure.deployment.controller import ForecastController, load_controller, METRICS_PATH


class FeaturePayload(BaseModel):
    features: dict[str, float] = Field(default_factory=dict)
    history: list[dict[str, float]] | None = None
    pm25: float | None = None
    pm2: float | None = None
    forecast_pm25: float | None = None


class HorizonConfig(BaseModel):
    h1: str = "stgnn"
    h2: str = "stgnn"
    h3: str = "stgnn"


app = FastAPI(title="Intelligent-IOT API", version="0.3.0")
controller: ForecastController = load_controller()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict:
    return controller.status()


@app.get("/metrics")
def metrics() -> dict:
    if controller.metrics:
        return controller.metrics
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="No metrics artifact found")


@app.post("/configure-horizons")
def configure_horizons(cfg: HorizonConfig) -> dict:
    """Select a different saved model per forecast horizon (+1h, +2h, +3h)."""
    global controller
    controller = load_controller(horizon_models={1: cfg.h1, 2: cfg.h2, 3: cfg.h3})
    return controller.status()


@app.post("/predict")
def predict(payload: FeaturePayload | dict) -> dict:
    body = payload.model_dump() if isinstance(payload, FeaturePayload) else payload
    if not controller.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run: python scripts/save_deployment_models.py",
        )
    return controller.predict(body)


@app.post("/alerts")
def alerts(payload: FeaturePayload | dict) -> dict:
    body = payload.model_dump() if isinstance(payload, FeaturePayload) else payload
    return controller.evaluate_alerts(body)


@app.post("/forecast-and-alert")
def forecast_and_alert(payload: FeaturePayload | dict) -> dict:
    body = payload.model_dump() if isinstance(payload, FeaturePayload) else payload
    if not controller.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    forecast = controller.predict(body)
    alert = controller.evaluate_alerts({**body, **forecast})
    return {"forecast": forecast, "alert": alert}
