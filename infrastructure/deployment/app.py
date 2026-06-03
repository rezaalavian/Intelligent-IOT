"""FastAPI application for forecasting and alerts."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from infrastructure.deployment.controller import ForecastController, load_controller


app = FastAPI(title="Intelligent-IOT API", version="0.1.0")
controller: ForecastController = load_controller()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict) -> dict:
    return controller.predict(payload)


@app.post("/alerts")
def alerts(payload: dict) -> dict:
    return controller.evaluate_alerts(payload)
