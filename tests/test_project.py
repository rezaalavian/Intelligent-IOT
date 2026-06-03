from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from analytics.flink_jobs.adjacency_matrix import compute_adjacency
from analytics.flink_jobs.feature_engineering import compute_rolling_features, introduce_raw_features
from analytics.flink_jobs.model_inference import predict_record
from analytics.recovery.kriging_spatial import spatial_interpolate
from infrastructure.deployment.controller import ForecastController
from infrastructure.kafka.scripts.data_downloader import DEFAULT_KEEP_COLUMNS, clean_raw_data
from models.model_io import load_model, save_model


def test_clean_raw_data_keeps_only_requested_columns(tmp_path: Path):
    source = tmp_path / "raw.csv"
    frame = pd.DataFrame({
        "Temp Definition °C": [1.0],
        "timestamp": ["2022-01-01 00:00:00"],
        "city_name": ["Toronto"],
        "extra": [123],
    })
    frame.to_csv(source, index=False)
    target = tmp_path / "clean.csv"
    clean_raw_data(source, target, DEFAULT_KEEP_COLUMNS)
    cleaned = pd.read_csv(target)
    assert list(cleaned.columns) == list(DEFAULT_KEEP_COLUMNS)
    assert "extra" not in cleaned.columns


def test_feature_engineering_adds_lags_and_rollings():
    frame = pd.DataFrame(
        [
            {"timestamp": "2022-01-01 00:00:00", "city_name": "Toronto", "pm2": 10.0, "no2": 3.0, "o3": 1.0, "co": 0.1, "so2": 0.02},
            {"timestamp": "2022-01-01 01:00:00", "city_name": "Toronto", "pm2": 12.0, "no2": 4.0, "o3": 1.2, "co": 0.12, "so2": 0.03},
        ]
    )
    engineered = compute_rolling_features(frame)
    assert "pm2_lag1" in engineered.columns
    assert "pm2_roll3" in engineered.columns


def test_raw_feature_introduction_keeps_base_columns():
    frame = pd.DataFrame(
        [
            {"timestamp": "2022-01-01 00:00:00", "city_name": "Toronto", "pm2": 10.0, "no2": 3.0, "o3": 1.0},
            {"timestamp": "2022-01-01 01:00:00", "city_name": "Toronto", "pm2": 12.0, "no2": 4.0, "o3": 1.2},
        ]
    )
    raw = introduce_raw_features(frame)
    assert "pm2" in raw.columns
    assert "no2" in raw.columns
    assert "pm2_lag1" not in raw.columns


def test_adjacency_matrix_shape():
    locations = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    winds = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    adjacency = compute_adjacency(locations, winds)
    assert adjacency.shape == (3, 3)


def test_recovery_fills_missing_values():
    frame = pd.DataFrame({"timestamp": ["2022-01-01 00:00:00", "2022-01-01 01:00:00"], "city_name": ["Toronto", "Toronto"], "pm2": [1.0, None]})
    recovered = spatial_interpolate(frame, ["pm2"])
    assert recovered["pm2"].isna().sum() == 0


def test_model_roundtrip_and_controller_alerts(tmp_path: Path):
    model = LinearRegression().fit([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], [3.0, 5.0, 7.0])
    artifact = save_model(model, tmp_path / "demo.pkl")
    loaded = load_model(artifact)
    prediction = predict_record(loaded, {"a": 1.0, "b": 2.0})
    assert len(prediction) == 1
    assert float(prediction[0]) == prediction[0]

    controller = ForecastController(model=None)
    response = controller.evaluate_alerts({"pm25": 200, "forecast_pm25": 210})
    assert response["alert"] is True
    assert response["level"] in {"warning", "critical"}
