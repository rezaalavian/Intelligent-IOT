from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

from models.forecast_bundle import ForecastBundle
from models.model_io import load_model, save_model


def test_forecast_bundle_predicts_all_horizons(tmp_path):
    scaler = RobustScaler().fit([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    model_h1 = LinearRegression().fit(scaler.transform([[1.0, 2.0], [2.0, 3.0]]), [3.0, 5.0])
    model_h2 = LinearRegression().fit(scaler.transform([[1.0, 2.0], [2.0, 3.0]]), [4.0, 6.0])
    bundle = ForecastBundle(
        feature_columns=["a", "b"],
        target_column="pm2",
        horizons=[1, 2],
        model_type="linear_regression",
        scaler=scaler,
        scalers={1: scaler, 2: scaler},
        models={1: model_h1, 2: model_h2},
    )
    preds = bundle.predict({"a": 1.0, "b": 2.0})
    assert "h1" in preds and "h2" in preds
    assert np.isfinite(preds["h1"])

    path = save_model(bundle, tmp_path / "bundle.pkl")
    loaded = load_model(path)
    assert isinstance(loaded, ForecastBundle)
    assert loaded.predict_horizon({"a": 1.0, "b": 2.0}, 1) == preds["h1"]
