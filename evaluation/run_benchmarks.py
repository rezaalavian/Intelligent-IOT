"""Run end-to-end benchmarks on held-out forecast windows."""

from pathlib import Path
import numpy as np
from evaluation.metrics import regression_metrics, timed_call
from infrastructure.deployment.controller import ForecastController
from models.spatiotemporal.train import build_windows, load_training_frame, train


def persistence_baseline(X_test: np.ndarray) -> np.ndarray:
    # Predict the next horizons using the latest observed pm2 value from each window.
    latest_pm2 = X_test[:, -1, _PM2_FEATURE_INDEX]
    return np.stack([latest_pm2, latest_pm2, latest_pm2, latest_pm2], axis=1)


_PM2_FEATURE_INDEX: int = -1


def main() -> None:
    dataset = Path("data/raw/historical_rawdata.csv")
    max_rows = 5000
    frame = load_training_frame(dataset, max_rows=max_rows)
    X, y, feature_columns = build_windows(frame)

    global _PM2_FEATURE_INDEX
    _PM2_FEATURE_INDEX = feature_columns.index("pm2")

    total = len(X)
    train_end = max(int(total * 0.7), 1)
    val_end = max(int(total * 0.85), train_end + 1)
    X_test = X[val_end:]
    y_test = y[val_end:]

    result = train(dataset, max_rows=max_rows)
    from models.model_io import load_model

    loaded_model = load_model(result.artifact_path)
    predictions, latency = timed_call(loaded_model.predict, X_test)
    metrics = regression_metrics(y_test.ravel(), predictions.ravel())

    baseline_predictions = persistence_baseline(X_test)
    baseline_metrics = regression_metrics(y_test.ravel(), baseline_predictions.ravel())

    controller = ForecastController(model=loaded_model)
    alert_payload = controller.evaluate_alerts({"pm25": float(np.nanmax(y_test)), "forecast_pm25": float(np.nanmax(predictions))})

    print({
        "artifact": str(result.artifact_path),
        "train_mae": result.train_mae,
        "val_mae": result.val_mae,
        "test_mae": result.test_mae,
        "train_rmse": result.train_rmse,
        "val_rmse": result.val_rmse,
        "test_rmse": result.test_rmse,
        "train_r2": result.train_r2,
        "val_r2": result.val_r2,
        "test_r2": result.test_r2,
        "test_mae_model": metrics.mae,
        "test_rmse_model": metrics.rmse,
        "test_r2_model": metrics.r2,
        "baseline_mae": baseline_metrics.mae,
        "baseline_rmse": baseline_metrics.rmse,
        "baseline_r2": baseline_metrics.r2,
        "latency_seconds": latency,
        "alert_level": alert_payload["level"],
    })


if __name__ == "__main__":
    main()
