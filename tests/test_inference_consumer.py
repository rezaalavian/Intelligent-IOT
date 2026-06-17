from datetime import datetime, timezone

from infrastructure.kafka.consumers.inference import build_prediction_record


class FakeBundle:
    def predict_pm25(self, features, history=None):
        assert "temp definition °c" in features
        return {"forecasts": {"h1": 12.0, "h2": 13.0, "h3": 14.0},
                "forecast_pm25": 12.0, "model_type": "stgnn"}


def test_build_prediction_record():
    ts = datetime(2026, 6, 16, 14, 0, tzinfo=timezone.utc)
    feature_rec = {"station_id": "swob-1", "timestamp": ts,
                   "features": {"temp definition °c": 20.0, "pm25": 14.0},
                   "history": [{"temp definition °c": 20.0, "pm25": 14.0}]}
    out = build_prediction_record(FakeBundle(), feature_rec)
    assert out["station_id"] == "swob-1"
    assert out["timestamp"] == ts
    assert out["forecasts"] == {"h1": 12.0, "h2": 13.0, "h3": 14.0}
    assert out["forecast_pm25"] == 12.0
    assert out["model_type"] == "stgnn"
