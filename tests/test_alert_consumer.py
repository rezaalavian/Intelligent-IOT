from datetime import datetime, timezone

from infrastructure.kafka.consumers.alerts import build_alert_record


class FakeController:
    def evaluate_alerts(self, payload):
        assert "forecast_pm25" in payload
        return {"level": "warning", "alert": True,
                "forecast_pm25": payload["forecast_pm25"],
                "recommendation": "Increase monitoring frequency."}


def test_build_alert_record():
    ts = datetime(2026, 6, 16, 14, 0, tzinfo=timezone.utc)
    pred = {"station_id": "swob-1", "timestamp": ts,
            "forecasts": {"h1": 40.0}, "forecast_pm25": 40.0, "model_type": "stgnn"}
    out = build_alert_record(FakeController(), pred)
    assert out["station_id"] == "swob-1"
    assert out["timestamp"] == ts
    assert out["level"] == "warning"
    assert out["alert"] is True
    assert out["forecast_pm25"] == 40.0
    assert out["recommendation"] == "Increase monitoring frequency."
