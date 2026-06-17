import os

from infrastructure.kafka import live_store


def test_read_missing_returns_empty(tmp_path):
    p = str(tmp_path / "s.json")
    assert live_store.read_state(p) == {"predictions": {}, "alerts": {}}


def test_update_upserts_latest_per_station(tmp_path):
    p = str(tmp_path / "s.json")
    live_store.update(p, "predictions", {"station_id": "openaq-7570", "forecast_pm25": 10.0})
    live_store.update(p, "predictions", {"station_id": "openaq-7570", "forecast_pm25": 12.0})
    live_store.update(p, "alerts", {"station_id": "openaq-7570", "level": "warning"})
    state = live_store.read_state(p)
    assert state["predictions"]["openaq-7570"]["forecast_pm25"] == 12.0
    assert state["alerts"]["openaq-7570"]["level"] == "warning"


def test_default_path_env(monkeypatch):
    monkeypatch.setenv("LIVE_STATE_PATH", "/tmp/x/live.json")
    assert live_store.default_path() == "/tmp/x/live.json"
