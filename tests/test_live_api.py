import json

from fastapi.testclient import TestClient


def test_live_endpoints(tmp_path, monkeypatch):
    store = tmp_path / "live.json"
    store.write_text(json.dumps({
        "predictions": {"openaq-7570": {"station_id": "openaq-7570", "forecast_pm25": 11.0}},
        "alerts": {"openaq-7570": {"station_id": "openaq-7570", "level": "warning"}},
    }))
    monkeypatch.setenv("LIVE_STATE_PATH", str(store))
    import importlib
    import infrastructure.deployment.app as app_mod
    importlib.reload(app_mod)
    client = TestClient(app_mod.app)

    r = client.get("/live/predictions")
    assert r.status_code == 200
    assert r.json()["openaq-7570"]["forecast_pm25"] == 11.0

    r = client.get("/live/alerts")
    assert r.status_code == 200
    assert r.json()["openaq-7570"]["level"] == "warning"
