from infrastructure.kafka import live_store
from infrastructure.kafka.consumers.live_state import apply_record


def test_apply_record_routes_by_topic(tmp_path):
    p = str(tmp_path / "s.json")
    apply_record(p, "aq.predictions", "aq.predictions", "aq.alerts",
                 {"station_id": "openaq-7570", "forecast_pm25": 9.0})
    apply_record(p, "aq.alerts", "aq.predictions", "aq.alerts",
                 {"station_id": "openaq-7570", "level": "critical"})
    state = live_store.read_state(p)
    assert state["predictions"]["openaq-7570"]["forecast_pm25"] == 9.0
    assert state["alerts"]["openaq-7570"]["level"] == "critical"
