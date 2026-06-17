from datetime import datetime, timezone

from infrastructure.kafka.rolling_buffer import RollingBuffer
from infrastructure.kafka.consumers.features import build_feature_record


def test_build_feature_record_shape():
    buf = RollingBuffer(lookback=12)
    ts = datetime(2026, 6, 16, 14, 0, tzinfo=timezone.utc)
    rec = {"station_id": "swob-1", "source": "envcanada", "timestamp": ts,
           "temperature": 20.0, "dew_point": 12.0, "humidity": 55.0,
           "wind_speed": 10.0, "wind_dir": 0.0, "pm25": 14.0}
    out = build_feature_record(rec, buf)
    assert out["station_id"] == "swob-1"
    assert out["source"] == "envcanada"
    assert out["timestamp"] == ts
    assert out["features"]["temp definition °c"] == 20.0
    assert len(out["history"]) == 1
    assert out["history"][0]["pm25"] == 14.0


def test_history_accumulates_per_station():
    buf = RollingBuffer(lookback=12)
    ts = datetime(2026, 6, 16, 14, 0, tzinfo=timezone.utc)
    base = {"station_id": "swob-1", "source": "envcanada", "timestamp": ts, "pm25": 1.0}
    build_feature_record(base, buf)
    out = build_feature_record({**base, "pm25": 2.0}, buf)
    assert [h["pm25"] for h in out["history"]] == [1.0, 2.0]
