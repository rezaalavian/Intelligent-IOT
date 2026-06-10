from datetime import datetime, timezone

import pytest

from infrastructure.kafka.consumers import normalizer as nm


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown source"):
        nm.normalize("foobar", {"station_id": "x", "datetime_utc": "2023-01-01T00:00:00Z"})


def test_missing_required_field_raises_value_error():
    with pytest.raises(ValueError, match="missing required field"):
        nm.normalize("openaq", {"parameter": "pm25", "value": 1.0})


def test_openaq_raw_normalizes_to_canonical():
    raw = {"station_id": "openaq-7570", "sensor_id": 100, "parameter": "pm25",
           "value": 14.0, "datetime_utc": "2023-01-01T14:37:00Z",
           "latitude": 43.7, "longitude": -79.4}
    rec = nm.normalize("openaq", raw, ingested_at=datetime(2023, 1, 1, 14, 40, tzinfo=timezone.utc))
    assert rec["station_id"] == "openaq-7570"
    assert rec["source"] == "openaq"
    assert rec["pm25"] == 14.0
    assert rec["timestamp"] == datetime(2023, 1, 1, 14, 0, tzinfo=timezone.utc)  # floored
    assert rec["temperature"] is None


def test_envcanada_raw_normalizes_weather():
    raw = {"station_id": "swob-6158359", "datetime_utc": "2023-01-01T14:00:00Z",
           "latitude": 43.7, "longitude": -79.4, "air_temp": -3.2, "rel_hum": 81.0,
           "wind_speed": 12.0, "wind_dir": 270.0, "pressure": 101.3}
    rec = nm.normalize("envcanada", raw, ingested_at=datetime(2023, 1, 1, 14, 5, tzinfo=timezone.utc))
    assert rec["temperature"] == -3.2
    assert rec["humidity"] == 81.0
    assert rec["wind_speed"] == 12.0
    assert rec["wind_dir"] == 270.0
    assert rec["pressure"] == 101.3
    assert rec["pm25"] is None


def test_iqair_raw_normalizes_pm25():
    raw = {"station_id": "iqair-toronto", "datetime_utc": "2023-01-01T14:00:00Z", "pm25": 7.5}
    rec = nm.normalize("iqair", raw)
    assert rec["pm25"] == 7.5
    assert rec["station_id"] == "iqair-toronto"
    assert rec["temperature"] is None


def test_dedup_key_and_same_hour_collapse():
    a = {"station_id": "s", "source": "openaq",
         "timestamp": datetime(2023, 1, 1, 14, 0, tzinfo=timezone.utc), "pm25": 1.0}
    b = {"station_id": "s", "source": "openaq",
         "timestamp": datetime(2023, 1, 1, 14, 0, tzinfo=timezone.utc), "pm25": 2.0}
    assert nm.dedup_key(a) == nm.dedup_key(b)
    kept = nm.collapse_same_hour([a, b])
    assert len(kept) == 1 and kept[0]["pm25"] == 2.0  # keep last
