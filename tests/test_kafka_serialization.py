from datetime import datetime, timezone

from infrastructure.kafka import serialization as ser


def test_floor_to_hour_utc():
    dt = datetime(2023, 1, 1, 14, 37, 9, tzinfo=timezone.utc)
    assert ser.floor_to_hour(dt) == datetime(2023, 1, 1, 14, 0, 0, tzinfo=timezone.utc)


def test_parse_iso_to_utc():
    dt = ser.to_utc("2023-01-01T14:37:00Z")
    assert dt.tzinfo == timezone.utc
    assert dt.hour == 14


def test_canonical_record_round_trips_through_avro():
    schema = ser.load_schema("measurement.avsc")
    record = {
        "station_id": "openaq-7570",
        "source": "openaq",
        "timestamp": datetime(2023, 1, 1, 14, 0, tzinfo=timezone.utc),
        "ingested_at": datetime(2023, 1, 1, 14, 5, tzinfo=timezone.utc),
        "latitude": 43.7, "longitude": -79.4,
        "pm25": 14.0, "pm10": None, "no": 0.012, "no2": 0.008, "nox": 0.025,
        "so2": None, "co": None, "o3": 0.019,
        "temperature": -3.2, "humidity": 81.0, "wind_speed": 12.0,
        "wind_dir": 270.0, "pressure": 101.3,
    }
    raw = ser.avro_encode(schema, record)
    back = ser.avro_decode(schema, raw)
    assert back["timestamp"] == record["timestamp"]
    assert back["pm25"] == 14.0
    assert "co2" not in back
