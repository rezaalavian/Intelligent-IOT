from infrastructure.kafka.data_sources import openaq


def test_build_sensor_map_extracts_parameter_names():
    locations_payload = {"results": [{
        "id": 7570, "coordinates": {"latitude": 43.7, "longitude": -79.4},
        "sensors": [
            {"id": 100, "parameter": {"name": "pm25"}},
            {"id": 101, "parameter": {"name": "no2"}},
        ],
    }]}
    smap, coords = openaq._build_sensor_map(locations_payload)
    assert smap == {100: "pm25", 101: "no2"}
    assert coords == (43.7, -79.4)


def test_latest_payload_maps_to_raw_records():
    smap = {100: "pm25", 101: "no2"}
    coords = (43.7, -79.4)
    latest = {"results": [
        {"sensorsId": 100, "value": 14.0, "datetime": {"utc": "2023-01-01T14:00:00Z"}},
        {"sensorsId": 101, "value": 0.008, "datetime": {"utc": "2023-01-01T14:00:00Z"}},
        {"sensorsId": 999, "value": 1.0, "datetime": {"utc": "2023-01-01T14:00:00Z"}},
    ]}
    recs = openaq._latest_to_raw(7570, smap, coords, latest)
    assert {r["parameter"] for r in recs} == {"pm25", "no2"}  # unknown sensor 999 dropped
    pm = next(r for r in recs if r["parameter"] == "pm25")
    assert pm["station_id"] == "openaq-7570"
    assert pm["value"] == 14.0
    assert pm["latitude"] == 43.7
    assert pm["datetime_utc"] == "2023-01-01T14:00:00Z"


def test_dedup_drops_repeated_sensor_datetime():
    recs = [
        {"sensor_id": 100, "datetime_utc": "t1"},
        {"sensor_id": 100, "datetime_utc": "t1"},
        {"sensor_id": 100, "datetime_utc": "t2"},
    ]
    assert len(openaq._dedup(recs)) == 2
