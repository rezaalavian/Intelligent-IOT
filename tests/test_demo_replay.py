import pandas as pd

from infrastructure.kafka.scripts.demo_replay import iter_feature_records, _select_window


def _frame():
    return pd.DataFrame({
        "timestamp": ["2026-02-02 00:00:00+00:00", "2026-02-02 01:00:00+00:00", "2026-07-01 00:00:00+00:00"],
        "temp definition °c": [1.0, 2.0, 3.0],
        "dew point definition °c": [0.0, 0.0, 0.0],
        "rel hum definition %": [50.0, 51.0, 52.0],
        "wind_u": [1.0, 1.0, 1.0],
        "wind_v": [0.0, 0.0, 0.0],
        "pm25": [12.0, 40.0, 200.0],
        "upwind_pm25": [5.0, 6.0, 7.0],
        "transport_potential": [0.1, 0.2, 0.3],
        "wind_alignment": [0.5, 0.5, 0.5],
        "no": [0.1, 0.1, 0.1], "no2": [1.0, 1.0, 1.0],
        "nox": [1.1, 1.1, 1.1], "o3": [30.0, 30.0, 30.0],
    })


def test_select_window_filters_by_timestamp():
    df = _select_window(_frame(), "2026-02-01", "2026-02-03")
    assert len(df) == 2  # the July row is excluded
    assert df["pm25"].tolist() == [12.0, 40.0]


def test_iter_feature_records_shape_and_history():
    df = _select_window(_frame(), "2026-02-01", "2026-02-03")
    recs = list(iter_feature_records(df))
    assert len(recs) == 2

    key, first = recs[0]
    assert key == "openaq-7570"                         # keyed by the target station
    assert first["source"] == "demo-replay"
    assert first["features"]["pm25"] == 12.0            # row pm2.5 -> current reading
    assert "upwind_pm25" in first["features"]           # diffusion feature carried
    assert first["history"] == []                       # first row has no prior history

    _, second = recs[1]
    assert len(second["history"]) == 1                  # rolling window grew
    assert second["history"][0]["pm25"] == 12.0
    assert hasattr(second["timestamp"], "tzinfo") and second["timestamp"].tzinfo is not None
