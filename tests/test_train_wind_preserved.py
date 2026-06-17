"""TDD: _load_base_frame must preserve pre-computed wind_u/wind_v columns."""
import io
import numpy as np
import pandas as pd
import pytest

from models.baselines.train_baselines import _load_base_frame


def _csv_with_wind_uv(wind_u: float = 3.5, wind_v: float = -1.2) -> str:
    """Build a minimal CSV that already has wind_u/wind_v but NO raw wind columns."""
    data = {
        "timestamp": ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
        "temp definition °c": [10.0, 11.0],
        "dew point definition °c": [5.0, 5.5],
        "rel hum definition %": [70.0, 71.0],
        "pm25": [15.0, 16.0],
        "wind_u": [wind_u, wind_u],
        "wind_v": [wind_v, wind_v],
        "upwind_pm25": [0.5, 0.6],
        "transport_potential": [0.3, 0.4],
        "wind_alignment": [0.1, 0.2],
    }
    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_wind_uv_preserved_when_no_raw_wind_columns(tmp_path):
    """wind_u and wind_v already in frame must NOT be zeroed by _load_base_frame."""
    expected_u = 3.5
    expected_v = -1.2

    csv_path = tmp_path / "mock_data.csv"
    csv_path.write_text(_csv_with_wind_uv(expected_u, expected_v))

    frame = _load_base_frame(csv_path)

    assert "wind_u" in frame.columns, "wind_u column missing"
    assert "wind_v" in frame.columns, "wind_v column missing"
    assert not (frame["wind_u"] == 0.0).all(), (
        f"wind_u was zeroed out; got {frame['wind_u'].tolist()}"
    )
    assert not (frame["wind_v"] == 0.0).all(), (
        f"wind_v was zeroed out; got {frame['wind_v'].tolist()}"
    )
    assert np.isclose(frame["wind_u"].iloc[0], expected_u), (
        f"Expected wind_u={expected_u}, got {frame['wind_u'].iloc[0]}"
    )
    assert np.isclose(frame["wind_v"].iloc[0], expected_v), (
        f"Expected wind_v={expected_v}, got {frame['wind_v'].iloc[0]}"
    )
