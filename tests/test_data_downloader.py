from __future__ import annotations

import pandas as pd

from infrastructure.kafka.scripts.data_downloader import DEFAULT_KEEP_COLUMNS, clean_raw_data


def test_clean_raw_data_normalizes_and_derives_wind_components() -> None:
    frame = pd.DataFrame(
        {
            " Timestamp ": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            "PM2.5": ["12.5", "13.0"],
            "Wind Speed": ["5", "10"],
            "Wind Direction": [90, 180],
            "Station ID": ["A", "A"],
        }
    )

    cleaned = clean_raw_data(frame)

    assert "timestamp" in cleaned.columns
    assert "wind_u" in cleaned.columns
    assert "wind_v" in cleaned.columns
    assert len(cleaned) == 2


def test_default_keep_columns_contains_core_fields() -> None:
    assert "timestamp" in DEFAULT_KEEP_COLUMNS
    assert "wind_speed" in DEFAULT_KEEP_COLUMNS