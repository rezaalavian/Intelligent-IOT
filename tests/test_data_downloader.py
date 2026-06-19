import pandas as pd

from infrastructure.kafka.scripts.data_downloader import DEFAULT_KEEP_COLUMNS, clean_raw_data


def test_clean_raw_data_keeps_requested_columns(tmp_path) -> None:
    src = tmp_path / "raw.csv"
    pd.DataFrame({"timestamp": ["2024-01-01 00:00:00"], "pm2": [12.5], "drop_me": [1]}).to_csv(src, index=False)

    out = clean_raw_data(src, tmp_path / "clean.csv", keep_columns=["timestamp", "pm2", "no2"])
    cleaned = pd.read_csv(out)

    assert list(cleaned.columns) == ["timestamp", "pm2", "no2"]  # only requested, in order
    assert cleaned["no2"].isna().all()                           # missing column created empty
    assert "drop_me" not in cleaned.columns
    assert len(cleaned) == 1


def test_default_keep_columns_contains_core_fields() -> None:
    assert "timestamp" in DEFAULT_KEEP_COLUMNS
    assert "pm2" in DEFAULT_KEEP_COLUMNS
