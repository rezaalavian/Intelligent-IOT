from datetime import datetime, timezone

import pandas as pd

from infrastructure.kafka.consumers import sink


def test_partition_path_by_date(tmp_path):
    rec = {"station_id": "s", "source": "openaq",
           "timestamp": datetime(2023, 1, 1, 14, 0, tzinfo=timezone.utc), "pm25": 1.0}
    p = sink.partition_path(str(tmp_path), rec)
    assert p.replace("\\", "/").endswith("date=2023-01-01/part.parquet")


def test_append_writes_and_dedups(tmp_path):
    base = str(tmp_path)
    rec = {"station_id": "s", "source": "openaq",
           "timestamp": datetime(2023, 1, 1, 14, 0, tzinfo=timezone.utc),
           "ingested_at": datetime(2023, 1, 1, 14, 5, tzinfo=timezone.utc), "pm25": 1.0}
    sink.append_record(base, rec)
    sink.append_record(base, {**rec, "pm25": 2.0})  # same dedup key -> last non-null wins
    df = pd.read_parquet(sink.partition_path(base, rec))
    assert len(df) == 1
    assert df.iloc[0]["pm25"] == 2.0


def test_append_merges_partial_records_same_key(tmp_path):
    base = str(tmp_path)
    ts = datetime(2023, 1, 1, 14, 0, tzinfo=timezone.utc)
    ing = datetime(2023, 1, 1, 14, 5, tzinfo=timezone.utc)
    common = {"station_id": "s", "source": "openaq", "timestamp": ts, "ingested_at": ing,
              "pm25": None, "no2": None, "o3": None}
    sink.append_record(base, {**common, "pm25": 14.0})
    sink.append_record(base, {**common, "no2": 0.02})
    sink.append_record(base, {**common, "o3": 0.019})
    df = pd.read_parquet(sink.partition_path(base, common))
    assert len(df) == 1
    assert df.iloc[0]["pm25"] == 14.0
    assert df.iloc[0]["no2"] == 0.02
    assert df.iloc[0]["o3"] == 0.019
