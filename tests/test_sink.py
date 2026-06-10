from datetime import datetime, timezone

import pandas as pd

from infrastructure.kafka.consumers import sink


def test_partition_path_by_date(tmp_path):
    rec = {"station_id": "s", "source": "openaq",
           "timestamp": datetime(2023, 1, 1, 14, 0, tzinfo=timezone.utc), "pm25": 1.0}
    p = sink.partition_path(str(tmp_path), rec)
    assert p.endswith("date=2023-01-01/part.parquet")


def test_append_writes_and_dedups(tmp_path):
    base = str(tmp_path)
    rec = {"station_id": "s", "source": "openaq",
           "timestamp": datetime(2023, 1, 1, 14, 0, tzinfo=timezone.utc),
           "ingested_at": datetime(2023, 1, 1, 14, 5, tzinfo=timezone.utc), "pm25": 1.0}
    sink.append_record(base, rec)
    sink.append_record(base, {**rec, "pm25": 2.0})  # same dedup key -> replace
    df = pd.read_parquet(sink.partition_path(base, rec))
    assert len(df) == 1
    assert df.iloc[0]["pm25"] == 2.0
