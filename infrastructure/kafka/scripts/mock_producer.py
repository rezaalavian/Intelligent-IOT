"""Mock producer for local testing.

The default behavior is to stream rows from the cleaned historical dataset as JSON
records. If Kafka settings are supplied, the same records can be sent to a topic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import pandas as pd


def iter_records(source_csv: str | Path, limit: int | None = None) -> Iterable[dict]:
    frame = pd.read_csv(source_csv)
    if limit is not None:
        frame = frame.head(limit)
    for _, row in frame.iterrows():
        payload = row.to_dict()
        if "timestamp" in payload and pd.notna(payload["timestamp"]):
            payload["timestamp"] = str(payload["timestamp"])
        yield payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock air-quality producer")
    parser.add_argument("--source", default="data/raw/historical_rawdata.csv")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    for record in iter_records(args.source, args.limit):
        print(json.dumps(record, ensure_ascii=False))
        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
