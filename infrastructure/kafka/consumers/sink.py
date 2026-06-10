from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .normalizer import dedup_key

log = logging.getLogger(__name__)


def partition_path(base_dir: str, rec: dict) -> str:
    date = rec["timestamp"].strftime("%Y-%m-%d")
    return str(Path(base_dir) / f"date={date}" / "part.parquet")


def append_record(base_dir: str, rec: dict) -> None:
    path = Path(partition_path(base_dir, rec))
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(rec)
    row["dedup_key"] = dedup_key(rec)
    new = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset="dedup_key", keep="last").reset_index(drop=True)
    combined.to_parquet(path, index=False)
