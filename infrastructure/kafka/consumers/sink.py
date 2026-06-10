from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .normalizer import dedup_key

log = logging.getLogger(__name__)


def partition_path(base_dir: str, rec: dict) -> str:
    ts = rec["timestamp"]
    if not isinstance(ts, datetime):
        raise TypeError(f"rec['timestamp'] must be a datetime, got {type(ts).__name__}")
    date = ts.strftime("%Y-%m-%d")
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


def run() -> None:  # pragma: no cover - integration path
    from confluent_kafka import Consumer
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroDeserializer
    from confluent_kafka.serialization import SerializationContext, MessageField
    from ..config import load_config
    from ..serialization import schema_str

    cfg = load_config()
    sr = SchemaRegistryClient({"url": cfg.schema_registry_url})
    deser = AvroDeserializer(sr, schema_str("measurement.avsc"))
    consumer = Consumer({"bootstrap.servers": cfg.bootstrap_servers,
                         "group.id": cfg.group_ids["sink"],
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    topic = cfg.topics["measurements"]
    consumer.subscribe([topic])
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            rec = deser(msg.value(), SerializationContext(topic, MessageField.VALUE))
            append_record(cfg.sink_dir, rec)
            consumer.commit(msg)
    finally:
        consumer.close()


if __name__ == "__main__":  # pragma: no cover
    run()
