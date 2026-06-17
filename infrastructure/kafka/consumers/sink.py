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
    combined = combined.groupby("dedup_key", as_index=False, sort=False).last()
    combined.to_parquet(path, index=False)


def run() -> None:  # pragma: no cover - integration path
    from confluent_kafka import Consumer, Producer
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
    dlq = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    topic = cfg.topics["measurements"]
    consumer.subscribe([topic])
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            try:
                rec = deser(msg.value(), SerializationContext(topic, MessageField.VALUE))
                append_record(cfg.sink_dir, rec)
            except Exception as exc:
                log.error("routing bad message to DLQ topic=%s offset=%s err=%s", topic, msg.offset(), exc)
                import base64, json
                raw_val = msg.value()
                dlq.produce(cfg.topics["deadletter"], key=msg.key(), value=json.dumps({
                    "source_topic": topic,
                    "offset": msg.offset(),
                    "error": str(exc),
                    "value_b64": base64.b64encode(raw_val).decode() if raw_val is not None else None,
                }).encode())
                dlq.poll(0)
            consumer.commit(msg)
    finally:
        dlq.flush(10.0)
        consumer.close()


if __name__ == "__main__":  # pragma: no cover
    run()
