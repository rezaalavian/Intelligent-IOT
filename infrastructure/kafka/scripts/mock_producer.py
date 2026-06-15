"""Offline mock producer: replays an existing CSV as canonical records.

Prints JSON by default; with --produce, publishes Avro to the canonical topic.
"""
import argparse
import json
import time
from datetime import datetime, timezone
import pandas as pd

DEFAULT_SOURCE = "data/external/openaq_7570/ml_wide_format.csv"


def iter_records(source_csv: str, limit: int | None = None):
    frame = pd.read_csv(source_csv)
    if limit is not None:
        frame = frame.head(limit)
    for _, row in frame.iterrows():
        payload = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        ts = payload.get("datetime") or payload.get("timestamp")
        payload["station_id"] = "mock-7570"
        payload["source"] = "mock"
        payload["timestamp"] = str(ts) if ts is not None else None
        yield payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock air-quality producer")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--produce", action="store_true", help="publish to Kafka canonical topic")
    args = parser.parse_args()

    if args.produce:
        from confluent_kafka import Producer
        from confluent_kafka.schema_registry import SchemaRegistryClient
        from confluent_kafka.schema_registry.avro import AvroSerializer
        from confluent_kafka.serialization import SerializationContext, MessageField
        from infrastructure.kafka.config import load_config
        from infrastructure.kafka.serialization import schema_str, to_utc, floor_to_hour
        cfg = load_config()
        sr = SchemaRegistryClient({"url": cfg.schema_registry_url})
        ser = AvroSerializer(sr, schema_str("measurement.avsc"),
                             conf={"auto.register.schemas": False, "use.latest.version": True})
        p = Producer({"bootstrap.servers": cfg.bootstrap_servers})
        topic = cfg.topics["measurements"]

    for record in iter_records(args.source, args.limit):
        if args.produce:
            now = datetime.now(timezone.utc)
            canon = {
                "station_id": record["station_id"], "source": "mock",
                "timestamp": floor_to_hour(to_utc(record["timestamp"])),
                "ingested_at": now, "latitude": None, "longitude": None,
                "pm25": record.get("pm25"), "pm10": None, "no": record.get("no"),
                "no2": record.get("no2"), "nox": record.get("nox"), "so2": None,
                "co": None, "o3": record.get("o3"), "temperature": None,
                "humidity": None, "wind_speed": None, "wind_dir": None, "pressure": None,
            }
            p.produce(topic, key=canon["station_id"].encode(),
                      value=ser(canon, SerializationContext(topic, MessageField.VALUE)))
            p.poll(0)
        else:
            print(json.dumps(record, ensure_ascii=False, default=str))
        if args.sleep:
            time.sleep(args.sleep)
    if args.produce:
        p.flush(10.0)


if __name__ == "__main__":
    main()
