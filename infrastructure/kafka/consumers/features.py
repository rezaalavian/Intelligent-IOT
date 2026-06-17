import logging

from ..feature_adapter import to_model_features
from ..rolling_buffer import RollingBuffer
from analytics.flink_jobs.diffusion_features import diffusion_features

log = logging.getLogger(__name__)


def enrich_with_diffusion(target_features: dict, target_coord: tuple, neighbor_pm: list[dict]) -> dict:
    diff = diffusion_features(target_coord[0], target_coord[1],
                              float(target_features.get("wind_u", 0.0)),
                              float(target_features.get("wind_v", 0.0)),
                              neighbor_pm)
    return {**target_features, **diff}


def build_feature_record(measurement: dict, buffer: RollingBuffer) -> dict:
    feats = to_model_features(measurement)
    history = buffer.append(measurement["station_id"], feats)
    return {
        "station_id": measurement["station_id"],
        "source": measurement.get("source", ""),
        "timestamp": measurement["timestamp"],
        "features": feats,
        "history": [dict(h) for h in history],
    }


def run() -> None:  # pragma: no cover - integration path
    from confluent_kafka import Consumer, Producer
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
    from confluent_kafka.serialization import SerializationContext, MessageField
    from ..config import load_config
    from ..serialization import schema_str

    cfg = load_config()
    sr = SchemaRegistryClient({"url": cfg.schema_registry_url})
    deser = AvroDeserializer(sr, schema_str("measurement.avsc"))
    out_ser = AvroSerializer(sr, schema_str("feature.avsc"),
                             conf={"auto.register.schemas": False, "use.latest.version": True})
    consumer = Consumer({"bootstrap.servers": cfg.bootstrap_servers,
                         "group.id": cfg.group_ids["features"],
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    in_topic = cfg.topics["measurements"]
    out_topic = cfg.topics["features"]
    consumer.subscribe([in_topic])
    out = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    dlq = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    buffer = RollingBuffer(lookback=12)
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            try:
                rec = deser(msg.value(), SerializationContext(in_topic, MessageField.VALUE))
                feature_rec = build_feature_record(rec, buffer)
                out.produce(out_topic, key=rec["station_id"].encode(),
                            value=out_ser(feature_rec, SerializationContext(out_topic, MessageField.VALUE)))
                out.poll(0)
            except Exception as exc:
                log.error("feature consumer error offset=%s err=%s", msg.offset(), exc)
                import base64, json
                raw_val = msg.value()
                dlq.produce(cfg.topics["deadletter"], key=msg.key(), value=json.dumps({
                    "source_topic": in_topic, "offset": msg.offset(), "error": str(exc),
                    "value_b64": base64.b64encode(raw_val).decode() if raw_val is not None else None,
                }).encode())
                dlq.poll(0)
            consumer.commit(msg)
    finally:
        out.flush(10.0)
        dlq.flush(10.0)
        consumer.close()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    run()
