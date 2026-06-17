import base64
import json
import logging

from .. import live_store

log = logging.getLogger(__name__)


def apply_record(path: str, topic: str, prediction_topic: str, alert_topic: str, rec: dict) -> None:
    if topic == prediction_topic:
        live_store.update(path, "predictions", rec)
    elif topic == alert_topic:
        live_store.update(path, "alerts", rec)


def run() -> None:  # pragma: no cover - integration path
    from confluent_kafka import Consumer, Producer
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroDeserializer
    from confluent_kafka.serialization import SerializationContext, MessageField
    from ..config import load_config
    from ..serialization import schema_str

    cfg = load_config()
    sr = SchemaRegistryClient({"url": cfg.schema_registry_url})
    pred_topic = cfg.topics["predictions"]
    alert_topic = cfg.topics["alerts"]
    deser = {
        pred_topic: AvroDeserializer(sr, schema_str("prediction.avsc")),
        alert_topic: AvroDeserializer(sr, schema_str("alert.avsc")),
    }
    consumer = Consumer({"bootstrap.servers": cfg.bootstrap_servers,
                         "group.id": cfg.group_ids.get("livestate", "aq-livestate"),
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    consumer.subscribe([pred_topic, alert_topic])
    dlq = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    path = live_store.default_path()
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            topic = msg.topic()
            try:
                rec = deser[topic](msg.value(), SerializationContext(topic, MessageField.VALUE))
                apply_record(path, topic, pred_topic, alert_topic, rec)
            except Exception as exc:
                log.error("live_state error topic=%s offset=%s err=%s", topic, msg.offset(), exc)
                raw_val = msg.value()
                dlq.produce(cfg.topics["deadletter"], key=msg.key(), value=json.dumps({
                    "source_topic": topic, "offset": msg.offset(), "error": str(exc),
                    "value_b64": base64.b64encode(raw_val).decode() if raw_val is not None else None,
                }).encode())
                dlq.poll(0)
            consumer.commit(msg)
    finally:
        dlq.flush(10.0)
        consumer.close()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    run()
