import logging

log = logging.getLogger(__name__)


def build_alert_record(controller, prediction_rec: dict) -> dict:
    result = controller.evaluate_alerts({
        "forecast_pm25": prediction_rec["forecast_pm25"],
        "current_pm25": prediction_rec.get("current_pm25"),
    })
    return {
        "station_id": prediction_rec["station_id"],
        "timestamp": prediction_rec["timestamp"],
        "level": result["level"],
        "alert": bool(result["alert"]),
        "current_pm25": prediction_rec.get("current_pm25"),
        "forecast_pm25": float(result["forecast_pm25"]),
        "recommendation": result["recommendation"],
    }


def run() -> None:  # pragma: no cover - integration path
    from confluent_kafka import Consumer, Producer
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
    from confluent_kafka.serialization import SerializationContext, MessageField
    from ..config import load_config
    from ..serialization import schema_str
    from infrastructure.deployment.controller import load_controller

    cfg = load_config()
    controller = load_controller()
    sr = SchemaRegistryClient({"url": cfg.schema_registry_url})
    deser = AvroDeserializer(sr, schema_str("prediction.avsc"))
    out_ser = AvroSerializer(sr, schema_str("alert.avsc"),
                             conf={"auto.register.schemas": False, "use.latest.version": True})
    consumer = Consumer({"bootstrap.servers": cfg.bootstrap_servers,
                         "group.id": cfg.group_ids["alerts"],
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    in_topic = cfg.topics["predictions"]
    out_topic = cfg.topics["alerts"]
    consumer.subscribe([in_topic])
    out = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    dlq = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            try:
                pred = deser(msg.value(), SerializationContext(in_topic, MessageField.VALUE))
                alert = build_alert_record(controller, pred)
                out.produce(out_topic, key=pred["station_id"].encode(),
                            value=out_ser(alert, SerializationContext(out_topic, MessageField.VALUE)))
                out.poll(0)
            except Exception as exc:
                log.error("alert consumer error offset=%s err=%s", msg.offset(), exc)
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
