import logging

log = logging.getLogger(__name__)


def build_prediction_record(bundle, feature_rec: dict) -> dict:
    result = bundle.predict_pm25(feature_rec["features"], history=feature_rec.get("history"))
    return {
        "station_id": feature_rec["station_id"],
        "timestamp": feature_rec["timestamp"],
        "forecasts": {k: float(v) for k, v in result["forecasts"].items()},
        "forecast_pm25": float(result["forecast_pm25"]),
        "model_type": result["model_type"],
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
    bundle = controller.bundle
    if bundle is None:
        raise RuntimeError("No model bundle loaded. Run scripts/save_deployment_models.py")
    sr = SchemaRegistryClient({"url": cfg.schema_registry_url})
    deser = AvroDeserializer(sr, schema_str("feature.avsc"))
    out_ser = AvroSerializer(sr, schema_str("prediction.avsc"),
                             conf={"auto.register.schemas": False, "use.latest.version": True})
    consumer = Consumer({"bootstrap.servers": cfg.bootstrap_servers,
                         "group.id": cfg.group_ids["inference"],
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    in_topic = cfg.topics["features"]
    out_topic = cfg.topics["predictions"]
    consumer.subscribe([in_topic])
    out = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    dlq = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            try:
                feat = deser(msg.value(), SerializationContext(in_topic, MessageField.VALUE))
                pred = build_prediction_record(bundle, feat)
                out.produce(out_topic, key=feat["station_id"].encode(),
                            value=out_ser(pred, SerializationContext(out_topic, MessageField.VALUE)))
                out.poll(0)
            except Exception as exc:
                log.error("inference consumer error offset=%s err=%s", msg.offset(), exc)
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
