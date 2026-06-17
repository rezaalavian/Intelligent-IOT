import logging
import time
from datetime import datetime, timezone

from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

from ..config import load_config
from ..serialization import schema_str
from ..data_sources import openaq, environment_canada
from confluent_kafka.serialization import SerializationContext, MessageField, SerializationError
from .base import BaseProducer, _on_delivery

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def enabled_sources(cfg) -> dict:
    sources: dict = {}
    if cfg.openaq_api_key:
        sources["openaq"] = lambda: openaq.poll(cfg.openaq_location_ids, cfg.openaq_api_key)
    else:
        log.warning("OpenAQ disabled (no API key)")
    sources["envcanada"] = lambda: environment_canada.poll(
        cfg.envcanada_bbox,
        datetime_window=environment_canada.recent_window(datetime.now(timezone.utc)),
    )
    return sources


def run_once(sources: dict, producer, raw_topic) -> None:
    for name, poll_fn in sources.items():
        try:
            records = poll_fn()
        except Exception as exc:
            log.error("poll failed source=%s err=%s", name, exc)
            continue
        for rec in records:
            producer.produce(raw_topic(name), rec)
    producer.flush(10.0)


def _build_producer(cfg) -> BaseProducer:
    sr = SchemaRegistryClient({"url": cfg.schema_registry_url})
    serializers = {}
    schema_by_source = {"openaq": "openaq_raw.avsc", "envcanada": "envcanada_raw.avsc",
                        "iqair": "iqair_raw.avsc"}
    for source, sfile in schema_by_source.items():
        serializers[cfg.raw_topic(source)] = AvroSerializer(
            sr, schema_str(sfile),
            conf={"auto.register.schemas": False, "use.latest.version": True},
        )

    main = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    dlq = Producer({"bootstrap.servers": cfg.bootstrap_servers})

    class _Routed(BaseProducer):
        def produce(self, topic, record):
            ser = serializers[topic]
            key = self._extract_key(record)
            try:
                value = ser(record, SerializationContext(topic, MessageField.VALUE))
            except SerializationError as exc:
                self._to_dlq(topic, record, str(exc))
                return
            self._main.produce(topic, key=key, value=value, on_delivery=_on_delivery)
            self._main.poll(0)

    return _Routed(main_producer=main, dlq_producer=dlq, serializer=None,
                   dlq_topic=cfg.topics["deadletter"])


def main() -> None:
    cfg = load_config()
    producer = _build_producer(cfg)
    sources = enabled_sources(cfg)
    interval = min(cfg.poll_intervals.values())
    log.info("starting ingestion loop interval=%ss sources=%s", interval, list(sources))
    while True:
        run_once(sources, producer, cfg.raw_topic)
        time.sleep(interval)


if __name__ == "__main__":
    main()
