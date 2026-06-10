from __future__ import annotations

import json
import logging

from confluent_kafka.serialization import SerializationError

log = logging.getLogger(__name__)


def _on_delivery(err, msg):
    if err is not None:
        log.error("delivery failed topic=%s err=%s", msg.topic() if msg else "?", err)


class BaseProducer:
    """Avro-serializing producer. Serializes before produce so failures route to a
    plain-text dead-letter producer (never re-Avro'd)."""

    def __init__(self, main_producer, dlq_producer, serializer, dlq_topic: str,
                 key_field: str = "station_id"):
        self._main = main_producer
        self._dlq = dlq_producer
        self._serialize = serializer
        self._dlq_topic = dlq_topic
        self._key_field = key_field

    def produce(self, topic: str, record: dict) -> None:
        key = str(record.get(self._key_field, "")).encode()
        try:
            value = self._serialize(record, None)
        except SerializationError as exc:
            self._to_dlq(topic, record, str(exc))
            return
        self._main.produce(topic, key=key, value=value, on_delivery=_on_delivery)
        self._main.poll(0)

    def _to_dlq(self, source_topic: str, record: dict, error: str) -> None:
        log.warning("routing record to DLQ topic=%s error=%s", source_topic, error)
        body = json.dumps(
            {"source_topic": source_topic, "error": error, "payload": record},
            default=str,
        ).encode()
        self._dlq.produce(self._dlq_topic, key=None, value=body, on_delivery=_on_delivery)
        self._dlq.poll(0)

    def flush(self, timeout: float = 10.0) -> None:
        self._main.flush(timeout)
        self._dlq.flush(timeout)
