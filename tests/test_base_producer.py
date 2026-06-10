import json
from datetime import datetime, timezone

from infrastructure.kafka.producers.base import BaseProducer


class FakeKafkaProducer:
    def __init__(self):
        self.produced = []

    def produce(self, topic, key=None, value=None, on_delivery=None):
        self.produced.append({"topic": topic, "key": key, "value": value})

    def poll(self, t):
        pass

    def flush(self, *a):
        return 0


def _serializer_ok(record, ctx):
    return json.dumps(record, default=str).encode()


def _serializer_bad(record, ctx):
    from confluent_kafka.serialization import SerializationError
    raise SerializationError("boom")


def _record():
    return {"station_id": "s1", "timestamp": datetime(2023, 1, 1, tzinfo=timezone.utc)}


def test_produce_sends_to_topic_keyed_by_station_id():
    main, dlq = FakeKafkaProducer(), FakeKafkaProducer()
    bp = BaseProducer(main_producer=main, dlq_producer=dlq,
                      serializer=_serializer_ok, dlq_topic="aq.deadletter")
    bp.produce("aq.openaq.raw", _record())
    assert len(main.produced) == 1
    assert main.produced[0]["topic"] == "aq.openaq.raw"
    assert main.produced[0]["key"] == b"s1"
    assert len(dlq.produced) == 0


def test_serialize_failure_routes_to_dlq_without_reavro():
    main, dlq = FakeKafkaProducer(), FakeKafkaProducer()
    bp = BaseProducer(main_producer=main, dlq_producer=dlq,
                      serializer=_serializer_bad, dlq_topic="aq.deadletter")
    bp.produce("aq.openaq.raw", _record())
    assert len(main.produced) == 0
    assert len(dlq.produced) == 1
    dlq_msg = dlq.produced[0]
    assert dlq_msg["topic"] == "aq.deadletter"
    body = json.loads(dlq_msg["value"])
    assert body["error"].startswith("boom") or "boom" in body["error"]
    assert body["source_topic"] == "aq.openaq.raw"
    assert body["payload"]["station_id"] == "s1"
