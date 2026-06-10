import socket
import time
import uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.integration


def _port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def stack():
    if not (_port_open("localhost", 9092) and _port_open("localhost", 8081)):
        pytest.skip("Kafka/Schema Registry not running; start docker compose first")


def test_end_to_end_mock_record_lands_in_sink(stack, tmp_path):
    from infrastructure.kafka import create_topics, register_schemas
    from infrastructure.kafka.config import load_config
    from infrastructure.kafka.consumers import sink
    create_topics.main()
    register_schemas.main()

    cfg = load_config()
    from confluent_kafka import Producer, Consumer
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
    from confluent_kafka.serialization import SerializationContext, MessageField
    from infrastructure.kafka.serialization import schema_str

    sr = SchemaRegistryClient({"url": cfg.schema_registry_url})
    ser = AvroSerializer(sr, schema_str("measurement.avsc"),
                         conf={"auto.register.schemas": False, "use.latest.version": True})
    topic = cfg.topics["measurements"]
    rec = {"station_id": "it-1", "source": "mock",
           "timestamp": datetime(2023, 1, 1, 14, 0, tzinfo=timezone.utc),
           "ingested_at": datetime.now(timezone.utc),
           "latitude": None, "longitude": None, "pm25": 9.9, "pm10": None,
           "no": None, "no2": None, "nox": None, "so2": None, "co": None, "o3": None,
           "temperature": None, "humidity": None, "wind_speed": None,
           "wind_dir": None, "pressure": None}
    p = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    p.produce(topic, key=b"it-1", value=ser(rec, SerializationContext(topic, MessageField.VALUE)))
    p.flush(10)

    deser = AvroDeserializer(sr, schema_str("measurement.avsc"))
    c = Consumer({"bootstrap.servers": cfg.bootstrap_servers,
                  "group.id": f"it-test-{uuid.uuid4()}",
                  "auto.offset.reset": "earliest"})
    c.subscribe([topic])
    deadline = time.time() + 30
    got = None
    while time.time() < deadline:
        msg = c.poll(1.0)
        if msg and not msg.error():
            got = deser(msg.value(), SerializationContext(topic, MessageField.VALUE))
            break
    c.close()
    assert got is not None and got["pm25"] == 9.9
    assert got["timestamp"] == rec["timestamp"]

    sink.append_record(str(tmp_path), got)
    import pandas as pd
    df = pd.read_parquet(sink.partition_path(str(tmp_path), got))
    assert df.iloc[0]["pm25"] == 9.9
