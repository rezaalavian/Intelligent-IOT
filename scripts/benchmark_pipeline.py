"""End-to-end streaming hop latency: time from publishing a feature record on
`aq.features` until its alert appears on `aq.alerts` (inference + alert hops).

This measures the *processing* latency of the live chain, excluding the deliberate
hourly feature tick. Requires the stack + inference + alert consumers running
(e.g. via scripts/demo_up.sh).

    python scripts/benchmark_pipeline.py --n 30
"""
import argparse
import statistics
import time

from confluent_kafka import Producer, Consumer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
from confluent_kafka.serialization import SerializationContext, MessageField

from infrastructure.kafka.config import load_config
from infrastructure.kafka.serialization import schema_str
from infrastructure.kafka.station_registry import target_id

# Representative non-alerting feature payload; pm25 is overwritten per iteration
# with a unique sentinel so each alert can be matched back to its publish time.
BASE_FEATURES = {
    "temp definition °c": 20.3, "dew point definition °c": 16.5, "rel hum definition %": 79.0,
    "wind_u": 12.267, "wind_v": -3.287, "pm25": 10.0,
    "upwind_pm25": 17.011, "transport_potential": -8.952, "wind_alignment": -0.848,
    "no": 0.012, "no2": 0.012, "nox": 0.025, "o3": 0.036,
}


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    f = int(k); c = min(f + 1, len(values) - 1)
    return values[f] if f == c else values[f] + (values[c] - values[f]) * (k - f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of records to time")
    ap.add_argument("--timeout", type=float, default=15.0, help="per-record wait (s)")
    args = ap.parse_args()

    cfg = load_config()
    sr = SchemaRegistryClient({"url": cfg.schema_registry_url})
    feat_ser = AvroSerializer(sr, schema_str("feature.avsc"),
                              conf={"auto.register.schemas": False, "use.latest.version": True})
    alert_deser = AvroDeserializer(sr, schema_str("alert.avsc"))
    producer = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    consumer = Consumer({"bootstrap.servers": cfg.bootstrap_servers, "group.id": "bench-pipeline",
                         "auto.offset.reset": "latest", "enable.auto.commit": False})
    feat_topic, alert_topic = cfg.topics["features"], cfg.topics["alerts"]
    consumer.subscribe([alert_topic])
    # prime the assignment so we only see alerts produced from now on
    consumer.poll(2.0)

    station = "openaq-%d" % target_id()
    from datetime import datetime, timezone
    latencies = []
    for i in range(args.n):
        sentinel = round(10.0 + i * 0.013, 4)   # unique, stays in the "normal" band
        feats = dict(BASE_FEATURES, pm25=sentinel)
        record = {"station_id": station, "source": "bench", "features": feats,
                  "history": [], "timestamp": datetime.now(timezone.utc)}
        t0 = time.perf_counter()
        producer.produce(feat_topic, key=station.encode(),
                         value=feat_ser(record, SerializationContext(feat_topic, MessageField.VALUE)))
        producer.flush(5.0)

        deadline = t0 + args.timeout
        while time.perf_counter() < deadline:
            msg = consumer.poll(0.5)
            if msg is None or msg.error():
                continue
            alert = alert_deser(msg.value(), SerializationContext(alert_topic, MessageField.VALUE))
            if abs((alert.get("current_pm25") or -1) - sentinel) < 1e-3:
                latencies.append((time.perf_counter() - t0) * 1000.0)
                break
    consumer.close()

    if not latencies:
        raise SystemExit("no matched alerts — are inference + alert consumers running?")
    print("\n===== PIPELINE HOP LATENCY (features -> alerts) =====\n")
    print(f"matched {len(latencies)}/{args.n} records")
    print(f"avg     {statistics.mean(latencies):8.2f} ms")
    print(f"median  {statistics.median(latencies):8.2f} ms")
    print(f"p95     {percentile(latencies, 95):8.2f} ms")
    print(f"min/max {min(latencies):.2f} / {max(latencies):.2f} ms")


if __name__ == "__main__":
    main()
