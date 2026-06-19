import logging
import math
import time
import os
from collections import deque
from datetime import datetime, timezone

from ..feature_adapter import to_model_features
from ..rolling_buffer import RollingBuffer
from analytics.features.diffusion_features import diffusion_features
from analytics.recovery.spatial_recovery import recover

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
    import base64, json
    from confluent_kafka import Consumer, Producer
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
    from confluent_kafka.serialization import SerializationContext, MessageField
    from ..config import load_config
    from ..serialization import schema_str
    from ..station_registry import target_id, neighbor_ids, coords
    from ..met_join import nearest_met

    cfg = load_config()
    tick_seconds = float(os.environ.get("FEATURE_TICK_SECONDS", "3600"))

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

    latest_pm: dict[int, float] = {}
    recent_met: list[dict] = []
    last_seen_hour: dict[int, float] = {}
    latest_pm_history: deque = deque(maxlen=12)
    tid = target_id()
    target_lat, target_lon = coords(tid)
    last_emit = 0.0

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                pass
            else:
                try:
                    rec = deser(msg.value(), SerializationContext(in_topic, MessageField.VALUE))
                    sid = rec.get("station_id", "")
                    if sid.startswith("openaq-"):
                        int_id = int(sid[len("openaq-"):])
                        pm = rec.get("pm25")
                        if pm is not None:
                            latest_pm[int_id] = float(pm)
                            if int_id == tid:
                                last_seen_hour[int_id] = time.time()
                                latest_pm_history.append(float(pm))
                    else:
                        recent_met.append(rec)
                        if len(recent_met) > 200:
                            recent_met = recent_met[-200:]
                except Exception as exc:
                    log.error("feature consumer deser error offset=%s err=%s", msg.offset(), exc)
                    raw_val = msg.value()
                    dlq.produce(cfg.topics["deadletter"], key=msg.key(), value=json.dumps({
                        "source_topic": in_topic, "offset": msg.offset(), "error": str(exc),
                        "value_b64": base64.b64encode(raw_val).decode() if raw_val is not None else None,
                    }).encode())
                    dlq.poll(0)
                consumer.commit(msg)

            now = time.time()
            if now - last_emit >= tick_seconds:
                try:
                    met = nearest_met(target_lat, target_lon, recent_met) or {}
                    target_pm = latest_pm.get(tid)
                    if target_pm is None:
                        gap_hours = (now - last_seen_hour.get(tid, 0.0)) / 3600.0
                        recovery_neighbor_pm = [
                            {"lat": coords(nid)[0], "lon": coords(nid)[1], "pm25": latest_pm.get(nid)}
                            for nid in neighbor_ids()
                        ]
                        wu = float(met.get("wind_speed") or 0.0) * math.cos(math.radians(float(met.get("wind_dir") or 0.0)))
                        wv = float(met.get("wind_speed") or 0.0) * math.sin(math.radians(float(met.get("wind_dir") or 0.0)))
                        target_pm, method = recover(target_lat, target_lon, wu, wv, recovery_neighbor_pm,
                                                    list(latest_pm_history), gap_hours, threshold_hours=3)
                        if target_pm is None:
                            last_emit = now
                            continue
                        log.info("recovered target pm25=%.2f via %s (gap=%.1fh)", target_pm, method, gap_hours)
                    target_measurement = {
                        "pm25": target_pm,
                        "temperature": met.get("temperature"),
                        "dew_point": met.get("dew_point"),
                        "humidity": met.get("humidity"),
                        "wind_speed": met.get("wind_speed"),
                        "wind_dir": met.get("wind_dir"),
                    }
                    base_feats = to_model_features(target_measurement)
                    neighbor_pm = [
                        {"lat": coords(nid)[0], "lon": coords(nid)[1], "pm25": latest_pm.get(nid)}
                        for nid in neighbor_ids()
                    ]
                    feats = enrich_with_diffusion(base_feats, coords(tid), neighbor_pm)
                    station_key = "openaq-%d" % tid
                    history = buffer.append(station_key, feats)
                    ts_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
                    feature_rec = {
                        "station_id": station_key,
                        "source": "openaq",
                        "timestamp": ts_hour,
                        "features": feats,
                        "history": [dict(h) for h in history],
                    }
                    out.produce(out_topic, key=station_key.encode(),
                                value=out_ser(feature_rec, SerializationContext(out_topic, MessageField.VALUE)))
                    out.poll(0)
                    log.info("emitted feature record station=%s ts=%s feats=%s", station_key, ts_hour, list(feats.keys()))
                except Exception as exc:
                    log.error("feature emit error err=%s", exc)
                last_emit = time.time()
    finally:
        out.flush(10.0)
        dlq.flush(10.0)
        consumer.close()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    run()
