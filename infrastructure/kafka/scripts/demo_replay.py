"""Demo replay producer: stream recorded feature rows into `aq.features` so the
live inference -> alerts -> live_state -> API/dashboard chain runs in real time.

Two windows (real data from the training CSV):
- `eval`     : a slice of the held-out test split (model never trained on it) that
               includes the early-Feb-2026 warning cluster -> normal/warning alerts.
- `wildfire` : the July-2025 extreme-smoke event (PM2.5 up to ~224) -> CRITICAL
               (red) alerts. Historical, NOT part of the eval set — label it as such.

Replaying precomputed features (not raw measurements) keeps the demo robust: no
network, no API key, no hourly tick wait, repeatable, and it exercises exactly the
parts worth showing live — the model forecast, the EPA-threshold alerts, and the
dashboard. Each row's own `pm25` becomes the alert's current reading.
"""
from datetime import datetime, timezone

import pandas as pd

from models.feature_recipes import RECIPES
from ..station_registry import target_id

# The feature keys the deployed bundle reads (extra keys are ignored by the model).
FEATURE_MAP_COLS = RECIPES["with_pollutants"]

# Default timestamp windows per mode (UTC). Override with --start/--end.
WINDOWS = {
    "eval": ("2026-02-02", "2026-02-12"),      # held-out test slice, warning cluster
    "wildfire": ("2025-07-13", "2025-07-17"),  # historical extreme event, critical
}


def _select_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    mask = (ts >= pd.Timestamp(start, tz="UTC")) & (ts < pd.Timestamp(end, tz="UTC"))
    return df.loc[mask].sort_values("timestamp").reset_index(drop=True)


def iter_feature_records(df: pd.DataFrame, lookback: int = 12):
    """Yield (station_key, feature_record) per row, with a rolling history window.

    Pure (no Kafka) so it can be unit-tested. `timestamp` is a tz-aware datetime,
    matching what the AvroSerializer expects for the timestamp-millis field.
    """
    station_key = "openaq-%d" % target_id()
    history: list[dict] = []
    for _, row in df.iterrows():
        feats = {c: float(row[c]) for c in FEATURE_MAP_COLS if c in row and pd.notna(row[c])}
        ts = pd.to_datetime(row["timestamp"], utc=True).to_pydatetime().astimezone(timezone.utc)
        record = {
            "station_id": station_key,
            "source": "demo-replay",
            "timestamp": ts,
            "features": feats,
            "history": [dict(h) for h in history[-lookback:]],
        }
        yield station_key, record
        history.append(feats)


def main() -> None:  # pragma: no cover - network I/O
    import argparse
    import time
    from confluent_kafka import Producer
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroSerializer
    from confluent_kafka.serialization import SerializationContext, MessageField
    from ..config import load_config
    from ..serialization import schema_str

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(WINDOWS), default="eval")
    ap.add_argument("--start", help="ISO date/time (UTC); overrides --mode window")
    ap.add_argument("--end", help="ISO date/time (UTC); overrides --mode window")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between records")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of rows")
    ap.add_argument("--path", default="data/external/multistation/train.csv")
    args = ap.parse_args()

    start, end = WINDOWS[args.mode]
    start, end = args.start or start, args.end or end
    df = _select_window(pd.read_csv(args.path), start, end)
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        raise SystemExit(f"no rows in window {start}..{end} of {args.path}")

    cfg = load_config()
    sr = SchemaRegistryClient({"url": cfg.schema_registry_url})
    ser = AvroSerializer(sr, schema_str("feature.avsc"),
                         conf={"auto.register.schemas": False, "use.latest.version": True})
    producer = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    topic = cfg.topics["features"]

    print(f"[demo-replay] mode={args.mode} window={start}..{end} rows={len(df)} "
          f"interval={args.interval}s -> {topic}")
    sent = 0
    for station_key, record in iter_feature_records(df):
        producer.produce(topic, key=station_key.encode(),
                         value=ser(record, SerializationContext(topic, MessageField.VALUE)))
        producer.poll(0)
        sent += 1
        pm = record["features"].get("pm25", 0.0)
        flag = "CRITICAL" if pm >= 125.5 else ("WARNING" if pm >= 35.5 else "normal")
        print(f"[{sent}/{len(df)}] {record['timestamp']:%Y-%m-%d %H:%M} pm2.5={pm:5.1f}  {flag}")
        time.sleep(args.interval)
    producer.flush(10.0)
    print(f"[demo-replay] done — {sent} feature records produced")


if __name__ == "__main__":  # pragma: no cover
    main()
