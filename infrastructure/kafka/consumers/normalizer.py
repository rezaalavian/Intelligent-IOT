from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..serialization import to_utc, floor_to_hour

log = logging.getLogger(__name__)

_CANON_FIELDS = (
    "latitude", "longitude", "pm25", "pm10", "no", "no2", "nox",
    "so2", "co", "o3", "temperature", "humidity", "wind_speed",
    "wind_dir", "pressure",
)


def _blank() -> dict:
    return {f: None for f in _CANON_FIELDS}


def normalize(source: str, raw: dict, ingested_at: datetime | None = None) -> dict:
    rec = _blank()
    try:
        rec["station_id"] = raw["station_id"]
        rec["timestamp"] = floor_to_hour(to_utc(raw["datetime_utc"]))
    except KeyError as exc:
        raise ValueError(f"[{source}] raw record missing required field: {exc}") from exc
    rec["source"] = source
    rec["ingested_at"] = ingested_at or datetime.now(timezone.utc)
    rec["latitude"] = raw.get("latitude")
    rec["longitude"] = raw.get("longitude")
    if source == "openaq":
        param = raw.get("parameter")
        if param in _CANON_FIELDS:
            rec[param] = raw.get("value")
        else:
            log.warning("openaq: unmapped parameter %r — dropping value", param)
    elif source == "envcanada":
        rec["temperature"] = raw.get("air_temp")
        rec["humidity"] = raw.get("rel_hum")
        rec["wind_speed"] = raw.get("wind_speed")
        rec["wind_dir"] = raw.get("wind_dir")
        rec["pressure"] = raw.get("pressure")
    elif source == "iqair":
        rec["pm25"] = raw.get("pm25")
    else:
        raise ValueError(f"unknown source: {source}")
    return rec


def dedup_key(rec: dict) -> str:
    return f"{rec['station_id']}|{rec['source']}|{rec['timestamp'].isoformat()}"


def collapse_same_hour(records: list[dict]) -> list[dict]:
    by_key = {dedup_key(r): r for r in records}  # later wins
    return list(by_key.values())
