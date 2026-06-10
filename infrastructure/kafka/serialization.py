from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import fastavro

SCHEMA_DIR = Path(__file__).parent / "schemas"


def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text())


def schema_str(name: str) -> str:
    return (SCHEMA_DIR / name).read_text()


def to_utc(value) -> datetime:
    """Parse an ISO8601 string or datetime into a tz-aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def avro_encode(schema: dict, record: dict) -> bytes:
    parsed = fastavro.parse_schema(schema)
    buf = io.BytesIO()
    fastavro.schemaless_writer(buf, parsed, record)
    return buf.getvalue()


def avro_decode(schema: dict, raw: bytes) -> dict:
    parsed = fastavro.parse_schema(schema)
    return fastavro.schemaless_reader(io.BytesIO(raw), parsed)
