import json
import os
from pathlib import Path

_EMPTY = {"predictions": {}, "alerts": {}}


def default_path() -> str:
    return os.getenv("LIVE_STATE_PATH", "data/stream/live_state.json")


def read_state(path: str) -> dict:
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"predictions": {}, "alerts": {}}
    data.setdefault("predictions", {})
    data.setdefault("alerts", {})
    return data


def update(path: str, kind: str, rec: dict) -> None:
    if kind not in ("predictions", "alerts"):
        raise ValueError(f"unknown kind: {kind}")
    state = read_state(path)
    state[kind][rec["station_id"]] = rec
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, default=str)
    os.replace(tmp, path)
