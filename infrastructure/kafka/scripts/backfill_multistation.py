import math

import pandas as pd

from ..station_registry import target_id, neighbor_ids, coords, STATIONS
from ..met_join import nearest_met
from analytics.flink_jobs.diffusion_features import diffusion_features


def parse_open_meteo(payload: dict, lat: float, lon: float) -> dict:
    """Open-Meteo archive JSON -> {hour_str: [met record]} matching build_training_frame's shape."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])

    def at(key: str, i: int):
        values = hourly.get(key) or []
        return values[i] if i < len(values) else None

    out: dict[str, list[dict]] = {}
    for i, t in enumerate(times):
        hour = str(pd.to_datetime(t, utc=True).floor("h"))
        out[hour] = [{
            "latitude": lat,
            "longitude": lon,
            "temperature": at("temperature_2m", i),
            "humidity": at("relativehumidity_2m", i),
            "dew_point": at("dewpoint_2m", i),
            "wind_speed": at("windspeed_10m", i),
            "wind_dir": at("winddirection_10m", i),
        }]
    return out


def build_training_frame(per_station_pm: dict, met_by_hour: dict) -> pd.DataFrame:
    tid = target_id()
    t_lat, t_lon = coords(tid)
    pm_by_hour: dict[int, dict[str, float]] = {}
    for sid, frame in per_station_pm.items():
        pm_by_hour[sid] = {str(r["datetime"]): float(r["pm25"]) for _, r in frame.iterrows()}
    rows = []
    for hour in sorted(pm_by_hour.get(tid, {})):
        met = nearest_met(t_lat, t_lon, met_by_hour.get(hour, [])) or {}
        speed = float(met.get("wind_speed") or 0.0)
        rad = math.radians(float(met.get("wind_dir") or 0.0))
        wind_u = speed * math.cos(rad)
        wind_v = speed * math.sin(rad)
        neighbors = [
            {"lat": coords(nid)[0], "lon": coords(nid)[1], "pm25": pm_by_hour.get(nid, {}).get(hour)}
            for nid in neighbor_ids()
        ]
        diff = diffusion_features(t_lat, t_lon, wind_u, wind_v, neighbors)
        rows.append({
            "datetime": hour,
            "temp definition °c": float(met.get("temperature") or 0.0),
            "dew point definition °c": float(met.get("dew_point") or 0.0),
            "rel hum definition %": float(met.get("humidity") or 0.0),
            "wind_u": wind_u,
            "wind_v": wind_v,
            "pm25": pm_by_hour[tid][hour],
            **diff,
        })
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def main() -> None:  # pragma: no cover - network I/O
    import argparse
    from pathlib import Path
    import requests
    from datetime import date
    from ..data_sources.openaq import fetch_openaq_location_ml
    from ..station_registry import STATIONS

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", default=date.today().isoformat(), help="YYYY-MM-DD")
    ap.add_argument("--out", default="data/external/multistation/train.csv")
    ap.add_argument("--tmp", default="data/external/multistation")
    args = ap.parse_args()

    per_station_pm = {}
    for sid in STATIONS:
        csv_path = fetch_openaq_location_ml(sid, args.start, f"{args.tmp}/loc_{sid}")
        df = pd.read_csv(csv_path)
        df = df.rename(columns={c: c.lower() for c in df.columns})
        pm_col = next((c for c in df.columns if c.replace(".", "").replace(" ", "") in ("pm25", "pm2")), None)
        if pm_col is None:
            raise SystemExit(f"no PM2.5 column for location {sid}; columns={list(df.columns)}")
        per_station_pm[sid] = df[["datetime", pm_col]].rename(columns={pm_col: "pm25"}).dropna()

    # Historical met from the Open-Meteo archive at the target station's coordinates.
    t_lat, t_lon = coords(target_id())
    resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": t_lat, "longitude": t_lon,
        "start_date": args.start, "end_date": args.end,
        "hourly": "temperature_2m,relativehumidity_2m,dewpoint_2m,windspeed_10m,winddirection_10m",
        "timezone": "UTC",
    }, timeout=60)
    resp.raise_for_status()
    met_by_hour = parse_open_meteo(resp.json(), t_lat, t_lon)

    frame = build_training_frame(per_station_pm, met_by_hour)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"wrote {len(frame)} rows -> {args.out}")
