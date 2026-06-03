"""Fetch PM2.5 measurements from OpenAQ and merge into the historical CSV.

This script will query the OpenAQ API v2 for `pm25` measurements for the cities
present in `data/raw/historical_rawdata.csv`, aggregate to hourly, and merge
the values into a new CSV `data/raw/historical_rawdata_with_pm2.csv`.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys

import pandas as pd
import requests


OPENAQ_MEASUREMENTS = "https://api.openaq.org/v2/measurements"


def fetch_pm25_for_city(city: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "city": city,
        "parameter": "pm25",
        "date_from": start,
        "date_to": end,
        "limit": 100,
        "page": 1,
        "sort": "asc",
    }
    rows = []
    while True:
        resp = requests.get(OPENAQ_MEASUREMENTS, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        for r in results:
            rows.append({"timestamp": r["date"]["utc"], "city_name": r.get("city"), "pm25": r.get("value")})
        meta = data.get("meta", {})
        found = meta.get("found", 0)
        limit = meta.get("limit", 100)
        page = params["page"]
        if page * limit >= found:
            break
        params["page"] = page + 1
        time.sleep(1)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "city_name", "pm25"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    return df


def main(input_csv: str | Path = "data/raw/historical_rawdata.csv", output_csv: str | Path = "data/raw/historical_rawdata_with_pm2.csv") -> int:
    inp = Path(input_csv)
    out = Path(output_csv)
    df = pd.read_csv(inp, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    cities = df["city_name"].dropna().unique().tolist()
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    if pd.isna(start) or pd.isna(end):
        print("Timestamps missing from input CSV")
        return 2

    # query OpenAQ in UTC ISO format
    start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    pm25_frames = []
    for city in cities:
        try:
            print(f"Fetching pm25 for {city} {start_iso} -> {end_iso}")
            city_df = fetch_pm25_for_city(city, start_iso, end_iso)
            if not city_df.empty:
                pm25_frames.append(city_df)
        except Exception as exc:
            print(f"Warning: failed to fetch for {city}: {exc}")

    if not pm25_frames:
        print("No PM2.5 data fetched from OpenAQ")
        return 3

    pm25_all = pd.concat(pm25_frames, ignore_index=True)
    # aggregate to hourly by city
    pm25_all["timestamp"] = pm25_all["timestamp"].dt.floor("H")
    hourly = pm25_all.groupby(["city_name", "timestamp"]).pm25.mean().reset_index()

    # merge into original frame on city and hour
    df["timestamp_hour"] = df["timestamp"].dt.floor("H")
    merged = df.merge(hourly, left_on=["city_name", "timestamp_hour"], right_on=["city_name", "timestamp"], how="left", suffixes=("", "_openaq"))
    # prefer existing pm2 (if any), otherwise use pm25 from OpenAQ
    merged["pm2"] = merged["pm2"].fillna(merged["pm25"])
    merged = merged.drop(columns=[c for c in ["timestamp_hour", "timestamp_openaq", "pm25"] if c in merged.columns])
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote merged CSV to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
