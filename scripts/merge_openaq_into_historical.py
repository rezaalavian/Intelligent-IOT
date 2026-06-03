"""Merge OpenAQ raw_long_format.csv into the project's historical raw CSV.

This script expects a folder like `data/external/openaq_7570/raw_long_format.csv`.
It will pivot the long-format file into hourly ML format, extract a PM2/PM2.5 column
and merge by hour into `data/raw/historical_rawdata.csv`, writing
`data/raw/historical_rawdata_with_openaq.csv`.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def merge(openaq_raw: str | Path = "data/external/openaq_7570/raw_long_format.csv",
          historical: str | Path = "data/raw/historical_rawdata.csv",
          output: str | Path = "data/raw/historical_rawdata_with_openaq.csv") -> Path:

    openaq_path = Path(openaq_raw)
    hist_path = Path(historical)
    out_path = Path(output)

    if not openaq_path.exists():
        raise FileNotFoundError(f"OpenAQ raw file not found: {openaq_path}")
    if not hist_path.exists():
        raise FileNotFoundError(f"Historical CSV not found: {hist_path}")

    raw = pd.read_csv(openaq_path)
    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["datetime", "parameter", "value"]) 
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["value"]) 

    agg = raw.groupby(["datetime", "parameter"], as_index=False).mean()
    ml = agg.pivot(index="datetime", columns="parameter", values="value")
    ml = ml.sort_index()
    ml = ml.resample("1H").mean()
    ml = ml.interpolate(method="time").ffill().bfill()

    # find PM2 column name
    pm_cols = [c for c in ml.columns if str(c).lower().replace(".", "").startswith("pm2")]
    if not pm_cols:
        raise RuntimeError("No PM2/PM2.5 column found in OpenAQ data")
    pm_col = pm_cols[0]

    pm_series = ml[[pm_col]].copy()
    pm_series = pm_series.rename(columns={pm_col: "pm2_openaq"})
    pm_series = pm_series.reset_index()
    pm_series["timestamp"] = pd.to_datetime(pm_series["datetime"]) 
    pm_series["timestamp"] = pm_series["timestamp"].dt.tz_convert(None)
    pm_series["timestamp"] = pm_series["timestamp"].dt.floor("H")

    hist = pd.read_csv(hist_path, low_memory=False)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist["timestamp_hour"] = hist["timestamp"].dt.floor("H")

    merged = hist.merge(pm_series[["timestamp", "pm2_openaq"]], left_on="timestamp_hour", right_on="timestamp", how="left")
    # prefer existing pm2, else use openaq
    if "pm2" not in merged.columns:
        merged["pm2"] = merged.get("pm2")
    merged["pm2"] = merged["pm2"].fillna(merged.get("pm2_openaq"))

    merged = merged.drop(columns=[c for c in ["timestamp_hour", "timestamp"] if c in merged.columns])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    import sys
    try:
        out = merge()
        print("Merged OpenAQ into:", out)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
