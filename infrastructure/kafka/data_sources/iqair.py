from pathlib import Path
from datetime import datetime
import logging
import re
import time
import pandas as pd
import requests

log = logging.getLogger(__name__)


IQAIR_CITY_URL = "https://www.iqair.com/canada/ontario/{slug}"


def slugify_city(city: str) -> str:
    s = city.lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = s.strip().replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    return s


def scrape_pm25_for_city(city: str) -> dict | None:
    slug = slugify_city(city.split(" ")[0])
    url = IQAIR_CITY_URL.format(slug=slug)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PM2Scraper/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text
        m = re.search(r"PM2\.5[^0-9\-\n\r]*([0-9]+(?:\.[0-9]+)?)", html, re.IGNORECASE)
        if not m:
            m = re.search(r"data\-pm25\-value\=\"([0-9]+(?:\.[0-9]+)?)\"", html, re.IGNORECASE)
        if not m:
            m = re.search(r'"pm25"\s*:\s*([0-9]+(?:\.[0-9]+)?)', html, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            return {"city_name": city, "timestamp": datetime.utcnow(), "pm25": val, "source_url": url}
        else:
            return None
    except Exception:
        return None


def scrape_pm25_cities(input_csv: str | Path, output_csv: str | Path, pause: float = 1.0) -> Path | None:
    inp = Path(input_csv)
    out = Path(output_csv)
    if not inp.exists():
        return None
    df = pd.read_csv(inp, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    cities = df["city_name"].dropna().unique().tolist()

    rows = []
    for city in cities:
        res = scrape_pm25_for_city(city)
        if res:
            rows.append(res)
        time.sleep(pause)

    if not rows:
        return None

    pm25_df = pd.DataFrame(rows)
    pm25_df["timestamp"] = pd.to_datetime(pm25_df["timestamp"]) 
    pm25_df["timestamp"] = pm25_df["timestamp"].dt.floor("H")
    hourly = pm25_df.groupby(["city_name", "timestamp"]).pm25.mean().reset_index()

    df["timestamp_hour"] = df["timestamp"].dt.floor("H")
    merged = df.merge(hourly, left_on=["city_name", "timestamp_hour"], right_on=["city_name", "timestamp"], how="left", suffixes=("", "_scrape"))
    merged["pm2"] = merged["pm2"].fillna(merged.get("pm25"))
    merged = merged.drop(columns=[c for c in ["timestamp_hour", "timestamp_scrape", "pm25"] if c in merged.columns])
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    return out


def poll(*args, **kwargs) -> list[dict]:
    """Live IQAir polling is not enabled in Phase 1.

    IQAir has no free realtime measurement API and HTML scraping is ToS-fragile,
    so it is a manual fallback only (see scrape_pm25_cities). The raw topic and
    schema (iqair_raw.avsc) exist as forward-compatible wiring; this returns no
    records so the ingestion loop treats IQAir as a disabled source.
    """
    log.info("IQAir live poll is disabled in Phase 1; returning no records")
    return []
