"""Data acquisition helpers for the project.

The project should prefer APIs for live updates and keep the raw historical CSV as
the training seed. Scraping remains optional and is not the default runtime path.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd
from pathlib import Path


# import optional data source implementations
try:
    from ..data_sources import openaq, environment_canada, iqair  # type: ignore
except Exception:
    openaq = None
    environment_canada = None
    iqair = None


@dataclass(frozen=True)
class DataSourceConfig:
    """Minimal configuration for live data acquisition."""

    name: str
    enabled: bool = True


DEFAULT_KEEP_COLUMNS: tuple[str, ...] = (
    "Temp Definition °C",
    "Dew Point Definition °C",
    "Rel Hum Definition %",
    "Precip. Amount Definition mm",
    "Wind Dir Definition 10's deg",
    "Wind Spd Definition km/h",
    "Visibility Definition km",
    "Stn Press Definition kPa",
    "Hmdx Definition",
    "Wind Chill Definition",
    "Weather Definition",
    "timestamp",
    "no",
    "no2",
    "nox",
    "o3",
    "pm2",
    "city_name",
    "co",
    "so2",
)


def clean_raw_data(input_csv: str | Path, output_csv: str | Path, keep_columns: Iterable[str] = DEFAULT_KEEP_COLUMNS) -> Path:
    """Keep only the requested columns and write a clean historical dataset.

    Missing requested columns are created with empty values so the downstream pipeline
    always receives a stable schema.
    """

    input_path = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path, low_memory=False)
    keep_columns = list(keep_columns)

    for column in keep_columns:
        if column not in frame.columns:
            frame[column] = pd.NA

    cleaned = frame.loc[:, keep_columns].copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]
    cleaned.to_csv(output_path, index=False)
    return output_path


def preferred_live_sources() -> list[DataSourceConfig]:
    """Return the default live sources for the project."""

    return [
        DataSourceConfig(name="openaq_api"),
        DataSourceConfig(name="environment_canada_api"),
    ]


def run_source(name: str, **kwargs):
    name = name.lower()
    if name.startswith("openaq") and openaq is not None:
        return openaq.fetch_openaq_location_ml(kwargs.get("location_id", 7570), kwargs.get("start_date", "2023-01-01"), kwargs.get("output_dir", "openaq_location"))
    if name.startswith("environment") and environment_canada is not None:
        return environment_canada.scrape_environment_canada(kwargs.get("climate_id", "6158359"), kwargs.get("province", "ON"), kwargs.get("start_year", 2022), kwargs.get("end_year", None), kwargs.get("output_file", "canada_climate_hourly.csv"))
    if name.startswith("iqair") and iqair is not None:
        return iqair.scrape_pm25_cities(kwargs.get("input_csv", "data/raw/historical_rawdata.csv"), kwargs.get("output_csv", "data/raw/historical_rawdata_with_pm2_scrape.csv"))
    raise RuntimeError(f"Unknown or unavailable source: {name}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("source", choices=["openaq", "environment_canada", "iqair"], help="which data source to run")
    p.add_argument("--location-id", type=int, default=7570)
    p.add_argument("--start-date", type=str, default="2023-01-01")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    if args.source == "openaq":
        out = run_source("openaq", location_id=args.location_id, start_date=args.start_date, output_dir=(args.output or f"openaq_location_{args.location_id}"))
    elif args.source == "environment_canada":
        out = run_source("environment_canada", climate_id="6158359", province="ON", start_year=2022, output_file=(args.output or "canada_climate_hourly.csv"))
    else:
        out = run_source("iqair", input_csv="data/raw/historical_rawdata.csv", output_csv=(args.output or "data/raw/historical_rawdata_with_pm2_scrape.csv"))

    print("Completed; output:", out)
