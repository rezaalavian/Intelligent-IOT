# Data Guide

## Canonical historical dataset
- Path: `data/raw/historical_rawdata.csv`
- Resolution: hourly
- Role: training seed for phases 2 and 3

## Expected columns
- Temperature, dew point, humidity, precipitation, wind direction, wind speed, visibility, pressure, humidex, wind chill, weather, timestamp, and pollutant columns (`no`, `no2`, `nox`, `o3`, `pm2`, `co`, `so2`)

## Live ingestion
- Prefer API ingestion for new data.
- Keep scraping only as a fallback.
