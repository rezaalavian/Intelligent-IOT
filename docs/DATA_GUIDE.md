# Data Guide

## Canonical training dataset
- Path: `data/external/multistation/train.csv`
- Built by: `infrastructure/kafka/scripts/backfill_multistation.py` (OpenAQ PM2.5 + co-pollutants, Open-Meteo archive meteorology)
- Resolution: hourly
- Role: training source for phases 2 and 3

## Columns
- `timestamp`, meteorology (`temp definition °c`, `dew point definition °c`, `rel hum definition %`), wind components (`wind_u`, `wind_v`), target `pm25`, wind-aware diffusion features (`upwind_pm25`, `transport_potential`, `wind_alignment`), and target co-pollutants (`no`, `no2`, `nox`, `o3`)

## Live ingestion
- Prefer API ingestion for new data.
- Keep scraping only as a fallback.
