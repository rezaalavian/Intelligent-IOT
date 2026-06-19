# Raw Data

The canonical training dataset is the assembled multi-station frame at
`data/external/multistation/train.csv`, built by
`infrastructure/kafka/scripts/backfill_multistation.py` from OpenAQ (PM2.5 +
co-pollutants) and the Open-Meteo archive (meteorology).

Live updates come from the API-based Kafka ingestion pipeline; this directory is
retained only as a placeholder for ad-hoc raw inputs.
