# Pipeline Overview

1. Historical data lives in `data/raw/historical_rawdata.csv`.
2. Live data should enter through the API-first ingestion layer and Kafka topics.
3. `analytics/flink_jobs/feature_engineering.py` builds the hourly feature table.
4. `models/spatiotemporal/train.py` trains the multi-horizon forecasting scaffold.
5. `infrastructure/deployment/controller.py` loads the saved model and emits alerts.

Recommended research direction:
- Keep the 1h truth source.
- Predict 1h, 2h, 3h, and 4h ahead first.
- Use interpolation only for gap filling and visualization.
