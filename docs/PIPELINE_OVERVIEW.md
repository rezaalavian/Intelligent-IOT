# Pipeline Overview

1. Training data lives in `data/external/multistation/train.csv`, assembled by `infrastructure/kafka/scripts/backfill_multistation.py`.
2. Live data enters through the API-first ingestion layer and Kafka topics.
3. `analytics/features/feature_engineering.py` builds the hourly feature table.
4. `models/baselines/train_baselines.py` trains the multi-horizon baselines (HA, LR, RF, LSTM, STGNN).
5. `infrastructure/deployment/controller.py` loads the saved model and emits alerts.

Recommended research direction:
- Keep the 1h truth source.
- Predict 1h, 2h, 3h, and 4h ahead first.
- Use interpolation only for gap filling and visualization.
