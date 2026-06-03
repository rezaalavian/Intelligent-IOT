# Intelligent-IOT

Distributed Real-Time Air Quality Forecasting and Proactive Industrial Response Using IoT Streams.

This repository implements a full end-to-end project flow for hourly air-quality data: ingestion, feature engineering, forecasting, recovery, and deployment.

## Project Goals
- Ingest live air-quality and weather data from API sources.
- Use the cleaned historical dataset as the training seed.
- Build hourly feature engineering for lagged and rolling predictors.
- Train a forecasting model for multi-horizon predictions.
- Add recovery logic for missing or unreliable values.
- Expose predictions and alerts through a small API.

## Current Data Strategy
- Historical source: `data/raw/historical_rawdata.csv`
- Live source: API-first ingestion through the Kafka layer
- Scraping: optional fallback only, not the main runtime path

The current raw dataset is hourly. The recommended research path is to keep the hourly truth data and predict multi-horizon hourly forecasts first.

## Repository Layout

```text
infrastructure/
  kafka/
    schemas/
    scripts/
  deployment/
    app.py
    controller.py
    dashboard/
analytics/
  flink_jobs/
  recovery/
models/
  baselines/
  spatiotemporal/
  saved_models/
data/
  raw/
docs/
scripts/
tests/
```

## Phases
- Phase 1: Kafka ingestion and schema validation
- Phase 2: Hourly feature engineering and adjacency building
- Phase 3: Spatiotemporal forecasting model
- Phase 4: Missing-data recovery and reconstruction
- Phase 5: Deployment, inference API, and proactive alerts

## Environment Setup
The project is configured for the `Intelligent-IOT` Conda environment.

```powershell
conda env create -f environment.yml
conda activate Intelligent-IOT
```

## Useful Commands
Clean and relocate the historical dataset:

```powershell
conda run -n Intelligent-IOT python scripts/clean_rawdata.py
```

Run the mock producer against the cleaned historical file:

```powershell
conda run -n Intelligent-IOT python infrastructure/kafka/scripts/mock_producer.py --limit 5
```

Run the feature engineering demo:

```powershell
conda run -n Intelligent-IOT python -m analytics.flink_jobs.feature_engineering
```

Train the baseline model:

```powershell
conda run -n Intelligent-IOT python -m models.baselines.train_baselines
```

Train the spatiotemporal model scaffold:

```powershell
conda run -n Intelligent-IOT python -m models.spatiotemporal.train
```

Start the API:

```powershell
conda run -n Intelligent-IOT uvicorn infrastructure.deployment.app:app --reload
```

## Notes
- GPU is optional and mainly useful for offline training.
- Phase 5 should consume the same prediction outputs produced by the model training pipeline.
- The project uses CPU-first inference so the pipeline remains easy to run and demo.
- The forecasting path now uses chronological train/validation/test splits and early stopping to reduce overfitting.
- The primary research target is hourly forecasting with 1h, 2h, snd 3h horizons.
- The benchmark harness uses a bounded sample for quick smoke tests, while the trainer can run on the full historical dataset.

## Documentation
- `docs/ENV_SETUP.md` explains the environment.
- `docs/DEV_ENV.md` contains developer setup notes.
- `models/saved_models/README.md` explains model artifact formats.
- `data/raw/README.md` explains historical data placement.
- `docs/PIPELINE_OVERVIEW.md` explains the end-to-end flow.
- `docs/DATA_GUIDE.md` explains the canonical data source and live ingestion policy.

## Evaluation and CI
- `evaluation/run_benchmarks.py` measures MAE, RMSE, and inference latency. The benchmark harness also computes R² and saves per-horizon summaries.
- Additional evaluation metrics available in the `evaluation` suite: early-warning lead time, alert precision/recall/F1 (for threshold-based alarms), calibration scores (CRPS), and throughput/latency under synthetic streaming loads.
- `.github/workflows/ci.yml` runs the smoke tests on every push and pull request.
