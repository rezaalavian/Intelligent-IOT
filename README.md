# Intelligent-IOT

Distributed Real-Time Air Quality Forecasting and Proactive Industrial Response Using IoT Streams.

This repository implements a full end-to-end project flow for hourly air-quality data: ingestion, feature engineering, forecasting, recovery, and deployment.

# System Overview

The project is a multi-stage streaming ML pipeline:
APIs → Kafka → Feature Engineering → Models → API → Dashboard

## Project Goals
- Ingest live air-quality and weather data from API sources.
- Use the cleaned historical dataset as the training seed.
- Build hourly feature engineering for lagged and rolling predictors.
- Train a forecasting model for multi-horizon predictions.
- Add recovery logic for missing or unreliable values.
- Expose predictions and alerts through a small API.

## Current Data Strategy
- Historical source: `data/raw/RawData.csv`
- Live source: 
  - OpenAQ API
  - Environment Canada API
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

# ⚙️ System Architecture

## Phase 1 — Ingestion (Kafka)
- API pulls real-time data
- Data normalized into canonical schema
- Kafka topics receive streaming measurements

## Phase 2 — Feature Engineering
- Hourly aggregation
- Lag features (12-hour window)
- Rolling statistics
- Wind-aware spatial features

## Phase 3 — Forecasting Models
Models supported:
- Historical Average
- Linear Regression
- Random Forest
- LSTM
- STGNN (primary model)

Outputs:
- 1-hour, 2-hour, 3-hour forecasts

## Phase 4 — Recovery
- Missing data interpolation (temporal + spatial)
- Kriging / graph-based reconstruction

## Phase 5 — Deployment
- FastAPI inference server
- Streamlit dashboard
- Alert generation system


## Environment Setup
## 1. Create environment

```powershell
conda env create -f environment.yml
conda activate Intelligent-IOT-blackwell
```
## 2. Start Kafka + Flink (REQUIRED for full system)

docker compose up -d
Check:
docker ps
## 3. Register Kafka Schemas
python -m infrastructure.kafka.register_schemas

## How to Run the System (Full Streaming Mode)
## Terminal 1 — Kafka already running via Docker
## Terminal 2 — Start data ingestion
python infrastructure/kafka/data_sources/openaq.py
## Terminal 3 — Start API
python -m uvicorn infrastructure.deployment.app:app --reload --port 8000
## Terminal 4 — Start Dashboard
python -m streamlit run infrastructure/deployment/dashboard/streamlit_app.py --server.port 8501

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
conda run -n Intelligent-IOT-blackwell uvicorn infrastructure.deployment.app:app --reload
```

Save deployment models and launch the dashboard:

```powershell
conda run -n Intelligent-IOT-blackwell python scripts/save_deployment_models.py
conda run -n Intelligent-IOT-blackwell streamlit run infrastructure/deployment/dashboard/streamlit_app.py
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
