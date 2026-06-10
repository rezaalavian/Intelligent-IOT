# Project Proposal — Summary, Status, and To-Do

## Short summary
This project proposes a distributed real-time air-quality forecasting system for industrial areas using IoT streams, Kafka + Flink ingestion, spatiotemporal graph-based forecasting (GAT + TCN), fault-tolerant sensor recovery (kriging + graph reconstruction), and a proactive industrial response module (rule-based + predictive control). Evaluation metrics include MAE/RMSE/R², early-warning lead time, and streaming performance.

## Repository Layout
- `infrastructure/`: Kafka + deployment scaffolding and data downloader scripts
- `analytics/`: Flink-style feature engineering and recovery helpers
- `models/`: baselines, spatiotemporal models, and `saved_models` artifacts
- `data/`: `raw/` (canonical CSV), `processed/`, and `external/` archives
- `scripts/`, `docs/`, `evaluation/`, `api/`, `tests/`

## Forecasting timeframe assumption
- Assumption: The available historical dataset contains hourly timestamps only. Therefore, forecasting and model training will be performed at hourly resolution (1-hour, 2-hour, 3-hour ahead). The original proposal's sub-hour horizons (5-minute and 30-minute) are not supported by the current dataset unless higher-frequency sensor data is acquired or a separate high-frequency ingestion stream is added.


## Phases & Steps (hourly-adapted)
- Phase 1 — IoT Stream Ingestion
  - Set up Kafka topics for OpenAQ, Environment Canada, and other sensors. (Goal: validated Avro schemas and ingestion pipelines): COMPLETED
  - Ensure ingestion includes a step to aggregate or downsample higher-frequency sources into hourly buckets where upstream data is irregular.
  - Note: truth resolution is hourly; the proposal's sub-minute ingestion target is not pursued in Phase 1 — this is a deliberate downsample, not a source limitation.

- Phase 2 — Real-time Feature Engineering (hourly)
  - Implement Flink jobs that compute hourly rolling/lag features, wind-conditioned spatial weighting, and direction-aware adjacency A(t) aggregated per hour.
  - Use hourly tumbling windows (1h) or pre-aggregate raw records into an hourly timeseries before feature creation.
  - Convert all lag and rolling sizes to hours (e.g., lag1 → 1 hour, roll3 → 3 hours).
  - Ensure `timestamp` alignment/truncation to the hour (consistent timezone/UTC handling) before joins and shifts.
  - Current repo structure now separates raw feature introduction from optional transforms; rolling features remain available as an opt-in helper for model-specific feature recipes.

- Phase 3 — Spatiotemporal Graph Forecasting Model (hourly)
  - Build model combining Graph Attention (GAT) for space + TCN for time, plus weather embeddings; inputs and adjacency are provided at hourly timesteps.
  - Forecast horizons mapped to hourly resolution: 1-hour, 2-hour, 3-hour ahead forecasts.
  - Interpret lookback and dilation parameters in hourly steps (e.g., lookback=24 → last 24 hours).
  - Train and evaluate model using hourly windows; ensure augmentation and interpolation are only applied to training folds.

- Phase 4 — Fault-Tolerant Sensor Recovery (hourly)
  - Implement two-stage recovery for hourly series: (1) Kriging spatial interpolation for short gaps measured in hours; (2) Graph-based reconstruction for longer outages using learned spatiotemporal dependencies on hourly windows.
  - Define gap thresholds in hours (example: short gap ≤ 3 hours → temporal interpolation; longer → spatial kriging/graph reconstruction).

- Phase 5 — Deployment & Proactive Response
  - Hybrid controller: rule-based thresholds + predictive-control triggers driven by hourly forecasts; real-time API and dashboard operate on hourly prediction outputs unless a higher-frequency ingestion stream is added.

## Completed vs Incomplete (current repo state, hourly-aware)
Assumption: "completed" means code artifacts or scripts exist and have been run locally; "partial" means some code exists but deployment/complete implementation/testing is missing.

- Phase 1 — IoT Stream Ingestion
  - Refactored data acquisition into `infrastructure/kafka/data_sources` (OpenAQ archive fetch, Environment Canada scraper, IQAir scraper): COMPLETED
  - Live OpenAQ v3 client (`data_sources/openaq.py` `poll()`): COMPLETED AND UNIT-TESTED
  - Live Environment Canada GeoMet SWOB client (`data_sources/environment_canada.py` `poll()`): COMPLETED AND UNIT-TESTED
  - `BaseProducer` with dead-letter queue: COMPLETED AND UNIT-TESTED
  - Raw→canonical `normalizer` consumer: COMPLETED AND UNIT-TESTED
  - Parquet audit `sink`: COMPLETED AND UNIT-TESTED
    - Note: the parquet sink is an audit/verification artifact and is NOT yet wired to model training (Phase 2 scope).
  - `create_topics.py` and `register_schemas.py`: COMPLETED AND UNIT-TESTED
  - Marker-gated end-to-end integration smoke test (`tests/test_kafka_integration.py`): COMPLETED
  - Full Kafka cluster + deployed topics: COMPLETED AND LOCAL TESTED
    - Docker-compose with Zookeeper, Kafka, Schema Registry (port 8081) and Flink JobManager/TaskManager (Flink UI port 8082) at root `docker-compose.yml`.
    - Canonical Avro schema at `infrastructure/kafka/schemas/measurement.avsc`; per-source raw schemas: `openaq_raw.avsc`, `envcanada_raw.avsc`, `iqair_raw.avsc`.
    - Schema registration helper at `infrastructure/kafka/register_schemas.py`.
    - Verified Schema Registry reachable and schemas registered during local bring-up.
    - Note: this provides a local test environment; production deployment requires provisioning, secure config and topic replication settings.
  - Schema and vocabulary notes:
    - `co2` is dropped from the canonical schema — no public source provides it.
    - The canonical schema uses correct sensor names (`pm25`, etc.); reconciling these to the training-data vocabulary (`pm2`, `Temp Definition °C`, wind in tens-of-degrees) is done via a boundary rename layer that is Phase 2 scope.

- Phase 2 — Real-time Feature Engineering (hourly)
  - Local Flink feature code under `analytics/flink_jobs/feature_engineering.py`: PARTIAL (raw feature introduction exists; rolling/lag transforms remain opt-in helpers and are not yet split into dedicated per-model recipes)
  - Raw feature introduction and optional transform helpers are consolidated in `analytics/flink_jobs/feature_engineering.py`: COMPLETED
    - Added `introduce_raw_features()` as the default raw hourly entrypoint.
    - Kept `compute_rolling_features()` as an explicit helper for later per-model/per-horizon recipes.
    - Smoke-tested locally: run `python analytics/flink_jobs/feature_engineering.py` prints raw and rolling feature samples.
  - Hourly aggregation / direction-aware adjacency production: PARTIAL (design and code fragments present; productionized hourly adjacency not confirmed)

- Phase 3 — Spatiotemporal Graph Forecasting Model (hourly)
  - `models/spatiotemporal` training pipeline and `SpatioTemporalModel` exist; training (quick + full) and Optuna HPO ran: COMPLETED
  - Hourly GAT+TCN weather-aware implementation: PARTIAL (model exists but may not fully implement time-varying hourly adjacency or all proposed variants)
  - Multi-horizon forecasts (1-hour, 2-hour, 3-hour) implemented and evaluated at hourly resolution: COMPLETED

- Phase 4 — Fault-Tolerant Sensor Recovery (hourly)
  - OpenAQ merge, timestamp repair, and `pm2` fill from OpenAQ: COMPLETED
  - Environment Canada and IQAir scrapers implemented as modules for fallback: COMPLETED (code present; live scrapes sometimes hit rate limits)
  - Kriging spatial interpolation and learned graph-reconstruction module for hourly series: SCAFFOLDING ADDED
    - Added `analysis/kriging.py` (pykrige wrapper + IDW fallback) and `tests/test_kriging.py`.
    - Added a simple graph-reconstruction stub at `models/reconstruction/graph_reconstruction.py` (placeholder for long-gap reconstruction).

- Phase 5 — Deployment & Proactive Response
  - Rule-based safety triggers (proposal rules are specified): PARTIAL (rules specified in proposal; not integrated into a deployed controller)
  - Predictive control layer and hybrid controller with dashboard & simulation: SCAFFOLDING ADDED
    - FastAPI serving skeleton at `api/serve.py` (health and `POST /predict` placeholder).
    - Streamlit dashboard skeleton at `dashboard/streamlit_app.py` that lists local MLflow experiments.

- Evaluation & Baselines
  - Baselines (HA, Linear, RF, LSTM) implemented and runnable via `models/baselines/train_baselines.py`: COMPLETED
    - Approach: separate per-horizon feature hooks are defined for each baseline family, then each horizon is trained and evaluated independently using MAE/RMSE/R².
      1. Historical Average (HA) — single-output constant predictor per horizon
      2. Linear Regression (LR) — single-output LinearRegression on flattened windows
      3. Random Forest (RF) — single-output RandomForestRegressor on flattened windows
      4. LSTM — single-output PyTorch LSTM trained on sequence windows
    - Outputs: saved models under `models/saved_models/` and a metrics summary at `models/saved_models/baseline_metrics.json`.
    - Run the benchmark (example):
      ```bash
      # all baselines, 5 epochs for LSTM (fast smoke)
      python models/baselines/train_baselines.py --model all --path data/raw/historical_rawdata_pm2_filled.csv --epochs 5
      ```
    - Results will be printed and written to `models/saved_models/baseline_metrics.json` for comparison and MLflow logging can be added later.
    - Note: a prior full run of `--model all` exists in the repo's `mlruns` directory and shows completed runs for LR/RF/LSTM/HA.
  - End-to-end streaming latency / throughput benchmarks and robustness tests under simulated sensor failure: PARTIAL (some experiments done locally, streaming tests/deployment-level benchmarks not completed)

## Key completed artifacts (repo evidence)
- `infrastructure/kafka/data_sources/openaq.py` — OpenAQ archive fetch and pivot: COMPLETED
- `scripts/merge_openaq_into_historical.py`, `scripts/fix_merged_timestamps.py`, `scripts/fill_pm2_from_openaq.py` — data merge/repair/fill: COMPLETED
- `data/raw/historical_rawdata_pm2_filled.csv` — filled pm2 dataset: COMPLETED
- `scripts/hpo_optuna_pm2.py` — Optuna HPO for pm2: COMPLETED
- `models/saved_models/hpo_best_pm2_*.txt` and `training_result_*.json` — HPO results and final full-run metrics: COMPLETED
- `models/spatiotemporal/train.py` MLflow logging — run parameters, metrics, and artifacts are tracked per training run: COMPLETED

## Assumptions used to mark status
- "Completed" = code present and executed locally with artifacts saved (CSV/model/hpo outputs). If a component only has design notes or partial code it is marked PARTIAL.
- No external cluster/production deploy is assumed unless there are explicit deploy scripts or CI config. I did not assume any cloud services are running.
- All processing, interpolation, and augmentation in this document are assumed to operate on hourly series unless noted otherwise.
- Rate-limited external APIs (IQAir) may prevent full automated recovery in real-time; code exists as fallback but live success varies.

## Recommended next steps (actionable sub-steps — hourly)
1. Deploy streaming stack (high priority)
   - Provision a local (or cloud) Kafka + Flink environment for end-to-end tests.
   - Create Avro schemas and topic config for each source and run ingestion consumer tests. Ensure topic consumers can aggregate to hourly buckets.
  - Local commands (tested):
    ```bash
    # start local stack (Schema Registry on port 8081, Flink UI on port 8082)
    docker compose up -d

    # register Avro schemas (requires Schema Registry on port 8081)
    python infrastructure/kafka/register_schemas.py

    # quick registry health check
    python scripts/check_schema_registry.py
    ```
2. Harden Flink feature engineering (high impact, hourly)
     - Smoke-test:
       ```bash
       python analytics/flink_jobs/feature_engineering.py
       ```
   - Aggregate raw inputs to hourly timestamps and truncate `timestamp` to the hour before feature computation.
   - Replace repeated DataFrame inserts with `pd.concat` to avoid fragmentation (fix `analytics/flink_jobs/feature_engineering.py`).
   - Implement efficient rolling/lag generation using hourly windows and test latency on a sample stream.
3. Complete direction-aware adjacency & model improvements (hourly)
   - Implement meteorology-conditioned adjacency A(t) aggregated hourly in the model input pipeline.
   - Add or verify GAT layers that accept time-varying hourly adjacency; test with ablation studies.
4. Implement full recovery pipeline (hourly)
   - Add kriging interpolation (Stage 1) for short gaps using hourly values (use `pykrige` or `skgstat`).
   - Implement graph-based reconstruction (Stage 2) using trained model or learned imputation network on hourly windows.
  - Scaffolding commands/tests are present: `pytest tests/test_kriging.py` (requires pytest and numpy installed).
5. Build the proactive response simulator & dashboard
   - Implement a small REST API to serve hourly forecasts and trigger rule/predictive controllers.
   - Create a simple UI or log-based simulator to compare reactive vs predictive policies.
6. Evaluation and robustness (hourly)
   - Run robustness tests with synthetic missingness (5,10,20%) on hourly series and record MAE/RMSE/R² degradation.
   - Measure end-to-end latency under expected ingestion rates, validating hourly aggregation and feature throughput.
7. Modeling improvements to meet R² target (exploratory)
   - Feature selection / addition: engineered meteorology features, local emission schedules, sensor calibration offsets.
   - Try stronger spatial model variants, ensembling, longer hourly lookbacks, or per-station fine-tuning.

## Data Interpolation & Oversampling (precision improvement, hourly)
This project will include a dedicated preprocessing + augmentation solution to improve forecast precision where data gaps or class imbalance exist. All interpolation/oversampling operates on hourly series unless higher-frequency data is introduced.

- Temporal interpolation: use conservative methods for short gaps (linear, spline, or time-aware interpolation) applied only to short contiguous missing windows on the hourly series.
- Spatial interpolation: use kriging (e.g., `pykrige`) or inverse-distance weighting for short-term spatial fill-ins leveraging neighboring stations' hourly values.
- Oversampling / augmentation: for rare high-pollution events or imbalanced patterns, apply time-series augmentation only on the training split. Options include:
  - Window bootstrap (resample temporal windows)
  - Time-series SMOTE / TS-DBA-based synthetic sequence generation
  - Generative models (seqGAN/TimeGAN) for synthetic sequences if necessary
- Implementation rules: never leak future information — perform interpolation and augmentation within train-only folds; validate on raw/unaugmented validation/test sets.

Recommended pipeline integration (hourly):
- Aggregate raw inputs to hourly resolution and implement interpolation hooks in the preprocessing module (`analytics/flink_jobs/feature_engineering.py` or a dedicated preprocessor) with configurable gap thresholds in hours.
- Apply augmentation during dataset window creation in `models/spatiotemporal/train.py` only to training windows (hourly windows).
- Add unit tests and ablation experiments to verify augmentation benefits and avoid overfitting.

Quick checklist addition:
- [ ] Add interpolation & oversampling pipeline and tests

## Quick actionable checklist (copyable)
- [ ] Aggregate raw inputs to hourly timestamps and truncate `timestamp` to the hour
- [ ] Deploy Kafka + Flink test cluster and validate ingestion topics
- [ ] Fix DataFrame fragmentation in `analytics/flink_jobs/feature_engineering.py`
- [ ] Implement direction-aware adjacency in preprocessing pipeline (hourly)
- [ ] Add kriging-based short-gap imputation (hourly)
- [ ] Implement graph-reconstruction imputation for long outages (hourly)
- [ ] Add REST endpoint for forecast serving and controller hooks
- [ ] Build a minimal dashboard / simulator for proactive response
- [ ] Run robustness + streaming latency benchmarks on hourly flows
- [ ] Experiment model variants to target R² 0.8–0.9 (if feasible)
- [x] Keep raw data downloader and raw hourly data in the repo for local feature/model work
- [x] Fold notebook graph helpers into `analytics/flink_jobs/feature_engineering.py`
- [x] Add missing project requirements for the notebook-style model helpers

## Notes / Risks
- External APIs (IQAir) are rate-limited; rely on archived OpenAQ or Environment Canada when possible.
- Improving R² to 0.8–0.9 may require richer inputs (higher-quality ground-truth PM2 sensors, external emission inventories, or per-source emissions data) beyond current historical coverage.

---
Generated by the repo assistant on 2026-05-29.

## Recent changes (2026-06-02)

- **Restored** `infrastructure/kafka/scripts/data_downloader.py` to a richer raw-data cleaner implementing `clean_raw_data(...)` and `DEFAULT_KEEP_COLUMNS` so scripts and tests continue to work with the local historical CSV.
- **Merged** notebook graph helpers into `analytics/flink_jobs/feature_engineering.py` (wind components, dynamic edge construction, LSTM/graph sequence builders) and made LR/RF/LSTM/STGNN feature pipelines use the notebook-style preprocessing.
- **Added** STGNN training branch to `models/baselines/train_baselines.py` and registered `stgnn` in feature builders and CLI choices.
- **Added** `models/baselines/horizon_plus.py` (per-horizon LightGBM / RF / LR fallback) and updated baseline wiring for per-horizon single-output training.
- **Updated** `requirements.txt` with optional notebook dependencies (e.g., `scikit-learn`, `torch-geometric` entries noted) — install optional packages only as needed for STGNN/LightGBM runs.
- **Tests:** added/updated small tests around the data downloader and feature engineering; syntax checks passed locally.

If you'd like, I can now run a short smoke test (one epoch, CPU) for a selected model to verify end-to-end training with the updated feature pipeline.

## Remaining before GitHub push

- [ ] Run a full benchmark in `Intelligent-IOT-blackwell` with `python scripts/run_baselines.py`
- [ ] Compare the full-horizon metrics against the reference results from `tested_models.py`
- [ ] Review `git status` and remove any remaining unneeded docs, notebooks, or artifacts
 - [ ] Confirm `data/raw/Raw_Data.csv` is the canonical raw dataset and the only retained raw dataset in `data/raw`
- [ ] Commit and push the cleaned project to GitHub

## Cleanup note

- MLflow support, helper scripts, and local tracking artifacts are being removed from the cleaned project path.
