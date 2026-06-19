# Intelligent-IOT — Architecture & Implementation Status

Real-time PM2.5 air-quality forecasting for Toronto using IoT streams, wind-aware
diffusion features, and a Kafka pipeline. This document describes the system as
actually implemented and is the single place to understand what runs, what's
tested, and what's deferred.

> **Status legend:** ✅ implemented + tested · ⚙️ implemented, operational/manual run pending · ⏳ deferred

---

## 1. System overview

```
 OpenAQ (4 PM2.5 stations) ─┐
 Env Canada SWOB (met) ─────┤
 IQAir (stub) ──────────────┘
        │  producers (Avro)
        ▼
  aq.{source}.raw ──► normalizer ──► aq.measurements ──► sink ──► Parquet (audit)
                                          │
            ┌─────────────────────────────┤
            ▼                             ▼
    feature consumer              recovery (in feature consumer)
   (per-station snapshot,         when target PM2.5 missing →
    diffusion features,           wind-aware neighbor estimate
    hourly tick)                  or temporal fallback
            │
            ▼
       aq.features ──► inference consumer ──► aq.predictions ──► alert consumer ──► aq.alerts
       (9 features +    (ForecastBundle:        (h1/h2/h3 +         (EPA thresholds      │
        12h history)     LR/RF/LSTM/STGNN)       forecast_pm25)      35.5 / 125.5)       │
                                                                                          ▼
                                                            live_state consumer ──► live_state.json
                                                                                          │
                                                          API GET /live/predictions, /live/alerts
                                                                     │
                                                              Streamlit dashboard (Live tab)
```

**Backbone:** Apache Kafka + Confluent Schema Registry + Avro. Every stage is a
Python `confluent-kafka` consumer following one shared loop pattern (poll →
deserialize → transform → produce → dead-letter on error → manual commit). No
Flink (the container exists but is unused; Python consumers were chosen for
lower risk).

**Resolution:** hourly. Timestamps are floored to the hour at normalization.

---

## 2. Kafka topics

| Topic | Producer | Consumer(s) | Payload |
|---|---|---|---|
| `aq.openaq.raw` / `aq.envcanada.raw` / `aq.iqair.raw` | source producers | normalizer | per-source raw Avro |
| `aq.measurements` | normalizer | feature consumer, sink | canonical `Measurement` |
| `aq.features` | feature consumer | inference consumer | 9 model features + 12h history |
| `aq.predictions` | inference consumer | alert consumer, live_state | h1/h2/h3 + forecast_pm25 |
| `aq.alerts` | alert consumer | live_state | level/alert/recommendation |
| `aq.deadletter` | all consumers | (manual inspection) | poison messages (JSON) |

---

## 3. Data sources & stations

- **OpenAQ (AirNow, reference-grade):** PM2.5 at 4 Toronto stations — Downtown `7570` (target), West `1274950`, North `1274949`, East `1210341`. Pollutants only (no wind).
- **Environment Canada SWOB:** meteorology — temperature, **dew point**, humidity, wind speed, wind direction, pressure.
- **IQAir:** stub (disabled).

Because OpenAQ PM2.5 records carry no wind, meteorology is **joined** per station
from the nearest SWOB reading. `infrastructure/kafka/station_registry.py` holds
the station id → (lat, lon, role) map.

---

## 4. The five phases

### Phase 1 — IoT Stream Ingestion ✅
Multi-source producers → per-source raw Avro topics → normalizer (raw→canonical
`Measurement`, dedup on `station_id|source|hour`, dead-letter on poison) →
Parquet audit sink. Dew point captured from SWOB `dwpt_temp`.
*Files:* `infrastructure/kafka/{producers,consumers,data_sources,schemas}`.
*Tested:* unit + a marker-gated integration test; **proven live e2e** (OpenAQ +
SWOB → canonical → parquet, DLQ empty).

### Phase 2 — Real-Time Feature Engineering ⚙️
Feature consumer keeps per-station rolling history; on an hourly tick it builds
the target's feature vector and computes **wind-aware diffusion features** —
`upwind_pm25`, `transport_potential`, `wind_alignment` — from the cross-station
snapshot, emitting a 9-feature record to `aq.features`. Distances use the
Haversine great-circle (not Euclidean); wind alignment uses a local north-east
bearing.
*Files:* `infrastructure/kafka/consumers/features.py`,
`analytics/features/{geo,diffusion_features}.py`, `infrastructure/kafka/{feature_adapter,met_join,rolling_buffer}.py`.
*Tested:* pure functions unit-tested; the live hourly-tick `run()` (with diffusion
+ recovery) has **not yet been run end-to-end** (see §7).

### Phase 3 — Spatiotemporal Forecasting ⚙️
Inference consumer loads a `ForecastBundle` (per-horizon h1/h2/h3) and predicts
PM2.5. Model families: Historical Average, Linear Regression, Random Forest,
LSTM, STGNN (GAT+TCN). All consume the 9-feature vector incl. diffusion features.
The STGNN graph uses real station coordinates and wind-aware edges.
*Files:* `models/`, `infrastructure/kafka/consumers/inference.py`.
*Tested:* model + bundle unit tests; multi-station retrain **demonstrated** (LR,
9 features, real backfilled data). Full all-family + STGNN retrain is env-gated
(see Deferred).

### Phase 4 — Fault-Tolerant Recovery ⚙️
When the target's PM2.5 is missing at tick time, recover it: **Stage 1** wind-aware
neighbor estimate (≤3h gap), **Stage 2** temporal fallback from the station's own
history (longer gaps). A degradation evaluation injects 5/10/20% missing and
scores MAE/RMSE.
*Files:* `analytics/recovery/{spatial_recovery,degradation_eval}.py`, wired into
`features.py`.
*Tested:* pure functions unit-tested; live recovery + real-data degradation run
deferred (needs neighbor columns in the backfill).

### Phase 5 — Deployment & Response ⚙️
FastAPI service: request/response `/predict`, `/alerts`, `/forecast-and-alert`,
plus **live** `GET /live/predictions` and `/live/alerts` fed by the `live_state`
consumer (materializes the stream into `live_state.json`). Streamlit dashboard
with a Live tab. Alert thresholds are EPA AQI breakpoints: **35.5** (warning),
**125.5** (critical).
*Files:* `infrastructure/deployment/`, `infrastructure/kafka/{live_store.py,consumers/live_state.py}`.
*Tested:* store/consumer/API unit-tested; live_state consumer + dashboard
end-to-end run pending (see §7).

---

## 5. Key design decisions

- **Wind-aware diffusion** is the project's core contribution: a neighbor's
  influence on the target is weighted by `1/distance × wind-alignment` (upwind
  neighbors amplified, downwind suppressed). The same weighting drives the
  diffusion *features*, the STGNN graph *edges*, and Phase 4 *recovery*.
- **Great-circle geometry** everywhere (`analytics/features/geo.py`) — never
  Euclidean on lat/lon.
- **Historical met** for training comes from the Open-Meteo archive (per-station
  coordinates); SWOB realtime only covers recent data.
- **Pure-Python consumers**, one shared loop + dead-letter pattern, no Flink.

---

## 6. How to run end-to-end (local)

```bash
docker compose up -d                                  # Kafka, Schema Registry :8081, Flink :8082 (idle)
python -m infrastructure.kafka.create_topics          # 8 topics
python -m infrastructure.kafka.register_schemas       # Avro schemas
export OPENAQ_API_KEY=...                              # required for OpenAQ
# consumers (separate processes):
python -m infrastructure.kafka.consumers.normalizer
python -m infrastructure.kafka.consumers.features
python -m infrastructure.kafka.consumers.inference
python -m infrastructure.kafka.consumers.alerts
python -m infrastructure.kafka.consumers.live_state
# producer:
python -m infrastructure.kafka.producers.run_ingestion
# API + dashboard:
uvicorn infrastructure.deployment.app:app
streamlit run infrastructure/deployment/dashboard/streamlit_app.py
```

---

## 6a. Training & retraining the models

**Pipeline:** backfill a multi-station training set → train per-horizon models →
the saved bundles become the active model the inference consumer / API load.

```bash
# 1. Backfill multi-station training data (OpenAQ archive PM2.5 + Open-Meteo historical met)
python -m infrastructure.kafka.scripts.backfill_multistation --start 2025-05-01 --end 2025-12-31
#    -> data/external/multistation/train.csv  (target pm25 + 6 base + 3 diffusion features)

# 2. Train all families and persist bundles + active model + registry
python scripts/save_deployment_models.py
#    (or a subset:  python scripts/run_baselines.py --model lr --path data/external/multistation/train.csv)

# 3. Evaluate recovery robustness (5/10/20% missing)
python -m analytics.recovery.degradation_eval --path data/external/multistation/train.csv
```

**Environment matters — use the py3.11 conda env for the full retrain.** The
trained `.pkl` artifacts pin `scikit-learn==1.5.1`, and the heavy ML stack
(TensorFlow for LSTM, torch/torch_geometric for STGNN, lightgbm for RF) must
load together without conflicting native OpenMP runtimes.

```bash
conda env create -f environment.yml     # name: Intelligent-IOT (Python 3.11)
conda activate Intelligent-IOT
pip install -r requirements.txt
python scripts/save_deployment_models.py # trains HA + LR + RF + LSTM + STGNN cleanly
```

**Known py3.13 pitfall.** On a Python 3.13 venv, training **all** families in one
process segfaults (exit 139) — TensorFlow + torch + lightgbm load incompatible
native libs together. Only LR trains cleanly there. Workarounds if you can't use
py3.11:
- `export KMP_DUPLICATE_LIB_OK=TRUE` (lets the duplicate OpenMP load — the common single fix).
- Train one family per process (`--model lstm`, then `--model stgnn`) so TF and torch never co-load.
- For RF: `pip uninstall lightgbm` to fall back to sklearn RandomForest (no libomp).

The canonical retrain is the py3.11 path above; it clears all three crashes and
the sklearn version skew at once.

---

## 7. Testing status

- **Unit tests:** ~60 across ingestion, geo, diffusion, feature adapter, rolling
  buffer, consumers (pure transforms), recovery, live store/API, config. All green
  in the dev venv.
- **Integration / e2e:** the **full current chain was run end-to-end via docker**
  (normalizer → features[diffusion+recovery] → inference → alerts → live_state +
  producer + API). Confirmed: `aq.features` records carry all 9 features incl. the
  diffusion features (`upwind_pm25` non-zero), predictions carry h1/h2/h3 +
  `current_pm25`, `live_state.json` populates, the API serves
  `GET /live/predictions` & `/live/alerts`, and `aq.deadletter` stayed empty
  throughout. The active model in that run was the pre-Path-A 6-feature STGNN
  bundle (so it doesn't yet consume the diffusion features — full 9-feature
  retrain is the deferred py3.11 item); the chain itself is verified end to end.
- **Deps note:** the deployment env is Python 3.11 with `scikit-learn==1.5.1`
  (matches the trained `.pkl`). The dev venv is 3.13 (loads the heavy ML stack
  with a known TF+torch segfault when all are imported together → full retrain
  runs on 3.11).

---

## 8. Deferred / known gaps

- Full all-family + STGNN retrain on the py3.11 env (dev-venv segfault on 3.13).
- STGNN *true* per-station graph nodes — the backfill must emit per-station feature
  windows (current fallback: target window across real-coord nodes, no perturbation).
- Phase 4 real-data degradation eval — backfill must surface neighbor `pm25_<id>` columns.
- Phase 4 Stage 2 learned graph reconstruction (currently temporal).
- CI full-suite green (the `python-version` parse bug is fixed; heavy-deps/test triage remains).
- Dashboard time-series/history view; endpoint auth.
```
