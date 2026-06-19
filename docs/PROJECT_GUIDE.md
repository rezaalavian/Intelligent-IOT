# Intelligent-IOT — Project Guide

> A complete, presentation-ready reference for the **PM2.5 air-quality forecasting** system.
> Written so a team member can present the whole project **without reading the code**, while still
> pointing to exact files (`path:line`) for anyone who wants to look.

---

## 1. Executive Summary

**What it is.** An end-to-end Internet-of-Things data system that ingests live air-quality and weather
data for Toronto, engineers physics-informed features, forecasts PM2.5 (fine particulate matter) **1, 2,
and 3 hours ahead**, recovers from missing sensor data, and serves predictions + health alerts through an
API and dashboard.

**How it is built.** Five connected phases running over **Apache Kafka** with **Avro** schemas and a
**Schema Registry**. Each phase is an independent, fault-tolerant Python consumer. Data flows through a
chain of Kafka topics from raw ingestion to served alerts.

**The headline result.** After benchmarking five model families at three horizons on three years of real
data, the deployed model is a **per-horizon Random Forest (gradient-boosted trees) on a 13-feature
"with_pollutants" feature set**, achieving test **R² = 0.903 / 0.787 / 0.682** at +1h / +2h / +3h —
far above the naive baseline and ahead of deep-learning alternatives (LSTM, graph neural network).

**The core technical contribution.** *Wind-aware diffusion features.* Instead of treating neighbouring
stations as equally relevant, the system weights them by **where the wind is blowing from** — an upwind
neighbour is a strong predictor of what is about to arrive at the target station. The same wind-physics
weighting powers three different parts of the system: the predictive features, the graph model's edges,
and the missing-data recovery.

---

## 2. System Architecture at a Glance

```
                    LIVE SOURCES                         BATCH / TRAINING
              ┌───────────────────────┐          ┌──────────────────────────┐
              │ OpenAQ v3 API (PM2.5  │          │ OpenAQ S3 archive (PM2.5 │
              │   + gases)            │          │   + gases per station)   │
              │ Env Canada SWOB (met) │          │ Open-Meteo archive (met) │
              └──────────┬────────────┘          └────────────┬─────────────┘
                         │ produce Avro                        │ backfill script
                         ▼                                     ▼
   PHASE 1   aq.openaq.raw / aq.envcanada.raw / aq.iqair.raw   data/external/multistation/
   INGEST            │  normalizer consumer                       train.csv  (25,559 rows,
                     ▼                                             3 years, 14 columns)
             aq.measurements ──────────► (audit) Parquet sink              │
                     │  features consumer (hourly tick)                    │ trains
   PHASE 2           ▼   + wind-diffusion + recovery                       ▼
   FEATURES   aq.features                                          5 model families
                     │  inference consumer (loads active model)    HA / LR / RF / LSTM / STGNN
   PHASE 3           ▼                                                     │ best per horizon
   FORECAST   aq.predictions                                       models/saved_models/
                     │  alerts consumer (EPA thresholds)              active_model.pkl
   PHASE 4           ▼                                                     ▲
   RECOVERY   aq.alerts                                                    │ loaded by
   + ALERTS           │  live_state consumer                              inference + API
                      ▼
   PHASE 5    data/stream/live_state.json ──► FastAPI  ──►  Streamlit dashboard
   DEPLOY                                      /live/*        (forecast, alerts, metrics)

   Any consumer failure on a bad message → aq.deadletter (poison-message quarantine)
```

**Topic chain:** `aq.{source}.raw → aq.measurements → aq.features → aq.predictions → aq.alerts`, plus
`aq.deadletter` for failures. Topics are created in `infrastructure/kafka/create_topics.py:13` and named
in `infrastructure/kafka/config.py:32`.

**Two data planes.** A **batch plane** builds the training CSV offline; a **live streaming plane** (the
Kafka consumers) serves real-time predictions. They use *different sources for the same quantities*
(archive vs live) — see §8.5.

---

## 3. Phase 1 — IoT Stream Ingestion

**Goal:** pull data from heterogeneous public sensors and land it as one clean, typed, hourly record.

### 3.1 Data sources (`infrastructure/kafka/data_sources/`)

| Source | Role | What it provides | Live? |
|---|---|---|---|
| **OpenAQ** (`openaq.py`) | primary AQ | PM2.5 **and** co-pollutants (NO, NO2, NOx, O3, SO2, CO) | **Live** v3 API (`openaq.py:197`) |
| **Environment Canada SWOB** (`environment_canada.py`) | meteorology | temperature, **dew point**, humidity, wind speed/direction, pressure | **Live** GeoMet API (`environment_canada.py:205`) |
| **Open-Meteo** | historical met | weather archive for training only | Archive only (`backfill_multistation.py:122`) |
| **IQAir** (`iqair.py`) | scaffold | (none — disabled stub) | No (`iqair.py:78`) |

- OpenAQ carries pollution but **no wind**; SWOB carries weather. The two are fused later — this split is
  *why* a per-station meteorology join exists.
- **Resilience built into the clients:** OpenAQ retries with exponential backoff and honours HTTP 429
  rate-limit resets (`openaq.py:183`); SWOB always queries a recent 2-hour window and follows pagination
  safely (`environment_canada.py:128`).
- IQAir is **forward-compatible scaffolding** — the topic, schema, and normalizer branch all exist, but
  `poll()` returns nothing because IQAir has no free realtime API (`iqair.py:78`). Good example to cite
  for "designed for extension."

### 3.2 Producers (`infrastructure/kafka/producers/`)

- **Entry point:** `run_ingestion.py:74`. Runs an infinite loop: poll every enabled source, produce, sleep
  (`run_ingestion.py:80`).
- **Polling cadence:** every source defaults to **300 s (5 min)** (`config.py:46`); the loop sleeps the
  minimum interval across sources (`run_ingestion.py:78`). SWOB's overlapping 2-hour window means no
  readings are missed between ticks.
- **Fault isolation:** each source's poll is wrapped in try/except — one source failing is logged and
  skipped, never stopping the others (`run_ingestion.py:34`).
- **Keyed by station:** records are keyed by `station_id` so all readings for a station land on the same
  Kafka partition (preserves per-station ordering) (`base.py:26`).
- **Safe serialization:** the base producer serializes to Avro **before** producing, so a bad record is
  caught and routed to the dead-letter topic as JSON instead of crashing (`base.py:31`).

### 3.3 Kafka topics, Avro, and the Schema Registry

- **8 topics** created in `create_topics.py:13`; names in `config.py:32`. Each is 3 partitions.
- **Why Avro + Schema Registry:** compact binary messages, a typed contract enforced centrally, and
  controlled schema evolution. Producers/consumers are set to **not auto-register** and to **use the
  latest registered version** (`run_ingestion.py:52`), so schema changes are a deliberate step, not
  accidental drift. Schemas registered with **BACKWARD** compatibility (`register_schemas.py:21`).
- **Schema files** (`infrastructure/kafka/schemas/`): per-source raw schemas, plus the canonical
  `measurement.avsc`, `feature.avsc`, `prediction.avsc`, `alert.avsc`.

### 3.4 The normalizer — turning many shapes into one

`consumers/normalizer.py` converts the three raw source formats into one canonical hourly **Measurement**
on `aq.measurements`.

- Floors every timestamp to the hour in UTC — the pipeline's fixed hourly resolution (`normalizer.py:21`).
- OpenAQ sends one parameter per message ("long format"); the normalizer **fans them back in** to one wide
  row per `station|source|hour`, letting non-null values win (`collapse_same_hour`, `normalizer.py:53`).
- Missing required fields (`station_id`, `timestamp`) → dead-letter, not a crash.

### 3.5 The shared consumer pattern + dead-letter design

Every consumer repeats the same loop (an intentional convention, not a base class):
**poll → deserialize (Avro) → transform → produce downstream → on error, dead-letter → manual commit.**
Canonical example: `normalizer.py:91`.

**Why a dead-letter queue (DLQ):** a single poison message (corrupt Avro, missing field) must not crash the
consumer or block its partition. Catching the error, writing the raw bytes to `aq.deadletter`, and
committing means the pipeline **keeps making progress** while preserving the bad message for inspection.
Manual commit (`enable.auto.commit: False`) gives **at-least-once** delivery — a message is only
acknowledged once it has been handled or safely quarantined (`normalizer.py:85`).

### 3.6 Why pure-Python consumers and not Apache Flink

The Docker stack includes a Flink container, but it is **idle and unused** (`docs/ARCHITECTURE.md:44`).
The workload is hourly, single-target, and low-volume. A JVM stream processor would add operational
complexity and a serialization boundary for no throughput benefit. Pure-Python consumers reuse the exact
same feature/model code (`analytics/`, `models/`) with no glue layer. **This is a deliberate
simplicity-over-machinery decision** worth stating on a slide.

---

## 4. Phase 2 — Real-Time Feature Engineering

**Goal:** turn raw hourly measurements into the feature vector the model expects, including the
wind-physics features.

### 4.1 Great-circle geometry — why not "flat-earth" distance (`analytics/features/geo.py`)

`haversine_m` computes true great-circle distance in metres (`geo.py:8`); `north_east_offsets_m` converts a
lat/lon difference into local north/east metres, scaling east by `cos(latitude)` (`geo.py:17`).

**Why:** latitude and longitude are angles on a sphere, not flat metres. At Toronto's latitude one degree
of longitude is only ~0.72× as long as one degree of latitude. Treating coordinates as a flat plane would
distort both the distances **and the directions** that the wind-alignment math depends on. (This was a
specific correction made during development: Euclidean distance on lat/lon is wrong; the earth is curved.)

### 4.2 Wind-aware diffusion features (`analytics/features/diffusion_features.py`)

For the target station, given the wind vector and the neighbours' PM2.5, the system computes three features
(`diffusion_features.py:6`):

- **`upwind_pm25`** — a weighted average of neighbour PM2.5, where each neighbour's weight is
  *inverse distance* **amplified if it is upwind** and *suppressed if downwind*. Concretely the alignment
  is the cosine between the neighbour→target direction and the wind vector; weight = `1/distance × (1 +
  alignment)` when upwind, `1/distance × exp(alignment)` when downwind (`diffusion_features.py:22`). This
  captures "pollution is being carried toward us from that direction."
- **`transport_potential`** — the strongest wind-driven transport vector across neighbours
  (`speed × alignment`) (`diffusion_features.py:27`).
- **`wind_alignment`** — the mean alignment across neighbours.

If no neighbour has a PM2.5 value that hour, all three default to 0 (`diffusion_features.py:29`).

**One physics formula, three uses.** The identical weighting appears in (1) these features, (2) the STGNN's
dynamic graph edges, and (3) Phase-4 recovery (`spatial_recovery.py:6`). Emphasise this unification —
it is the project's central idea.

### 4.3 The streaming feature consumer (`consumers/features.py`)

- Keeps per-station state: latest PM2.5 per station, recent meteorology, and a 12-hour history buffer
  (`features.py:62`).
- **Emits on an hourly tick.** `FEATURE_TICK_SECONDS` defaults to **3600 s** (`features.py:48`); the
  consumer ingests continuously but only emits one feature record per tick, matching the hourly data
  resolution. *(For demos/tests, set `FEATURE_TICK_SECONDS=5` to see flow immediately.)*
- On each tick it joins the nearest meteorology reading (`met_join.py:4`), runs recovery if the target's
  PM2.5 is missing (§6.1), computes the diffusion features, attaches the 12-hour history, and produces to
  `aq.features` (`features.py:104`).

### 4.4 The exact feature columns

The live `aq.features` record carries **9 features**: 6 base
(`temp definition °c`, `dew point definition °c`, `rel hum definition %`, `wind_u`, `wind_v`, `pm25`) +
3 diffusion (`upwind_pm25`, `transport_potential`, `wind_alignment`) (`feature_adapter.py:3`). The training
CSV adds the 4 co-pollutant gases (`no`, `no2`, `nox`, `o3`) for **13** model features total.

---

## 5. Phase 3 — Forecasting & Modeling

**Goal:** predict PM2.5 at +1h, +2h, +3h, choosing the best model and features by evidence.
All metrics below are authoritative from `docs/MODEL_RESULTS.md`.

### 5.1 The five model families

All trained in `models/baselines/train_baselines.py`; inference wrappers in `models/predictors.py`.

1. **Historical Average (HA)** — predicts the training-set mean for every row (`train_baselines.py:353`).
   *Why:* the floor. Any model that cannot beat "always guess the average" (R² ≈ 0) has learned nothing.
2. **Linear Regression (LR)** — OLS on scaled features (`train_baselines.py:372`). *Why:* simplest learner
   that uses the features; fast, interpretable, no version-skew risk. Its autoregressive `pm25` input
   (persistence) gives it most of its short-horizon power.
3. **Random Forest (RF)** — **LightGBM gradient-boosted trees** under the hood (200 trees, depth 6, LR
   0.04, subsampling) when LightGBM is available; a scikit-learn RandomForest only as a fallback
   (`train_baselines.py:391`). *Why:* captures non-linear feature interactions LR cannot (e.g. wind
   direction × upwind PM2.5). **This is the deployed model.**
4. **LSTM** — stacked LSTM over a 12-hour lookback (TensorFlow primary, PyTorch fallback that is actually
   serialized) (`train_baselines.py:428`, `predictors.py:81`). *Why:* PM2.5 is a time series; an LSTM
   should exploit temporal dynamics across the window.
5. **STGNN** — Spatiotemporal Graph Neural Network: GAT (graph attention) over the 3-station network with
   **wind-aware edges**, followed by a temporal convolution stack (`predictors.py:133`,
   `train_baselines.py:499`). *Why:* pollution physically transports between stations along the wind; a
   graph net should model that spatial advection.

### 5.2 The comparison — who won and why

**Test-split metrics (R² / MAE / RMSE), `docs/MODEL_RESULTS.md`:**

| Horizon | Linear Regression | **Random Forest** | LSTM | STGNN | Historical Avg |
|---|---|---|---|---|---|
| **+1 h** | 0.902 / 1.33 / 2.16 | **0.901 / 1.36 / 2.18** | 0.801 / 2.07 / 3.09 | 0.792 / 2.12 / 3.15 | ~0 / 4.90 / 6.91 |
| **+2 h** | 0.776 / 2.08 / 3.28 | **0.784 / 2.05 / 3.21** | 0.698 / 2.57 / 3.81 | −2.835 / 9.00 / 13.56 | ~0 / 4.91 / 6.92 |
| **+3 h** | 0.649 / 2.65 / 4.10 | **0.678 / 2.53 / 3.93** | 0.605 / 2.94 / 4.35 | −2.762 / 9.04 / 13.43 | ~0 / 4.91 / 6.92 |

- **LR and RF are both excellent and far ahead of the baseline.** RF edges LR at the harder +2h/+3h
  horizons; both crush Historical Average.
- **LSTM is consistently behind the tabular models.** On this dataset the persistence (`pm25`) feature plus
  the engineered diffusion features capture the temporal signal already, so the LSTM's extra machinery does
  not pay off — and its artifact is far larger (~390 KB vs LR's ~6.5 KB).
- **STGNN is structurally broken at +2h/+3h** (R² ≈ −2.8, *worse than guessing*). This is a genuine finding
  with two diagnosed root causes (`docs/MODEL_RESULTS.md`):
  1. **Replicated-node graph.** Neighbour PM2.5 columns are not yet in the training frame, so all three
     graph nodes carry the *same* target-station window — the "spatial" graph has no real spatial signal to
     learn (`train_baselines.py:489`).
  2. **Split-index misalignment.** The train/val/test indices computed on the full frame are reused on the
     shorter graph-sequence list; the mismatch grows with horizon, exactly matching h1-fine / h2-h3-collapse
     (`train_baselines.py:491`).
  A sweep confirmed it is structural, not under-training (h3 stays negative at 5/25/80 epochs).

**Decision:** deploy tabular. STGNN is excluded (broken); LSTM is dominated. The chosen model is
**per-horizon Random Forest on the 13-feature `with_pollutants` recipe.**

> Honest framing for the deck: the diffusion *features* (inside LR/RF) clearly work; the STGNN *graph*, in
> its current form, does not. Fixing it requires real per-station node features (a known next step).

### 5.3 Training pipeline

- **70 / 15 / 15 chronological split** (`train_baselines.py:329`). The frame is sorted by time; train =
  earliest, test = most recent ~5–6 months. **Why chronological, not random:** a random shuffle would let
  the model see the future during training (temporal leakage) and inflate scores. The scaler is fit on the
  training split only — preprocessing never sees test statistics either.
- **Per-horizon models** (`models/per_horizon.py`, `scripts/train_per_horizon.py`): each horizon trains an
  independent model with its **own feature set**, because each lead time is a different problem (longer
  horizons lean more on diffusion/transport as persistence decays). Inference returns
  `{h1, h2, h3}` (`forecast_bundle.py:53`).
- **Feature recipes** (`models/feature_recipes.py`): named feature sets — `base6` (6), `diffusion9` (9),
  `with_pollutants` (13, deployed), `base+pollutants` (10), plus single-feature isolation probes.

### 5.4 The ablation harness — automated feature selection (`scripts/run_ablation.py`)

Rather than hand-pick features, the harness sweeps the full cross-product of **feature-set × model ×
horizon**, trains each combination, and writes `experiments.csv` (`run_ablation.py:37`). This produces a
reproducible, evidence-based record of which feature set helps which model at which horizon — the basis for
the deployment decision.

### 5.5 The co-pollutant question (NO / NO2 / NOx / O3)

A teammate suggested adding the gases. The ablation answered it with data (`docs/MODEL_RESULTS.md`):

| recipe | LR h1/h2/h3 | RF h1/h2/h3 |
|---|---|---|
| diffusion9 (9) | 0.902 / 0.774 / 0.647 | 0.900 / 0.783 / 0.682 |
| with_pollutants (13) | 0.902 / 0.774 / 0.645 | **0.903 / 0.787 / 0.682** |
| base+pollutants (10) | 0.901 / 0.771 / 0.643 | 0.899 / 0.769 / 0.654 |

- **LR:** no benefit. **RF:** a small lift at +1h/+2h (+0.003 / +0.004 R²) — **but only combined with
  diffusion** (gases without diffusion are *worse*). Diffusion features dominate.
- **Decision:** keep the gases and deploy RF `with_pollutants` (banks the small lift); a reasonable,
  data-backed answer to the suggestion rather than a guess.

### 5.6 Metrics, meaning, and artifacts

- **R²** = fraction of variance explained (1 perfect, 0 = no better than the mean, negative = worse). **MAE**
  = average absolute error in µg/m³ (directly interpretable). **RMSE** = penalises large misses (e.g.
  wildfire spikes) more heavily. Computed in `train_baselines.py:173`; all results saved to
  `models/saved_models/baseline_metrics.json`.
- **Artifacts** (`models/saved_models/`): per-family and per-horizon `.pkl` bundles, `active_model.pkl`
  (the deployed bundle), `model_registry.json`. Saved/loaded via `models/model_io.py`.
- **Switching the active model:** `scripts/switch_active_model.py` (one family for all horizons),
  `scripts/set_horizon_models.py` (different family per horizon), `scripts/train_per_horizon.py` (train +
  deploy fresh per-horizon models — the path used for the current deployment).

---

## 6. Phase 4 — Recovery + Alerts

### 6.1 Missing-data recovery (`analytics/recovery/spatial_recovery.py`)

When the target sensor reports nothing for an hour, the pipeline must still produce a value. `recover()`
(`spatial_recovery.py:36`) is a **two-stage estimator**:

1. **Wind-weighted spatial estimate** (only if the gap ≤ 3 hours): the same upwind-weighted average of
   neighbour PM2.5 used by the diffusion features — upwind, nearby neighbours dominate
   (`spatial_recovery.py:6`).
2. **Temporal fallback** (longer gaps, or if no neighbours are usable): the most recent known-good value
   (persistence) (`spatial_recovery.py:29`).

If neither yields a value, the consumer skips that hour. **Why the 3-hour switch:** neighbours are a good
spatial proxy only while the gap is short; once it is long, local conditions have likely diverged and
persistence is safer. Recovery is wired into the live feature consumer (`features.py:108`), so downstream
phases never see a hole.

**Robustness is measured** by `analytics/recovery/degradation_eval.py`, which injects 5% / 10% / 20%
missing data and reports recovery MAE/RMSE — evidence the system degrades gracefully (`make eval`).

### 6.2 Alerts — EPA thresholds, and predictive not reactive

Thresholds are the **EPA PM2.5 AQI breakpoints** (`infrastructure/deployment/controller.py:21`):

- **Warning ≥ 35.5 µg/m³** (start of "Unhealthy for Sensitive Groups")
- **Critical ≥ 125.5 µg/m³** (start of "Very Unhealthy")

> Note: the original proposal listed 150/250; those were **wrong** and are not used. The code's EPA
> breakpoints (35.5 / 125.5) are authoritative.

`evaluate_alerts` (`controller.py:74`) sets the level to `critical` / `warning` / `normal` and — crucially —
fires when **either the current or the forecast** value crosses a threshold. **The alerting is predictive:**
a *forecasted* breach raises an alert before the sensor itself crosses the line. Each alert carries a plain
recommendation (e.g. "Increase monitoring and prepare emission controls before the forecast peak")
(`controller.py:190`). The live alert consumer is `consumers/alerts.py`.

---

## 7. Phase 5 — Deployment

### 7.1 FastAPI service (`infrastructure/deployment/app.py`)

| Endpoint | Method | Returns |
|---|---|---|
| `/health` | GET | liveness `{"status":"ok"}` |
| `/status` | GET | active model, per-horizon map, feature columns, metrics availability |
| `/metrics` | GET | saved benchmark metrics |
| `/configure-horizons` | POST | hot-swap the per-horizon model map |
| `/predict` | POST | per-horizon forecasts for a feature payload |
| `/alerts` | POST | alert level + recommendation for a payload |
| `/forecast-and-alert` | POST | predict **and** alert in one call |
| `/live/predictions` | GET | latest prediction per station (from `live_state.json`) |
| `/live/alerts` | GET | latest alert per station |

### 7.2 The ForecastController (`infrastructure/deployment/controller.py`)

`load_controller()` (`controller.py:123`) loads the active model: it reads `active_model.pkl` (the deployed
per-horizon bundle), or composes one per-horizon from the registry, with sensible fallbacks. This is the
single place that decides "which model is serving."

### 7.3 Live serving (`consumers/live_state.py` + `infrastructure/kafka/live_store.py`)

The `live_state` consumer reads both `aq.predictions` and `aq.alerts` and writes the **latest record per
station** to `data/stream/live_state.json` atomically (`live_store.py:23`). The API's `/live/*` endpoints
just read this file — so the dashboard and API **never touch Kafka directly**, decoupling serving from the
stream.

### 7.4 Streamlit dashboard (`infrastructure/deployment/dashboard/streamlit_app.py`)

Five tabs: **Forecast** (enter features, see per-horizon predictions), **Alert simulator** (sliders, colour-
coded level, reactive-vs-predictive comparison), **Benchmark metrics** (the model table), **API probe**, and
**Live** (latest per-station predictions/alerts).

### 7.5 Audit sink (`consumers/sink.py`)

A second consumer of `aq.measurements` writes every measurement to **date-partitioned Parquet** under
`data/stream/measurements/`, deduplicated — a durable system-of-record for replay, retraining, and
backtesting, separate from the live serving path.

---

## 8. The Data Pipeline (training data)

### 8.1 The canonical dataset

`data/external/multistation/train.csv` — **25,559 hourly rows, 2023-07-16 → 2026-06-15** (~3 years), built
by `infrastructure/kafka/scripts/backfill_multistation.py`. Columns: `timestamp`, meteorology, `wind_u/
wind_v`, `pm25`, the three diffusion features, and the four gases (14 columns total).

### 8.2 Stations (`infrastructure/kafka/station_registry.py:7`)

| Station | OpenAQ ID | Role |
|---|---|---|
| Toronto Downtown | 7570 | **target** |
| Toronto West | 1274950 | neighbour |
| Toronto North | 1274949 | neighbour |
| Toronto East | 1210341 | **dropped** (only ~5 days of data) |

**Why multi-station:** three stations are what make the diffusion features and neighbour recovery possible —
a single station cannot express upwind transport. East was dropped because its sensor died days after the
usable window began.

### 8.3 Training sources

- **OpenAQ S3 archive** for PM2.5 (all stations) and gases (target only): `fetch_openaq_location_ml`
  downloads daily files, pivots to hourly wide format, interpolates and fills (`openaq.py:16`).
- **Open-Meteo archive** for historical meteorology at the target's coordinates
  (`backfill_multistation.py:122`) — used because SWOB realtime only covers recent data.

### 8.4 How missing/joined data is handled

- **Everything floored to the hour** so PM, met, and gas keys align (`backfill_multistation.py:41`).
- **Nearest-station met join** by Haversine distance (`met_join.py:4`).
- **Absent values default to 0.0** (met, wind, gases); diffusion features default to 0 when no neighbour has
  PM2.5 that hour.
- **Rows are anchored to the target's PM2.5 hours** — only hours where the target reports PM2.5 produce a
  training row.

### 8.5 Live vs training data (same quantities, different sources)

| | Training (batch) | Live (stream) |
|---|---|---|
| PM2.5 | OpenAQ **S3 archive** | OpenAQ **v3 live API** |
| Meteorology | **Open-Meteo archive** | **Env Canada SWOB** (live) |
| Shape | wide hourly CSV, features pre-computed | raw Avro → normalized → features computed on the hourly tick |

---

## 9. Design Decisions & Rationale (the "why X over Y" slide)

| Decision | Chosen | Over | Why |
|---|---|---|---|
| Stream processor | Pure-Python Kafka consumers | Apache Flink | Hourly, low-volume, single-target; reuse ML code directly; no JVM/serialization overhead |
| Messaging format | Avro + Schema Registry | JSON | Typed contract, compact, controlled evolution |
| Distance metric | Haversine (great-circle) | Euclidean on lat/lon | Earth is curved; lon/lat are angles — flat distance distorts wind alignment |
| Deployed model | Per-horizon Random Forest (LightGBM) | LSTM, STGNN, LR | Best accuracy; non-linear interactions; LSTM dominated; STGNN broken |
| Feature set | `with_pollutants` (13) | diffusion9 / base | Ablation showed a small but real RF lift from gases (with diffusion) |
| Dataset | Multi-station (3) | Single-station | Enables diffusion features + neighbour recovery |
| Train/test split | 70/15/15 chronological | Random shuffle | Prevents temporal leakage; mirrors production |
| Per-horizon models | Independent model+features per horizon | One model for all | Each lead time is a different problem |
| Alert thresholds | EPA 35.5 / 125.5 | Proposal's 150/250 | Proposal values were wrong; EPA breakpoints are standard |
| Alerting | Predictive (forecast-triggered) | Reactive (current-only) | Warns *before* the sensor crosses the line |
| Recovery | Wind-weighted spatial → temporal fallback | Simple interpolation | Adds advection physics; switches to persistence on long gaps |
| Resilience | Per-message dead-letter + manual commit | Crash on bad message | At-least-once; one poison message never blocks the pipeline |

---

## 10. How to Run

### 10.1 Environments (important)

There are **two** Python environments, and which one you use matters:

- **Training / model-serving tier → Python 3.11** (conda env via `make setup`, or the `Dockerfile.train`
  container). Needed for the full ML stack.
- **Ingestion + tests → the `.venv` (Python 3.13)** via `make venv`. Fine for producers and the model-free
  consumers (normalizer, features, live_state, sink).

> **Known issue:** loading the full ML model (TensorFlow + PyTorch + LightGBM together) **segfaults on
> Python 3.13**. Therefore the consumers/services that load the model — **inference, alerts, and the API** —
> must run on **Python 3.11** (and need `lightgbm` installed there). This was confirmed during end-to-end
> testing. The model-free consumers run fine on 3.13.

### 10.2 Required environment variables (`.env`)

- `OPENAQ_API_KEY` — required for live OpenAQ (without it, ingestion runs Env Canada only).
- `KAFKA_BOOTSTRAP=localhost:9092`, `SCHEMA_REGISTRY_URL=http://localhost:8081` (defaults match the Docker
  stack). See `.env.example`.

### 10.3 End-to-end, step by step

```bash
make bootstrap                       # docker compose up + create topics + register schemas
export OPENAQ_API_KEY=...            # or put it in .env

# consumers — model-free ones can run on .venv (py3.13):
make normalizer
FEATURE_TICK_SECONDS=15 make features   # 15s tick for a live demo (default is hourly/3600)
make live-state

# model-loading consumers + API — run on Python 3.11:
python3.11 -m infrastructure.kafka.consumers.inference
python3.11 -m infrastructure.kafka.consumers.alerts
python3.11 -m uvicorn infrastructure.deployment.app:app --port 8000

make producer                        # start live ingestion
make dashboard                       # Streamlit UI
```

Then check the flow: each topic's message count should climb
`aq.measurements → aq.features → aq.predictions → aq.alerts`, `aq.deadletter` should stay **0**, and
`GET /live/predictions` should return per-station forecasts. *(This exact run was verified: full chain
flowed with zero dead-letters; a sample prediction served `per_horizon:rf` with h1/h2/h3.)*

### 10.4 Complete Makefile reference (`Makefile`)

| Target | What it does |
|---|---|
| `help` | List all targets |
| `setup` | Create the py3.11 conda env (canonical, for training) |
| `venv` | Create the py3.13 `.venv` (ingestion + tests) |
| `test` | Unit tests (excludes integration) |
| `test-int` | Integration tests (needs Kafka up) |
| `lint` | `ruff` + `black --check` |
| `up` / `down` / `ps` | Start / stop / list the Docker stack |
| `topics` | Create the 8 Kafka topics |
| `schemas` | Register Avro schemas |
| `bootstrap` | `up` + `topics` + `schemas` |
| `producer` | Run live ingestion (needs `OPENAQ_API_KEY`) |
| `normalizer` | raw → `aq.measurements` |
| `features` | `aq.measurements` → `aq.features` (`FEATURE_TICK_SECONDS` overridable) |
| `inference` | `aq.features` → `aq.predictions` |
| `alerts` | `aq.predictions` → `aq.alerts` |
| `live-state` | predictions/alerts → `live_state.json` |
| `sink` | `aq.measurements` → Parquet audit |
| `api` | Run the FastAPI service |
| `dashboard` | Run the Streamlit dashboard |
| `backfill` | Build the multi-station training CSV (`START`/`END` overridable) |
| `train` | Train all families + save bundles (use py3.11) |
| `eval` | Recovery degradation eval (5/10/20% missing) |
| `clean` | Remove caches |

---

## 11. How to Test

- **Quick unit tests:** `make test` (excludes integration). ~99 tests across `tests/`.
- **Full suite (with the ML models) → Python 3.11 container** (avoids the py3.13 segfault):
  ```bash
  docker build -t iiot-train -f Dockerfile.train .
  docker run --rm -v "$PWD":/app iiot-train python -m pytest -q     # 98 passed, 1 skipped
  ```
- **Integration tests:** `make test-int` (requires the Kafka stack running).
- **CI:** `.github/workflows/ci.yml` installs CPU PyTorch + `requirements.txt`, then runs `pytest` on every
  push and pull request.

---

## 12. Training / Retraining

```bash
make backfill START=2023-07-16 END=2026-06-15   # rebuild data/external/multistation/train.csv
make train                                       # train all families (py3.11) -> bundles + metrics
# deploy the chosen per-horizon model:
python scripts/train_per_horizon.py --model rf --map "1=with_pollutants,2=with_pollutants,3=with_pollutants"
make eval                                         # recovery robustness
```

The inference consumer and API then load the new `active_model.pkl` automatically.

---

## 13. Known Limitations & Honest Caveats (good to pre-empt in Q&A)

- **STGNN is broken at +2h/+3h** — a real, diagnosed finding (replicated-node graph + split-index
  misalignment), not deployed. Fixing it needs neighbour PM2.5 columns in the training frame.
- **Python 3.13 segfault** loading the full ML stack — the model-serving tier must run on 3.11.
- **Co-pollutant lift is marginal** (RF only, +0.003–0.004 R²) — kept because it helps slightly and is now
  data-backed, not because it is a big win.
- **East station dropped** — only 3 of the 4 candidate stations have usable data.
- **Model selection:** the controller loads the served bundle directly from `active_model.pkl` (the
  per-horizon RF). `model_registry.json` is an informational label kept in sync with it.

---

## 14. Quick File / Code Map

| Area | Key files |
|---|---|
| Ingestion sources | `infrastructure/kafka/data_sources/{openaq,environment_canada,iqair}.py` |
| Producers | `infrastructure/kafka/producers/{run_ingestion,base}.py` |
| Topics / schemas | `infrastructure/kafka/{create_topics,register_schemas,config}.py`, `schemas/*.avsc` |
| Consumers | `infrastructure/kafka/consumers/{normalizer,features,inference,alerts,live_state,sink}.py` |
| Feature engineering | `analytics/features/{geo,diffusion_features,feature_engineering,adjacency_matrix}.py` |
| Recovery | `analytics/recovery/{spatial_recovery,degradation_eval}.py` |
| Models | `models/baselines/train_baselines.py`, `models/{predictors,per_horizon,forecast_bundle,feature_recipes,model_io,model_registry}.py` |
| Training scripts | `scripts/{run_baselines,train_per_horizon,run_ablation,save_deployment_models,switch_active_model,set_horizon_models}.py` |
| Deployment | `infrastructure/deployment/{app,controller}.py`, `dashboard/streamlit_app.py`, `infrastructure/kafka/live_store.py` |
| Data | `infrastructure/kafka/scripts/backfill_multistation.py`, `infrastructure/kafka/station_registry.py`, `data/external/multistation/train.csv` |
| Ops | `Makefile`, `docker-compose.yml`, `Dockerfile.train`, `environment.yml`, `requirements.txt`, `.env.example` |
| Docs | `docs/{PROJECT_GUIDE,ARCHITECTURE,MODEL_RESULTS,PIPELINE_OVERVIEW,DATA_GUIDE}.md` |

---

*This guide consolidates the architecture, model results, and end-to-end run procedure. For the deepest
model detail see `docs/MODEL_RESULTS.md`; for the layered architecture rationale see `docs/ARCHITECTURE.md`.*
