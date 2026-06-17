# Report & Slides Update Guide

How to update `report-draft5.docx` and `Report_ppt-draft1.pptx` to match the
system as actually built. Each item: **current draft says → change to**, with the
report section / slide it affects. Pair this with `docs/ARCHITECTURE.md` for
full detail.

> ⚠️ **Numbers gate:** the Results tables need a re-run on the multi-station data.
> Only Linear Regression has been retrained on it so far (py3.13 dev venv). Run
> the full retrain on the py3.11 env (`make train`) to regenerate MAE/RMSE/R²
> for all families before finalizing §4. See "Results numbers" below.

---

## 1. Data sources & variables  (Report §3.3 · Slide 6)

- **Draft says:** OpenAQ provides PM2.5 **and** temperature, humidity, wind speed,
  wind direction, pressure.
- **Change to:** **OpenAQ provides pollutants only** (PM2.5, NO, NO₂, NOx, O₃).
  **Meteorology comes from Environment Canada SWOB** — temperature, **dew point**,
  relative humidity, wind speed, wind direction, station pressure. Wind is **not**
  from OpenAQ. Each PM2.5 station is paired with its **nearest SWOB station** for met
  (a met-join, because OpenAQ records carry no wind).
- **Also add:** **dew point** is now a captured variable (SWOB `dwpt_temp`).

## 2. Monitoring stations  (Report §3.5, Fig 4 · Slide 6 table)

- **Draft says:** 3 Toronto stations (A/B/C); single conceptual target + 2 neighbors.
- **Change to:** **4 real OpenAQ AirNow PM2.5 stations** — Downtown `7570` (target),
  West `1274950`, North `1274949`, East `1210341`. The wind-aware graph uses their
  **real coordinates**. (Earlier work used only the single Downtown station; the
  spatial story is now backed by real multi-station data.)

## 3. Distance metric  (Report §3.4 / §3.5)

- **Draft says:** geographical distance via the **Haversine** formula.
- **Status:** ✅ **now true.** The code originally used Euclidean distance on raw
  lat/lon; it was corrected to **Haversine great-circle distance** + a local
  north-east bearing for wind alignment. Keep the Haversine claim — it is now accurate.

## 4. Diffusion features in the models  (Report §3.5–3.6, §4.3, §5 · Slide 5/7)

- **Draft says:** "all models received the full wind-aware diffusion feature set."
- **Change to:** this is **now true of the deployed model.** The feature vector is
  **9 features**: 6 base (`temp`, `dew point`, `rel hum`, `wind_u`, `wind_v`, `pm25`)
  + **3 diffusion features**: `upwind_pm25` (edge-weighted neighbour PM2.5),
  `transport_potential` (wind speed × alignment), `wind_alignment` (mean inbound
  alignment). The same wind-aware weighting drives the diffusion features, the STGNN
  graph edges, and the Phase-4 recovery estimator. (Be accurate: the *active* model
  is currently Linear Regression — see Results gate.)

## 5. Streaming architecture & topics  (Report §3.2 · Slide 4)

- **Draft says:** "3 Kafka topics: raw → features → predictions"; consumer does
  preprocessing + feature engineering; ~60-minute polling.
- **Change to:** the real chain is
  `aq.{source}.raw → aq.measurements → aq.features → aq.predictions → aq.alerts`
  plus a dead-letter topic `aq.deadletter` (8 topics total), all **Avro + Confluent
  Schema Registry**. Stages are **pure-Python Kafka consumers** (normalizer → feature
  consumer → inference → alert → live_state), each with dead-letter handling — **not**
  a Flink job (the Flink container exists but is unused). Reconcile the **polling
  interval**: code default is 300 s (5 min), configurable; the feature consumer emits
  on an hourly tick. State 5-min poll / hourly forecast resolution consistently.

## 6. Forecasting models  (Report §3.6, §4)

- **Draft says:** HA, LR, RF, LSTM, STGNN, all on the diffusion feature set.
- **Change to:** all five exist. **STGNN** uses real station coordinates and
  wind-aware edges (GAT+TCN). Note honestly: the **active deployed model is LR**
  (retrained on multi-station data, consumes the 9 features); the full RF/LSTM/STGNN
  retrain on the multi-station set runs on the py3.11 env (`make train`).

## 7. Fault-tolerant recovery  (Report — Phase 4 / new subsection · Slide 9)

- **Draft says (proposal):** kriging spatial interpolation (Stage 1) + learned graph
  reconstruction (Stage 2).
- **Change to:** implemented as **Stage 1 — wind-aware spatial estimate** of a missing
  station's PM2.5 from neighbours (same diffusion weighting), **Stage 2 — temporal
  fallback** from the station's own recent history, with a **gap-threshold router**
  (≤3 h → spatial; longer → temporal), wired into the live feature consumer. A
  **degradation evaluation** (inject 5/10/20 % missing, score MAE/RMSE) exists.
  *Deferred:* a learned graph-reconstruction model (Stage 2 is temporal for now).

## 8. Alert thresholds  (Report §Phase 5 / Discussion · Slide 8/9)

- **Draft/proposal says:** PM2.5 > 250 µg/m³ emergency; > 150 for >10 min warning.
- **Change to:** the system uses **EPA AQI breakpoints — 35.5 µg/m³ (warning),
  125.5 µg/m³ (critical)**. These are the intended thresholds; the proposal's
  150/250 are not used. (The ">10 min duration" rule is not implemented — alerts are
  per-hour.)

## 9. Deployment / API & dashboard  (Report §Phase 5, §4.5 · Slide 8)

- **Change to:** the API now serves the **live stream**, not just request/response:
  `GET /live/predictions` and `GET /live/alerts` return the latest per-station
  forecast/alert, fed by a `live_state` consumer that materializes the stream to a
  JSON store; the Streamlit dashboard has a **Live tab**. (Existing `POST /predict`,
  `/alerts`, `/forecast-and-alert` remain.)

## 10. Historical met for training  (Report §3.3 / §4.1)

- **Add:** the training set is built by a **backfill** — OpenAQ archive PM2.5 for the
  4 stations + **Open-Meteo archive** historical hourly met (per-station coordinates;
  SWOB realtime only covers recent data). Diffusion features are computed per hour
  into the training frame.

---

## Results numbers  (Report §4.2 Table 1, §4.3 · Slide 7)

The draft's R²/MAE/RMSE table is from the **old single-station** data and must be
**regenerated** on the multi-station set. To produce real numbers:

```bash
make backfill START=2024-01-01 END=2025-12-31   # wider window than the 6-week demo
make train                                       # py3.11 env: prints the full metrics matrix, saves bundles
make eval                                        # recovery degradation MAE/RMSE (needs neighbour pm25 cols)
```

`make train` prints a per-horizon, per-model matrix (R²/MAE/MSE/RMSE for train/val/test)
— copy those into Table 1 and the slide-7 chart.

**What we have so far (LR only, ~6-week/3-station demo — replace with the full run):**

| Horizon | Model | R² (test) | MAE | RMSE |
|---|---|---|---|---|
| 1 h | Linear Regression | ~0.61 | ~? | ~? |
| 2 h | Linear Regression | ~0.03 | 4.32 | 5.72 |
| 3 h | Linear Regression | -0.36 | 5.22 | 6.76 |

These are weak because the window was short and only 3 stations had archive data —
a longer backfill will improve them. **Do not publish these**; run the full retrain.

**System-performance / load-test numbers (Report §4.5, Table 2 · Slide 8):** the
draft's latency/throughput figures should be **re-measured** against the current API
(now with the live endpoints) before publishing — `scripts/benchmark_api.py` is the
starting point.

---

## Slide-by-slide quick map

- **Slide 4 (Architecture):** fix topic chain (5+ topics, Avro/Schema Registry, Python consumers); 4 stations.
- **Slide 5 (Diffusion features):** the 3 features are now real + deployed; mention Haversine + wind alignment.
- **Slide 6 (Datasets):** split OpenAQ (pollutants) vs SWOB (met incl. dew point); 4 AirNow stations; Open-Meteo for training met.
- **Slide 7 (Performance):** regenerate from `make train`.
- **Slide 8 (System performance):** re-measure load test; add the live API endpoints.
- **Slide 9 (Challenges):** add the real ones — wind not on OpenAQ (met-join), single-vs-multi station data availability, py3.13 native-lib segfaults, SWOB realtime retention (Open-Meteo for history).
- **Slide 11 (Team contributions):** unchanged.

## Honest caveats to keep in the report
- Active model is LR; full STGNN/RF/LSTM multi-station retrain pending the py3.11 env.
- STGNN graph uses real coordinates but per-station node features are a fallback (target window replicated) until the backfill emits per-station windows.
- Phase 4 Stage 2 is temporal (no learned graph model yet); real-data degradation eval needs neighbour PM2.5 columns surfaced in the backfill.
