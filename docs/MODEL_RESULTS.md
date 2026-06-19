# Model Results — Multi-Station Retrain (3-year data)

Results of the full retrain of all model families on the multi-station dataset,
run in the isolated Python 3.11 container (`Dockerfile.train`). These are the
numbers to use for the report.

## Dataset
- **`data/external/multistation/train.csv`** — 25,535 hourly rows, **2023-07-16 → 2026-06-14**.
- **3 stations:** Toronto Downtown `7570` (target) + West `1274950` + North `1274949`
  (East `1210341` dropped — only ~5 days of OpenAQ data). Sources: OpenAQ archive PM2.5,
  Open-Meteo archive meteorology, wind-aware diffusion features.
- **Features:** 9 core — `temp definition °c`, `dew point definition °c`, `rel hum definition %`,
  `wind_u`, `wind_v`, `pm25`, `upwind_pm25`, `transport_potential`, `wind_alignment` — plus the
  4 target co-pollutants `no`, `no2`, `nox`, `o3` (default `FEATURE_COLS` is now 13; see ablation).
- **Target:** PM2.5 (µg/m³), range 0–224 (incl. 2023 wildfire-smoke spikes).
- **Split:** 70 / 15 / 15 **chronological** (train = earliest, test = most recent ~5–6 months).

## Test-split metrics (R² / MAE / RMSE)

| Horizon | Linear Regression | Random Forest | LSTM | STGNN | Historical Avg |
|---|---|---|---|---|---|
| **+1 h** | 0.902 / 1.34 / 2.16 | **0.903 / 1.34 / 2.15** | 0.795 / 2.12 / 3.11 | 0.865 / 1.77 / 2.53 | ~0 / 4.88 / 6.89 |
| **+2 h** | 0.774 / 2.10 / 3.28 | **0.787 / 2.04 / 3.18** | 0.688 / 2.67 / 3.84 | 0.767 / 2.35 / 3.33 | ~0 / 4.88 / 6.89 |
| **+3 h** | 0.645 / 2.68 / 4.10 | **0.682 / 2.51 / 3.88** | 0.587 / 2.98 / 4.42 | 0.615 / 2.97 / 4.28 | ~0 / 4.88 / 6.89 |

*(All five models retrained on the deployed 13-feature `with_pollutants` config; numbers match
`models/saved_models/baseline_metrics.json`. **RF (bold) is the deployed model.** STGNN is the
**fixed** model — see the STGNN section; its original h2/h3 were −2.835 / −2.762.)*

## Findings
- **Linear Regression ≈ Random Forest, both best.** RF edges LR slightly at +2 h / +3 h;
  LR is near-identical and simpler/more interpretable. Both far exceed the Historical
  Average baseline.
- **LSTM** is solid but consistently behind the tabular models (0.80 / 0.69 / 0.59).
- **The wind-aware diffusion features carry real signal** for the tabular models — the
  ablation (`scripts/run_ablation.py`) shows `upwind_pm25` giving a small, consistent lift,
  growing with horizon.
- **STGNN now works** (0.865 / 0.767 / 0.615) after a two-part fix — competitive with the tabular
  models but still just behind RF at every horizon. See the STGNN section below.

## Deployed model
**Active model = per-horizon Random Forest on `with_pollutants` (13 features)** — test R²
**0.903 / 0.787 / 0.682** (h1/h2/h3). This supersedes the earlier `composite:h1=lr,h2=rf,h3=rf`
(9-feature) deployment after the co-pollutant ablation (below) showed RF gains a small,
consistent lift from the gases. STGNN now trains correctly (see below) but still trails RF at
every horizon, so RF remains the deployed model.

### Co-pollutant ablation (NO/NO2/NOx/O3)
The target station's co-pollutants are now kept in `train.csv` and exposed as the
`with_pollutants` (diffusion + gases) and `base+pollutants` (base + gases) recipes.

| recipe | LR h1/h2/h3 | RF h1/h2/h3 |
|---|---|---|
| diffusion9 (9) | 0.902 / 0.774 / 0.647 | 0.900 / 0.783 / 0.682 |
| with_pollutants (13) | 0.902 / 0.774 / 0.645 | **0.903 / 0.787 / 0.682** |
| base+pollutants (10) | 0.901 / 0.771 / 0.643 | 0.899 / 0.769 / 0.654 |

- **LR:** gases give no benefit (≈ identical).
- **RF:** a small lift at h1/h2 (+0.003 / +0.004 R²), but **only combined with diffusion** —
  gases alone (`base+pollutants`) are worse than diffusion alone. Diffusion features dominate.
- Decision: deploy RF `with_pollutants`; the default `FEATURE_COLS` now includes the gases.

## STGNN: was broken, now fixed (still not deployed — RF wins)
The STGNN originally collapsed to **R² ≈ −2.8** at +2h/+3h. The cause was two layers of bugs,
both now fixed; the model trains correctly and is competitive (0.865 / 0.767 / 0.615), just
behind RF at every horizon.

**Root cause (confirmed in code):**
1. **Degenerate graph — replicated nodes.** Every graph node carried the *same* target-station
   window (`scaled_all = {sid: scaled_target ...}`), so the "spatial" graph had no spatial signal.
   Neighbour PM2.5 columns were not in the training frame.
2. **Split-index misalignment.** `train_end/val_end` were row indices on the full frame but applied
   to the shorter graph-sequence list (length `n − lookback − horizon + 1`), shifting the splits and
   worsening with horizon.
3. **Training instability.** Even after (1)+(2), some horizons *diverged* (negative **train** R²) —
   no gradient clipping, no early stopping, fixed 25 epochs at lr 0.002.

**The fix (`models/baselines/train_baselines.py`, `infrastructure/kafka/scripts/backfill_multistation.py`):**
- The backfill now emits per-neighbour `pm25_<id>` columns; the STGNN branch builds **real per-station
  nodes** (each node gets its own PM2.5, shared met/wind/diffusion). Tabular models ignore the extra
  columns via `select_feature_cols`, so they are unaffected.
- The graph-sequence split is aligned to the tabular row split (offset by `lookback − 1`).
- The graph scaler is fit on **train rows only** (was fit on all rows — a leak).
- Training adds **gradient clipping (max-norm 1.0) + early stopping on validation** (patience 10, up to
  80 epochs, lr 0.001), seeded for reproducibility.

**Result (test R²):** h1 0.865 (train 0.941), h2 0.767 (train 0.872), h3 0.615 (train 0.832) — all
converge, all positive. **Still below RF** (0.903 / 0.787 / 0.682), so RF stays deployed — but the
comparison is now fair: the diffusion features carry the signal, and the graph model, once correct, is
competitive rather than broken.

## Reproduce (Python 3.11 container — avoids the py3.13 TF+torch+lightgbm segfault)
```bash
docker build -t iiot-train -f Dockerfile.train .
# full retrain (writes bundles + baseline_metrics.json to models/saved_models/):
docker run --rm -v "$PWD":/app -w /app iiot-train python -W ignore -c \
  "from pathlib import Path; from models.baselines.train_baselines import train_and_eval; \
   train_and_eval('all', Path('data/external/multistation/train.csv'), save_models=True)"
# feature ablation (feature-set x model x horizon):
docker run --rm -v "$PWD":/app -w /app iiot-train python scripts/run_ablation.py --models all --out experiments.csv
# set the deployed active model (per-horizon RF on the co-pollutant recipe):
docker run --rm -v "$PWD":/app -w /app iiot-train python scripts/train_per_horizon.py \
  --model rf --map "1=with_pollutants,2=with_pollutants,3=with_pollutants"
```
The trained pkls match the deploy `scikit-learn==1.5.1` pin (no version skew).

## Caveats / next
- STGNN is fixed and competitive but still behind RF; further tuning (deeper graph, attention on the
  temporal axis, neighbour met) could close the gap — optional future work.
- **Co-pollutants** (NO/NO2/NOx/O3) are now in `train.csv` and deployed (see the ablation
  above); the lift is small. Lag/rolling versions of the gases remain a possible enhancement.
