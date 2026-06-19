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
| **+1 h** | **0.902** / 1.33 / 2.16 | 0.901 / 1.36 / 2.18 | 0.801 / 2.07 / 3.09 | 0.792 / 2.12 / 3.15 | ~0 / 4.90 / 6.91 |
| **+2 h** | 0.776 / 2.08 / 3.28 | **0.784** / 2.05 / 3.21 | 0.698 / 2.57 / 3.81 | −2.835 / 9.00 / 13.56 | ~0 / 4.91 / 6.92 |
| **+3 h** | 0.649 / 2.65 / 4.10 | **0.678** / 2.53 / 3.93 | 0.605 / 2.94 / 4.35 | −2.762 / 9.04 / 13.43 | ~0 / 4.91 / 6.92 |

## Findings
- **Linear Regression ≈ Random Forest, both best.** RF edges LR slightly at +2 h / +3 h;
  LR is near-identical and simpler/more interpretable. Both far exceed the Historical
  Average baseline.
- **LSTM** is solid but consistently behind the tabular models (0.80 / 0.70 / 0.61).
- **The wind-aware diffusion features carry real signal** for the tabular models — the
  ablation (`scripts/run_ablation.py`) shows `upwind_pm25` giving a small, consistent lift,
  growing with horizon.
- **STGNN collapses at +2 h / +3 h** (R² ≈ −2.8, worse than guessing) — see below.

## Deployed model
**Active model = per-horizon Random Forest on `with_pollutants` (13 features)** — test R²
**0.903 / 0.787 / 0.682** (h1/h2/h3). This supersedes the earlier `composite:h1=lr,h2=rf,h3=rf`
(9-feature) deployment after the co-pollutant ablation (below) showed RF gains a small,
consistent lift from the gases. STGNN is excluded because it is broken at the longer horizons.

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

## STGNN: known-broken at h2/h3 (a finding, not deployed)
- h1 is fine (R² 0.79); **h2/h3 collapse to R² ≈ −2.8** with very large prediction variance.
- **It is structural, not under-training** — a sweep showed h3 R² stays negative at 5 / 25 / 80
  epochs (it does not converge). Suspected causes (under investigation): the STGNN graph uses a
  **fallback with replicated identical node features** (real station coordinates + edges, but
  every node is the same target window — neighbour PM2.5 columns aren't yet in the training
  frame), and a likely **train/val/test split-index misalignment** between the per-horizon
  frame length and the shorter graph-sequence list, which worsens with horizon.
- **Interpretation for the report:** the diffusion **features** (in LR/RF) work; the STGNN
  **graph**, in its current single-station-replicated form, does not help and hurts at longer
  horizons. A real per-station graph needs the backfill to emit neighbour PM2.5 columns
  (deferred).

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
- STGNN fix is deferred (needs real per-station node features → neighbour PM2.5 columns in the backfill).
- **Co-pollutants** (NO/NO2/NOx/O3) are now in `train.csv` and deployed (see the ablation
  above); the lift is small. Lag/rolling versions of the gases remain a possible enhancement.
