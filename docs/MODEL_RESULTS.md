# Model Results — Multi-Station Retrain (3-year data)

Results of the full retrain of all model families on the multi-station dataset,
run in the isolated Python 3.11 container (`Dockerfile.train`). These are the
numbers to use for the report.

## Dataset
- **`data/external/multistation/train.csv`** — 25,535 hourly rows, **2023-07-16 → 2026-06-14**.
- **3 stations:** Toronto Downtown `7570` (target) + West `1274950` + North `1274949`
  (East `1210341` dropped — only ~5 days of OpenAQ data). Sources: OpenAQ archive PM2.5,
  Open-Meteo archive meteorology, wind-aware diffusion features.
- **Features (9):** `temp definition °c`, `dew point definition °c`, `rel hum definition %`,
  `wind_u`, `wind_v`, `pm25`, `upwind_pm25`, `transport_potential`, `wind_alignment`.
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
**Active model = `composite:h1=lr, h2=rf, h3=rf`** — the best model per horizon (uses the
per-horizon model selection). The retrain initially defaulted the active model to STGNN;
that was corrected because STGNN is broken at the longer horizons.

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
# set the deployed active model (best per horizon):
docker run --rm -v "$PWD":/app -w /app iiot-train python -c \
  "from models.model_registry import set_active_horizons; set_active_horizons({1:'lr',2:'rf',3:'rf'})"
```
The trained pkls match the deploy `scikit-learn==1.5.1` pin (no version skew).

## Caveats / next
- STGNN fix is deferred (needs real per-station node features → neighbour PM2.5 columns in the backfill).
- **Co-pollutants** (NO/NO2/NOx/O3) are present in the OpenAQ archive for the target station
  but not yet in `train.csv` (only PM2.5 was kept) — adding them as target features is a
  planned enhancement that may improve accuracy and matches the original RawData.csv feature set.
