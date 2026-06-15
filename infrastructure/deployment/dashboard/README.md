# Dashboard

Streamlit UI for Phase 5: per-horizon model selection, forecasts, alert simulation, and API probing.

## Run (Windows — recommended)

```powershell
conda activate Intelligent-IOT-blackwell
cd Intelligent-IOT
powershell -File scripts/run_dashboard.ps1
```

Or:

```powershell
python -m streamlit run infrastructure/deployment/dashboard/streamlit_app.py --server.port 8501
```

Open **http://localhost:8501**

> Do **not** use `conda run streamlit ...` on Windows — it often exits immediately.
> Do **not** wrap the app in `if __name__ == "__main__"` — Streamlit must execute UI code at import time.

## Per-horizon model selection

Use the sidebar to pick a different saved model for +1h, +2h, and +3h (e.g. STGNN / LR / RF mix).

To persist the selection for the API:

```powershell
python scripts/set_horizon_models.py --h1 stgnn --h2 lr --h3 rf
```

## API (separate terminal)

```powershell
uvicorn infrastructure.deployment.app:app --reload --port 8000
```
