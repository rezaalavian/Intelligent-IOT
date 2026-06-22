# Live Demo Runbook

Stream **recorded real data** through the running pipeline so the dashboard shows
forecasts and alerts updating live. Two windows (both real, from `train.csv`):

- **`eval`** — a slice of the **held-out test split** (the model never trained on it),
  early-Feb-2026, including a cluster where PM2.5 crosses the warning line →
  **normal → warning** alerts.
- **`wildfire`** — the **July-2025 extreme-smoke event** (PM2.5 up to ~224) →
  **critical (red)** alerts. This is historical data, *not* the eval set — say so.

The replay publishes precomputed feature records into `aq.features`; the live
inference → alerts → live_state → API → dashboard chain runs unchanged. Each row's
own PM2.5 becomes the "current" reading, and the model's forecast drives the alert —
so a **forecasted** breach can raise an alert *before* the current reading crosses
the line (the predictive-alerting story).

## Prerequisites (once)

- Docker running.
- `.env` present (the demo works without `OPENAQ_API_KEY` — it replays recorded data).
- **Python 3.11** with the project deps + `streamlit` (the model-loading services
  segfault on 3.13). `python3.11` must be on your PATH.

## Run it (two commands)

```bash
scripts/demo_up.sh                 # brings up Kafka + all 5 services in the background
make demo DEMO_MODE=eval           # stream the held-out eval window (normal → warning)
```

Open the **dashboard at http://localhost:8501** (Live tab) — or the API directly at
http://localhost:8000/live/alerts. Watch the alert level change as the stream plays.

For the dramatic red alert:

```bash
make demo DEMO_MODE=wildfire        # July-2025 smoke event → critical
```

Stop everything:

```bash
scripts/demo_down.sh
```

## Knobs

- `DEMO_INTERVAL` — seconds between records (default `1.0`). `make demo DEMO_MODE=eval DEMO_INTERVAL=0.3`
  plays faster; a larger value lets you narrate each step.
- `make demo-reset` — clears the dashboard board (`live_state.json`) for a clean start.
  (The launcher already does this each time.)
- Custom window: `python -m infrastructure.kafka.scripts.demo_replay --mode eval --start 2026-02-03 --end 2026-02-05 --interval 0.5`

## What's happening (for narration)

1. `demo_replay` reads recorded rows and publishes them to **`aq.features`**.
2. The **inference** consumer loads the deployed per-horizon Random Forest and emits
   1h/2h/3h forecasts to `aq.predictions`.
3. The **alerts** consumer applies the **EPA thresholds** (warning ≥ 35.5, critical
   ≥ 125.5 µg/m³) and emits to `aq.alerts`.
4. The **live_state** consumer materializes the latest per-station prediction + alert
   into `live_state.json`.
5. The **API** serves `/live/predictions` and `/live/alerts`; the **dashboard** polls
   them and renders the live board.

## Notes

- The dashboard shows the **latest** reading per station. The wildfire window peaks at
  ~224 then tapers; to freeze on the red critical alert, use a slower `DEMO_INTERVAL`
  or stop the replay (Ctrl-C) at the peak.
- Manual alternative to the launcher (each in its own terminal):
  `make bootstrap` · `make PY=python3.11 inference` · `make PY=python3.11 alerts` ·
  `make live-state` · `make PY=python3.11 api` · `make PY=python3.11 dashboard`.
- Verified end-to-end: eval window → 13 warning alerts; wildfire window → 59 critical
  alerts; `aq.deadletter` empty throughout.
