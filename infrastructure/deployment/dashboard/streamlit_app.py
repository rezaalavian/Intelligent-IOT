"""Streamlit dashboard for Intelligent-IOT Phase 5 deployment."""

import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Repo root: .../Intelligent-IOT
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRICS_PATH = ROOT / "models" / "saved_models" / "baseline_metrics.json"
DEFAULT_API = "http://127.0.0.1:8000"
MODEL_CHOICES = ["stgnn", "lr", "rf", "lstm", "ha"]


@st.cache_resource
def _get_controller(horizon_map: tuple[tuple[int, str], ...]):
    from infrastructure.deployment.controller import load_controller

    mapping = {int(h): str(m) for h, m in horizon_map}
    return load_controller(horizon_models=mapping)


def _load_metrics_file() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def _default_features(controller) -> dict[str, float]:
    cols = controller.bundle.feature_columns if controller.bundle else [
        "temp definition °c",
        "dew point definition °c",
        "rel hum definition %",
        "wind_u",
        "wind_v",
        "pm2",
    ]
    defaults = {
        "temp definition °c": 5.0,
        "dew point definition °c": 0.0,
        "rel hum definition %": 70.0,
        "wind_u": 2.0,
        "wind_v": 1.0,
        "pm25": 35.0,
        "pm2": 35.0,
    }
    return {col: float(defaults.get(col, 0.0)) for col in cols}


def _metrics_table(metrics: dict) -> pd.DataFrame | None:
    results = metrics.get("results", {})
    if not results:
        return None
    rows = []
    for horizon_key, models in results.items():
        for model_name, splits in models.items():
            test = splits.get("test", {})
            rows.append(
                {
                    "Horizon": horizon_key,
                    "Model": model_name,
                    "Test R²": test.get("r2"),
                    "Test MAE": test.get("mae"),
                    "Test RMSE": test.get("rmse"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Intelligent-IOT Dashboard", layout="wide")
    st.title("Intelligent-IOT — Air Quality Forecast & Alerts")

    st.sidebar.markdown("### Per-horizon model selection")
    st.sidebar.caption("Pick a different saved model for each forecast window.")
    h1_model = st.sidebar.selectbox("+1 hour model", MODEL_CHOICES, index=0, key="sel_h1")
    h2_model = st.sidebar.selectbox("+2 hour model", MODEL_CHOICES, index=0, key="sel_h2")
    h3_model = st.sidebar.selectbox("+3 hour model", MODEL_CHOICES, index=0, key="sel_h3")

    horizon_tuple = ((1, h1_model), (2, h2_model), (3, h3_model))
    try:
        controller = _get_controller(horizon_tuple)
    except Exception as exc:
        st.error(f"Failed to load models: {exc}")
        st.info("Train first: `python scripts/run_baselines.py --model all --path data/raw/Raw_Data.csv`")
        return

    api_base = st.sidebar.text_input("API base URL", DEFAULT_API)

    st.sidebar.markdown("### System status")
    status = controller.status()
    st.sidebar.write("Model loaded:", "Yes" if status["model_loaded"] else "No")
    if status.get("active_horizons"):
        st.sidebar.json(status["active_horizons"])
    if status.get("lookback"):
        st.sidebar.write("Lookback (h):", status["lookback"])

    tab_forecast, tab_alerts, tab_metrics, tab_api = st.tabs(
        ["Forecast", "Alert simulator", "Benchmark metrics", "API probe"]
    )

    features = _default_features(controller)
    feature_cols = list(features.keys())

    with tab_forecast:
        st.subheader("Multi-horizon forecast")
        if not controller.is_ready():
            st.error("No model loaded. Run training first.")
        else:
            cols = st.columns(2)
            inputs: dict[str, float] = {}
            for idx, col_name in enumerate(feature_cols):
                with cols[idx % 2]:
                    inputs[col_name] = st.number_input(
                        col_name,
                        value=float(features[col_name]),
                        key=f"feat_{col_name}",
                    )

            if st.button("Run forecast", type="primary"):
                lookback = status.get("lookback") or 12
                history = [dict(inputs) for _ in range(int(lookback))]
                result = controller.predict({"features": inputs, "history": history})
                st.success("Forecast complete")
                fc = result.get("forecasts", {})
                c1, c2, c3 = st.columns(3)
                for col, (label, val) in zip([c1, c2, c3], sorted(fc.items())):
                    col.metric(label.upper(), f"{val:.2f} µg/m³")
                st.json(result)

    with tab_alerts:
        st.subheader("Proactive response simulator")
        from infrastructure.deployment.controller import THRESHOLD_WARNING, THRESHOLD_CRITICAL

        alert_inputs: dict[str, float] = {}
        if controller.is_ready():
            acols = st.columns(2)
            for idx, col_name in enumerate(feature_cols):
                with acols[idx % 2]:
                    alert_inputs[col_name] = st.number_input(
                        f"Alert · {col_name}",
                        value=float(features[col_name]),
                        key=f"alert_{col_name}",
                    )

        current = st.slider("Current PM2.5 (µg/m³)", 0.0, 400.0, 80.0, 1.0)
        use_model = st.checkbox("Use model forecast from inputs above", value=controller.is_ready())
        manual_forecast = st.slider("Manual forecast PM2.5", 0.0, 400.0, 120.0, 1.0)

        payload: dict = {"pm25": current, "pm2": current}
        if use_model and controller.is_ready():
            lookback = status.get("lookback") or 12
            hist = [dict(alert_inputs or _default_features(controller)) for _ in range(int(lookback))]
            payload["features"] = alert_inputs or _default_features(controller)
            payload["history"] = hist
        else:
            payload["forecast_pm25"] = manual_forecast

        alert = controller.evaluate_alerts(payload)
        level = alert["level"]
        color = {"normal": "green", "warning": "orange", "critical": "red"}.get(level, "gray")
        st.markdown(f"**Alert level:** :{color}[{level.upper()}]")
        st.write(f"Current: **{alert['current_pm25']:.1f}** | Forecast: **{alert['forecast_pm25']:.1f}**")
        st.info(alert["recommendation"])
        st.caption(f"Warning ≥ {THRESHOLD_WARNING} µg/m³ · Critical ≥ {THRESHOLD_CRITICAL} µg/m³")

        st.markdown("#### Reactive vs predictive")
        st.dataframe(
            pd.DataFrame(
                {"Policy": ["Reactive", "Predictive"], "Action": [
                    "Wait until threshold exceeded, then react.",
                    alert["recommendation"],
                ]}
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab_metrics:
        st.subheader("Saved benchmark metrics (all trained models)")
        metrics = _load_metrics_file()
        table = _metrics_table(metrics)
        if table is not None and not table.empty:
            st.dataframe(table.sort_values(["Horizon", "Test R²"], ascending=[True, False]), use_container_width=True)
        else:
            st.warning("No baseline_metrics.json found. Run baselines first.")

    with tab_api:
        st.subheader("Live API probe")
        st.caption("Start API: `uvicorn infrastructure.deployment.app:app --reload`")
        try:
            import requests
        except ImportError:
            st.warning("Install requests to use API probe tab.")
            requests = None

        if requests is not None:
            if st.button("GET /health"):
                try:
                    r = requests.get(f"{api_base.rstrip('/')}/health", timeout=5)
                    st.code(r.json())
                except Exception as exc:
                    st.error(str(exc))
            if st.button("GET /status"):
                try:
                    r = requests.get(f"{api_base.rstrip('/')}/status", timeout=5)
                    st.code(r.json())
                except Exception as exc:
                    st.error(str(exc))


# Streamlit executes this file top-to-bottom on every interaction — do not guard with __main__.
main()
