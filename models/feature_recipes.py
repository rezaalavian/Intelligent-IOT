_BASE = ["temp definition °c", "dew point definition °c", "rel hum definition %",
         "wind_u", "wind_v", "pm25"]
_GASES = ["no", "no2", "nox", "o3"]
_TIME = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
_LAGS = ["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag6", "pm25_lag12", "pm25_roll3_mean", "pm25_roll6_mean"]

RECIPES = {
    "base6": list(_BASE),
    "diffusion9": _BASE + ["upwind_pm25", "transport_potential", "wind_alignment"],
    "base+upwind": _BASE + ["upwind_pm25"],
    "base+transport": _BASE + ["transport_potential"],
    "base+alignment": _BASE + ["wind_alignment"],
    "base+pollutants": _BASE + _GASES,
    "with_pollutants": _BASE + ["upwind_pm25", "transport_potential", "wind_alignment"] + _GASES,
    "with_lags_pollutants": _BASE + ["upwind_pm25", "transport_potential", "wind_alignment"] + _GASES + _TIME + _LAGS,
}

LOG_SKEWED_COLS = ["pm25", "upwind_pm25", "transport_potential", "no", "no2", "nox"]

def get_features_for_model_and_horizon(model_name: str, horizon: int) -> list[str]:
    model_key = model_name.lower()
    base = list(_BASE)
    gases = list(_GASES)
    time_feats = list(_TIME)
    diffusion = ["upwind_pm25", "transport_potential", "wind_alignment"]
    
    if model_key in ("ha", "historical_average"):
        return ["pm25"]
        
    if model_key in ("lr", "linear_regression"):
        if horizon == 1:
            return base + gases + time_feats + ["pm25_lag1", "pm25_lag2", "pm25_roll3_mean"]
        elif horizon == 2:
            return base + gases + time_feats + diffusion + ["pm25_lag2", "pm25_lag3", "pm25_roll3_mean"]
        else:
            return base + gases + time_feats + diffusion + ["pm25_lag3", "pm25_lag6", "pm25_roll6_mean"]
            
    if model_key in ("rf", "gradient_boosting_rf", "tree_regressor", "gradient boosting/rf"):
        return base + gases + time_feats + diffusion + _LAGS
        
    if model_key in ("lstm", "stgnn", "lstm sequential", "spatiotemporal graph", "graph_stgnn"):
        return base + gases + time_feats + diffusion + _LAGS

    return base + gases + time_feats + diffusion + _LAGS


