from pathlib import Path

from models.baselines.train_baselines import train_and_eval
from models.model_io import load_model
from models.forecast_bundle import ForecastBundle

_ROOT = Path(__file__).resolve().parents[1]


def build_per_horizon_bundle(path, model, horizon_features):
    models, scalers, feat_by_h = {}, {}, {}
    target = "pm25"
    lookback = 24
    for h, cols in sorted(horizon_features.items()):
        train_and_eval(model, Path(path), horizon=h, features=cols, save_models=True)
        fam = load_model(str(_ROOT / "models" / "saved_models" / f"{model}_bundle.pkl"))
        models[h] = fam.models[h]
        scalers[h] = (fam.scalers or {}).get(h, fam.scaler)
        feat_by_h[h] = list(cols)  # the feature set this horizon was trained on
        target, lookback = fam.target_column, fam.lookback
    horizons = sorted(models)
    return ForecastBundle(
        feature_columns=feat_by_h[horizons[0]],
        target_column=target,
        horizons=horizons,
        model_type=f"per_horizon:{model}",
        scaler=scalers[horizons[0]],
        models=models,
        scalers=scalers,
        feature_columns_by_horizon=feat_by_h,
        lookback=lookback,
    )
