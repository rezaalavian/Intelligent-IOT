"""Save and register all trained model families as .pkl artifacts."""

import json
import shutil
from pathlib import Path
from typing import Any
from models.forecast_bundle import ForecastBundle, is_forecast_bundle
from models.model_io import load_model, save_model

SAVED_MODELS_DIR = Path("models/saved_models")
ACTIVE_MODEL_KEY = "stgnn"
REGISTRY_PATH = SAVED_MODELS_DIR / "model_registry.json"
MODEL_KEYS = ("ha", "lr", "rf", "lstm", "stgnn")
DEFAULT_HORIZONS = (1, 2, 3)


def default_active_horizons(model_key: str = ACTIVE_MODEL_KEY) -> dict[str, str]:
    return {str(h): model_key for h in DEFAULT_HORIZONS}


def save_model_family(
    model_key: str,
    model_type: str,
    *,
    feature_columns: list[str],
    target_column: str,
    horizon_models: dict[int, Any],
    horizon_scalers: dict[int, Any],
    metrics: dict[str, Any] | None = None,
    lookback: int = 24,
    save_models: bool = True,
) -> Path | None:
    """Persist per-horizon and combined bundles for one model family."""
    if not save_models or not horizon_models:
        return None

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for horizon, predictor in horizon_models.items():
        bundle = ForecastBundle(
            feature_columns=feature_columns,
            target_column=target_column,
            horizons=[horizon],
            model_type=model_type,
            scaler=horizon_scalers.get(horizon),
            models={horizon: predictor},
            scalers=horizon_scalers,
            metrics=metrics,
            lookback=lookback,
        )
        save_model(bundle, SAVED_MODELS_DIR / f"{model_key}_h{horizon}.pkl")

    # Merge other horizons already present on disk to support parallel training execution
    merged_models = dict(horizon_models)
    merged_scalers = dict(horizon_scalers)
    for h in [1, 2, 3]:
        if h not in merged_models:
            try:
                loaded = load_horizon_bundle(model_key, h)
                if loaded and loaded.models and h in loaded.models:
                    merged_models[h] = loaded.models[h]
                    if loaded.scalers and h in loaded.scalers:
                        merged_scalers[h] = loaded.scalers[h]
                    elif loaded.scaler is not None:
                        merged_scalers[h] = loaded.scaler
            except Exception:
                pass

    combined = ForecastBundle(
        feature_columns=feature_columns,
        target_column=target_column,
        horizons=sorted(merged_models.keys()),
        model_type=model_type,
        scaler=merged_scalers[min(merged_scalers.keys())],
        models=merged_models,
        scalers=merged_scalers,
        metrics=metrics,
        lookback=lookback,
    )
    bundle_path = SAVED_MODELS_DIR / f"{model_key}_bundle.pkl"
    save_model(combined, bundle_path)
    return bundle_path


def load_horizon_bundle(model_key: str, horizon: int) -> ForecastBundle | None:
    path = SAVED_MODELS_DIR / f"{model_key}_h{horizon}.pkl"
    if not path.exists():
        return None
    loaded = load_model(path)
    return loaded if is_forecast_bundle(loaded) else None


def load_composite_bundle(horizon_models: dict[int, str]) -> ForecastBundle | None:
    """Build one ForecastBundle using a different model family per horizon."""
    merged_models: dict[int, Any] = {}
    merged_scalers: dict[int, Any] = {}
    feature_columns: list[str] = []
    target_column = "pm2"
    lookback = 24
    model_labels: dict[int, str] = {}

    for horizon in sorted(horizon_models.keys()):
        model_key = horizon_models[horizon]
        bundle = load_horizon_bundle(model_key, int(horizon))
        if bundle is None:
            raise FileNotFoundError(
                f"Missing artifact: {model_key}_h{horizon}.pkl — train with run_baselines.py first."
            )
        h = int(horizon)
        merged_models[h] = bundle.models[h]
        if bundle.scalers and h in bundle.scalers:
            merged_scalers[h] = bundle.scalers[h]
        elif bundle.scaler is not None:
            merged_scalers[h] = bundle.scaler
        if not feature_columns:
            feature_columns = list(bundle.feature_columns)
            target_column = bundle.target_column
            lookback = bundle.lookback
        model_labels[h] = model_key

    if not merged_models:
        return None

    label = "composite:" + ",".join(f"h{h}={model_labels[h]}" for h in sorted(model_labels))
    return ForecastBundle(
        feature_columns=feature_columns,
        target_column=target_column,
        horizons=sorted(merged_models.keys()),
        model_type=label,
        scaler=merged_scalers[min(merged_scalers.keys())],
        models=merged_models,
        scalers=merged_scalers,
        lookback=lookback,
    )


def parse_horizon_map(raw: dict[str, Any] | None, fallback_key: str = ACTIVE_MODEL_KEY) -> dict[int, str]:
    if not raw:
        return {h: fallback_key for h in DEFAULT_HORIZONS}
    parsed: dict[int, str] = {}
    for key, value in raw.items():
        parsed[int(key)] = str(value)
    return parsed


def write_registry(
    families: dict[str, dict[str, Any]],
    *,
    active_model: str = ACTIVE_MODEL_KEY,
    active_horizons: dict[str, str] | None = None,
) -> Path:
    """Write model_registry.json describing all saved families."""
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    horizons = active_horizons or default_active_horizons(active_model)
    registry = {
        "active_model": active_model,
        "active_horizons": horizons,
        "active_path": str(SAVED_MODELS_DIR / "active_model.pkl"),
        "families": families,
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return REGISTRY_PATH


def set_active_horizons(horizon_models: dict[int, str]) -> Path:
    """Save a composite active_model.pkl and update registry active_horizons."""
    composite = load_composite_bundle(horizon_models)
    if composite is None:
        raise ValueError("Could not build composite bundle — no horizons resolved.")
    target = SAVED_MODELS_DIR / "active_model.pkl"
    save_model(composite, target)

    registry: dict[str, Any] = {}
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["active_horizons"] = {str(h): horizon_models[h] for h in sorted(horizon_models)}
    registry["active_path"] = str(target)
    unique = sorted(set(horizon_models.values()))
    registry["active_model"] = unique[0] if len(unique) == 1 else "composite"
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return target


def set_active_model(model_key: str) -> Path | None:
    """Set the same model family for all horizons."""
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unknown model key: {model_key}")
    horizon_map = {h: model_key for h in DEFAULT_HORIZONS}
    try:
        return set_active_horizons(horizon_map)
    except FileNotFoundError:
        source = SAVED_MODELS_DIR / f"{model_key}_bundle.pkl"
        target = SAVED_MODELS_DIR / "active_model.pkl"
        if not source.exists():
            return None
        shutil.copy2(source, target)
        if REGISTRY_PATH.exists():
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            registry["active_model"] = model_key
            registry["active_horizons"] = default_active_horizons(model_key)
            registry["active_path"] = str(target)
            REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        return target
