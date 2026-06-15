#!/usr/bin/env python3
"""Train all model families and write .pkl artifacts + STGNN active model for Phase 5."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.baselines.train_baselines import train_and_eval, _resolve_path


def main() -> int:
    data_path = _resolve_path(Path("data/raw/Raw_Data.csv"))
    print(f"Training all models and saving .pkl artifacts from {data_path}")
    train_and_eval("all", data_path, epochs=125, save_models=True)
    active = ROOT / "models" / "saved_models" / "active_model.pkl"
    registry = ROOT / "models" / "saved_models" / "model_registry.json"
    if not active.exists():
        print("ERROR: active_model.pkl was not created")
        return 1
    print(f"Phase 5 active model (STGNN): {active}")
    print(f"Model registry: {registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
