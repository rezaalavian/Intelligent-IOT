#!/usr/bin/env python3
"""Switch the Phase 5 streaming model without retraining."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_registry import REGISTRY_PATH, SAVED_MODELS_DIR, set_active_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Set active streaming model from saved .pkl bundles")
    parser.add_argument(
        "model_key",
        choices=["ha", "lr", "rf", "lstm", "stgnn"],
        help="Model family key matching saved bundles (e.g. stgnn_bundle.pkl)",
    )
    args = parser.parse_args()

    active = set_active_model(args.model_key)
    if active is None:
        print(f"ERROR: {args.model_key}_bundle.pkl not found. Train first with run_baselines.py")
        return 1

    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        registry["active_model"] = args.model_key
        registry["active_path"] = str(active)
        REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"Active streaming model set to '{args.model_key}' -> {active}")
    print("Restart API/dashboard or set IOT_ACTIVE_MODEL=%s" % args.model_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
