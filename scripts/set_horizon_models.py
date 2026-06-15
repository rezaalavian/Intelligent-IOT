#!/usr/bin/env python3
"""Set a different model per forecast horizon for Phase 5 streaming."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_registry import MODEL_KEYS, set_active_horizons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a model family per forecast horizon (+1h, +2h, +3h)"
    )
    parser.add_argument("--h1", choices=MODEL_KEYS, default="stgnn", help="Model for +1-hour forecast")
    parser.add_argument("--h2", choices=MODEL_KEYS, default="stgnn", help="Model for +2-hour forecast")
    parser.add_argument("--h3", choices=MODEL_KEYS, default="stgnn", help="Model for +3-hour forecast")
    args = parser.parse_args()

    horizon_map = {1: args.h1, 2: args.h2, 3: args.h3}
    try:
        active = set_active_horizons(horizon_map)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Per-horizon streaming models:")
    for h, key in horizon_map.items():
        print(f"  +{h}h -> {key}")
    print(f"Saved composite active model: {active}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
