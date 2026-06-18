import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.feature_recipes import RECIPES  # noqa: E402
from models.per_horizon import build_per_horizon_bundle  # noqa: E402
from models.model_io import save_model  # noqa: E402


def parse_map(spec, recipes):
    out = {}
    for part in spec.split(","):
        h, name = part.split("=", 1)
        out[int(h.strip())] = recipes[name.strip()]   # KeyError on unknown recipe
    return out


def main():  # pragma: no cover - training I/O
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lr")
    ap.add_argument("--map", required=True, help="e.g. 1=base6,2=diffusion9,3=diffusion9")
    ap.add_argument("--path", default="data/external/multistation/train.csv")
    args = ap.parse_args()
    horizon_features = parse_map(args.map, RECIPES)
    bundle = build_per_horizon_bundle(args.path, args.model, horizon_features)
    out = Path("models/saved_models/active_model.pkl")
    save_model(bundle, str(out))
    counts = {h: len(c) for h, c in bundle.feature_columns_by_horizon.items()}
    print(f"per-horizon bundle saved: {bundle.model_type}, feature_columns_by_horizon={counts}")


if __name__ == "__main__":  # pragma: no cover
    main()
