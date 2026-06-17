import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baselines.train_baselines import train_and_eval  # noqa: E402

_BASE = ["temp definition °c", "dew point definition °c", "rel hum definition %",
         "wind_u", "wind_v", "pm25"]
RECIPES = {
    "base6": list(_BASE),
    "diffusion9": _BASE + ["upwind_pm25", "transport_potential", "wind_alignment"],
    "base+upwind": _BASE + ["upwind_pm25"],
    "base+transport": _BASE + ["transport_potential"],
    "base+alignment": _BASE + ["wind_alignment"],
}


def flatten_results(results_master: dict, feature_set: str) -> list[dict]:
    rows = []
    for horizon, models in results_master.items():
        for model_name, splits in models.items():
            for split_name, m in splits.items():
                rows.append({
                    "feature_set": feature_set,
                    "model": model_name,
                    "horizon": int(horizon),
                    "split": split_name.lower(),
                    "r2": m.get("R2"), "mae": m.get("MAE"),
                    "mse": m.get("MSE"), "rmse": m.get("RMSE"),
                })
    return rows


def main():  # pragma: no cover - heavy training I/O
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/external/multistation/train.csv")
    ap.add_argument("--models", default="all")
    ap.add_argument("--recipes", default=",".join(RECIPES))
    ap.add_argument("--out", default="experiments.csv")
    args = ap.parse_args()

    selected = [r.strip() for r in args.recipes.split(",") if r.strip()]
    all_rows = []
    for name in selected:
        cols = RECIPES[name]
        print(f"[ablation] recipe={name} features={cols}")
        rm = train_and_eval(args.models, Path(args.path), features=cols, save_models=False)
        all_rows.extend(flatten_results(rm, name))

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature_set", "model", "horizon", "split",
                                           "r2", "mae", "mse", "rmse"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {args.out}")
    # Print a compact test-split matrix
    for r in all_rows:
        if r["split"] == "test":
            print(f"  {r['feature_set']:16} {r['model']:20} h{r['horizon']}  R2={r['r2']:.3f}  RMSE={r['rmse']:.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()
