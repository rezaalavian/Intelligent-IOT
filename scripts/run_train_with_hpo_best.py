"""Run training with the best HPO params found earlier (o3).
This script is a thin wrapper that calls models.spatiotemporal.train.train
with the tuned hyperparameters and writes a JSON summary to models/saved_models.
"""
from datetime import datetime
from pathlib import Path
import json

from models.spatiotemporal.train import train

BEST = {
    "hidden_dim": 247,
    "lr": 0.007214190462732783,
    "weight_decay": 6.670624447158425e-05,
    "epochs": 22,
    "patience": 11,
}

if __name__ == "__main__":
    out_dir = Path("models/saved_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    artifact = out_dir / f"spatiotemporal_model_hpo_best_{ts}.pt"

    print(f"Starting full training with HPO-best params on target=o3")
    result = train(path="data/raw/historical_rawdata.csv",
                   output_path=str(artifact),
                   max_rows=None,
                   target_column="o3",
                   hidden_dim=BEST["hidden_dim"],
                   lr=BEST["lr"],
                   weight_decay=BEST["weight_decay"],
                   epochs=BEST["epochs"],
                   patience=BEST["patience"],
                   )

    result_dict = {
        "artifact": str(result.artifact_path),
        "target": result.target,
        "n_samples": result.n_samples,
        "train_mae": result.train_mae,
        "val_mae": result.val_mae,
        "test_mae": result.test_mae,
        "train_rmse": result.train_rmse,
        "val_rmse": result.val_rmse,
        "test_rmse": result.test_rmse,
        "train_r2": result.train_r2,
        "val_r2": result.val_r2,
        "test_r2": result.test_r2,
        "per_horizon": getattr(result, "per_horizon_metrics", None),
    }

    out_json = out_dir / f"training_result_hpo_best_{ts}.json"
    out_json.write_text(json.dumps(result_dict, indent=2))

    print(f"Training finished. Summary written to: {out_json}")
    print(json.dumps(result_dict, indent=2))
