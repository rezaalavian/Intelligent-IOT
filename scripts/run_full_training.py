"""Run full training on the historical dataset and save metrics to disk."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import sys
from pathlib import Path as _Path

# ensure project root is on sys.path so `models` imports resolve when invoked via tooling
_proj_root = str(_Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from models.spatiotemporal.train import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/raw/historical_rawdata.csv")
    parser.add_argument("--target", default=None, help="Target numeric column to forecast (e.g. pm2, o3)")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args(argv)

    out_dir = Path("models/saved_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    artifact_default = out_dir / f"spatiotemporal_model_{ts}.pt"

    print(f"Starting training on dataset={args.dataset} target={args.target}")
    # build kwargs for optional hyperparameters (only pass those provided)
    train_kwargs = {}
    if args.hidden_dim is not None:
        train_kwargs["hidden_dim"] = args.hidden_dim
    if args.lr is not None:
        train_kwargs["lr"] = args.lr
    if args.weight_decay is not None:
        train_kwargs["weight_decay"] = args.weight_decay
    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs
    if args.patience is not None:
        train_kwargs["patience"] = args.patience

    result = train(path=args.dataset, output_path=str(artifact_default), max_rows=None, target_column=args.target, **train_kwargs)

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
        "per_horizon": result.per_horizon_metrics if hasattr(result, 'per_horizon_metrics') else None,
    }

    out_json = out_dir / f"training_result_{ts}.json"
    out_json.write_text(json.dumps(result_dict, indent=2))

    print(f"Training finished. Results written to: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
