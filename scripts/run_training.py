"""Simple CLI to run spatiotemporal training.

Usage examples:
    python scripts/run_training.py --path data/raw/historical_rawdata_pm2_filled.csv --max-rows 2000 --epochs 5 --mlflow-experiment "Intelligent-IOT" --mlflow-run "manual-run"
"""
import argparse
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.spatiotemporal.train import train

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run spatiotemporal model training")
    p.add_argument("--path", default="data/raw/historical_rawdata_pm2_filled.csv")
    p.add_argument("--output", default="models/saved_models/spatiotemporal_cli.pt")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--no-mlflow", dest="log_to_mlflow", action="store_false")
    p.add_argument("--mlflow-experiment", default="Intelligent-IOT-spatiotemporal")
    p.add_argument("--mlflow-run", default=None)
    p.add_argument("--mlflow-tracking-dir", default="mlruns")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = train(
        path=Path(args.path),
        output_path=Path(args.output),
        max_rows=args.max_rows,
        target_column=args.target,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        log_to_mlflow=args.log_to_mlflow,
        mlflow_experiment_name=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run,
        mlflow_tracking_dir=Path(args.mlflow_tracking_dir),
    )
    print(result)


if __name__ == "__main__":
    main()
