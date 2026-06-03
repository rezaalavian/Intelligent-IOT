"""Quick training run on pm2 for fast feedback (small subset + fewer epochs)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import sys
from pathlib import Path as _Path

# ensure project root is on sys.path
_proj_root = str(_Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from models.spatiotemporal.train import train


def main():
    dataset = Path("data/raw/historical_rawdata_with_openaq_fixed.csv")
    out_dir = Path("models/saved_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    artifact = out_dir / f"spatiotemporal_quick_pm2_{ts}.pt"

    # small quick-run for feedback
    result = train(path=str(dataset), output_path=str(artifact), max_rows=8000, target_column="pm2", hidden_dim=64, epochs=8, patience=3)

    out_json = out_dir / f"training_result_quick_pm2_{ts}.json"
    out_json.write_text(json.dumps({
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
        "per_horizon": result.per_horizon_metrics,
    }, indent=2))

    print("Quick training finished. Results:")
    print(out_json)


if __name__ == "__main__":
    main()
