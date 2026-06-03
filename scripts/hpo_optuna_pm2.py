import os
import optuna
from datetime import datetime
import sys
from pathlib import Path as _Path

# ensure project root on sys.path
_proj_root = str(_Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from models.spatiotemporal.train import train


def objective(trial: optuna.Trial, max_rows: int) -> float:
    hidden_dim = trial.suggest_int("hidden_dim", 16, 256)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    epochs = trial.suggest_int("epochs", 10, 40)
    patience = trial.suggest_int("patience", 3, 12)

    out_name = f"models/saved_models/hpo_trial_{trial.number}_pm2.pt"
    try:
        result = train(
            path="data/raw/historical_rawdata_pm2_filled.csv",
            output_path=out_name,
            max_rows=max_rows,
            target_column="pm2",
            hidden_dim=hidden_dim,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
        )
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return -999.0

    # We maximize validation R2
    print(f"Trial {trial.number}: val_r2={result.val_r2}, params={{'hidden_dim':{hidden_dim}, 'lr':{lr}, 'weight_decay':{weight_decay}, 'epochs':{epochs}, 'patience':{patience}}}")
    return float(result.val_r2)


def main():
    n_trials = int(os.environ.get("OPT_TRIALS", "8"))
    max_rows = int(os.environ.get("MAX_ROWS", "16000"))
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, max_rows=max_rows), n_trials=n_trials)

    best = study.best_trial
    print("Best trial:")
    print(f"  Value: {best.value}")
    print(f"  Params: {best.params}")

    # save best params
    out_dir = "models/saved_models"
    os.makedirs(out_dir, exist_ok=True)
    params_path = os.path.join(out_dir, f"hpo_best_pm2_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.txt")
    with open(params_path, "w") as fh:
        fh.write(str(best.params))


if __name__ == "__main__":
    main()
