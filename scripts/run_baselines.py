#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from models.baselines.train_baselines import train_and_eval
import argparse

p = argparse.ArgumentParser()
p.add_argument('--model', default='all')
p.add_argument('--path', type=Path, default=Path('data/external/multistation/train.csv'))
p.add_argument('--epochs', type=int, default=125)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--hidden-dim', type=int, default=64)
p.add_argument('--device', default='auto')
p.add_argument('--rf-backend', choices=['sklearn','xgboost'], default='sklearn')
p.add_argument('--horizon', type=int, choices=[1, 2, 3], default=None, help='Train a single forecast horizon instead of all horizons')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--weight-decay', type=float, default=1e-4)
p.add_argument('--patience', type=int, default=5)
args = p.parse_args()

train_and_eval(
	args.model,
	args.path,
	epochs=args.epochs,
	lr=args.lr,
	hidden_dim=args.hidden_dim,
	device=args.device,
	rf_backend=args.rf_backend,
	horizon=args.horizon,
	seed=args.seed,
	weight_decay=args.weight_decay,
	patience=args.patience,
)
