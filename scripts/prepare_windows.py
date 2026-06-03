"""Prepare multivariate sliding windows from hourly raw data.

This script:
- loads the repo's cleaned/filled raw CSV
- introduces raw hourly features without rolling transforms
- constructs per-city sliding windows (history -> horizon)
- saves flattened X, y arrays to an .npz file for quick experiments

Usage:
    python scripts/prepare_windows.py --max-rows 1000 --history 24 --horizon 1 --target pm2 --out data/processed/windows.npz
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from analytics.flink_jobs.feature_engineering import introduce_raw_features


def build_windows_from_group(df: pd.DataFrame, feature_cols: List[str], target_col: str, history: int, horizon: int):
    X_list = []
    y_list = []
    n_features = len(feature_cols)
    for i in range(history, len(df) - horizon + 1):
        window = df.iloc[i - history : i][feature_cols].to_numpy()
        # flatten time dimension into features
        X_list.append(window.flatten())
        # take target at horizon (i + horizon - 1)
        y_list.append(df.iloc[i + horizon - 1][target_col])
    if X_list:
        return np.stack(X_list), np.array(y_list)
    return np.empty((0, history * n_features)), np.empty((0,))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--history", type=int, default=24)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--target", type=str, default="pm2")
    p.add_argument("--out", type=str, default="data/processed/windows.npz")
    args = p.parse_args()

    raw_path = "data/raw/historical_rawdata_pm2_filled.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw CSV not found at {raw_path}")

    df = pd.read_csv(raw_path)
    if args.max_rows:
        df = df.head(args.max_rows)

    # start from raw hourly features only; rolling transforms remain opt-in
    features = introduce_raw_features(df)

    # determine numeric feature columns to use (exclude timestamp, city_name and target)
    exclude = {"timestamp", "city_name"}
    all_cols = [c for c in features.columns if c not in exclude]
    # ensure target included as a column (we use original target, not flattened)
    if args.target not in all_cols:
        all_cols.append(args.target)
    # feature columns: drop the target from inputs
    feature_cols = [c for c in all_cols if c != args.target and not c.startswith("timestamp") and not c.endswith("_lag0")]

    Xs = []
    ys = []
    group_col = "city_name" if "city_name" in features.columns else None
    if group_col:
        for city, group in features.groupby(group_col, sort=False):
            group_sorted = group.sort_values("timestamp").reset_index(drop=True)
            X, y = build_windows_from_group(group_sorted, feature_cols, args.target, args.history, args.horizon)
            if X.size:
                Xs.append(X)
                ys.append(y)
    else:
        sorted_df = features.sort_values("timestamp").reset_index(drop=True)
        X, y = build_windows_from_group(sorted_df, feature_cols, args.target, args.history, args.horizon)
        if X.size:
            Xs.append(X)
            ys.append(y)

    if not Xs:
        print("No windows generated (dataset too small after history/horizon).")
        return

    X = np.vstack(Xs)
    y = np.concatenate(ys)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, feature_cols=np.array(feature_cols), history=args.history, horizon=args.horizon)
    print(f"Saved windows: X.shape={X.shape}, y.shape={y.shape} -> {args.out}")


if __name__ == "__main__":
    main()
