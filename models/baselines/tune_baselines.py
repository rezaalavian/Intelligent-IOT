"""Hyperparameter optimization and fine-tuning system per model and horizon."""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    RandomForestRegressor = None

try:
    from lightgbm import LGBMRegressor
    USE_LGBM = True
except Exception:
    LGBMRegressor = None
    USE_LGBM = False

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
    USE_PYG = True
except Exception:
    Data = None
    PyGDataLoader = None
    USE_PYG = False

from infrastructure.kafka.station_registry import STATIONS as _STATIONS
from infrastructure.kafka.station_registry import target_id as _target_id
from models.baselines.train_baselines import (
    DEFAULT_PATH,
    LOOKBACK_STEPS,
    NUM_STATIONS,
    STATION_COORDS,
    build_graph_sequences,
    build_lstm_sequences,
    compile_metrics,
    get_scaled_splits_global,
)
from models.feature_recipes import RECIPES, get_features_for_model_and_horizon
from models.model_registry import (
    ACTIVE_MODEL_KEY,
    default_active_horizons,
    save_model_family,
    set_active_horizons,
    set_active_model,
    write_registry,
)
from models.predictors import (
    AirQualitySTGNN,
    ConstantPredictor,
    LSTMPredictor,
    STGNNPredictor,
    TabularPredictor,
    TorchLSTMRegressor,
)

try:
    import mlflow
except ImportError:
    mlflow = None


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tune_ridge(
    frame: pd.DataFrame, h: int, features: List[str] | None
) -> Tuple[Dict[str, Any], Any, RobustScaler, List[str]]:
    print(f"\n--- Tuning Ridge Regression for Horizon +{h}h ---")
    X_tr, X_va, X_te, y_tr, y_va, y_te, scaler, feats, _, _, _ = get_scaled_splits_global(
        frame, "lr", h, features
    )

    if h == 1:
        base_alpha = 0.0
    elif h == 2:
        base_alpha = 5.0
    else:
        base_alpha = 15.0

    # Ensure baseline is evaluated and search in its neighborhood
    alphas = sorted(list(set([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0, base_alpha])))
    best_val_r2 = -float("inf")
    best_alpha = None
    best_model = None

    for alpha in alphas:
        # alpha=0 is linear regression (OLS)
        if alpha == 0.0:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        else:
            model = Ridge(alpha=alpha)
        model.fit(X_tr, y_tr)
        val_preds = model.predict(X_va)
        val_r2 = r2_score(y_va, val_preds)
        print(f"Alpha: {alpha:5.1f} (Baseline={base_alpha}) | Val R2: {val_r2:8.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = alpha
            best_model = model

    print(f"Best Alpha: {best_alpha} | Best Val R2: {best_val_r2:.4f}")
    
    # Evaluate final
    train_metrics = compile_metrics(y_tr, best_model.predict(X_tr))
    val_metrics = compile_metrics(y_va, best_model.predict(X_va))
    test_metrics = compile_metrics(y_te, best_model.predict(X_te))
    
    metrics = {"Train": train_metrics, "Val": val_metrics, "Test": test_metrics}
    return metrics, best_model, scaler, feats


def tune_lgbm_or_rf(
    frame: pd.DataFrame, h: int, features: List[str] | None, seed: int
) -> Tuple[Dict[str, Any], Any, RobustScaler, List[str]]:
    print(f"\n--- Tuning Gradient Boosting / RF for Horizon +{h}h ---")
    X_tr, X_va, X_te, y_tr, y_va, y_te, scaler, feats, _, _, _ = get_scaled_splits_global(
        frame, "rf", h, features
    )

    best_val_r2 = -float("inf")
    best_params = {}
    best_model = None

    if USE_LGBM:
        # Determine baseline hyperparameters
        if h == 1:
            base_p = {"n_est": 250, "lr": 0.04, "depth": 6, "reg_lam": 0.0, "reg_alp": 0.0}
        elif h == 2:
            base_p = {"n_est": 250, "lr": 0.04, "depth": 6, "reg_lam": 1.0, "reg_alp": 0.0}
        else:
            base_p = {"n_est": 200, "lr": 0.03, "depth": 5, "reg_lam": 2.0, "reg_alp": 0.0}

        # Build regularized search space around base_p
        lrs = sorted(list(set([0.01, 0.02, 0.03, 0.04, base_p["lr"]])))
        depths = sorted(list(set([3, 4, 5, 6, base_p["depth"]])))
        estimators = sorted(list(set([150, 200, 250, 300, 450, base_p["n_est"]])))
        lambdas = sorted(list(set([0.0, 1.0, 2.0, 5.0, 10.0, 15.0, base_p["reg_lam"]])))
        alphas = [0.0, 0.1, 0.5, 1.0, 2.0]

        grid = []
        for lr in lrs:
            for depth in depths:
                for n_est in estimators:
                    for reg_lam in lambdas:
                        for reg_alp in alphas:
                            grid.append({
                                "lr": lr,
                                "depth": depth,
                                "n_est": n_est,
                                "reg_lam": reg_lam,
                                "reg_alp": reg_alp
                            })
        
        # Deduplicate and ensure baseline is first
        unique_grid = []
        seen = set()
        
        # Add base_p first
        base_key = (base_p["lr"], base_p["depth"], base_p["n_est"], base_p["reg_lam"], base_p["reg_alp"])
        unique_grid.append(base_p)
        seen.add(base_key)
        
        # Add rest randomly
        random.seed(seed)
        random.shuffle(grid)
        for item in grid:
            key = (item["lr"], item["depth"], item["n_est"], item["reg_lam"], item["reg_alp"])
            if key not in seen:
                unique_grid.append(item)
                seen.add(key)
        
        trials = unique_grid[:18]  # Test 18 configurations total (first is baseline)

        for idx, p in enumerate(trials):
            is_base = " (Baseline)" if idx == 0 else ""
            model = LGBMRegressor(
                n_estimators=p["n_est"],
                learning_rate=p["lr"],
                max_depth=p["depth"],
                subsample=0.7,             # Slightly more stochastic (regularized)
                colsample_bytree=0.7,      # Slightly more stochastic (regularized)
                reg_lambda=p["reg_lam"],
                reg_alpha=p["reg_alp"],
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            val_preds = model.predict(X_va)
            val_r2 = r2_score(y_va, val_preds)
            print(f"Trial {idx+1:2d}/18{is_base} | LR: {p['lr']:.3f}, Depth: {p['depth']}, Est: {p['n_est']}, Lambda: {p['reg_lam']:.1f}, Alpha: {p['reg_alp']:.1f} | Val R2: {val_r2:8.4f}")

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_params = p
                best_model = model
    else:
        # Fallback Random Forest tuning anchored to horizon-specific baseline
        if h == 1:
            base_p = {"n_est": 40, "depth": 12, "leaf": 1}
        elif h == 2:
            base_p = {"n_est": 60, "depth": 8, "leaf": 4}
        else:
            base_p = {"n_est": 80, "depth": 6, "leaf": 8}

        depths = sorted(list(set([max(2, base_p["depth"] - 2), base_p["depth"], base_p["depth"] + 2, base_p["depth"] + 4])))
        estimators = sorted(list(set([max(10, base_p["n_est"] - 20), base_p["n_est"], base_p["n_est"] + 20, base_p["n_est"] + 40])))
        leaves = sorted(list(set([max(1, base_p["leaf"] - 2), base_p["leaf"], base_p["leaf"] + 2, base_p["leaf"] + 4])))

        grid = []
        for depth in depths:
            for n_est in estimators:
                for leaf in leaves:
                    grid.append({"depth": depth, "n_est": n_est, "leaf": leaf})

        unique_grid = [base_p]
        seen = {(base_p["depth"], base_p["n_est"], base_p["leaf"])}
        
        random.seed(seed)
        random.shuffle(grid)
        for item in grid:
            key = (item["depth"], item["n_est"], item["leaf"])
            if key not in seen:
                unique_grid.append(item)
                seen.add(key)
                
        trials = unique_grid[:12]

        for idx, p in enumerate(trials):
            if RandomForestRegressor is None:
                continue
            is_base = " (Baseline)" if idx == 0 else ""
            model = RandomForestRegressor(
                n_estimators=p["n_est"],
                max_depth=p["depth"],
                min_samples_leaf=p["leaf"],
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            val_preds = model.predict(X_va)
            val_r2 = r2_score(y_va, val_preds)
            print(f"Trial {idx+1:2d}/12{is_base} | Depth: {p['depth']}, Est: {p['n_est']}, Leaf: {p['leaf']} | Val R2: {val_r2:8.4f}")

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_params = p
                best_model = model

    print(f"Best Params: {best_params} | Best Val R2: {best_val_r2:.4f}")
    
    # Evaluate final
    train_metrics = compile_metrics(y_tr, best_model.predict(X_tr))
    val_metrics = compile_metrics(y_va, best_model.predict(X_va))
    test_metrics = compile_metrics(y_te, best_model.predict(X_te))
    
    metrics = {"Train": train_metrics, "Val": val_metrics, "Test": test_metrics}
    return metrics, best_model, scaler, feats


def tune_lstm(
    frame: pd.DataFrame,
    h: int,
    features: List[str] | None,
    n_trials: int,
    epochs: int,
    final_epochs: int,
    device: str,
    seed: int,
    batch_size: int = 64,
) -> Tuple[Dict[str, Any], LSTMPredictor, RobustScaler, List[str]]:
    print(f"\n--- Tuning LSTM Network for Horizon +{h}h ---")
    X_tr, X_va, X_te, y_tr, y_va, y_te, scaler, feats, _, _, _ = get_scaled_splits_global(
        frame, "lstm", h, features
    )
    
    X_tr_3d, y_tr_3d = build_lstm_sequences(X_tr, y_tr, LOOKBACK_STEPS)
    X_va_3d, y_va_3d = build_lstm_sequences(X_va, y_va, LOOKBACK_STEPS)
    X_te_3d, y_te_3d = build_lstm_sequences(X_te, y_te, LOOKBACK_STEPS)

    best_val_r2 = -float("inf")
    best_params = {}
    
    # Baseline hyperparameters
    if h == 1:
        base_p = {"h1": 32, "h2": 16, "dropout": 0.2, "wd": 5e-4, "lr": 1e-3}
    elif h == 2:
        base_p = {"h1": 32, "h2": 16, "dropout": 0.4, "wd": 5e-3, "lr": 1e-3}
    else:
        base_p = {"h1": 32, "h2": 16, "dropout": 0.5, "wd": 1e-2, "lr": 1e-3}

    # Search Space Options - Wider search as requested
    dropout_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    wd_options = [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2]
    lr_options = [1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
    h1_options = [16, 32, 64, 128]
    h2_options = [8, 16, 32, 64]

    # Pre-select trials starting with the exact baseline parameters
    trials_params = [base_p]
    seen = {(base_p["h1"], base_p["h2"], base_p["dropout"], base_p["wd"], base_p["lr"])}
    
    set_seeds(seed)
    # Generate the remaining trials randomly in the neighborhood of the baseline
    while len(trials_params) < n_trials:
        p = {
            "dropout": random.choice(dropout_options),
            "wd": random.choice(wd_options),
            "lr": random.choice(lr_options),
            "h1": random.choice(h1_options),
            "h2": random.choice(h2_options)
        }
        key = (p["h1"], p["h2"], p["dropout"], p["wd"], p["lr"])
        if key not in seen:
            trials_params.append(p)
            seen.add(key)

    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import TensorDataset

    ds_tr = TensorDataset(
        torch.tensor(X_tr_3d, dtype=torch.float32),
        torch.tensor(y_tr_3d, dtype=torch.float32).unsqueeze(1)
    )
    loader_tr = TorchDataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    
    xva = torch.tensor(X_va_3d, dtype=torch.float32).to(device)
    yva = torch.tensor(y_va_3d, dtype=torch.float32).unsqueeze(1).to(device)

    for idx, p in enumerate(trials_params):
        set_seeds(seed)
        model = TorchLSTMRegressor(
            X_tr_3d.shape[2], hidden_dim1=p["h1"], hidden_dim2=p["h2"], dropout=p["dropout"]
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["wd"])
        loss_fn = nn.MSELoss()

        best_trial_val_loss = float("inf")
        best_state = None
        patience = 8
        bad_epochs = 0

        for epoch in range(epochs):
            model.train()
            for bx, by in loader_tr:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = loss_fn(model(bx), by)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(xva), yva).item()
            
            if val_loss < best_trial_val_loss:
                best_trial_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            val_preds = model(xva).detach().cpu().numpy().flatten()
        
        val_r2 = r2_score(y_va_3d, val_preds)
        is_base = " (Baseline)" if idx == 0 else ""
        print(f"Trial {idx+1:2d}/{n_trials}{is_base} | H1/H2: {p['h1']}/{p['h2']}, LR: {p['lr']:.4f}, WD: {p['wd']:.4f}, Drop: {p['dropout']:.2f} | Val R2: {val_r2:8.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_params = p

    print(f"Best LSTM Params: {best_params} | Best Val R2: {best_val_r2:.4f}")
    print(f"Running final training for {final_epochs} epochs on best config...")

    # Final training run
    set_seeds(seed)
    best_model = TorchLSTMRegressor(
        X_tr_3d.shape[2],
        hidden_dim1=best_params["h1"],
        hidden_dim2=best_params["h2"],
        dropout=best_params["dropout"]
    ).to(device)
    opt = torch.optim.Adam(best_model.parameters(), lr=best_params["lr"], weight_decay=best_params["wd"])
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 12
    bad_epochs = 0

    for epoch in range(final_epochs):
        best_model.train()
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = loss_fn(best_model(bx), by)
            loss.backward()
            opt.step()

        best_model.eval()
        with torch.no_grad():
            val_loss = loss_fn(best_model(xva), yva).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in best_model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        best_model.load_state_dict(best_state)

    best_model.eval()
    with torch.no_grad():
        xtr = torch.tensor(X_tr_3d, dtype=torch.float32).to(device)
        train_preds = best_model(xtr).detach().cpu().numpy().flatten()
        val_preds = best_model(xva).detach().cpu().numpy().flatten()
        
        xte = torch.tensor(X_te_3d, dtype=torch.float32).to(device)
        test_preds = best_model(xte).detach().cpu().numpy().flatten()

    train_metrics = compile_metrics(y_tr_3d, train_preds)
    val_metrics = compile_metrics(y_va_3d, val_preds)
    test_metrics = compile_metrics(y_te_3d, test_preds)
    metrics = {"Train": train_metrics, "Val": val_metrics, "Test": test_metrics}

    best_model = best_model.cpu()
    predictor = LSTMPredictor(best_model, scaler, feats, lookback=LOOKBACK_STEPS, log_target=False)
    return metrics, predictor, scaler, feats


def tune_stgnn(
    frame: pd.DataFrame,
    h: int,
    features: List[str] | None,
    n_trials: int,
    epochs: int,
    final_epochs: int,
    device: str,
    seed: int,
    batch_size: int = 64,
) -> Tuple[Dict[str, Any], STGNNPredictor, RobustScaler, List[str]]:
    print(f"\n--- Tuning STGNN for Horizon +{h}h ---")
    if not USE_PYG:
        print("PyTorch Geometric not available. Skipping STGNN tuning.")
        return {}, None, None, []

    X_tr, X_va, X_te, y_tr, y_va, y_te, graph_scaler, stgnn_feats, df_model, tr_end, va_end = get_scaled_splits_global(
        frame, "stgnn", h, features
    )
    
    target_sid = _target_id()

    def _node_matrix(sid):
        feat = df_model[stgnn_feats].copy()
        if sid != target_sid:
            ncol = f"pm25_{sid}"
            if ncol in df_model.columns:
                feat["pm25"] = np.log1p(np.clip(df_model[ncol].values, 0.0, None))
            for col_base, col_neighbor in [
                ("temp definition °c", f"temp_{sid}"),
                ("dew point definition °c", f"dew_point_{sid}"),
                ("rel hum definition %", f"rel_hum_{sid}"),
                ("wind_u", f"wind_u_{sid}"),
                ("wind_v", f"wind_v_{sid}"),
            ]:
                if col_neighbor in df_model.columns:
                    feat[col_base] = df_model[col_neighbor].values
        
        return graph_scaler.transform(feat)

    scaled_all = {sid: _node_matrix(sid) for sid in STATION_COORDS}
    X_graphs = build_graph_sequences(scaled_all, df_model, lookback=LOOKBACK_STEPS, horizon=h)
    
    lb = LOOKBACK_STEPS
    g_tr_end = max(0, tr_end - lb + 1)
    g_va_end = max(g_tr_end, va_end - lb + 1)
    g_train = X_graphs[:g_tr_end]
    g_val = X_graphs[g_tr_end:g_va_end]
    g_test = X_graphs[g_va_end:]

    loader_tr = PyGDataLoader(g_train, batch_size=batch_size, shuffle=False)
    loader_va = PyGDataLoader(g_val, batch_size=batch_size * 2, shuffle=False)
    loader_te = PyGDataLoader(g_test, batch_size=batch_size * 2, shuffle=False)

    best_val_r2 = -float("inf")
    best_params = {}

    # Baseline hyperparameters
    if h == 1:
        base_p = {"dropout": 0.1, "wd": 1e-3, "lr": 1e-3}
    elif h == 2:
        base_p = {"dropout": 0.15, "wd": 3e-3, "lr": 1e-3}
    else:
        base_p = {"dropout": 0.2, "wd": 5e-3, "lr": 1e-3}

    # Search Space Options - Wider search as requested
    dropout_options = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    wd_options = [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2]
    lr_options = [1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3]

    # Pre-select trials starting with the exact baseline parameters
    trials_params = [base_p]
    seen = {(base_p["dropout"], base_p["wd"], base_p["lr"])}
    
    set_seeds(seed)
    # Generate the remaining trials randomly in the neighborhood of the baseline
    while len(trials_params) < n_trials:
        p = {
            "dropout": random.choice(dropout_options),
            "wd": random.choice(wd_options),
            "lr": random.choice(lr_options)
        }
        key = (p["dropout"], p["wd"], p["lr"])
        if key not in seen:
            trials_params.append(p)
            seen.add(key)

    criterion = nn.MSELoss()

    def _val_loss(model):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for batch in loader_va:
                batch = batch.to(device)
                out = model(batch)
                tot += criterion(out, batch.y.flatten()).item() * out.numel()
                cnt += out.numel()
        return tot / max(cnt, 1)

    for idx, p in enumerate(trials_params):
        set_seeds(seed)
        model = AirQualitySTGNN(
            num_features=len(stgnn_feats), num_timesteps_input=LOOKBACK_STEPS, dropout=p["dropout"]
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["wd"])

        best_trial_val_loss = float("inf")
        best_state = None
        patience = 8
        bad_epochs = 0

        for epoch in range(epochs):
            model.train()
            for batch in loader_tr:
                batch = batch.to(device)
                opt.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y.flatten())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            
            vl = _val_loss(model)
            if vl < best_trial_val_loss:
                best_trial_val_loss = vl
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        
        if best_state is not None:
            model.load_state_dict(best_state)

        # Get val R2
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader_va:
                batch = batch.to(device)
                out_numpy = model(batch).detach().cpu().numpy()
                y_numpy = batch.y.flatten().detach().cpu().numpy()
                preds.append(out_numpy[0::NUM_STATIONS])
                trues.append(y_numpy[0::NUM_STATIONS])
        val_r2 = r2_score(np.concatenate(trues), np.concatenate(preds))
        is_base = " (Baseline)" if idx == 0 else ""
        print(f"Trial {idx+1:2d}/{n_trials}{is_base} | LR: {p['lr']:.4f}, WD: {p['wd']:.4f}, Drop: {p['dropout']:.2f} | Val R2: {val_r2:8.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_params = p

    print(f"Best STGNN Params: {best_params} | Best Val R2: {best_val_r2:.4f}")
    print(f"Running final training for {final_epochs} epochs on best config...")

    # Final training
    set_seeds(seed)
    best_model = AirQualitySTGNN(
        num_features=len(stgnn_feats), num_timesteps_input=LOOKBACK_STEPS, dropout=best_params["dropout"]
    ).to(device)
    opt = torch.optim.AdamW(best_model.parameters(), lr=best_params["lr"], weight_decay=best_params["wd"])

    best_val_loss = float("inf")
    best_state = None
    patience = 15
    bad_epochs = 0

    for epoch in range(final_epochs):
        best_model.train()
        for batch in loader_tr:
            batch = batch.to(device)
            opt.zero_grad()
            out = best_model(batch)
            loss = criterion(out, batch.y.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(best_model.parameters(), 1.0)
            opt.step()
        
        vl = _val_loss(best_model)
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.detach().clone() for k, v in best_model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        best_model.load_state_dict(best_state)

    best_model.eval()

    def _preds(loader):
        preds_arr, trues_arr = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out_numpy = best_model(batch).detach().cpu().numpy()
                y_numpy = batch.y.flatten().detach().cpu().numpy()
                preds_arr.append(out_numpy[0::NUM_STATIONS])
                trues_arr.append(y_numpy[0::NUM_STATIONS])
        return np.concatenate(trues_arr), np.concatenate(preds_arr)

    y_tr_true, tr_p = _preds(loader_tr)
    y_va_true, va_p = _preds(loader_va)
    y_te_true, te_p = _preds(loader_te)

    train_metrics = compile_metrics(y_tr_true, tr_p)
    val_metrics = compile_metrics(y_va_true, va_p)
    test_metrics = compile_metrics(y_te_true, te_p)
    metrics = {"Train": train_metrics, "Val": val_metrics, "Test": test_metrics}

    best_model = best_model.cpu()
    predictor = STGNNPredictor(best_model, graph_scaler, stgnn_feats, lookback=LOOKBACK_STEPS, log_target=False)
    return metrics, predictor, graph_scaler, stgnn_feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lr", "rf", "lstm", "stgnn", "all"], default="all")
    p.add_argument("--path", type=Path, default=DEFAULT_PATH)
    p.add_argument("--n-trials", type=int, default=8, help="Number of trials for deep models")
    p.add_argument("--epochs", type=int, default=50, help="Epochs per trial for deep models")
    p.add_argument("--final-epochs", type=int, default=125, help="Final training epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for deep model loaders")
    p.add_argument("--device", default="auto")
    p.add_argument("--horizon", type=int, choices=[1, 2, 3], default=None)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--mlflow-experiment-name", default="Intelligent-IOT-baselines-tuning")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-save", action="store_true", help="Skip writing .pkl bundles")
    args = p.parse_args()

    set_seeds(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    resolved_path = Path(args.path)
    if not resolved_path.exists():
        resolved_path = Path("data/external/multistation/train.csv")
    frame = pd.read_csv(resolved_path, low_memory=False)
    frame.columns = frame.columns.str.strip().str.lower()
    
    # We need to process date features / hour / cyclic variables
    if "time lst" in frame.columns:
        if frame["time lst"].dtype == "object":
            frame["clean_hour"] = frame["time lst"].astype(str).str.extract(r"(\d+)").astype(float).fillna(0).astype(int)
        else:
            frame["clean_hour"] = pd.to_numeric(frame["time lst"], errors="coerce").fillna(0).astype(int)
    else:
        frame["clean_hour"] = 0

    if "timestamp" in frame.columns:
        frame["datetime_combined"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    else:
        date_strings = (
            frame["year"].astype(str).str.extract(r"(\d+)")[0].fillna("2026")
            + "-"
            + frame["month"].astype(str).str.extract(r"(\d+)")[0].fillna("1")
            + "-"
            + frame["day"].astype(str).str.extract(r"(\d+)")[0].fillna("1")
            + " "
            + frame["clean_hour"].astype(str)
            + ":00:00"
        )
        frame["datetime_combined"] = pd.to_datetime(date_strings, errors="coerce")

    if frame["datetime_combined"].isna().all():
        frame["datetime_combined"] = pd.date_range(start="2026-01-01", periods=len(frame), freq="h")

    frame = frame.dropna(subset=["datetime_combined"]).sort_values("datetime_combined")

    cols_to_numeric = [
        "temp definition °c",
        "dew point definition °c",
        "rel hum definition %",
        "wind spd definition km/h",
        "wind dir definition 10's deg",
        "pm25",
        "pm2",
        "upwind_pm25", "transport_potential", "wind_alignment",
    ]
    for c in cols_to_numeric:
        if c in frame.columns:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")

    frame = frame.ffill().bfill().fillna(0.0)

    if "wind dir definition 10's deg" in frame.columns and "wind spd definition km/h" in frame.columns:
        rad = np.deg2rad(frame["wind dir definition 10's deg"] * 10)
        frame["wind_u"] = frame["wind spd definition km/h"] * np.cos(rad)
        frame["wind_v"] = frame["wind spd definition km/h"] * np.sin(rad)
    elif "wind_u" not in frame.columns or "wind_v" not in frame.columns:
        frame["wind_u"] = 0.0
        frame["wind_v"] = 0.0

    if "datetime_combined" in frame.columns:
        hour = frame["datetime_combined"].dt.hour
        dayofweek = frame["datetime_combined"].dt.dayofweek
        frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        frame["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
        frame["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    else:
        frame["hour_sin"] = 0.0
        frame["hour_cos"] = 0.0
        frame["dow_sin"] = 0.0
        frame["dow_cos"] = 0.0

    if "pm25" in frame.columns:
        for lag in [1, 2, 3, 6, 12]:
            frame[f"pm25_lag{lag}"] = frame["pm25"].shift(lag)
        frame["pm25_roll3_mean"] = frame["pm25"].rolling(window=3, min_periods=1).mean()
        frame["pm25_roll6_mean"] = frame["pm25"].rolling(window=6, min_periods=1).mean()
        
        lag_cols = [f"pm25_lag{lag}" for lag in [1, 2, 3, 6, 12]] + ["pm25_roll3_mean", "pm25_roll6_mean"]
        for col in lag_cols:
            frame[col] = frame[col].ffill().bfill().fillna(0.0)

    target_col = "pm25" if "pm25" in frame.columns else "pm2"
    horizons = [int(args.horizon)] if args.horizon is not None else [1, 2, 3]
    selected = args.model.lower()

    if mlflow is not None and not args.no_mlflow:
        try:
            mlflow.set_experiment(args.mlflow_experiment_name)
        except Exception:
            pass

    results_master: Dict[int, Dict[str, Any]] = {1: {}, 2: {}, 3: {}}
    lr_models = {}
    rf_models = {}
    lstm_models = {}
    stgnn_models = {}
    lr_scalers = {}
    rf_scalers = {}
    lstm_scalers = {}
    stgnn_scalers = {}

    for h in horizons:
        results_master[h] = {}

        # 1. Ridge Regression (LR)
        if selected in ("lr", "all"):
            best_val_r2 = -float("inf")
            best_model_overall = None
            best_scaler_overall = None
            best_feats_overall = None
            best_metrics_overall = None
            
            for recipe_name, recipe_cols in RECIPES.items():
                print(f"\n>>> Tuning LR with Recipe: {recipe_name} <<<")
                m_res, model, scaler, feats = tune_ridge(frame, h, recipe_cols)
                
                mkey = f"Linear Regression (Recipe: {recipe_name})"
                results_master[h][mkey] = m_res
                
                val_r2 = m_res["Val"]["R2"]
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_model_overall = model
                    best_scaler_overall = scaler
                    best_feats_overall = feats
                    best_metrics_overall = m_res
            
            lr_models[h] = TabularPredictor(best_model_overall, best_scaler_overall, best_feats_overall, log_target=False)
            lr_scalers[h] = best_scaler_overall

        # 2. Gradient Boosting / RF
        if selected in ("rf", "all"):
            best_val_r2 = -float("inf")
            best_model_overall = None
            best_scaler_overall = None
            best_feats_overall = None
            best_metrics_overall = None
            
            for recipe_name, recipe_cols in RECIPES.items():
                print(f"\n>>> Tuning RF with Recipe: {recipe_name} <<<")
                m_res, model, scaler, feats = tune_lgbm_or_rf(frame, h, recipe_cols, args.seed)
                
                mkey = f"Gradient Boosting/RF (Recipe: {recipe_name})"
                results_master[h][mkey] = m_res
                
                val_r2 = m_res["Val"]["R2"]
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_model_overall = model
                    best_scaler_overall = scaler
                    best_feats_overall = feats
                    best_metrics_overall = m_res
                    
            rf_models[h] = TabularPredictor(best_model_overall, best_scaler_overall, best_feats_overall, log_target=False)
            rf_scalers[h] = best_scaler_overall

        # 3. LSTM
        if selected in ("lstm", "all"):
            best_val_r2 = -float("inf")
            best_predictor_overall = None
            best_scaler_overall = None
            best_feats_overall = None
            best_metrics_overall = None
            
            for recipe_name, recipe_cols in RECIPES.items():
                print(f"\n>>> Tuning LSTM with Recipe: {recipe_name} <<<")
                m_res, predictor, scaler, feats = tune_lstm(
                    frame, h, recipe_cols, args.n_trials, args.epochs, args.final_epochs, device, args.seed, args.batch_size
                )
                
                mkey = f"LSTM Sequential (Recipe: {recipe_name})"
                results_master[h][mkey] = m_res
                
                val_r2 = m_res["Val"]["R2"]
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_predictor_overall = predictor
                    best_scaler_overall = scaler
                    best_feats_overall = feats
                    best_metrics_overall = m_res
                    
            lstm_models[h] = best_predictor_overall
            lstm_scalers[h] = best_scaler_overall

        # 4. STGNN
        if selected in ("stgnn", "all") and USE_PYG:
            best_val_r2 = -float("inf")
            best_predictor_overall = None
            best_scaler_overall = None
            best_feats_overall = None
            best_metrics_overall = None
            
            for recipe_name, recipe_cols in RECIPES.items():
                print(f"\n>>> Tuning STGNN with Recipe: {recipe_name} <<<")
                m_res, predictor, scaler, feats = tune_stgnn(
                    frame, h, recipe_cols, args.n_trials, args.epochs, args.final_epochs, device, args.seed, args.batch_size
                )
                
                if predictor is not None:
                    mkey = f"Spatiotemporal Graph (Recipe: {recipe_name})"
                    results_master[h][mkey] = m_res
                    
                    val_r2 = m_res["Val"]["R2"]
                    if val_r2 > best_val_r2:
                        best_val_r2 = val_r2
                        best_predictor_overall = predictor
                        best_scaler_overall = scaler
                        best_feats_overall = feats
                        best_metrics_overall = m_res
                        
            if best_predictor_overall is not None:
                stgnn_models[h] = best_predictor_overall
                stgnn_scalers[h] = best_scaler_overall

    # Log metrics to JSON
    # Structure match baseline_metrics.json for compatibility
    def _normalize_results(results):
        normalized = {}
        for hz, models in results.items():
            hkey = f"h{hz}"
            normalized[hkey] = {}
            for name, splits in models.items():
                normalized[hkey][name] = {
                    "train": {"r2": splits["Train"]["R2"], "mae": splits["Train"]["MAE"], "mse": splits["Train"]["MSE"], "rmse": splits["Train"]["RMSE"]},
                    "val": {"r2": splits["Val"]["R2"], "mae": splits["Val"]["MAE"], "mse": splits["Val"]["MSE"], "rmse": splits["Val"]["RMSE"]},
                    "test": {"r2": splits["Test"]["R2"], "mae": splits["Test"]["MAE"], "mse": splits["Test"]["MSE"], "rmse": splits["Test"]["RMSE"]},
                }
        return normalized

    normalized_metrics = _normalize_results(results_master)

    # Save outputs if not no_save
    if not args.no_save:
        out_path = Path("models/saved_models/baseline_metrics.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing metrics safely with retries to support parallel terminal runs
        existing = {}
        if out_path.exists():
            for retry in range(10):
                try:
                    with open(out_path, "r", encoding="utf-8") as fh:
                        existing = json.load(fh).get("results", {})
                    break
                except Exception:
                    time.sleep(random.uniform(0.1, 0.5))
        
        # Merge new results on top of existing ones
        for hkey, models in normalized_metrics.items():
            if hkey not in existing:
                existing[hkey] = {}
            for mkey, split_dict in models.items():
                existing[hkey][mkey] = split_dict
                
        # Write back safely with retries
        for retry in range(10):
            try:
                temp_path = out_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as fh:
                    json.dump({"timestamp": time.time(), "results": existing}, fh, indent=2)
                if out_path.exists():
                    out_path.unlink()
                temp_path.rename(out_path)
                break
            except Exception:
                time.sleep(random.uniform(0.1, 0.5))

        # Save model bundles
        family_specs = [
            ("lr", "linear_regression", lr_models, lr_scalers),
            ("rf", "gradient_boosting_rf", rf_models, rf_scalers),
            ("lstm", "lstm", lstm_models, lstm_scalers),
            ("stgnn", "stgnn", stgnn_models, stgnn_scalers),
        ]
        
        registry_families = {}
        for key, model_type, horizon_models, horizon_scalers in family_specs:
            if not horizon_models:
                continue
            
            # Fetch generic feature columns for saving
            feat_cols = get_features_for_model_and_horizon(key, 1)
            bundle_path = save_model_family(
                    key,
                    model_type,
                    feature_columns=feat_cols,
                    target_column=target_col,
                    horizon_models=horizon_models,
                    horizon_scalers=horizon_scalers,
                    metrics=existing,
                    lookback=LOOKBACK_STEPS,
                    save_models=True,
                )
            registry_families[key] = {
                "model_type": model_type,
                "bundle": str(bundle_path) if bundle_path else None,
                "horizons": sorted(horizon_models.keys()),
                "per_horizon": [f"{key}_h{hz}.pkl" for hz in sorted(horizon_models.keys())],
            }
            print(f"Saved optimized {key} bundle ({len(horizon_models)} horizons)")

        # Write/Update the model registry safely
        if registry_families:
            registry_path = Path("models/saved_models/model_registry.json")
            existing_registry = {}
            if registry_path.exists():
                for retry in range(10):
                    try:
                        with open(registry_path, "r", encoding="utf-8") as fh:
                            existing_registry = json.load(fh)
                        break
                    except Exception:
                        time.sleep(random.uniform(0.1, 0.5))
            
            # Merge families
            families = existing_registry.get("families", {})
            for key, new_fam in registry_families.items():
                if key in families:
                    # Merge horizons and per_horizon list
                    existing_fam = families[key]
                    merged_horizons = sorted(list(set(existing_fam.get("horizons", []) + new_fam["horizons"])))
                    merged_per_horizon = sorted(list(set(existing_fam.get("per_horizon", []) + new_fam["per_horizon"])))
                    families[key] = {
                        "model_type": new_fam["model_type"],
                        "bundle": new_fam["bundle"] or existing_fam.get("bundle"),
                        "horizons": merged_horizons,
                        "per_horizon": merged_per_horizon
                    }
                else:
                    families[key] = new_fam
            
            # Write registry safely with retries
            for retry in range(10):
                try:
                    write_registry(
                        families,
                        active_model=existing_registry.get("active_model", ACTIVE_MODEL_KEY if stgnn_models else "lr"),
                        active_horizons=existing_registry.get("active_horizons", default_active_horizons(ACTIVE_MODEL_KEY if stgnn_models else "lr")),
                    )
                    break
                except Exception:
                    time.sleep(random.uniform(0.1, 0.5))
            print("Model registry updated.")

    print("\n" + "=" * 95)
    print("             MASTER HEAD-TO-HEAD COMPREHENSIVE PERFORMANCE EVALUATION")
    print("=" * 95)
    for h in horizons:
        print(f"\n[TIMEFRAME EVALUATION MATRIX]: +{h}-HOUR FORECASTING WINDOW")
        print("-" * 95)
        rows = []
        for model_name_row, split_dict in results_master[h].items():
            for split_name, metrics in split_dict.items():
                rows.append(
                    {
                        "Model Name": model_name_row,
                        "Data Split": split_name,
                        "R2": metrics["R2"],
                        "MAE": metrics["MAE"],
                        "MSE": metrics["MSE"],
                        "RMSE": metrics["RMSE"],
                    }
                )
        if rows:
            print(pd.DataFrame(rows).set_index(["Model Name", "Data Split"]).round(4))
        else:
            print("No models were tuned for this horizon.")
        print("-" * 95)


if __name__ == "__main__":
    main()
