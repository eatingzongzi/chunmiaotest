#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ensemble v2: per-model scaler correction + robust aggregation.

Fixes two flaws of ensemble_test.py:
  1. each model's prediction is un-normalised with ITS OWN label scaler
     (the first version wrongly used the evaluation seed's scaler)
  2. a model that collapses on a foreign cell poisons the mean — compare
     mean / median / trimmed mean (drop best+worst of 5)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import rmse

device = get_device('cuda')
SEEDS = [42, 43, 44, 45, 46]

models, loaders = [], []
for seed in SEEDS:
    config = Config()
    config.epochs = 100
    config.random_seed = seed
    config.moe_balance_coeff = 0.05
    config.route_sup_coeff = 1.0
    config.router_mse_coeff = 0.1
    train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=seed)
    model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
    model.eval()
    models.append((model, label_scaler))
    loaders.append(test_cells)
    print(f"trained seed {seed}", flush=True)

print("\n=== Robust ensemble (own-scaler, mean/median/trimmed) ===")
summaries = {k: [] for k in ['mean', 'median', 'trimmed', 'single_best']}
for si, test_cells in enumerate(loaders):
    cell_ens = {k: [] for k in summaries}
    for cell in test_cells:
        x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
        y_orig = cell['y_orig']
        preds = []
        with torch.no_grad():
            for m, ls in models:
                g, *_ = m(x)
                preds.append(ls.inverse_transform(
                    g.detach().cpu().numpy().reshape(-1, 1)).ravel())
        P = np.array(preds)                      # (5, N)
        mean_p = P.mean(axis=0)
        med_p = np.median(P, axis=0)
        sorted_p = np.sort(P, axis=0)
        trim_p = sorted_p[1:4].mean(axis=0)      # drop best & worst of 5
        best_single = min(rmse(y_orig, P[j]) for j in range(5))
        r = {
            'mean': rmse(y_orig, mean_p),
            'median': rmse(y_orig, med_p),
            'trimmed': rmse(y_orig, trim_p),
            'single_best': best_single,
        }
        for k in summaries:
            cell_ens[k].append(r[k])
        print(f"  seed{SEEDS[si]} cell {cell['cell_id']:2d} eol={cell['eol']:4d}: "
              f"mean={r['mean']:6.1f} median={r['median']:6.1f} "
              f"trimmed={r['trimmed']:6.1f} best_single={best_single:6.1f}", flush=True)
    for k in summaries:
        summaries[k].append(np.mean(cell_ens[k]))

print("\n=== 5-seed averages ===")
for k, v in summaries.items():
    print(f"  {k:12s}: {np.mean(v):6.2f}  (per-seed: {[f'{x:.1f}' for x in v]})")
print("\nReference: single-model own-test 32.33 | paper 16.11")
