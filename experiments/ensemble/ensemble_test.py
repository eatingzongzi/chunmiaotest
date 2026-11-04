#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ensemble of 5 hard-sup models trained on different splits.

Each model trains on its own seed's 24 cells; predictions are averaged
(equal weights, normalised scale) and evaluated on the seed-42 test cells
and on each model's own test cells. Error cancellation across splits is
the hypothesis (paper's own ensemble baseline: 17.03 +/- 4.05).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import rmse, compute_all_metrics

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
    models.append(model)
    loaders.append((test_cells, label_scaler))
    print(f"trained seed {seed}", flush=True)

# ── Ensemble on each seed's own test cells ──────────────────────────
print("\n=== Ensemble evaluation per seed (each model predicts all seeds' tests) ===")
for si, (test_cells, label_scaler) in enumerate(loaders):
    for cell in test_cells:
        x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
        y_orig = cell['y_orig']
        preds = []
        for m in models:
            with torch.no_grad():
                g, *_ = m(x)
            preds.append(label_scaler.inverse_transform(
                g.detach().cpu().numpy().reshape(-1, 1)).ravel())
        ens = np.mean(preds, axis=0)
        single = [rmse(y_orig, p) for p in preds]
        print(f"  seed{SEEDS[si]} cell {cell['cell_id']:2d} eol={cell['eol']:4d}: "
              f"ensemble={rmse(y_orig, ens):6.2f}  singles={[f'{s:.1f}' for s in single]}",
              flush=True)
