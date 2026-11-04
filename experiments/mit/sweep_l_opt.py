#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Window length re-sweep under the optimised config (hard routing sup).

Baseline: l=30 + hard-sup = 20.90 (seed 42). Earlier (pre-optimisation)
l=20 beat l=30 on several seeds, so retest with the optimised training.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell

device = get_device('cuda')

for l in [20, 40]:
    config = Config()
    config.epochs = 100
    config.random_seed = 42
    config.moe_balance_coeff = 0.05
    config.route_sup_coeff = 1.0
    config.router_mse_coeff = 0.1

    train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=l, n_cells=30, n_test=6, seed=42)
    model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
    per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
    print(f"l={l}: avg RMSE={avg['rmse']:.2f} MAE={avg['mae']:.2f} "
          f"R2={avg['r2']:.4f}", flush=True)
    for i, pc in enumerate(per_cell):
        print(f"    #{i+1} cell_id={pc['cell_id']:2d} eol={pc['eol']:4d} "
              f"RMSE={pc['metrics']['rmse']:.2f}", flush=True)

print("\nReference: l=30 hard-sup 20.90 | paper 16.11")
