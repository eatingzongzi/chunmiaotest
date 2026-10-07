#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train on 42 of the 48 cells (vs the paper's 24) — coverage attack.

The 5-seed bottleneck is that each split puts a cell type in the test
set that the 24 train cells never covered. Training on 42 cells covers
nearly all degradation patterns. Evaluation stays per-cell on 6 test
cells per seed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell

device = get_device('cuda')

for seed in [42, 43, 44, 45, 46]:
    config = Config()
    config.epochs = 100
    config.random_seed = seed
    config.moe_balance_coeff = 0.05
    config.route_sup_coeff = 1.0
    config.router_mse_coeff = 0.1

    train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=seed)
    model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
    per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
    print(f"\nseed {seed}: avg RMSE={avg['rmse']:.2f} MAE={avg['mae']:.2f} "
          f"R2={avg['r2']:.4f}", flush=True)
    for i, pc in enumerate(per_cell):
        print(f"  #{i+1} cell_id={pc['cell_id']:2d} eol={pc['eol']:4d} "
              f"RMSE={pc['metrics']['rmse']:.2f}", flush=True)

print("\nReference: 24-cell training 5-seed 32.33+/-7.03 | paper 16.11")
