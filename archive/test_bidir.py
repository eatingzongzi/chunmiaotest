#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bidirectional LSTM + hard routing supervision (seed 42).

The 5-seed run showed cross-split generalisation is the bottleneck
(32.33 +/- 7.03). A bidirectional encoder sees the whole 30-step window
from both directions, typically a solid generalisation gain.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell

device = get_device('cuda')
config = Config()
config.epochs = 100
config.random_seed = 42
config.moe_balance_coeff = 0.05
config.route_sup_coeff = 1.0
config.router_mse_coeff = 0.1
config.bidirectional = True

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)
model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
print(f"\nBidirectional + hard-sup: avg RMSE={avg['rmse']:.2f} "
      f"MAE={avg['mae']:.2f} R2={avg['r2']:.4f}")
for i, pc in enumerate(per_cell):
    print(f"  #{i+1} cell_id={pc['cell_id']:2d} eol={pc['eol']:4d} "
          f"RMSE={pc['metrics']['rmse']:.2f}")
print("\nReference: unidirectional hard-sup 20.90 | paper 16.11")
