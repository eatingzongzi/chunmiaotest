#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leakage probe: fit the Min-Max scaler on train+test combined.

If this drops RMSE from ~23.9 toward the paper's 16.11, then normalisation
leakage (fitting scalers on all cells) could explain the paper's number.
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

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6,
    seed=42, scaler_fit='all')

model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
print(f"\nLeaky scaler (fit on all): avg RMSE={avg['rmse']:.2f} "
      f"MAE={avg['mae']:.2f} R2={avg['r2']:.4f}")
for i, pc in enumerate(per_cell):
    print(f"  #{i+1} cell_id={pc['cell_id']:2d} eol={pc['eol']:4d} "
          f"RMSE={pc['metrics']['rmse']:.2f}")
print("\nReference: clean scaler (fit on train only): 23.89 | paper: 16.11")
