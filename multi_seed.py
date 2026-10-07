#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the paper protocol across multiple cell-split seeds (1 trial each)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell

SEEDS = [42, 43, 44, 45, 46]
device = get_device('cuda')
config = Config()
config.epochs = 100
config.window_length = 30

all_avg = []
for seed in SEEDS:
    train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=seed)
    config.random_seed = seed
    model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
    per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
    all_avg.append(avg['rmse'])
    print(f"seed {seed}: avg RMSE={avg['rmse']:.2f} MAE={avg['mae']:.2f} "
          f"MAPE={avg['mape']:.2f}% R2={avg['r2']:.4f}")
    for i, pc in enumerate(per_cell):
        print(f"    #{i+1} cell_id={pc['cell_id']:2d} eol={pc['eol']:4d} "
              f"RMSE={pc['metrics']['rmse']:.2f}")

print(f"\nMean over {len(SEEDS)} seeds: {np.mean(all_avg):.2f} +/- {np.std(all_avg):.2f}")
print(f"Paper (Table 5) average: 16.11")
