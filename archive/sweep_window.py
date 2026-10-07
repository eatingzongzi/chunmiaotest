#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Window-length sensitivity: l in {20,30,40,50} x seeds {42,43}.

The paper never gives a numeric value for the sliding-window length l
(Section 3.1, Eq. 9), so sweep the plausible range to see which setting
best reproduces the paper's MIT-Stanford RMSE (Table 5: 16.11).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell

L_VALUES = [20, 30, 40, 50]
SEEDS = [42, 43]
device = get_device('cuda')
config = Config()
config.epochs = 100

results = {}
for l in L_VALUES:
    for seed in SEEDS:
        config.random_seed = seed
        train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
            'data/mit_stanford', window_length=l, n_cells=30, n_test=6,
            seed=seed)
        model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
        per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
        results[(l, seed)] = avg
        print(f"l={l} seed={seed}: avg RMSE={avg['rmse']:.2f} "
              f"MAE={avg['mae']:.2f} MAPE={avg['mape']:.2f}% "
              f"R2={avg['r2']:.4f}", flush=True)

print("\n=== Summary (mean over seeds) ===")
for l in L_VALUES:
    vals = [results[(l, s)]['rmse'] for s in SEEDS]
    print(f"l={l}: RMSE {np.mean(vals):.2f} +/- {np.std(vals):.2f}  ({[f'{v:.2f}' for v in vals]})")
print("Paper (Table 5) average: 16.11")
