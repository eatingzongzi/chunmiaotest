#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature augmentation for anomalous-cell generalisation.

Baseline hard-sup: seed 42 = 20.90, seed 43 = 40.03 (cell 19 -> 80.2).
Perturbing F1/F2/F4 during training should teach the model that feature
offsets are not lifetime offsets.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell

device = get_device('cuda')

for seed, aug in [(42, 0.1), (43, 0.1), (43, 0.2)]:
    config = Config()
    config.epochs = 100
    config.random_seed = seed
    config.moe_balance_coeff = 0.05
    config.route_sup_coeff = 1.0
    config.router_mse_coeff = 0.1
    config.aug_scale = aug

    train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=seed)
    model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
    per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
    print(f"\nseed={seed} aug={aug}: avg RMSE={avg['rmse']:.2f} "
          f"MAE={avg['mae']:.2f} R2={avg['r2']:.4f}", flush=True)
    for i, pc in enumerate(per_cell):
        print(f"  #{i+1} cell_id={pc['cell_id']:2d} eol={pc['eol']:4d} "
              f"RMSE={pc['metrics']['rmse']:.2f}", flush=True)

print("\nReference: seed42 no-aug 20.90 | seed43 no-aug 40.03 | paper 16.11")
