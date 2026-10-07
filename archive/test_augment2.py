#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cell-consistent augmentation + l re-check, on the hard-sup base.

Anomalous cells shift the WHOLE lifetime curve (capacity, IC peak), so
the augmentation offset must be one shared factor per cell, not per
sample. Also re-test l=20/40 which looked better pre-optimisation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell

device = get_device('cuda')

CASES = [
    dict(tag='l30_aug0_s43', seed=43, l=30, aug=0.0),
    dict(tag='l30_aug01_s43', seed=43, l=30, aug=0.1),
    dict(tag='l20_aug0_s43', seed=43, l=20, aug=0.0),
    dict(tag='l20_aug0_s42', seed=42, l=20, aug=0.0),
]

for case in CASES:
    config = Config()
    config.epochs = 100
    config.random_seed = case['seed']
    config.moe_balance_coeff = 0.05
    config.route_sup_coeff = 1.0
    config.router_mse_coeff = 0.1

    train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=case['l'], n_cells=30, n_test=6,
        seed=case['seed'], aug_cell_scale=case['aug'])
    model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
    per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
    print(f"\n{case['tag']}: avg RMSE={avg['rmse']:.2f} MAE={avg['mae']:.2f} "
          f"R2={avg['r2']:.4f}", flush=True)
    for i, pc in enumerate(per_cell):
        print(f"  #{i+1} cell_id={pc['cell_id']:2d} eol={pc['eol']:4d} "
              f"RMSE={pc['metrics']['rmse']:.2f}", flush=True)

print("\nReference: s42 l30 no-aug 20.90 | s43 l30 no-aug 40.03 | paper 16.11")
