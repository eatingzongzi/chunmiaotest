#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""5-seed validation of the optimised model (hard routing supervision).

Best seed-42 result so far: RMSE 20.90. This checks cross-split stability
of the optimised config (vs the paper's 16.49 +/- 5.62 over 10 trials).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell

SEEDS = [42, 43, 44, 45, 46]
device = get_device('cuda')

# optimised config (seed-42 best: 20.90)
KWARGS = dict(moe_balance_coeff=0.05, route_sup_coeff=1.0,
              router_mse_coeff=0.1, gate_hidden=0)

all_avg = []
for seed in SEEDS:
    config = Config()
    config.epochs = 100
    config.random_seed = seed
    for k, v in KWARGS.items():
        setattr(config, k, v)
    train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=seed)
    model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
    per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
    all_avg.append(avg['rmse'])
    print(f"seed {seed}: avg RMSE={avg['rmse']:.2f} MAE={avg['mae']:.2f} "
          f"R2={avg['r2']:.4f}", flush=True)
    for i, pc in enumerate(per_cell):
        print(f"    #{i+1} cell_id={pc['cell_id']:2d} eol={pc['eol']:4d} "
              f"RMSE={pc['metrics']['rmse']:.2f}", flush=True)

print(f"\nOptimised config over {len(SEEDS)} seeds: "
      f"{np.mean(all_avg):.2f} +/- {np.std(all_avg):.2f}")
print("Baselines: paper 16.11 | paper-protocol repro 39.07 | seed42-only 20.90")
