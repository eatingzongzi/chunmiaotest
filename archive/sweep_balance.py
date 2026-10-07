#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sweep the MoE load-balancing coefficient on a single seed-42 run.

Reports expert load share and test RMSE per coefficient, so we can see
whether stronger balancing helps or hurts (0.01 already tested: RMSE 26.30).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell

COEFFS = [0.05, 0.1, 0.2]
device = get_device('cuda')
config = Config()
config.epochs = 100
config.random_seed = 42

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)

for coeff in COEFFS:
    config.moe_balance_coeff = coeff
    model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
    model.eval()

    # Expert load share on the training set (clean logits)
    loads = np.zeros(config.num_experts)
    with torch.no_grad():
        for i in range(0, len(train_ds.X), 512):
            x = train_ds.X[i:i + 512].to(device)
            _, (h_n, _) = model.lstm(x)
            clean = model.moe.gate.w_g(h_n[-1])
            _, idx = torch.topk(clean, config.top_k, dim=-1)
            for k in range(config.top_k):
                np.add.at(loads, idx[:, k].cpu().numpy(), 1)
    load_str = " ".join(f"E{e}:{loads[e]/loads.sum():.2f}" for e in range(config.num_experts))

    per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
    print(f"coeff={coeff}: avg RMSE={avg['rmse']:.2f} MAE={avg['mae']:.2f} "
          f"R2={avg['r2']:.4f}", flush=True)
    print(f"  loads: {load_str}", flush=True)

print("\nReference: no balancing (collapse): 23.89 | coeff=0.01: 26.30 | paper: 16.11")
