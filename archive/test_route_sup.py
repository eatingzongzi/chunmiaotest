#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train with routing supervision and measure gate agreement + RMSE.

Baselines (seed 42): gate-mixture 24.70, all-expert mean 23.43,
per-sample oracle expert 19.7, paper 16.11.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

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
config.route_sup_tau = 0.02

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)
model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
model.eval()

# ── Gated mixture RMSE ──────────────────────────────────────────────
per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
print(f"\nGated mixture (route-sup): avg RMSE={avg['rmse']:.2f} "
      f"MAE={avg['mae']:.2f} R2={avg['r2']:.4f}")
for i, pc in enumerate(per_cell):
    print(f"  #{i+1} cell_id={pc['cell_id']:2d} eol={pc['eol']:4d} "
          f"RMSE={pc['metrics']['rmse']:.2f}")

# ── Gate agreement ──────────────────────────────────────────────────
agree, n = 0, 0
for cell in test_cells:
    x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
    y_orig = cell['y_orig']
    with torch.no_grad():
        _, (h_n, _) = model.lstm(x)
        h = h_n[-1]
        errs = []
        for e in range(10):
            out = model.moe.experts[e](h)
            gamma_e, *_ = model.evidence(out)
            pred = label_scaler.inverse_transform(
                gamma_e.detach().cpu().numpy().reshape(-1, 1)).ravel()
            errs.append((pred - y_orig) ** 2)
        errs = np.array(errs).T
        best = errs.argmin(axis=1)
        clean = model.moe.gate.w_g(h)
        _, idx = torch.topk(clean, config.top_k, dim=-1)
        routed = idx.cpu().numpy()
        hit = np.array([best[j] in routed[j] for j in range(len(x))])
        agree += hit.sum(); n += len(x)
print(f"\nGate agreement (best expert among routed top-4): {agree/n:.1%} "
      f"(random = 40%, no-sup = 25.2%)")

# ── Per-sample oracle (upper bound with THIS model) ─────────────────
oracle_rmse = []
for cell in test_cells:
    x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
    y_orig = cell['y_orig']
    with torch.no_grad():
        _, (h_n, _) = model.lstm(x)
        h = h_n[-1]
        errs = []
        for e in range(10):
            out = model.moe.experts[e](h)
            gamma_e, *_ = model.evidence(out)
            pred = label_scaler.inverse_transform(
                gamma_e.detach().cpu().numpy().reshape(-1, 1)).ravel()
            errs.append((pred - y_orig) ** 2)
        oracle_rmse.append(float(np.sqrt(np.min(np.array(errs).T, axis=1).mean())))
print(f"Oracle per-sample expert (this model): avg={np.mean(oracle_rmse):.2f} "
      f"per-cell={[f'{v:.1f}' for v in oracle_rmse]}")
print("\nReference: no-sup gate 24.70 | all-mean 23.43 | paper 16.11")
