#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gate with explicit rough-RUL input (sparse top-4 kept).

The gate previously failed to learn routing from the LSTM hidden state
alone (33.7% agreement < random 40%). Now the gate input is
[h, rough_RUL], where rough_RUL comes from a supervised 1-dim router
head — experts specialise by RUL stage, so this signal should make
routing learnable while keeping the sparse top-4 mechanism.
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
config.router_mse_coeff = 0.1

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)
model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
model.eval()

per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
print(f"\nGated mixture (rough-RUL gate): avg RMSE={avg['rmse']:.2f} "
      f"MAE={avg['mae']:.2f} R2={avg['r2']:.4f}")
for i, pc in enumerate(per_cell):
    print(f"  #{i+1} cell_id={pc['cell_id']:2d} eol={pc['eol']:4d} "
          f"RMSE={pc['metrics']['rmse']:.2f}")

# gate agreement (with the new gate input)
agree, n = 0, 0
for cell in test_cells:
    x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
    y_orig = cell['y_orig']
    with torch.no_grad():
        _, (h_n, _) = model.lstm(x)
        h = h_n[-1]
        rough = model.router(h)
        gate_in = torch.cat([h, rough], dim=-1)
        errs = []
        for e in range(10):
            out = model.moe.experts[e](h)
            gamma_e, *_ = model.evidence(out)
            pred = label_scaler.inverse_transform(
                gamma_e.detach().cpu().numpy().reshape(-1, 1)).ravel()
            errs.append((pred - y_orig) ** 2)
        errs = np.array(errs).T
        best = errs.argmin(axis=1)
        clean = model.moe.gate.w_g(gate_in)
        _, idx = torch.topk(clean, config.top_k, dim=-1)
        routed = idx.cpu().numpy()
        hit = np.array([best[j] in routed[j] for j in range(len(x))])
        agree += hit.sum(); n += len(x)
print(f"Gate agreement: {agree/n:.1%} (random=40%, h-only gate=33.7%)")

# oracle bound with this model
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
print(f"Oracle per-sample expert: avg={np.mean(oracle_rmse):.2f} "
      f"per-cell={[f'{v:.1f}' for v in oracle_rmse]}")
print("\nReference: h-only gate 22.77 | oracle 11.52 | paper 16.11")
