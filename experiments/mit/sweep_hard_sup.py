#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sweep hard routing-supervision strength and gate capacity.

Baseline hard-sup (coeff=1.0, linear gate): RMSE 20.90, agreement 44.6%.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell

device = get_device('cuda')
train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)

CASES = [
    dict(name='coeff3_lin', route_sup_coeff=3.0, gate_hidden=0),
    dict(name='coeff5_lin', route_sup_coeff=5.0, gate_hidden=0),
    dict(name='coeff3_mlp64', route_sup_coeff=3.0, gate_hidden=64),
]

for case in CASES:
    config = Config()
    config.epochs = 100
    config.random_seed = 42
    config.moe_balance_coeff = 0.05
    config.route_sup_coeff = case['route_sup_coeff']
    config.gate_hidden = case['gate_hidden']
    model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
    model.eval()
    per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)

    # gate agreement
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
            clean = model.moe.gate.clean_logits(gate_in)
            _, idx = torch.topk(clean, config.top_k, dim=-1)
            routed = idx.cpu().numpy()
            hit = np.array([best[j] in routed[j] for j in range(len(x))])
            agree += hit.sum(); n += len(x)
    print(f"{case['name']}: RMSE={avg['rmse']:.2f} MAE={avg['mae']:.2f} "
          f"R2={avg['r2']:.4f} agreement={agree/n:.1%}", flush=True)
    pcs = [f"{c['metrics']['rmse']:.1f}" for c in per_cell]
    print(f"  per-cell: {pcs}", flush=True)

print("\nReference: coeff1.0 lin gate: 20.90 (44.6%) | paper 16.11")
