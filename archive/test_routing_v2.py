#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregation matrix on route-sup-sharpened experts.

The learned gate is a bad router (33.7% agreement, random=40%), so
compare deterministic alternatives on the sharpened experts:
  A. gate top-4 hard            (current model)
  B. all-10 equal mean
  C. gate full-soft weighting   (dense routing, no sparsity)
  D. RUL-bucket routing         (rough RUL -> nearest territory experts)
  O. oracle per-sample expert   (upper bound, was 11.52)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import rmse, evaluate_per_cell

device = get_device('cuda')
config = Config()
config.epochs = 100
config.random_seed = 42
config.moe_balance_coeff = 0.05
config.route_sup_coeff = 1.0

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)
model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
model.eval()

# ── Expert territories (clean top-4, train set) ─────────────────────
routed_ruls = {e: [] for e in range(10)}
with torch.no_grad():
    for i in range(0, len(train_ds.X), 512):
        x = train_ds.X[i:i + 512].to(device)
        rul = label_scaler.inverse_transform(
            train_ds.y[i:i + 512].numpy().reshape(-1, 1)).ravel()
        _, (h_n, _) = model.lstm(x)
        clean = model.moe.gate.w_g(h_n[-1])
        _, idx = torch.topk(clean, config.top_k, dim=-1)
        for j, row in enumerate(idx.cpu().numpy()):
            for e in row:
                routed_ruls[e].append(rul[j])
territory = {e: float(np.median(routed_ruls[e])) for e in range(10)}
print("Territory medians:", {e: round(v) for e, v in territory.items()})

def to_rul(g):
    return label_scaler.inverse_transform(g.detach().cpu().numpy().reshape(-1, 1)).ravel()

results = {k: [] for k in ['A_gate', 'B_all_mean', 'C_soft', 'D_bucket', 'O_oracle']}
for cell in test_cells:
    x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
    y_orig = cell['y_orig']
    with torch.no_grad():
        _, (h_n, _) = model.lstm(x)
        h = h_n[-1]
        outs = torch.stack([model.moe.experts[e](h) for e in range(10)], dim=1)  # (N,10,H)
        clean = model.moe.gate.w_g(h)
        prob = F.softmax(clean, dim=-1)                          # (N,10)

    # A: model's own sparse mixture
    gammaA, *_ = model(x)
    results['A_gate'].append(rmse(y_orig, to_rul(gammaA)))

    # B: all-10 equal mean
    gammaB, *_ = model.evidence(outs.mean(dim=1))
    results['B_all_mean'].append(rmse(y_orig, to_rul(gammaB)))

    # C: full-soft weighted mixture (dense routing)
    mixed = (outs * prob.unsqueeze(-1)).sum(dim=1)
    gammaC, *_ = model.evidence(mixed)
    results['C_soft'].append(rmse(y_orig, to_rul(gammaC)))

    # D: RUL-bucket routing — rough RUL from the soft mixture, then the
    #    4 experts with nearest territory medians, equal weights
    rough = to_rul(gammaC)
    meds = np.array([territory[e] for e in range(10)])
    selD = np.argsort(np.abs(meds[None, :] - rough[:, None]), axis=1)[:, :4]
    selD = torch.tensor(selD, device=device)
    picked = torch.gather(outs, 1, selD.unsqueeze(-1).expand(-1, -1, outs.shape[-1]))
    gammaD, *_ = model.evidence(picked.mean(dim=1))
    results['D_bucket'].append(rmse(y_orig, to_rul(gammaD)))

    # O: oracle per-sample best expert
    errs = []
    for e in range(10):
        gammaE, *_ = model.evidence(outs[:, e])
        pred = to_rul(gammaE)
        errs.append((pred - y_orig) ** 2)
    oracle = np.min(np.array(errs).T, axis=1)
    results['O_oracle'].append(float(np.sqrt(oracle.mean())))

print("\n=== Aggregation comparison (avg over 6 test cells) ===")
for k, v in results.items():
    print(f"  {k:10s}: {np.mean(v):6.2f}   (per-cell: {[f'{x:.1f}' for x in v]})")
print("\nReference: paper 16.11 | pure LSTM 22.76 | learned gate 22.77")
