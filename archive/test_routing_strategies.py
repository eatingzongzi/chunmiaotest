#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare expert combination strategies with one trained model.

The gate routes worse than random (34% vs 40% agreement), so test
simpler/stronger aggregation of the 10 specialised experts:
  A. gate-weighted top-4        (current)
  B. mean of ALL 10 experts
  C. mean of top-4 experts (by clean logits, no gate weights)
  D. RUL-bucket routing: predict a rough RUL first (gamma from all-expert
     mean), then average the experts whose training RUL territory covers
     that bucket
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import rmse

device = get_device('cuda')
config = Config()
config.epochs = 100
config.random_seed = 42
config.moe_balance_coeff = 0.05

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)
model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
model.eval()

# ── Expert RUL territory medians (from the train set, clean top-4) ──
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

# ── Evaluate combination strategies per test cell ───────────────────
def expert_outputs(x):
    with torch.no_grad():
        _, (h_n, _) = model.lstm(x)
        h = h_n[-1]
        outs = torch.stack([model.moe.experts[e](h) for e in range(10)], dim=1)  # (N,10,H)
        return h, outs

def to_rul(gamma):
    return label_scaler.inverse_transform(
        gamma.detach().cpu().numpy().reshape(-1, 1)).ravel()

results = {k: [] for k in ['A_gate', 'B_all_mean', 'C_top4_mean', 'D_rul_bucket']}
for cell in test_cells:
    x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
    y_orig = cell['y_orig']  # un-normalised RUL
    h, outs = expert_outputs(x)
    N = len(x)

    # A: gate-weighted top-4 (the model's own mixture)
    gamma, nu, alpha, beta = model(x)
    predA = to_rul(gamma)
    results['A_gate'].append(rmse(y_orig, predA))

    # B: mean of all 10 expert outputs -> evidence head
    mean_all = outs.mean(dim=1)
    gammaB, *_ = model.evidence(mean_all)
    results['B_all_mean'].append(rmse(y_orig, to_rul(gammaB)))

    # C: mean of top-4 expert outputs by clean logits (equal weights)
    with torch.no_grad():
        clean = model.moe.gate.w_g(h)
        topk_idx = torch.topk(clean, config.top_k, dim=-1).indices  # (N,4)
        sel = torch.gather(outs, 1, topk_idx.unsqueeze(-1).expand(-1, -1, outs.shape[-1]))
    gammaC, *_ = model.evidence(sel.mean(dim=1))
    results['C_top4_mean'].append(rmse(y_orig, to_rul(gammaC)))

    # D: RUL-bucket routing — rough RUL from all-expert mean, then average
    #    the 4 experts whose territory median is nearest to that RUL
    rough = to_rul(gammaB)
    meds = np.array([territory[e] for e in range(10)])
    selD = []
    for r in rough:
        order = np.argsort(np.abs(meds - r))[:4]
        selD.append(order)
    selD = torch.tensor(np.array(selD), device=device)
    picked = torch.gather(outs, 1, selD.unsqueeze(-1).expand(-1, -1, outs.shape[-1]))
    gammaD, *_ = model.evidence(picked.mean(dim=1))
    results['D_rul_bucket'].append(rmse(y_orig, to_rul(gammaD)))

print("\n=== Strategy comparison (avg over 6 test cells) ===")
for k, v in results.items():
    print(f"  {k:14s}: {np.mean(v):6.2f}   (per-cell: {[f'{x:.1f}' for x in v]})")
print("\nReference: paper 16.11 | pure LSTM 22.76")
