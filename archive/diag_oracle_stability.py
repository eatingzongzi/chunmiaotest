#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Is the per-sample 'best expert' a learnable signal or noise?

If neighbouring samples (same cell, adjacent cycles) consistently pick
the same best expert, routing is learnable. If the best expert jumps
randomly between neighbours, the oracle target is noise-champion
selection and no gate can learn it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device

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
    errs = np.array(errs).T              # (N, 10)
    best = errs.argmin(axis=1)           # (N,)

    # neighbour agreement: same best expert at adjacent samples
    same = (best[:-1] == best[1:]).mean()
    # gap between best and second-best error
    sorted_err = np.sort(errs, axis=1)
    gap = (sorted_err[:, 1] - sorted_err[:, 0]) / sorted_err[:, 0]
    # distribution of best experts
    import collections
    cnt = collections.Counter(best.tolist())
    top = cnt.most_common(3)
    print(f"cell {cell['cell_id']:2d} eol={cell['eol']:4d}: "
          f"neighbour-agreement={same:.1%}  "
          f"median gap-to-2nd={np.median(gap):.3f}  "
          f"top best-experts={[(e, f'{c/len(best):.0%}') for e, c in top]}")

# aggregate
all_best, all_adj_same = [], []
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
    all_best.append(best)
    all_adj_same.append((best[:-1] == best[1:]).mean())
print(f"\nOverall neighbour agreement: {np.mean(all_adj_same):.1%} "
      f"(random baseline ~{1/10:.0%})")
