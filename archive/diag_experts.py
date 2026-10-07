#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Expert specialisation analysis (optimisation-oriented).

Trains a balanced model (coeff=0.05), then for each of the 10 experts:
  A. RUL territory on the training set (which RUL range routes to it)
  B. Standalone per-cell RMSE on the test set (how good is each expert
     alone, cell by cell)
  C. Gate agreement: does the gate route each sample to experts that
     are actually good for that sample?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import rmse, evaluate_per_cell

device = get_device('cuda')
config = Config()
config.epochs = 100
config.random_seed = 42
config.moe_balance_coeff = 0.05

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)
model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
model.eval()

# ── A. Expert RUL territories (train set) ────────────────────────────
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

print("\n=== A. Expert RUL territories on train set (clean top-4) ===")
for e in range(10):
    r = np.array(routed_ruls[e])
    if len(r) == 0:
        print(f"  E{e}: no samples"); continue
    print(f"  E{e}: n={len(r):5d}  RUL mean={r.mean():6.1f}  "
          f"p25={np.percentile(r,25):5.1f}  p50={np.percentile(r,50):5.1f}  "
          f"p75={np.percentile(r,75):5.1f}")

# ── B. Per-expert standalone RMSE per test cell ─────────────────────
print("\n=== B. Standalone per-cell RMSE: each expert alone vs combined ===")
print("  (rows: test cells, cols: E0..E9 | comb = gated mixture | best = best single expert)")
hdr = "  cell  eol | " + " ".join(f"E{e:>6d}" for e in range(10)) + " |   comb |   best"
print(hdr)
for cell in test_cells:
    x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
    y_orig = cell['y_orig']  # un-normalised RUL
    with torch.no_grad():
        _, (h_n, _) = model.lstm(x)
        h = h_n[-1]
        per = []
        for e in range(10):
            out = model.moe.experts[e](h)
            gamma, nu, alpha, beta = model.evidence(out)
            pred = label_scaler.inverse_transform(
                gamma.cpu().numpy().reshape(-1, 1)).ravel()
            per.append(rmse(y_orig, pred))
    per_cell_res, avg = evaluate_per_cell(model, [cell], device, label_scaler)
    comb = per_cell_res[0]['metrics']['rmse']
    best = min(per)
    cells = f"  {cell['cell_id']:4d} {cell['eol']:4d} |"
    for p in per:
        cells += f" {p:6.1f}"
    print(f"{cells} | {comb:6.1f} | {best:6.1f}")

# ── C. Gate agreement on test samples ───────────────────────────────
print("\n=== C. Gate agreement: does routing hit the best experts? ===")
# For each test sample: rank experts by standalone squared error vs
# gate's top-4. Report how often the best expert is among the routed 4.
all_agree = 0
all_n = 0
for cell in test_cells:
    x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
    y_orig = cell['y_orig']  # un-normalised RUL
    with torch.no_grad():
        _, (h_n, _) = model.lstm(x)
        h = h_n[-1]
        errs = []
        for e in range(10):
            out = model.moe.experts[e](h)
            gamma, nu, alpha, beta = model.evidence(out)
            pred = label_scaler.inverse_transform(
                gamma.cpu().numpy().reshape(-1, 1)).ravel()
            errs.append((pred - y_orig) ** 2)
        errs = np.array(errs).T          # (N, 10)
        best_exp = errs.argmin(axis=1)   # best expert per sample
        clean = model.moe.gate.w_g(h)
        _, idx = torch.topk(clean, config.top_k, dim=-1)
        routed = idx.cpu().numpy()       # (N, 4)
        hit = np.array([best_exp[j] in routed[j] for j in range(len(x))])
        all_agree += hit.sum(); all_n += len(x)
print(f"  Best expert is among the routed top-4: {all_agree/all_n:.1%} of test samples")
print(f"  (random routing would give {config.top_k/10:.0%})")
