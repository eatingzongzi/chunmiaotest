#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure whether the sparse gate actually routes dynamically.

Trains with balance_coeff=0 (the collapsing setting), then reports:
  1. How many distinct top-4 expert COMBINATIONS occur across samples.
     A static gate shows 1 combination for ~100% of samples.
  2. Per test cell: the expert combination chosen at the start of life
     (high RUL) vs near EOL (low RUL) — a real router should differ.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from collections import Counter

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device

device = get_device('cuda')
config = Config()
config.epochs = 100
config.random_seed = 42
config.moe_balance_coeff = 0.0   # reproduce the collapse

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)
model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
model.eval()


def top4_combo(x):
    """Return the sorted top-4 expert tuple for input x (clean logits)."""
    with torch.no_grad():
        _, (h_n, _) = model.lstm(x.to(device))
        clean = model.moe.gate.w_g(h_n[-1])
        _, idx = torch.topk(clean, model.moe.top_k, dim=-1)
        return tuple(sorted(idx[0].cpu().numpy().tolist()))


# ── 1. Combination diversity on the training set ────────────────────
combos = Counter()
with torch.no_grad():
    for i in range(0, len(train_ds.X), 256):
        x = train_ds.X[i:i + 256]
        _, (h_n, _) = model.lstm(x.to(device))
        clean = model.moe.gate.w_g(h_n[-1])
        _, idx = torch.topk(clean, model.moe.top_k, dim=-1)
        for row in idx.cpu().numpy():
            combos[tuple(sorted(row.tolist()))] += 1
print(f"\nDistinct top-4 combinations on train set: {len(combos)}")
for combo, cnt in combos.most_common(8):
    print(f"  {combo}: {cnt} ({cnt/sum(combos.values()):.1%})")

# ── 2. Routing across RUL stages per test cell ──────────────────────
print("\nPer test cell: routing at early life (RUL high) vs near EOL (RUL low)")
for cell in test_cells:
    X = torch.tensor(cell['X'], dtype=torch.float32)
    y = cell['y']
    n = len(X)
    early = top4_combo(X[max(0, n - 30):])          # first window = highest RUL
    late = top4_combo(X[:min(30, n)])               # last window = lowest RUL
    mid = top4_combo(X[n // 2 - 15:n // 2 + 15])
    print(f"  cell {cell['cell_id']:2d} eol={cell['eol']:4d}: "
          f"early={early} mid={mid} late={late}")
