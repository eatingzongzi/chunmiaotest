#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose RUL distribution + per-cell feature stats (no training needed)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from src.data_loader import load_mit_stanford_paper

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)

print("\n=== label_scaler (fit on train RUL) ===")
print(f"  RUL min={label_scaler.data_min_[0]:.2f}  max={label_scaler.data_max_[0]:.2f}")

train_y_orig = label_scaler.inverse_transform(train_ds.y.numpy().reshape(-1, 1)).ravel()
print("\n=== train RUL distribution (original units) ===")
print(f"  n={len(train_y_orig)}  min={train_y_orig.min():.1f}  mean={train_y_orig.mean():.1f}  max={train_y_orig.max():.1f}")
hist, edges = np.histogram(train_y_orig, bins=10)
for i in range(len(hist)):
    print(f"  [{edges[i]:6.0f}, {edges[i+1]:6.0f}) : {hist[i]:5d}")

print("\n=== test cells RUL range (original units) ===")
for i, c in enumerate(test_cells):
    y = c['y_orig']
    print(f"  #{i+1} cell_id={c['cell_id']:2d} eol={c['eol']:4d}  RUL=[{y.min():6.0f}, {y.max():6.0f}]")

print("\n=== per-cell feature deviation vs train (normalized space) ===")
train_X = train_ds.X.numpy()
train_mean = train_X.reshape(-1, 16).mean(0)
train_std = train_X.reshape(-1, 16).std(0)
feat_names = ['v1','v2','v3','v4','v5','v6','c1','c2','c3','c4','c5','c6',
              'ic_peak','ic_pos','dcir','maxq']
for i, c in enumerate(test_cells):
    flat = c['X'].reshape(-1, 16)
    mn, mx = flat.min(0), flat.max(0)
    out_of_range = int(((mx > 1.0) | (mn < 0.0)).sum())
    zscore = (flat.mean(0) - train_mean) / (train_std + 1e-8)
    top = np.argsort(-np.abs(zscore))[:4]
    top_str = ", ".join(f"{feat_names[j]}(z={zscore[j]:+.1f})" for j in top)
    print(f"  #{i+1} cell_id={c['cell_id']:2d}: out_of_range={out_of_range}/16  |  top-deviating: {top_str}")
