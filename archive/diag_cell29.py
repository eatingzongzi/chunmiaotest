#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep-dive on the anomalous test cell 29 (eol=498, RMSE ~48.9).

It is the only test cell far from paper level (oracle bound 16.8). Quantify:
  1. feature-space distance of cell 29 to the 24 train cells (z-scores)
  2. per-RUL-bucket prediction bias of cell 29
  3. which experts handle cell 29 best and what they see
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import h5py
import torch
from sklearn.preprocessing import MinMaxScaler

from config import Config
from src.data_loader import (sliding_window_sampling, _extract_cell_features_paper,
                              load_mit_stanford_paper)
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

# ── 1. Feature distance: cell 29 vs train cells ─────────────────────
batch_file = '2017-06-30_batchdata_updated_struct_errorcorrect.mat'
f = h5py.File(os.path.join('data/mit_stanford', batch_file), 'r')
batch = f['batch']
all_feats = {}
for i in range(batch['summary'].shape[0]):
    try:
        summ = f[batch['summary'][i, 0]]
        cyc = f[batch['cycles'][i, 0]]
        cl = float(f[batch['cycle_life'][i, 0]][()].flatten()[0])
        vdlin = None
        try:
            vdlin = f[batch['Vdlin'][i, 0]][()].flatten().astype(np.float64)
        except Exception:
            pass
        feats, cn = _extract_cell_features_paper(f, summ, cyc, cl, vdlin)
        valid = cn <= int(cl)
        all_feats[i] = feats[valid]
    except Exception:
        continue
f.close()

feat_names = ['V1','V2','V3','V4','V5','V6','I1','I2','I3','I4','I5','I6',
              'F1_peak','F2_peakV','F3_IR','F4_Qmax']
cell29 = all_feats[29].mean(axis=0)
train_ids = [i for i in all_feats if i != 29 and i not in
             (13, 17, 24, 25, 34, 29)]
train_means = np.array([all_feats[i].mean(axis=0) for i in train_ids])
train_std = train_means.std(axis=0) + 1e-9
z = (cell29 - train_means.mean(axis=0)) / train_std
print("\n=== 1. Cell 29 mean-feature z-scores vs train cells ===")
for name, zi in zip(feat_names, z):
    bar = '*' * int(min(abs(zi) * 5, 30))
    print(f"  {name:9s}: z={zi:+6.1f}  {bar}")

# ── 2. Per-RUL-bucket bias for cell 29 ──────────────────────────────
cell29_test = [c for c in test_cells if c['cell_id'] == 29][0]
x = torch.tensor(cell29_test['X'], dtype=torch.float32).to(device)
y_orig = cell29_test['y_orig']
with torch.no_grad():
    gamma, nu, alpha, beta = model(x)
    pred = label_scaler.inverse_transform(gamma.detach().cpu().numpy().reshape(-1, 1)).ravel()
print("\n=== 2. Cell 29 per-RUL-bucket bias (pred - true) ===")
buckets = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]
for lo, hi in buckets:
    m = (y_orig >= lo) & (y_orig < hi)
    if m.sum():
        bias = (pred[m] - y_orig[m]).mean()
        rmse_b = float(np.sqrt(((pred[m] - y_orig[m]) ** 2).mean()))
        print(f"  RUL [{lo:3d},{hi:3d}): n={m.sum():3d} bias={bias:+7.1f} rmse={rmse_b:6.1f}")

# ── 3. Best experts for cell 29 by RUL bucket ───────────────────────
with torch.no_grad():
    _, (h_n, _) = model.lstm(x)
    h = h_n[-1]
    errs = []
    for e in range(10):
        out = model.moe.experts[e](h)
        gamma_e, *_ = model.evidence(out)
        pred_e = label_scaler.inverse_transform(
            gamma_e.detach().cpu().numpy().reshape(-1, 1)).ravel()
        errs.append((pred_e - y_orig) ** 2)
errs = np.array(errs).T
print("\n=== 3. Best expert for cell 29 per RUL bucket ===")
for lo, hi in buckets:
    m = (y_orig >= lo) & (y_orig < hi)
    if m.sum():
        best = errs[m].argmin(axis=1)
        import collections
        cnt = collections.Counter(best.tolist())
        print(f"  RUL [{lo:3d},{hi:3d}): best expert distribution: "
              f"{dict(sorted(cnt.items()))}")
