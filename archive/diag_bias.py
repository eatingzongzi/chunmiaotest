#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train 1 model (trial 0) and report per-cell prediction bias direction."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.model import SMoELSTM

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)

device = get_device('cuda')
config = Config()
config.epochs = 100
config.window_length = 30
model, _ = train_model(config, train_ds, device, trial=0, verbose=False)

model.eval()

def predict_cell(c):
    X = torch.tensor(c['X'], dtype=torch.float32).to(device)
    gamma, nu, alpha, beta = model(X)
    pred_norm = SMoELSTM.predict(gamma, nu, alpha, beta)['pred'].cpu().numpy().ravel()
    return label_scaler.inverse_transform(pred_norm.reshape(-1, 1)).ravel()

print("\n=== per-cell prediction bias (pred - true, orig units) ===")
for i, c in enumerate(test_cells):
    pred = predict_cell(c)
    y = c['y_orig']
    resid = pred - y
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    bias = float(resid.mean())
    print(f"  #{i+1} cell_id={c['cell_id']:2d} eol={c['eol']:4d}: "
          f"bias={bias:+7.2f}  RMSE={rmse:6.2f}  "
          f"pred=[{pred.min():.0f},{pred.max():.0f}] true=[{y.min():.0f},{y.max():.0f}]")

print("\n=== RUL-binned residual for #5 and #6 ===")
bins = [0, 100, 200, 300, 400, 500]
for idx in [4, 5]:
    c = test_cells[idx]
    pred = predict_cell(c)
    y = c['y_orig']
    print(f"  #{idx+1} (cell_id={c['cell_id']}, eol={c['eol']}):")
    for b in range(len(bins) - 1):
        m = (y >= bins[b]) & (y < bins[b + 1])
        if m.sum() > 0:
            r = (pred - y)[m]
            print(f"    RUL [{bins[b]:3d},{bins[b+1]:3d}): n={m.sum():3d}  "
                  f"mean_bias={r.mean():+7.2f}  rmse={np.sqrt(np.mean(r**2)):6.2f}")
