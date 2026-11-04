#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Our model under the SAME per-battery protocol as the NASA model.

Per battery: sliding-window 16 features (l=30) -> capacity label
(QDischarge/QDischarge[0], the SoH view). Train a compact LSTM on the
first 60% of windows, predict capacity for the last 40% of cycles.
Directly comparable to bayes_pinn_pytorch.py (RMSE in Ah + MAPE).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import h5py
import torch
import torch.nn as nn

from src.data_loader import sliding_window_sampling, _extract_cell_features_paper
from src.train import set_seed, get_device

device = get_device('cuda')


class SmallLSTM(nn.Module):
    """Compact LSTM capacity predictor (comparable capacity to the NASA MLP)."""

    def __init__(self, input_dim=16, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


# ── Build per-cell window features with capacity labels ─────────────
f = h5py.File('data/mit_stanford/2017-06-30_batchdata_updated_struct_errorcorrect.mat', 'r')
batch = f['batch']
cells = {}
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
        qd = summ['QDischarge'][()].flatten().astype(np.float64)
        valid = cn <= int(cl)
        if np.sum(valid) < 60:
            continue
        feats, cn, qd = feats[valid], cn[valid], qd[valid]
        # capacity labels aligned with features (same cycle index)
        X, _ = sliding_window_sampling(feats, cn, int(cl), 30)
        # label for window i = capacity at window END cycle = qd[i+29]
        y = np.array([qd[i + 29] for i in range(len(X))]) / qd[0]
        cells[i] = (X, y)
    except Exception:
        continue
f.close()
print(f"{len(cells)} cells with window samples")

rows = []
for cid in sorted(cells):
    X, y = cells[cid]
    n = len(X)
    n_tr = max(2, int(n * 0.6))
    Xtr = torch.tensor(X[:n_tr], dtype=torch.float32)
    ytr = torch.tensor(y[:n_tr], dtype=torch.float32).unsqueeze(-1)

    set_seed(cid)
    model = SmallLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(300):
        model.train()
        opt.zero_grad()
        pred = model(Xtr.to(device))
        loss = nn.functional.mse_loss(pred, ytr.to(device))
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_all = model(torch.tensor(X, dtype=torch.float32).to(device)
                         ).cpu().squeeze(-1).numpy()
    y_tail = y[n_tr:]
    p_tail = pred_all[n_tr:]
    rmse = float(np.sqrt(np.mean((p_tail - y_tail) ** 2)))
    mape = float(np.mean(np.abs((p_tail - y_tail) / y_tail)) * 100)
    rows.append((cid, rmse, mape))
    print(f"cell {cid:2d}: tail RMSE={rmse:.4f} (SOH) MAPE={mape:.2f}%", flush=True)

rmse_all = np.mean([r[1] for r in rows])
mape_all = np.mean([r[2] for r in rows])
print(f"\nOur model per-battery protocol on MIT data ({len(rows)} cells):")
print(f"  mean RMSE={rmse_all:.4f} (SOH fraction)  mean MAPE={mape_all:.2f}%")
np.save('our_model_results.npy', np.array(rows), allow_pickle=True)
