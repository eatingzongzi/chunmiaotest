#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leak-free eVTOL ensemble.

Previous eVTOL/MIT ensembles mixed models that had TRAINED on the test
cells (different random splits), inflating scores. Here each split fixes
the test cells for ALL ensemble members; diversity comes only from
initialisation/training order (5 seeds per split). Two fixed splits.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from config import Config
from src.train import train_model, get_device
from src.evaluate import rmse
from src.data_loader import BatteryDataset

EOL_THR = 0.85
L = 30
device = get_device('cuda')


def load_cells():
    d = np.load('data/evtol_features.npy', allow_pickle=True).item()
    cells = {}
    for cid, v in d.items():
        soh, cyc = v['soh'], v['cycle']
        below = np.where(soh < EOL_THR)[0]
        if len(below) == 0:
            continue
        i = below[0]
        eol = (cyc[i - 1] + (soh[i - 1] - EOL_THR) /
               (soh[i - 1] - soh[i]) * (cyc[i] - cyc[i - 1])) if i > 0 else cyc[0]
        keep = cyc <= eol
        X, c = v['X'][keep].copy(), cyc[keep]
        if len(X) < L + 10:
            continue
        cum_q = np.cumsum(X[:, 24]); cum_e = np.cumsum(X[:, 25])
        X = np.concatenate([X, np.stack([c.astype(float), cum_q, cum_e], 1)], 1)
        rul = eol - c
        Xw = np.stack([X[j:j + L] for j in range(len(X) - L + 1)], 0)
        yw = np.array([rul[j + L - 1] for j in range(len(X) - L + 1)])
        cells[cid] = {'X': Xw.astype(np.float32), 'y': yw.astype(np.float32),
                      'eol': float(eol)}
    return cells


cells = load_cells()
ids = sorted(cells)
print(f"{len(ids)} cells")

# Two fixed, disjoint-test splits (test cells never seen by any member)
SPLITS = {
    'A': ['VAH01', 'VAH02', 'VAH13', 'VAH10'],
    'B': ['VAH28', 'VAH10', 'VAH05', 'VAH24'],
}
N_MODELS = 5

for sname, test_ids in SPLITS.items():
    train_ids = [c for c in ids if c not in test_ids]
    Xtr = np.concatenate([cells[c]['X'] for c in train_ids])
    ytr = np.concatenate([cells[c]['y'] for c in train_ids])
    N, Lw, D = Xtr.shape
    fs = MinMaxScaler().fit(Xtr.reshape(-1, D))
    Xtr_n = fs.transform(Xtr.reshape(-1, D)).reshape(N, Lw, D)
    ls = MinMaxScaler().fit(ytr.reshape(-1, 1))
    ytr_n = ls.transform(ytr.reshape(-1, 1)).ravel()

    models = []
    for k in range(N_MODELS):
        config = Config()
        config.input_dim = D
        config.epochs = 100
        config.random_seed = 42 + k
        config.moe_balance_coeff = 0.05
        config.route_sup_coeff = 1.0
        config.router_mse_coeff = 0.1
        m, _ = train_model(config, BatteryDataset(Xtr_n, ytr_n), device,
                           trial=0, verbose=False)
        m.eval()
        models.append(m)
        print(f"  split {sname}: model {k+1}/{N_MODELS} trained", flush=True)

    singles, means, medians = [], [], []
    for cid in test_ids:
        X, y_orig = cells[cid]['X'], cells[cid]['y']
        x = torch.tensor(fs.transform(X.reshape(-1, D)).reshape(len(X), Lw, D),
                         dtype=torch.float32).to(device)
        preds = []
        with torch.no_grad():
            for m in models:
                g, *_ = m(x)
                preds.append(ls.inverse_transform(
                    g.cpu().numpy().reshape(-1, 1)).ravel())
        P = np.array(preds)
        singles.append(rmse(y_orig, P[0]))
        means.append(rmse(y_orig, P.mean(0)))
        medians.append(rmse(y_orig, np.median(P, 0)))
        print(f"  {cid}: single={singles[-1]:.1f} mean={means[-1]:.1f} "
              f"median={medians[-1]:.1f}", flush=True)
    print(f"SPLIT {sname}: single={np.mean(singles):.2f} "
          f"ens_mean={np.mean(means):.2f} ens_median={np.mean(medians):.2f}\n",
          flush=True)
