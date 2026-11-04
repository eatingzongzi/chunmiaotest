#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eVTOL RUL v3: per-cell baseline normalisation.

Each cell's 28 raw features are divided by the mean of its first 10
mission cycles (sign-preserving), removing absolute differences between
the 22 protocols. Cumulative throughput + cycle index are kept raw.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import Config
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell
from src.data_loader import BatteryDataset

EOL_THR = 0.85
L = 30
device = get_device('cuda')


def load_cells(rel=True):
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
        if rel:
            base = X[:10].mean(axis=0)
            sign = np.sign(base)
            sign[sign == 0] = 1.0
            X = X / (np.abs(base) + 1e-9) * 1.0
        cum_q = np.cumsum(X[:, 24])
        cum_e = np.cumsum(X[:, 25])
        extra = np.stack([c.astype(float), cum_q, cum_e], axis=1)
        X = np.concatenate([X, extra], axis=1)
        rul = eol - c
        Xw = np.stack([X[j:j + L] for j in range(len(X) - L + 1)], 0)
        yw = np.array([rul[j + L - 1] for j in range(len(X) - L + 1)])
        cells[cid] = {'X': Xw.astype(np.float32), 'y': yw.astype(np.float32),
                      'eol': float(eol)}
    return cells


def run_seed(cells, seed, n_test=4):
    rng = np.random.RandomState(seed)
    ids = sorted(cells)
    perm = rng.permutation(len(ids))
    test_ids = [ids[i] for i in perm[:n_test]]
    train_ids = [ids[i] for i in perm[n_test:]]
    Xtr = np.concatenate([cells[c]['X'] for c in train_ids])
    ytr = np.concatenate([cells[c]['y'] for c in train_ids])
    N, Lw, D = Xtr.shape
    fs = MinMaxScaler().fit(Xtr.reshape(-1, D))
    Xtr_n = fs.transform(Xtr.reshape(-1, D)).reshape(N, Lw, D)
    ls = MinMaxScaler().fit(ytr.reshape(-1, 1))
    ytr_n = ls.transform(ytr.reshape(-1, 1)).ravel()

    config = Config()
    config.input_dim = D
    config.epochs = 100
    config.random_seed = seed
    config.moe_balance_coeff = 0.05
    config.route_sup_coeff = 1.0
    config.router_mse_coeff = 0.1
    model, _ = train_model(config, BatteryDataset(Xtr_n, ytr_n), device,
                           trial=0, verbose=False)
    test_cells = []
    for c in test_ids:
        X = cells[c]['X']
        test_cells.append({
            'cell_id': c, 'eol': cells[c]['eol'],
            'X': fs.transform(X.reshape(-1, D)).reshape(len(X), Lw, D),
            'y': ls.transform(cells[c]['y'].reshape(-1, 1)).ravel(),
            'y_orig': cells[c]['y'],
        })
    _, avg = evaluate_per_cell(model, test_cells, device, ls)
    return avg['rmse'], test_ids


if __name__ == '__main__':
    cells = load_cells(rel=True)
    print(f"{len(cells)} cells, {cells[sorted(cells)[0]]['X'].shape[-1]} features (per-cell normalised)")
    vals = []
    for seed in [42, 43, 44, 45, 46]:
        r, t = run_seed(cells, seed)
        vals.append(r)
        print(f"seed {seed} ({t}): SMoE={r:.2f}", flush=True)
    print(f"\n5-seed: {np.mean(vals):.2f}+/-{np.std(vals):.2f}  (v2 raw: 60.80)")
