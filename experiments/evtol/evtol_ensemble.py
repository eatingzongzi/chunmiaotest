#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eVTOL RUL ensemble: 10 models (seeds 42-51, 14/4 splits), median/mean.

Mirrors the MIT ensemble success: each model trains on a different
14-cell split; predictions for each seed's test cells are aggregated.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from config import Config
from src.train import train_model, get_device
from src.evaluate import rmse
from src.data_loader import BatteryDataset

EOL_THR = 0.85
L = 30
SEEDS = list(range(42, 52))
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
        cells[cid] = {'X': Xw.astype(np.float32), 'y': yw.astype(np.float32)}
    return cells


cells = load_cells()
ids = sorted(cells)
print(f"{len(ids)} cells")

model_list = []
for seed in SEEDS:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    test_ids = [ids[i] for i in perm[:4]]
    train_ids = [ids[i] for i in perm[4:]]
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
    m, _ = train_model(config, BatteryDataset(Xtr_n, ytr_n), device,
                       trial=0, verbose=False)
    m.eval()
    model_list.append({'model': m, 'fs': fs, 'ls': ls, 'seed': seed,
                       'test_ids': set(test_ids)})
    print(f"seed {seed} trained (test: {test_ids})", flush=True)

summaries = {'single': [], 'mean': [], 'median': []}
for entry in model_list:
    seed = entry['seed']
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    test_ids = [ids[i] for i in perm[:4]]
    cell_s, cell_mean, cell_med = [], [], []
    for cid in test_ids:
        X = cells[cid]['X']
        y_orig = cells[cid]['y']
        preds = []
        with torch.no_grad():
            for ml in model_list:
                x = torch.tensor(
                    ml['fs'].transform(X.reshape(-1, D)).reshape(len(X), Lw, D),
                    dtype=torch.float32).to(device)
                g, *_ = ml['model'](x)
                preds.append(ml['ls'].inverse_transform(
                    g.cpu().numpy().reshape(-1, 1)).ravel())
        P = np.array(preds)
        # single = this seed's own model
        own_pred = preds[seed - 42]
        cell_s.append(rmse(y_orig, own_pred))
        cell_mean.append(rmse(y_orig, P.mean(0)))
        cell_med.append(rmse(y_orig, np.median(P, 0)))
    summaries['single'].append(np.mean(cell_s))
    summaries['mean'].append(np.mean(cell_mean))
    summaries['median'].append(np.mean(cell_med))
    print(f"seed {seed}: single={np.mean(cell_s):.1f} "
          f"ens_mean={np.mean(cell_mean):.1f} ens_median={np.mean(cell_med):.1f}",
          flush=True)

print("\n=== eVTOL ensemble (5-seed average) ===")
for k, v in summaries.items():
    print(f"  {k:8s}: {np.mean(v):.2f}  (per-seed: {[f'{x:.1f}' for x in v]})")
