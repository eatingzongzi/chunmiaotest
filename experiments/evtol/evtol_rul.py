#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eVTOL RUL prediction with our SMoE-LSTM pipeline.

Task: RUL = EOL(85% SOH) - current cycle, sliding window l=30 over the
28 per-cycle features. Cross-cell splits (14 train / 4 test) across 5
seeds, per-cell RMSE (same protocol as our MIT experiments).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from config import Config
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell
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
        X, c = v['X'][keep], cyc[keep]
        if len(X) < L + 10:
            continue
        rul = eol - c
        # sliding windows over cycles (contiguous mission cycles)
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
    per_cell, avg = evaluate_per_cell(model, test_cells, device, ls)
    return avg, per_cell, test_ids


if __name__ == '__main__':
    cells = load_cells()
    print(f"{len(cells)} cells with EOL@{EOL_THR:.0%}: "
          f"{sorted(cells)}")
    print(f"windows per cell: {[len(cells[c]['y']) for c in sorted(cells)]}")

    all_avg = []
    for seed in [42, 43, 44, 45, 46]:
        avg, per_cell, test_ids = run_seed(cells, seed)
        all_avg.append(avg['rmse'])
        print(f"\nseed {seed} (test: {test_ids}): RMSE={avg['rmse']:.2f} "
              f"MAE={avg['mae']:.2f} MAPE={avg['mape']:.2f}% R2={avg['r2']:.4f}")
        for pc in per_cell:
            print(f"   {pc['cell_id']} eol={pc['eol']:.0f} "
                  f"RMSE={pc['metrics']['rmse']:.1f}", flush=True)
    print(f"\n=== eVTOL RUL, 5 seeds: {np.mean(all_avg):.2f} +/- {np.std(all_avg):.2f} ===")
