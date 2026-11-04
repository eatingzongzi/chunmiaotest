#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eVTOL RUL v2: add cumulative-throughput features + LSTM-only control.

Adds three BMS-available ageing features per sample:
  cycle index, cumulative mission charge (Ah), cumulative energy (Wh).
Compares SMoE-LSTM (hard routing sup) vs plain LSTM under the same
cross-cell splits.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from config import Config
from src.train import train_model, get_device, set_seed, prepare_dataloaders
from src.evaluate import evaluate_per_cell, rmse
from src.data_loader import BatteryDataset
from src.loss import EvidentialRegressionLoss
from src.model import EvidenceLayer

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
        # cumulative throughput (per mission cycle)
        cum_q = np.cumsum(X[:, 24])      # mission_Q_Ah
        cum_e = np.cumsum(X[:, 25])      # mission_E_Wh
        extra = np.stack([c.astype(float), cum_q, cum_e], axis=1)
        X = np.concatenate([X, extra], axis=1)
        rul = eol - c
        Xw = np.stack([X[j:j + L] for j in range(len(X) - L + 1)], 0)
        yw = np.array([rul[j + L - 1] for j in range(len(X) - L + 1)])
        cells[cid] = {'X': Xw.astype(np.float32), 'y': yw.astype(np.float32),
                      'eol': float(eol)}
    return cells


class LSTMOnly(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, 2, batch_first=True, dropout=0.1)
        self.evidence = EvidenceLayer(hidden, 64)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.evidence(h_n[-1])


def train_lstm_only(Xtr, ytr, seed):
    set_seed(seed)
    loader, _ = prepare_dataloaders(BatteryDataset(Xtr, ytr), 64, 0.0, seed)
    m = LSTMOnly(Xtr.shape[-1]).to(device)
    crit = EvidentialRegressionLoss(coeff=2.0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
    for ep in range(100):
        m.train()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            g, nu, al, be = m(X)
            loss = crit(y, g, nu, al, be)
            loss.backward(); opt.step()
    m.eval()
    return m


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
    lstm = train_lstm_only(Xtr_n, ytr_n, seed)

    test_cells = []
    for c in test_ids:
        X = cells[c]['X']
        test_cells.append({
            'cell_id': c, 'eol': cells[c]['eol'],
            'X': fs.transform(X.reshape(-1, D)).reshape(len(X), Lw, D),
            'y': ls.transform(cells[c]['y'].reshape(-1, 1)).ravel(),
            'y_orig': cells[c]['y'],
        })
    # LSTM-only evaluation
    lres = []
    with torch.no_grad():
        for tc in test_cells:
            x = torch.tensor(tc['X'], dtype=torch.float32).to(device)
            g, nu, al, be = lstm(x)
            p = ls.inverse_transform(g.cpu().numpy().reshape(-1, 1)).ravel()
            lres.append(rmse(tc['y_orig'], p))
    _, avg = evaluate_per_cell(model, test_cells, device, ls)
    return avg['rmse'], float(np.mean(lres)), test_ids


if __name__ == '__main__':
    cells = load_cells()
    print(f"{len(cells)} cells, {cells[sorted(cells)[0]]['X'].shape[-1]} features")
    smo, lst = [], []
    for seed in [42, 43, 44, 45, 46]:
        a, b, t = run_seed(cells, seed)
        smo.append(a); lst.append(b)
        print(f"seed {seed} ({t}): SMoE={a:.2f}  LSTM={b:.2f}", flush=True)
    print(f"\n5-seed: SMoE-LSTM {np.mean(smo):.2f}+/-{np.std(smo):.2f} | "
          f"LSTM-only {np.mean(lst):.2f}+/-{np.std(lst):.2f}")
