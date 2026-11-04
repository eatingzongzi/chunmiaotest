#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Which experts are strong on hard (anomalous) cells?

Trains the best config (hard routing sup, seed 42), then evaluates every
non-train cell with: the gated mixture and each of the 10 experts alone.
Goal: find whether a stable 'hard-cell expert' exists (same expert best
across several hard cells) — if so, routing can be biased toward it for
anomalous samples.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import h5py
import torch
from sklearn.preprocessing import MinMaxScaler
import collections

from config import Config
from src.data_loader import (sliding_window_sampling, _extract_cell_features_paper,
                             BatteryDataset)
from src.train import train_model, get_device
from src.model import SMoELSTM
from src.evaluate import rmse

device = get_device('cuda')
config = Config()
config.epochs = 100
config.random_seed = 42
config.moe_balance_coeff = 0.05
config.route_sup_coeff = 1.0
config.router_mse_coeff = 0.1

# ── Load all 48 cells ───────────────────────────────────────────────
f = h5py.File(os.path.join('data/mit_stanford',
                           '2017-06-30_batchdata_updated_struct_errorcorrect.mat'), 'r')
batch = f['batch']
all_cells = []
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
        if len(feats) == 0:
            continue
        eol = int(cl)
        valid = cn <= eol
        if np.sum(valid) < 30:
            continue
        all_cells.append({'cell_id': i, 'eol': eol,
                          'features': feats[valid], 'cycle_nums': cn[valid]})
    except Exception:
        continue
f.close()

# ── Seed-42 split ───────────────────────────────────────────────────
rng = np.random.RandomState(42)
selected = rng.permutation(len(all_cells))[:30]
selected = rng.permutation(selected)
test_ids, train_ids = selected[:6], selected[6:]

train_X, train_y = [], []
for idx in train_ids:
    X, y = sliding_window_sampling(all_cells[idx]['features'],
                                   all_cells[idx]['cycle_nums'],
                                   all_cells[idx]['eol'], 30)
    train_X.append(X); train_y.append(y)
X_train = np.concatenate(train_X); y_train = np.concatenate(train_y)
N, L, D = X_train.shape
feat_scaler = MinMaxScaler().fit(X_train.reshape(-1, D))
X_norm = feat_scaler.transform(X_train.reshape(-1, D)).reshape(N, L, D)
lab_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))
y_norm = lab_scaler.transform(y_train.reshape(-1, 1)).ravel()

model, _ = train_model(config, BatteryDataset(X_norm, y_norm), device,
                       trial=0, verbose=False)
model.eval()

# ── Evaluate every non-train cell: gated vs each expert alone ───────
others = [i for i in range(len(all_cells)) if i not in train_ids]
rows = []
for idx in others:
    X, y = sliding_window_sampling(all_cells[idx]['features'],
                                   all_cells[idx]['cycle_nums'],
                                   all_cells[idx]['eol'], 30)
    if len(X) == 0:
        continue
    Xn = feat_scaler.transform(X.reshape(-1, D)).reshape(len(X), L, D)
    x = torch.tensor(Xn, dtype=torch.float32).to(device)
    y_orig = y
    with torch.no_grad():
        g, *_ = model(x)
        comb_pred = lab_scaler.inverse_transform(
            g.detach().cpu().numpy().reshape(-1, 1)).ravel()
        _, (h_n, _) = model.lstm(x)
        h = h_n[-1]
        per = []
        for e in range(10):
            out = model.moe.experts[e](h)
            ge, *_ = model.evidence(out)
            pe = lab_scaler.inverse_transform(
                ge.detach().cpu().numpy().reshape(-1, 1)).ravel()
            per.append(rmse(y_orig, pe))
    rows.append({
        'cell_id': all_cells[idx]['cell_id'],
        'eol': all_cells[idx]['eol'],
        'comb': rmse(y_orig, comb_pred),
        'per': per,
        'best_exp': int(np.argmin(per)),
    })

# ── Report ──────────────────────────────────────────────────────────
rows.sort(key=lambda r: r['comb'])
print("=== All 24 non-train cells: gated RMSE vs best single expert ===")
for r in rows:
    hard = "HARD" if r['comb'] > 40 else "    "
    print(f"  [{hard}] cell {r['cell_id']:2d} eol={r['eol']:4d}: "
          f"gated={r['comb']:6.1f}  best=E{r['best_exp']}({min(r['per']):6.1f})  "
          f"per={[f'{p:.0f}' for p in r['per']]}")

hard_rows = [r for r in rows if r['comb'] > 40]
easy_rows = [r for r in rows if r['comb'] <= 40]
print(f"\nHard cells (gated>40): {len(hard_rows)}")
if hard_rows:
    best_cnt = collections.Counter(r['best_exp'] for r in hard_rows)
    print(f"  best-expert distribution on hard cells: {dict(sorted(best_cnt.items()))}")
    for r in hard_rows:
        # is the best expert's standalone RMSE on hard cells better than gated?
        print(f"  cell {r['cell_id']:2d} eol={r['eol']:4d}: gated={r['comb']:.1f} "
              f"best E{r['best_exp']}={min(r['per']):.1f} "
              f"(gain {r['comb'] - min(r['per']):+.1f})")
print(f"\nEasy cells (gated<=40): {len(easy_rows)}")
if easy_rows:
    best_cnt_e = collections.Counter(r['best_exp'] for r in easy_rows)
    print(f"  best-expert distribution on easy cells: {dict(sorted(best_cnt_e.items()))}")
