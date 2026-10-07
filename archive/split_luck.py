#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""How lucky must a 6-cell test split be to reach the paper's 16.11?

Trains on the seed-42 24 train cells, evaluates per-cell RMSE on ALL
remaining 24 cells of the batch, then bootstraps: what fraction of random
6-cell test draws gives a mean RMSE <= 16.11 (paper) or <= 20?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler

from config import Config
from src.data_loader import sliding_window_sampling, _extract_cell_features_paper, BatteryDataset
from src.train import train_model, get_device
from src.evaluate import compute_all_metrics

device = get_device('cuda')
config = Config()
config.epochs = 100
config.random_seed = 42

# ── 1. Load all 48 cells ─────────────────────────────────────────────
batch_file = '2017-06-30_batchdata_updated_struct_errorcorrect.mat'
f = h5py.File(os.path.join('data/mit_stanford', batch_file), 'r')
batch = f['batch']
n_available = batch['summary'].shape[0]

all_cells = []
for i in range(n_available):
    try:
        summ = f[batch['summary'][i, 0]]
        cyc = f[batch['cycles'][i, 0]]
        cycle_life = float(f[batch['cycle_life'][i, 0]][()].flatten()[0])
        vdlin = None
        try:
            vdlin = f[batch['Vdlin'][i, 0]][()].flatten().astype(np.float64)
        except Exception:
            pass
        features, cycle_nums = _extract_cell_features_paper(
            f, summ, cyc, cycle_life, vdlin)
        if len(features) == 0:
            continue
        eol = int(cycle_life)
        valid = cycle_nums <= eol
        if np.sum(valid) < 30:
            continue
        all_cells.append({'cell_id': i, 'eol': eol,
                          'features': features[valid],
                          'cycle_nums': cycle_nums[valid]})
    except Exception:
        continue
f.close()
print(f"{len(all_cells)} valid cells loaded")

# ── 2. Seed-42 split (same as the paper protocol) ────────────────────
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

# ── 3. Train ─────────────────────────────────────────────────────────
train_ds = BatteryDataset(X_norm, y_norm)
model, _ = train_model(config, train_ds, device, trial=0, verbose=False)

# ── 4. Per-cell RMSE on every non-train cell ─────────────────────────
others = [i for i in range(len(all_cells)) if i not in train_ids]
per_cell_rmse = {}
for idx in others:
    X, y = sliding_window_sampling(all_cells[idx]['features'],
                                   all_cells[idx]['cycle_nums'],
                                   all_cells[idx]['eol'], 30)
    if len(X) == 0:
        continue
    Xn = feat_scaler.transform(X.reshape(-1, D)).reshape(len(X), L, D)
    import torch
    with torch.no_grad():
        gamma, nu, alpha, beta = model(torch.tensor(Xn, dtype=torch.float32).to(device))
        from src.model import SMoELSTM
        pred = SMoELSTM.predict(gamma, nu, alpha, beta)['pred'].cpu().numpy().ravel()
    y_orig = y  # sliding_window_sampling returns raw (un-normalised) RUL
    pred_orig = lab_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
    m = compute_all_metrics(y_orig, pred_orig)
    per_cell_rmse[idx] = (m['rmse'], all_cells[idx]['eol'])

print("\nPer-cell RMSE of the 24 non-train cells:")
for idx, (r, eol) in sorted(per_cell_rmse.items(), key=lambda kv: kv[1][0]):
    tag = "TEST" if idx in test_ids else "    "
    print(f"  [{tag}] cell {idx:2d} eol={eol:4d} RMSE={r:6.2f}")

vals = np.array([v[0] for v in per_cell_rmse.values()])
print(f"\nAll 24 non-train cells: mean={vals.mean():.2f} median={np.median(vals):.2f} "
      f"min={vals.min():.2f} max={vals.max():.2f}")

# ── 5. Bootstrap: P(random 6-cell draw reaches paper-level) ──────────
rng = np.random.RandomState(0)
n_boot = 20000
counts = {16.11: 0, 20: 0, 24: 0}
for _ in range(n_boot):
    draw = rng.choice(len(vals), 6, replace=False)
    m = vals[draw].mean()
    for thr in counts:
        if m <= thr:
            counts[thr] += 1
print(f"\nBootstrap over {n_boot} random 6-cell test draws:")
for thr, c in counts.items():
    print(f"  P(mean RMSE <= {thr}) = {c/n_boot:.3%}")
