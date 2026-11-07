#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Knee-aware degradation model: causal fit + comparison with NN.

Model:  L(t) = a*t + b*max(0, t - t_k)^2      (SEI/linear + accelerated knee)
Target: capacity loss reaching cap0 - 0.88 Ah.
At each cycle t, fit parameters (a, b, t_k) on the observed prefix only,
predict EOL and RUL_phys = EOL - t. Also outputs fit quality, knee flag
and an approximate 1-sigma EOL uncertainty.

The script then trains ONE baseline SMoE-LSTM on the fixed 24-cell split
and compares physics vs NN per true-RUL bucket.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import h5py
from scipy.optimize import least_squares
from sklearn.preprocessing import MinMaxScaler
import torch

from src.data_loader import _extract_cell_features_paper
from src.train import train_model, get_device
from src.evaluate import rmse
from src.data_loader import BatteryDataset

EOL_CAP = 0.88
L = 30
STEP = 5              # refit every 5 cycles (causal)
T_MIN = 30
BUCKETS = [(0, 50), (50, 100), (100, 150), (150, 300), (300, 10**9)]
device = get_device('cuda')


def model_loss(s, a, b, tk):
    return a * s + b * np.maximum(0.0, s - tk) ** 2


def fit_knee(cap, t):
    """Fit on cycles 1..t; return (a, b, tk, r2, sigma_eol, knee_flag)."""
    s = np.arange(1, t + 1, dtype=float)
    loss = cap[0] - cap[:t]
    ok = loss > 1e-5
    if ok.sum() < 20:
        return None
    s, loss = s[ok], loss[ok]
    try:
        res = least_squares(
            lambda p: model_loss(s, *p) - loss,
            x0=[loss[-1] / s[-1], 0.0, s[-1] * 0.7],
            bounds=([0, 0, 0], [1.0, 1e-3, s[-1]]),
            loss='soft_l1', f_scale=1e-3, max_nfev=200)
    except Exception:
        return None
    a, b, tk = res.x
    pred = model_loss(s, a, b, tk)
    ss_res = np.sum((pred - loss) ** 2)
    ss_tot = np.sum((loss - loss.mean()) ** 2) + 1e-12
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    sigma_res = float(np.sqrt(ss_res / max(len(loss) - 3, 1)))
    knee = 1.0 if (b > 1e-7 and tk < s[-1]) else 0.0
    # EOL: solve a*s + b*(s-tk)^2 = target
    target = cap[0] - EOL_CAP
    eol = np.nan
    if target > 0:
        if b > 1e-12:
            A, B, C = b, a - 2 * b * tk, b * tk * tk - target
            disc = B * B - 4 * A * C
            if disc >= 0:
                roots = [(-B + np.sqrt(disc)) / (2 * A),
                         (-B - np.sqrt(disc)) / (2 * A)]
                roots = [r for r in roots if r > s[-1]]
                if roots:
                    eol = min(roots)
        elif a > 1e-12:
            eol = target / a
        if not np.isfinite(eol) or eol > 5000:
            eol = np.nan
    # approximate sigma_EOL via residual / local slope at EOL
    sigma_eol = np.nan
    if np.isfinite(eol):
        slope = a + 2 * b * max(0.0, eol - tk) + 1e-9
        sigma_eol = float(sigma_res / slope)
    return a, b, tk, r2, sigma_eol, knee, eol


def physics_predictions(cap):
    """Causal predictions per cycle: (N,5) [rul, r2, sigma, knee, valid]."""
    N = len(cap)
    P = np.zeros((N, 5), dtype=np.float64)
    last = None
    for t in range(T_MIN, N + 1):
        if (t - T_MIN) % STEP == 0 or last is None:
            last = fit_knee(cap, t)
        if last is None:
            continue
        a, b, tk, r2, sig, knee, eol = last
        if np.isfinite(eol) and eol > t:
            P[t - 1] = [eol - t, r2, sig, knee, 1.0]
    return P


def load_cells():
    f = h5py.File('data/mit_stanford/2017-06-30_batchdata_updated_struct_errorcorrect.mat', 'r')
    b = f['batch']
    cells = {}
    for i in range(b['summary'].shape[0]):
        try:
            summ = f[b['summary'][i, 0]]; cyc = f[b['cycles'][i, 0]]
            cl = float(f[b['cycle_life'][i, 0]][()].flatten()[0])
            vd = f[b['Vdlin'][i, 0]][()].flatten().astype(np.float64)
            feats, cn = _extract_cell_features_paper(f, summ, cyc, cl, vd)
            if len(feats) == 0:
                continue
            eol = int(cl)
            valid = cn <= eol
            if valid.sum() < L + 10:
                continue
            cells[i] = {'cap': feats[valid, 15].astype(float),
                        'features': feats[valid], 'cycle': cn[valid], 'eol': eol}
        except Exception:
            continue
    f.close()
    return cells


print("loading cells...")
cells = load_cells()
print(f"{len(cells)} cells; fitting knee-aware physics...")

# ── Physics evaluation by RUL bucket ────────────────────────────────
bucket = {b: [] for b in BUCKETS}
per_cell = []
for cid in sorted(cells):
    c = cells[cid]
    P = physics_predictions(c['cap'])
    errs = []
    for k in range(len(P)):
        if P[k, 4] < 1:
            continue
        rul_true = c['eol'] - c['cycle'][k]
        if rul_true < 0:
            continue
        e = (P[k, 0] - rul_true) ** 2
        errs.append(e)
        for lo, hi in BUCKETS:
            if lo <= rul_true < hi:
                bucket[(lo, hi)].append(e); break
    if errs:
        per_cell.append(np.sqrt(np.mean(errs)))
print("\n=== Knee-aware physics (causal) ===")
for b, v in bucket.items():
    if v:
        hi = b[1] if b[1] < 10**8 else 'inf'
        print(f"  RUL [{b[0]:>3},{hi:>4}): n={len(v):6d} RMSE={np.sqrt(np.mean(v)):8.1f}")
print(f"  per-cell mean RMSE={np.mean(per_cell):.1f}")

# ── NN baseline on the fixed 24-cell split ──────────────────────────
rng = np.random.RandomState(42)
ids = sorted(cells)
sel = rng.permutation(len(ids))[:30]
sel = rng.permutation(sel)
test_ids = [ids[i] for i in sel[:6]]
train_ids = [ids[i] for i in sel[6:]]
print(f"\nNN split: test {test_ids}")


def windows(c):
    feats, cyc, eol = c['features'], c['cycle'], c['eol']
    N = len(feats)
    X = np.stack([feats[i:i + L] for i in range(N - L + 1)], 0)
    y = np.array([eol - cyc[i + L - 1] for i in range(N - L + 1)])
    return X.astype(np.float32), y.astype(np.float32)


Xtr = np.concatenate([windows(cells[c])[0] for c in train_ids])
ytr = np.concatenate([windows(cells[c])[1] for c in train_ids])
N, Lw, D = Xtr.shape
fs = MinMaxScaler().fit(Xtr.reshape(-1, D))
Xtr_n = fs.transform(Xtr.reshape(-1, D)).reshape(N, Lw, D)
ls = MinMaxScaler().fit(ytr.reshape(-1, 1))
ytr_n = ls.transform(ytr.reshape(-1, 1)).ravel()

from config import Config
config = Config()
config.epochs = 100
config.random_seed = 42
config.moe_balance_coeff = 0.05
config.route_sup_coeff = 1.0
config.router_mse_coeff = 0.1
model, _ = train_model(config, BatteryDataset(Xtr_n, ytr_n), device,
                       trial=0, verbose=False)
model.eval()

nn_bucket = {b: [] for b in BUCKETS}
nn_cell = []
with torch.no_grad():
    for cid in test_ids:
        X, y = windows(cells[cid])
        Xn = fs.transform(X.reshape(-1, D)).reshape(len(X), Lw, D)
        pred = model(torch.tensor(Xn, dtype=torch.float32).to(device))[0]
        pred = ls.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).ravel()
        errs = []
        for i, yt in enumerate(y):
            e = (pred[i] - yt) ** 2
            errs.append(e)
            for lo, hi in BUCKETS:
                if lo <= yt < hi:
                    nn_bucket[(lo, hi)].append(e); break
        nn_cell.append(np.sqrt(np.mean(errs)))
print("\n=== NN baseline (same split) ===")
for b, v in nn_bucket.items():
    if v:
        hi = b[1] if b[1] < 10**8 else 'inf'
        print(f"  RUL [{b[0]:>3},{hi:>4}): n={len(v):6d} RMSE={np.sqrt(np.mean(v)):8.1f}")
print(f"  per-cell mean RMSE={np.mean(nn_cell):.1f}")
