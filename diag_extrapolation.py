#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-cell diagnostic: capacity-curve extrapolation vs neural network.

For each MIT cell, at cycle t fit a degradation model to capacity[0:t]
(causal, no leakage), extrapolate to the EOL threshold, and predict
RUL = EOL_est - t. Compare per-cell RMSE with the clean NN results.

Models: linear (all / last 100 cycles), log-linear (exponential decay),
quadratic. Capacity smoothed with a moving average before fitting.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import h5py

from src.data_loader import _extract_cell_features_paper

MA_WIN = 10
T_MIN = 30


def fit_predict(cap, t, thr, kind):
    """Fit on cap[0:t] (causally smoothed) and return estimated EOL."""
    y = cap[:t]
    if len(y) < 20:
        return np.nan
    # causal moving average (only past data)
    k = min(MA_WIN, len(y) // 2)
    if k >= 2:
        kernel = np.ones(k) / k
        ys = np.convolve(y, kernel, mode='valid')
        xs = np.arange(k - 1, len(y), dtype=float)
    else:
        ys, xs = y.astype(float), np.arange(len(y), dtype=float)
    try:
        if kind == 'linear_all':
            slope, intercept = np.polyfit(xs, ys, 1)
            if slope >= -1e-9:
                return np.nan
            return (thr - intercept) / slope
        if kind == 'linear_last':
            n = min(100, len(xs))
            slope, intercept = np.polyfit(xs[-n:], ys[-n:], 1)
            if slope >= -1e-9:
                return np.nan
            return (thr - intercept) / slope
        if kind == 'exp':
            pos = ys > 1e-3
            if pos.sum() < 20:
                return np.nan
            b, a = np.polyfit(xs[pos], np.log(ys[pos]), 1)
            if b >= -1e-9:
                return np.nan
            return (np.log(thr) - a) / b
        if kind == 'quad':
            c2, c1, c0 = np.polyfit(xs, ys, 2)
            roots = np.roots([c2, c1, c0 - thr])
            real = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > xs[-1]]
            return min(real) if real else np.nan
    except Exception:
        return np.nan
    return np.nan


def load_cells():
    f = h5py.File('data/mit_stanford/2017-06-30_batchdata_updated_struct_errorcorrect.mat', 'r')
    batch = f['batch']
    cells = {}
    for i in range(batch['summary'].shape[0]):
        try:
            summ = f[batch['summary'][i, 0]]
            cyc = f[batch['cycles'][i, 0]]
            cl = float(f[batch['cycle_life'][i, 0]][()].flatten()[0])
            vd = None
            try:
                vd = f[batch['Vdlin'][i, 0]][()].flatten().astype(np.float64)
            except Exception:
                pass
            feats, cn = _extract_cell_features_paper(f, summ, cyc, cl, vd)
            if len(feats) == 0:
                continue
            eol = int(cl)
            valid = cn <= eol
            if valid.sum() < T_MIN + 10:
                continue
            cells[i] = {'cap': feats[valid, 15].astype(float),
                        'cycle': cn[valid], 'eol': eol}
        except Exception:
            continue
    f.close()
    return cells


cells = load_cells()
print(f"{len(cells)} cells\n")

# EOL threshold: capacity at cycle_life (check consistency)
for cid in [47, 29, 24]:
    if cid in cells:
        c = cells[cid]
        print(f"cell {cid}: len={len(c['cap'])}, eol={c['eol']}, "
              f"cap[0]={c['cap'][0]:.3f}, cap[eol]={c['cap'][-1]:.3f}, "
              f"0.8*cap0={0.8*c['cap'][0]:.3f}")

print("\nPer-cell RMSE (cycles), causal extrapolation:")
print(f"{'cell':>4} {'eol':>5} {'lin_all':>8} {'lin_last':>8} {'exp':>8} {'quad':>8}")
results = {}
for cid in sorted(cells):
    c = cells[cid]
    cap, eol = c['cap'], c['eol']
    thr = 0.8 * cap[0]
    N = min(eol, len(cap) - 1)
    row = {}
    for kind in ['linear_all', 'linear_last', 'exp', 'quad']:
        errs = []
        for t in range(T_MIN, N + 1):
            eol_est = fit_predict(cap, t, thr, kind)
            if not np.isfinite(eol_est):
                continue
            rul_pred = max(0.0, eol_est - t)
            errs.append((rul_pred - (eol - t)) ** 2)
        row[kind] = np.sqrt(np.mean(errs)) if errs else np.nan
    results[cid] = row
    print(f"{cid:>4} {eol:>5} {row['linear_all']:>8.1f} {row['linear_last']:>8.1f} "
          f"{row['exp']:>8.1f} {row['quad']:>8.1f}")

# Focus cells: clean NN per-cell RMSE
print("\nComparison on the clean test cells:")
nn42 = {47: 147.1, 32: 22.5, 26: 21.5, 21: 20.1, 27: 32.4, 30: 29.2}
nn24 = {13: 27.7, 17: 20.5, 24: 15.2, 25: 19.6, 34: 21.6, 29: 38.3}
for tag, nn in [('42-cell split', nn42), ('24-cell split', nn24)]:
    print(f"  {tag}:")
    for cid, nnv in nn.items():
        if cid in results:
            r = results[cid]
            best = min(r.values(), key=lambda v: v if np.isfinite(v) else 1e9)
            print(f"    cell {cid:2d}: NN={nnv:6.1f} | extrap best={best:6.1f} "
                  f"(lin_all={r['linear_all']:.1f}, lin_last={r['linear_last']:.1f}, "
                  f"exp={r['exp']:.1f}, quad={r['quad']:.1f})")

np.save('data/extrapolation_results.npy', results, allow_pickle=True)
print("\nsaved data/extrapolation_results.npy")
