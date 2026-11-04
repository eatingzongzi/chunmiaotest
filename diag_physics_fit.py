#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Physics-based degradation laws: causal fit + extrapolation test.

Laws tested (capacity loss L(t) = cap0 - cap(t)):
  SEI diffusion : L = a * sqrt(t)
  power law     : L = a * t^b   (b > 1 captures the knee)
  SEI + linear  : L = a*sqrt(t) + c*t
Fit on cycles [0, T] only; predict the cycle where cap = 0.88 Ah.
Errors bucketed by true RUL to see where each law is usable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import h5py
from scipy.optimize import curve_fit

from src.data_loader import _extract_cell_features_paper

EOL_CAP = 0.88
T_MIN = 30
BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 10**9)]


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
            valid = cn <= int(cl)
            if valid.sum() < T_MIN + 10:
                continue
            cells[i] = {'cap': feats[valid, 15].astype(float), 'eol': int(cl)}
        except Exception:
            continue
    f.close()
    return cells


def f_sei(t, a):
    return a * np.sqrt(t)


def f_power(t, a, b):
    return a * np.power(t, b)


def f_sei_lin(t, a, c):
    return a * np.sqrt(t) + c * t


def predict_eol(cap, t, law):
    cap0 = cap[0]
    xs = np.arange(1, t + 1, dtype=float)
    loss = cap0 - cap[:t]
    ok = loss > 1e-4
    if ok.sum() < 15:
        return np.nan
    target = cap0 - EOL_CAP
    if target <= 0:
        return np.nan
    try:
        if law == 'sei':
            popt, _ = curve_fit(f_sei, xs[ok], loss[ok], p0=[1e-3], maxfev=2000)
            a = popt[0]
            if a <= 0: return np.nan
            return (target / a) ** 2
        if law == 'power':
            popt, _ = curve_fit(f_power, xs[ok], loss[ok], p0=[1e-3, 1.0],
                                bounds=([0, 0.1], [np.inf, 5.0]), maxfev=4000)
            a, b = popt
            if a <= 0: return np.nan
            return (target / a) ** (1.0 / b)
        if law == 'sei_lin':
            popt, _ = curve_fit(f_sei_lin, xs[ok], loss[ok], p0=[1e-3, 1e-4],
                                maxfev=4000)
            a, c = popt
            if a <= 0 and c <= 0: return np.nan
            # solve a*sqrt(t) + c*t = target
            if abs(c) < 1e-12:
                return (target / a) ** 2
            disc = a * a + 4 * c * target
            if disc < 0: return np.nan
            return ((-a + np.sqrt(disc)) / (2 * c)) ** 2
    except Exception:
        return np.nan
    return np.nan


cells = load_cells()
print(f"{len(cells)} cells")

for law in ['sei', 'power', 'sei_lin']:
    print(f"\n=== {law} ===")
    bucket = {b: [] for b in BUCKETS}
    per_cell = []
    for cid in sorted(cells):
        c = cells[cid]; cap, eol = c['cap'], c['eol']
        errs = []
        for t in range(T_MIN, min(eol, len(cap) - 1) + 1):
            e = predict_eol(cap, t, law)
            if not np.isfinite(e) or e <= t:
                continue
            err = (max(0.0, e - t) - (eol - t)) ** 2
            errs.append(err)
            for lo, hi in BUCKETS:
                if lo <= eol - t < hi:
                    bucket[(lo, hi)].append(err); break
        if errs:
            per_cell.append(np.sqrt(np.mean(errs)))
    for b, v in bucket.items():
        if v:
            hi = b[1] if b[1] < 10**8 else 'inf'
            print(f"  RUL [{b[0]:>3},{hi:>4}): n={len(v):6d} RMSE={np.sqrt(np.mean(v)):8.1f}")
    print(f"  per-cell mean RMSE={np.mean(per_cell):.1f}")
