#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extrapolation v2: fixed 0.88 Ah EOL threshold + RUL-bucket analysis.

The capacity curve has a knee: nearly flat for the first half of life,
then accelerating fade. Extrapolation should only be trusted late.
Reports RMSE per true-RUL bucket to locate where it helps.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import h5py

from src.data_loader import _extract_cell_features_paper

EOL_CAP = 0.88  # 80% of 1.1 Ah nominal, consistent with dataset cycle_life
T_MIN = 20
BUCKETS = [(0, 100), (100, 200), (200, 400), (400, 10**9)]


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


def est_eol(cap, t, kind='lin_last', window=60):
    y = cap[:t]
    if len(y) < 30:
        return np.nan
    xs = np.arange(len(y), dtype=float)
    try:
        if kind == 'lin_last':
            n = min(window, len(y))
            slope, intercept = np.polyfit(xs[-n:], y[-n:], 1)
        elif kind == 'lin_all':
            slope, intercept = np.polyfit(xs, y, 1)
        elif kind == 'exp_last':
            n = min(window, len(y))
            ys, xw = y[-n:], xs[-n:]
            pos = ys > 1e-3
            b, a = np.polyfit(xw[pos], np.log(ys[pos]), 1)
            if b >= -1e-12:
                return np.nan
            return (np.log(EOL_CAP) - a) / b
        else:
            return np.nan
        if slope >= -1e-12:
            return np.nan
        return (EOL_CAP - intercept) / slope
    except Exception:
        return np.nan


cells = load_cells()
print(f"{len(cells)} cells; EOL threshold {EOL_CAP} Ah\n")

for kind in ['lin_last', 'lin_all', 'exp_last']:
    print(f"=== {kind} ===")
    bucket_errs = {b: [] for b in BUCKETS}
    per_cell = []
    for cid in sorted(cells):
        c = cells[cid]; cap, eol = c['cap'], c['eol']
        errs = []
        for t in range(T_MIN, min(eol, len(cap) - 1) + 1):
            e = est_eol(cap, t, kind)
            if not np.isfinite(e):
                continue
            rul_true = eol - t
            rul_pred = max(0.0, e - t)
            err = (rul_pred - rul_true) ** 2
            errs.append(err)
            for lo, hi in BUCKETS:
                if lo <= rul_true < hi:
                    bucket_errs[(lo, hi)].append(err)
                    break
        if errs:
            per_cell.append(np.sqrt(np.mean(errs)))
    for b, v in bucket_errs.items():
        if v:
            print(f"  RUL [{b[0]:>3},{b[1] if b[1] < 10**8 else 'inf':>4}): "
                  f"n={len(v):6d} RMSE={np.sqrt(np.mean(v)):8.1f}")
    print(f"  overall per-cell mean RMSE={np.mean(per_cell):.1f}\n")
