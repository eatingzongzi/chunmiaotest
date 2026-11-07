#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eVTOL enhanced features: 28 base + 8 IR + 3 cumulative = 39.

IR comes from VAH##_impedance.csv (4 measurements every ~51 cycles:
20%/60% SOC, 1s/30s). Measurements are mapped to the reconstructed
global cycle index (tester numbers are 1-based) and forward-filled
causally onto mission cycles. Ratios to the first measurement capture
relative resistance growth.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import pandas as pd

DATA = 'data/evtol'
OUT = 'data/evtol_features_v3.npy'


def load_ir(cid):
    fn = os.path.join(DATA, f'{cid}_impedance.csv')
    if not os.path.exists(fn):
        return None
    df = pd.read_csv(fn)
    cols = ['20%_1_second', '20%_30_second', '60%_1_second', '60%_30_second']
    cyc = df['cycle numbers'].values.astype(float) - 1.0  # 1-based -> 0-based
    vals = df[cols].values.astype(float)
    return cyc, vals


def build():
    d = np.load('data/evtol_features.npy', allow_pickle=True).item()

    # ── Pass 1: measured IR series (absolute values) per cell ───────
    ir_series = {}
    for cid in d:
        ir = load_ir(cid)
        if ir is not None:
            ir_series[cid] = ir

    # ── Donor pool for missing-IR imputation (normalised lifetime) ──
    donors = []
    for cid, (ir_cyc, ir_val) in ir_series.items():
        n = d[cid]['cycle']
        f = (ir_cyc - ir_cyc.min()) / max(ir_cyc.max() - ir_cyc.min(), 1e-9)
        donors.append((f, ir_val))

    def impute_ir(N):
        """Median IR trajectory over donors at each normalised lifetime."""
        fr = np.linspace(0, 1, N)
        out = np.zeros((N, 4))
        for j in range(4):
            for k, f in enumerate(fr):
                vals = [np.interp(f, fd, iv[:, j]) for fd, iv in donors]
                out[k, j] = float(np.median(vals))
        return out

    out = {}
    for cid, v in d.items():
        X = v['X'].astype(np.float64)
        cyc = v['cycle'].astype(np.float64)
        N = len(X)

        ir = load_ir(cid)
        if ir is not None:
            ir_cyc, ir_val = ir
            filled = np.zeros((N, 4))
            for k in range(N):
                idx = np.searchsorted(ir_cyc, cyc[k], side='right') - 1
                filled[k] = ir_val[max(idx, 0)]
            missing = 0.0
        else:
            filled = impute_ir(N)      # population-median trajectory
            missing = 1.0
        first = filled[0]
        ir_ratio = filled / (np.abs(first) + 1e-12)
        miss_col = np.full((N, 1), missing)

        # cumulative throughput
        cum_q = np.cumsum(X[:, 24])
        cum_e = np.cumsum(X[:, 25])
        extra = np.stack([cyc, cum_q, cum_e], axis=1)

        X2 = np.concatenate([X, filled, ir_ratio, miss_col, extra], axis=1)
        out[cid] = {'X': X2.astype(np.float32), 'cycle': v['cycle'],
                    'soh': v['soh'], 'test_cycles': v['test_cycles'],
                    'test_cap': v['test_cap']}
        print(f"{cid}: {X2.shape[0]} cycles, {X2.shape[1]} features, "
              f"IR={'yes' if ir is not None else 'imputed'}")
    np.save(OUT, out, allow_pickle=True)
    print(f"\nsaved {OUT}")


if __name__ == '__main__':
    build()
