#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eVTOL features v5: v4 (46) + discharge-curve shape (6) + IC peak (2).

For each mission cycle, from the raw voltage/current time series:
  - integrate current to obtain discharged capacity Q(t);
  - resample voltage at 6 equal-Q points (curve shape);
  - compute dQ/dV and extract the dominant peak amplitude / voltage.
All features are per-cycle and causal (no future information).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import pandas as pd

DATA = 'data/evtol'
IN = 'data/evtol_features_v4.npy'
OUT = 'data/evtol_features_v5.npy'
N_V = 6


def cycle_curve_features(v, i, t):
    """v: voltage, i: current (mA), t: time (s) for one cycle's discharge."""
    if len(v) < 50:
        return None
    dt = np.diff(t, prepend=t[0])
    q = np.cumsum(-i / 1000.0 * dt / 3600.0)   # Ah
    if q[-1] < 0.3:
        return None
    # resample V at 6 equal-capacity points
    frac = np.linspace(0, 1, N_V)
    qq = q[-1] * frac
    v_r = np.interp(qq, q, v)
    # dQ/dV on the discharge curve
    dv = np.diff(v)
    dq = np.diff(q)
    ok = np.abs(dv) > 1e-4
    if ok.sum() < 10:
        return None
    dqdv = dq[ok] / dv[ok]
    k = int(np.argmax(np.abs(dqdv)))
    peak_val = float(np.abs(dqdv[k]))
    peak_v = float(v[:-1][ok][k])
    return np.concatenate([v_r, [peak_val, peak_v]])


def build():
    d = np.load(IN, allow_pickle=True).item()
    out = {}
    for cid, v4 in d.items():
        fn = os.path.join(DATA, f'{cid}.csv')
        df = pd.read_csv(fn, usecols=['time_s', 'Ecell_V', 'I_mA',
                                      'Ns', 'cycleNumber'])
        ns = df['Ns'].values
        start = (ns == 0) & (np.r_[True, ns[:-1] != 0])
        g = np.cumsum(start) - 1

        feats = np.zeros((g.max() + 1, N_V + 2), dtype=np.float64)
        for c, sub in df.groupby(g):
            dis = sub[(sub['I_mA'] < -100) & (sub['Ecell_V'] > 2.0)]
            if len(dis) < 50:
                continue
            r = cycle_curve_features(dis['Ecell_V'].values.astype(float),
                                     dis['I_mA'].values.astype(float),
                                     dis['time_s'].values.astype(float))
            if r is not None and c < len(feats):
                feats[c] = r

        cyc = v4['cycle'].astype(int)
        cyc = np.clip(cyc, 0, len(feats) - 1)
        new = feats[cyc]                        # align to mission cycles
        X = np.concatenate([v4['X'], new], axis=1)
        out[cid] = {'X': X.astype(np.float32), 'cycle': v4['cycle'],
                    'soh': v4['soh'], 'test_cycles': v4['test_cycles'],
                    'test_cap': v4['test_cap']}
        print(f"{cid}: {X.shape[0]} cycles, {X.shape[1]} features", flush=True)
    np.save(OUT, out, allow_pickle=True)
    print(f"\nsaved {OUT}")


if __name__ == '__main__':
    build()
