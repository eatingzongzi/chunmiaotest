#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eVTOL features v4: v3 (40) + SOH history (2) + T-normalised IR (4) = 46.

New features
  soh_last          last measured capacity-test SOH (forward-filled)
  cycles_since_test cycles since that measurement
  ir_corr_ratio_*   4 IR ratios corrected to 25 C via Arrhenius
                    R_ref = R(T) * exp(Ea/kB * (1/T_ref - 1/T))
                    with Ea = 0.4 eV (typical SEI/charge-transfer value)
All features are causal (only past measurements).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np

IN = 'data/evtol_features_v3.npy'
OUT = 'data/evtol_features_v4.npy'
EA_EV = 0.4
KB = 8.617333e-5   # eV/K
T_REF = 298.15     # 25 C


def build():
    d = np.load(IN, allow_pickle=True).item()
    out = {}
    for cid, v in d.items():
        X = v['X'].astype(np.float64)          # (N, 40)
        cyc = v['cycle'].astype(np.float64)
        tcyc = np.asarray(v['test_cycles'], dtype=float)
        tcap = np.asarray(v['test_cap'], dtype=float)
        N = len(X)

        # SOH history (causal forward-fill from capacity tests)
        soh = np.zeros(N)
        since = np.zeros(N)
        for k in range(N):
            j = np.searchsorted(tcyc, cyc[k], side='right') - 1
            j = max(j, 0)
            soh[k] = tcap[j]
            since[k] = cyc[k] - tcyc[j]

        # temperature-corrected IR ratios
        # T_rep = mean of charge/takeoff/cruise/landing Tmax and T_start
        T_c = X[:, [5, 11, 17, 23, 27]].mean(axis=1)
        T_k = T_c + 273.15
        ir = X[:, 28:32]
        corr = ir * np.exp(EA_EV / KB * (1.0 / T_REF - 1.0 / T_k))[:, None]
        corr_ratio = corr / (np.abs(corr[0]) + 1e-12)

        X4 = np.concatenate([X,
                             np.stack([soh, since], axis=1),
                             corr_ratio], axis=1)
        out[cid] = {'X': X4.astype(np.float32), 'cycle': v['cycle'],
                    'soh': v['soh'], 'test_cycles': v['test_cycles'],
                    'test_cap': v['test_cap']}
        print(f"{cid}: {N} cycles, {X4.shape[1]} features")
    np.save(OUT, out, allow_pickle=True)
    print(f"\nsaved {OUT}")


if __name__ == '__main__':
    build()
