#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract per-cycle features from the CMU eVTOL dataset.

Cycle structure (from exploration):
  capacity-test cycles (every ~51): segments [0,1,2,3,4,5,6,7,8],
      all discharge segments at 0.2C -> true capacity Q.
  mission cycles: [0(CC charge),1(CV),3,4(takeoff ~15A),
      5(cruise ~4A),6(landing ~17A),7(rest)].
SOH at mission cycles is linearly interpolated from the periodic
capacity tests. Saves per-cell feature matrices + SOH labels.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

DATA = 'data/evtol'
OUT = 'data/evtol_features.npz'

FEATURES = [
    'cc_dur_s', 'cc_Imean_mA', 'cv_dur_s', 'cv_Q_mAh', 'charge_E_Wh', 'charge_Tmax_C',
    'to_dur_s', 'to_Imean_mA', 'to_Imax_mA', 'to_Vmin_V', 'to_Vmean_V', 'to_Tmax_C',
    'cr_dur_s', 'cr_Imean_mA', 'cr_Imax_mA', 'cr_Vmin_V', 'cr_Vmean_V', 'cr_Tmax_C',
    'ld_dur_s', 'ld_Imean_mA', 'ld_Imax_mA', 'ld_Vmin_V', 'ld_Vmean_V', 'ld_Tmax_C',
    'mission_Q_Ah', 'mission_E_Wh', 'rest_dur_s', 'T_start_C',
]


def cell_features(fn):
    df = pd.read_csv(os.path.join(DATA, fn))
    ns = df['Ns'].values
    # global cycle reconstruction (tester resets cycleNumber)
    start = (ns == 0) & (np.r_[True, ns[:-1] != 0])
    g = np.cumsum(start) - 1
    ncyc = g.max() + 1

    # capacity-test cycles: contain Ns==2 and Ns==8
    has2 = pd.Series(ns == 2).groupby(g).any().values
    has8 = pd.Series(ns == 8).groupby(g).any().values
    is_test = has2 & has8

    grp = df.groupby(g)
    cap_all = grp['QDischarge_mA_h'].max().values / 1000.0
    test_cycles = np.where(is_test)[0]
    test_cap = cap_all[test_cycles]

    rows, cyc_idx = [], []
    for c in range(ncyc):
        if is_test[c]:
            continue
        sub = df[g == c]
        if len(sub) < 100:
            continue
        f = {}
        for name, seg in [('cc', 0), ('cv', 1), ('to', 4), ('cr', 5), ('ld', 6)]:
            s = sub[sub['Ns'] == seg]
            if len(s) == 0:
                continue
            dur = float(s['time_s'].max() - s['time_s'].min())
            if name == 'cc':
                f.update(cc_dur_s=dur, cc_Imean_mA=float(s['I_mA'].mean()))
            elif name == 'cv':
                f.update(cv_dur_s=dur, cv_Q_mAh=float(s['QCharge_mA_h'].max()),
                         charge_E_Wh=float(s['EnergyCharge_W_h'].max()),
                         charge_Tmax_C=float(s['Temperature__C'].max()))
            else:
                f.update({
                    f'{name}_dur_s': dur,
                    f'{name}_Imean_mA': float(s['I_mA'].mean()),
                    f'{name}_Imax_mA': float(-s['I_mA'].min()),
                    f'{name}_Vmin_V': float(s['Ecell_V'].min()),
                    f'{name}_Vmean_V': float(s['Ecell_V'].mean()),
                    f'{name}_Tmax_C': float(s['Temperature__C'].max()),
                })
        rest = sub[sub['Ns'] == 7]
        f['rest_dur_s'] = float(rest['time_s'].max() - rest['time_s'].min()) if len(rest) else 0.0
        f['mission_Q_Ah'] = float(sub['QDischarge_mA_h'].max()) / 1000.0
        f['mission_E_Wh'] = float(sub['EnergyDischarge_W_h'].max())
        f['T_start_C'] = float(sub['Temperature__C'].iloc[0])
        rows.append([f.get(k, 0.0) for k in FEATURES])
        cyc_idx.append(c)

    X = np.array(rows, dtype=np.float64)
    cyc_idx = np.array(cyc_idx)
    # interpolate SOH (capacity/initial) onto mission cycles
    soh_at_tests = test_cap / test_cap[0]
    # mission cycles between tests: linear interp in cycle index
    soh = np.interp(cyc_idx, test_cycles, soh_at_tests)
    return X, cyc_idx, soh, test_cycles, test_cap


if __name__ == '__main__':
    files = sorted(f for f in os.listdir(DATA)
                   if f.endswith('.csv') and 'impedance' not in f)
    out = {}
    for fn in files:
        cid = fn.replace('.csv', '')
        X, cyc, soh, tcyc, tcap = cell_features(fn)
        out[cid] = {'X': X, 'cycle': cyc, 'soh': soh,
                    'test_cycles': tcyc, 'test_cap': tcap}
        print(f"{cid}: {len(cyc)} mission cycles, {len(tcyc)} tests, "
              f"SOH {soh[0]:.3f}->{soh[-1]:.3f}", flush=True)
    np.save(OUT, out, allow_pickle=True)
    print(f"\nsaved {OUT} ({len(out)} cells, {len(FEATURES)} features)")
