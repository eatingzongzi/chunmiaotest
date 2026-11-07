#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract per-cycle features from the NASA accelerated-life dataset.

A mission = a contiguous block with mission_type set (0 = 2.5 A reference
discharge, 1 = regular/random mission). For each mission:
  - discharge stats: Ah, mean/max current, mean/min voltage, max temp,
    duration, energy;
  - preceding charge stats: duration, max charger voltage, max temp;
  - cumulative Ah and cycle index.
Relative features are formed w.r.t. the first mission, so the values are
comparable across packs and (approximately) across datasets.

Usage:
  python experiments/evtol/nasa_alt_features.py --pack battery00
  python experiments/evtol/nasa_alt_features.py --all
"""
import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import pandas as pd

ROOT = 'data/nasa_alt/battery_alt_dataset'


def blocks(mask):
    """Start/end indices of contiguous True blocks."""
    m = np.asarray(mask)
    if not m.any():
        return []
    d = np.diff(np.r_[False, m, False].astype(int))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))


def process_pack(path):
    df = pd.read_csv(path, usecols=['time', 'mode', 'voltage_charger',
                                    'temperature_battery', 'voltage_load',
                                    'current_load', 'mission_type'])
    t = df['time'].values.astype(float)
    mode = df['mode'].values
    mt = df['mission_type'].values
    is_m = ~np.isnan(mt)

    # charge blocks (mode == 1)
    ch_blocks = blocks(mode == 1)
    ch_stats = []
    for s, e in ch_blocks:
        dur = t[e - 1] - t[s]
        ch_stats.append((t[e - 1], dur,
                         np.nanmax(df['voltage_charger'].values[s:e]),
                         np.nanmax(df['temperature_battery'].values[s:e])))

    # mission blocks
    m_blocks = blocks(is_m)
    rows = []
    for s, e in m_blocks:
        seg = df.iloc[s:e]
        tt = seg['time'].values.astype(float)
        cur = seg['current_load'].values.astype(float)
        vol = seg['voltage_load'].values.astype(float)
        tmp = seg['temperature_battery'].values.astype(float)
        dur = tt[-1] - tt[0] if len(tt) > 1 else 0.0
        dt = np.diff(tt, prepend=tt[0]) if len(tt) else np.array([])
        ah = float(np.nansum(cur * dt) / 3600.0) if len(dt) else 0.0
        wh = float(np.nansum(vol * cur * dt) / 3600.0) if len(dt) else 0.0
        # preceding charge
        t0 = tt[0]
        prev = [c for c in ch_stats if c[0] <= t0]
        cs = prev[-1] if prev else (np.nan, np.nan, np.nan, np.nan)
        rows.append([ah, float(np.nanmean(cur)), float(np.nanmax(cur)),
                     float(np.nanmean(vol)), float(np.nanmin(vol)),
                     float(np.nanmax(tmp)), dur, wh,
                     cs[1], cs[2], cs[3], int(mt[s])])
    A = np.array(rows, dtype=np.float64)
    if len(A) < 10:
        return None
    # relative features w.r.t. the first reference mission (or first mission)
    ref_idx = np.where(A[:, 11] == 0)[0]
    b = A[ref_idx[0]] if len(ref_idx) else A[0]
    eps = 1e-9
    X = np.stack([
        A[:, 0] / (abs(b[0]) + eps),            # Ah retention proxy
        A[:, 1] / (abs(b[1]) + eps),            # mean-current ratio
        A[:, 2] / (abs(b[2]) + eps),            # max-current ratio
        A[:, 3] / (abs(b[3]) + eps),            # mean-voltage ratio
        A[:, 4] / (abs(b[4]) + eps),            # min-voltage ratio
        A[:, 5] - b[5],                          # temperature delta
        A[:, 6] / (abs(b[6]) + eps),            # duration ratio
        A[:, 7] / (abs(b[7]) + eps),            # energy ratio
        A[:, 8] / (abs(b[8]) + eps),            # charge duration ratio
        A[:, 9] / (abs(b[9]) + eps),            # charge Vmax ratio
        A[:, 10] - b[10],                        # charge temp delta
        np.arange(len(A)) / 1000.0,             # cycle index (scaled)
        np.cumsum(A[:, 0]) / (abs(b[0]) * 100 + eps),  # cumulative Ah
    ], axis=1)
    # reference capacity series and EOL at 80% (drop invalid tests)
    ref_cap = A[ref_idx, 0]
    ref_cyc = ref_idx
    ok = ref_cap > 0.5
    ref_cap, ref_cyc = ref_cap[ok], ref_cyc[ok]
    eol = np.nan
    if len(ref_cap) >= 3:
        below = np.where(ref_cap < 0.8 * ref_cap[0])[0]
        if len(below):
            i = below[0]
            eol = (ref_cyc[i - 1] + (ref_cap[i - 1] - 0.8 * ref_cap[0]) /
                   (ref_cap[i - 1] - ref_cap[i]) * (ref_cyc[i] - ref_cyc[i - 1])
                   ) if i > 0 else ref_cyc[0]
    return {'X': X.astype(np.float32), 'cycle': np.arange(len(A)),
            'ref_cap': ref_cap, 'ref_cyc': ref_cyc, 'eol': eol,
            'mission_type': A[:, 11].astype(int)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', default=None)
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    if args.pack:
        path = os.path.join(ROOT, 'regular_alt_batteries', f'{args.pack}.csv')
        r = process_pack(path)
        print(args.pack, 'cycles', None if r is None else len(r['X']),
              'eol', None if r is None else r['eol'])
    elif args.all:
        out = {}
        for folder in ['regular_alt_batteries', 'recommissioned_batteries',
                       'second_life_batteries']:
            for fn in sorted(os.listdir(os.path.join(ROOT, folder))):
                if not fn.endswith('.csv'):
                    continue
                pack = fn[:-4]
                try:
                    r = process_pack(os.path.join(ROOT, folder, fn))
                    if r is None:
                        continue
                    out[pack] = r
                    print(f"{pack}: {len(r['X'])} missions, "
                          f"ref tests={len(r['ref_cap'])}, eol={r['eol']}",
                          flush=True)
                except Exception as e:
                    print(f"{pack}: FAIL {e}")
        np.save('data/nasa_alt/nasa_alt_features.npy', out, allow_pickle=True)
        print(f"saved {len(out)} packs")
