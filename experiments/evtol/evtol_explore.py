#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Explore the CMU eVTOL dataset: segments, cycles, capacity fade."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import pandas as pd

DATA = 'data/evtol'
files = sorted(f for f in os.listdir(DATA) if f.endswith('.csv') and 'impedance' not in f)

print(f"{len(files)} cell files\n")
for fn in files[:3]:
    df = pd.read_csv(os.path.join(DATA, fn), nrows=200000)
    print(f"=== {fn}: {len(df)} rows ===")
    print("  columns:", list(df.columns))
    print("  Ns values:", sorted(df['Ns'].dropna().unique().tolist())[:20])
    print("  cycles: ", df['cycleNumber'].min(), "->", df['cycleNumber'].max())
    print("  V range:", round(df['Ecell_V'].min(), 3), "-", round(df['Ecell_V'].max(), 3))
    print("  T range:", round(df['Temperature__C'].min(), 2), "-", round(df['Temperature__C'].max(), 2))
    print()

# Per-cell capacity fade summary
print("=== Per-cell capacity fade (discharge Ah per cycle) ===")
print(f"{'cell':8s} {'cycles':>7s} {'Q0(Ah)':>8s} {'Qend(Ah)':>9s} {'drop%':>7s}")
rows = []
for fn in files:
    df = pd.read_csv(os.path.join(DATA, fn))
    # capacity per cycle: max cumulative discharge charge within a cycle
    g = df.groupby('cycleNumber')['QDischarge_mA_h']
    q = g.max() / 1000.0  # Ah
    q = q[q > 0.1]  # drop cycles without discharge data
    if len(q) < 10:
        continue
    q0, qe = q.iloc[0], q.iloc[-1]
    rows.append((fn.replace('.csv', ''), len(q), q0, qe, (q0 - qe) / q0 * 100))
    print(f"{rows[-1][0]:8s} {len(q):7d} {q0:8.3f} {qe:9.3f} {rows[-1][4]:7.1f}")
