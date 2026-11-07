#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NASA hybrid Bayesian PINN on the eVTOL data (per-cell protocol).

For each cell: input = cumulative mission discharge energy (kWh) at the
capacity-test points, output = capacity (Ah). Fit the variational MLP on
the first 60% of capacity tests, predict the remaining 40%, then derive:
  - capacity RMSE / MAPE on the held-out tests
  - EOL error and RUL RMSE (EOL at 85% of initial capacity)
Same data and EOL definition as our SMoE-LSTM eVTOL experiments.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.comparison.bayes_pinn_pytorch import fit_per_battery

EOL_THR = 0.85
FRAC = 0.6


def build_energy_series():
    d = np.load('data/evtol_features.npy', allow_pickle=True).item()
    out = {}
    for cid, v in d.items():
        X, cyc = v['X'], v['cycle']
        tcyc = np.asarray(v['test_cycles'], dtype=float)
        tcap = np.asarray(v['test_cap'], dtype=float)
        # cumulative discharge throughput (Ah); the raw "energy" column is
        # negative during discharge, so use charge throughput instead
        mission_q = X[:, 24]                       # Ah per mission cycle
        cum_q = np.cumsum(mission_q)
        cum_at_test = []
        for tc in tcyc:
            idx = np.searchsorted(cyc, tc, side='right') - 1
            cum_at_test.append(cum_q[max(idx, 0)])
        out[cid] = {'cum_kwh': np.array(cum_at_test, dtype=float),
                    'cap': tcap, 'test_cycles': tcyc}
    return out


def crossing_cycle(x, y, level):
    """First x where y crosses below level (linear interpolation)."""
    below = np.where(y < level)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return x[0]
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    if y0 == y1:
        return x1
    return x0 + (y0 - level) / (y0 - y1) * (x1 - x0)


def main():
    data = build_energy_series()
    print(f"{len(data)} cells")
    # sanity check
    for cid, v in list(data.items())[:3]:
        print(f"  {cid}: n_test={len(v['cap'])}, "
              f"cap nan={np.isnan(v['cap']).sum()}, cum nan={np.isnan(v['cum_kwh']).sum()}, "
              f"cum range={np.nanmin(v['cum_kwh']):.3f}-{np.nanmax(v['cum_kwh']):.3f}")

    rows = []
    for cid, v in data.items():
        cum, cap, tcyc = v['cum_kwh'], v['cap'], v['test_cycles']
        n = len(cap)
        n_tr = max(3, int(n * FRAC))
        if n - n_tr < 2:
            continue
        if not (np.isfinite(cum).all() and np.isfinite(cap).all()):
            print(f"{cid}: skip (non-finite data)")
            continue
        m, s, _ = fit_per_battery(cum, cap, frac_train=FRAC, epochs=2000, seed=0)
        # held-out capacity metrics
        cap_rmse = float(np.sqrt(np.mean((m[n_tr:] - cap[n_tr:]) ** 2)))
        cap_mape = float(np.mean(np.abs((m[n_tr:] - cap[n_tr:]) / cap[n_tr:])) * 100)
        # EOL at 85% of initial capacity (true and predicted)
        q0 = cap[0]
        eol_true = crossing_cycle(tcyc, cap, EOL_THR * q0)
        eol_pred = crossing_cycle(tcyc, m, EOL_THR * q0)
        # RUL errors over held-out test points
        if np.isfinite(eol_true) and np.isfinite(eol_pred):
            rul_true = eol_true - tcyc[n_tr:]
            rul_pred = eol_pred - tcyc[n_tr:]
            rul_rmse = float(np.sqrt(np.mean((rul_pred - rul_true) ** 2)))
            eol_err = float(eol_pred - eol_true)
        else:
            rul_rmse = np.nan
            eol_err = np.nan
        rows.append((cid, cap_rmse, cap_mape, eol_err, rul_rmse))
        print(f"{cid}: capRMSE={cap_rmse:.4f} Ah MAPE={cap_mape:5.2f}% "
              f"EOLerr={eol_err:7.1f} RUL_RMSE={rul_rmse:7.1f}", flush=True)

    valid = [r for r in rows if np.isfinite(r[4])]
    print(f"\nNASA Bayesian PINN on eVTOL (per-cell, 60/40):")
    print(f"  cells={len(rows)}, valid EOL={len(valid)}")
    print(f"  mean cap RMSE={np.mean([r[1] for r in rows]):.4f} Ah, "
          f"MAPE={np.mean([r[2] for r in rows]):.2f}%")
    if valid:
        print(f"  mean EOL error={np.mean([abs(r[3]) for r in valid]):.1f} cycles, "
              f"mean RUL RMSE={np.mean([r[4] for r in valid]):.1f}")
    else:
        print("  no valid EOL crossings")


if __name__ == '__main__':
    main()
