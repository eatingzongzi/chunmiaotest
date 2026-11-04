#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert MIT data to the NASA-model input format: cumulative discharge
energy (kWh) vs max discharge capacity per cycle, per battery.

Their model: per-battery Bayesian MLP, X = cumulative energy, Y = q_max.
Discharge energy per cycle: E_k = sum_i Vdlin_i * dQdlin_i (numeric
integral of the V-Q curve on the 1000-point grid).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import h5py

f = h5py.File('data/mit_stanford/2017-06-30_batchdata_updated_struct_errorcorrect.mat', 'r')
batch = f['batch']
vdlin = f[batch['Vdlin'][0, 0]][()].flatten().astype(np.float64)

cells = {}
for i in range(batch['summary'].shape[0]):
    try:
        summ = f[batch['summary'][i, 0]]
        cyc = f[batch['cycles'][i, 0]]
        qd_all = summ['QDischarge'][()].flatten().astype(np.float64)
        cycle_nums = summ['cycle'][()].flatten().astype(np.int32)
        qdlin_refs = cyc['Qdlin'][()].flatten()

        energies = []
        for j in range(len(qd_all)):
            try:
                qd = f[qdlin_refs[j]][()].flatten().astype(np.float64)
                dq = np.diff(qd, prepend=0.0)
                # E = sum V * dQ  (Ah*V -> Wh), divide by 1000 for kWh
                energies.append(float(np.sum(vdlin * dq) / 1000.0))
            except Exception:
                energies.append(np.nan)
        energies = np.array(energies)
        ok = ~np.isnan(energies)
        if ok.sum() < 30:
            continue
        cum_kwh = np.cumsum(energies[ok])
        cells[i] = {
            'cum_kwh': cum_kwh,
            'q_max': qd_all[ok],          # Ah per cycle
            'cycle': cycle_nums[ok],
        }
    except Exception:
        continue
f.close()

out = {k: cells[k] for k in sorted(cells)}
np.save('mit_energy_curves.npy', out, allow_pickle=True)
print(f"{len(out)} cells converted")
for k in [0, 8, 29]:
    if k in out:
        c = out[k]
        print(f"  cell {k:2d}: {len(c['cum_kwh'])} cycles, "
              f"energy range {c['cum_kwh'][0]:.3f}->{c['cum_kwh'][-1]:.3f} kWh, "
              f"q range {c['q_max'].min():.3f}->{c['q_max'].max():.3f} Ah")
