#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build CMU eVTOL features in the SAME 12-dim generic space as NASA ALT.

Matches the NASA features (after dropping the uninformative charge-Vmax
ratio):
  [Ah ratio, Imean ratio, Imax ratio, Vmean ratio, Vmin ratio, T delta,
   duration ratio, energy ratio, charge-duration ratio, charge-T delta,
   cycle/1000, cumulative-Ah ratio]
This enables pretraining on NASA ALT and fine-tuning on CMU.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np

EOL_THR = 0.85
OUT = 'data/evtol_cmu_generic.npy'


def build():
    d = np.load('data/evtol_features_v4.npy', allow_pickle=True).item()
    out = {}
    for cid, v in d.items():
        X28 = v['X'][:, :28].astype(np.float64)   # base features
        cyc = v['cycle'].astype(np.float64)
        soh = v['soh']
        below = np.where(soh < EOL_THR)[0]
        if len(below) == 0:
            continue
        i = below[0]
        eol = (cyc[i - 1] + (soh[i - 1] - EOL_THR) /
               (soh[i - 1] - soh[i]) * (cyc[i] - cyc[i - 1])) if i > 0 else cyc[0]
        keep = cyc <= eol
        X28, cyc = X28[keep], cyc[keep]
        n = len(X28)

        q = X28[:, 24]                                  # mission Ah
        dur = X28[:, 6] + X28[:, 12] + X28[:, 18]       # discharge duration
        imax = np.maximum.reduce([X28[:, 8], X28[:, 14], X28[:, 20]])
        imean = np.where(dur > 0, q * 3600.0 / np.maximum(dur, 1e-9), 0.0)
        vmean = (X28[:, 10] + X28[:, 16] + X28[:, 22]) / 3.0
        vmin = np.minimum.reduce([X28[:, 9], X28[:, 15], X28[:, 21]])
        tmax = np.maximum.reduce([X28[:, 11], X28[:, 17], X28[:, 23]])
        energy = q * vmean
        ch_dur = X28[:, 0] + X28[:, 2]
        ch_tmax = X28[:, 5]

        b = 0  # first mission baseline
        eps = 1e-9
        F = np.stack([
            q / (abs(q[b]) + eps),
            imean / (abs(imean[b]) + eps),
            imax / (abs(imax[b]) + eps),
            vmean / (abs(vmean[b]) + eps),
            vmin / (abs(vmin[b]) + eps),
            tmax - tmax[b],
            dur / (abs(dur[b]) + eps),
            energy / (abs(energy[b]) + eps),
            ch_dur / (abs(ch_dur[b]) + eps),
            ch_tmax - ch_tmax[b],
            cyc / 1000.0,
            np.cumsum(q) / (abs(q[0]) * 100 + eps),
        ], axis=1)
        rul = eol - cyc
        out[cid] = {'X': F.astype(np.float32), 'cycle': cyc,
                    'rul': rul.astype(np.float32), 'eol': float(eol)}
        print(f"{cid}: {n} cycles, eol={eol:.0f}, X={F.shape}")
    np.save(OUT, out, allow_pickle=True)
    print(f"\nsaved {OUT}")


if __name__ == '__main__':
    build()
