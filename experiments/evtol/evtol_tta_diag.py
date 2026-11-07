#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose a selector for TTA: does plain-model epistemic uncertainty
predict whether TTA will help?

For every test cell (5-fold CV) report:
  plain RUL RMSE, TTA RUL RMSE, benefit (plain - TTA),
  mean epistemic on prefix / eval, protocol family.
Then Spearman(epistemic, benefit).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.evtol.evtol_tta import (
    load_cells, train_base, predict_rul, L, N_FOLDS, SEED, PREFIX_FRAC)
from experiments.evtol.evtol_tta_selective import prefix_adapt
from experiments.evtol.evtol_final import PROTOCOL
from src.evaluate import rmse
from src.model import SMoELSTM

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict_with_uq(model, fs, ls, cell, idx):
    D = cell['X'].shape[-1]
    Xn = fs.transform(cell['X'][idx].reshape(-1, D)).reshape(len(idx), L, D)
    x = torch.tensor(Xn, dtype=torch.float32).to(device)
    with torch.no_grad():
        g, nu, al, be = model(x)
        out = SMoELSTM.predict(g, nu, al, be)
        epis = out['epistemic'].cpu().numpy().ravel()
    # scale epistemic variance to original RUL units
    scale = float(ls.data_range_[0]) ** 2
    return float(np.mean(epis) * scale)


if __name__ == '__main__':
    cells = load_cells()
    ids = sorted(cells)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ids))
    folds = [[ids[i] for i in perm[k::N_FOLDS]] for k in range(N_FOLDS)]
    print("cell      protocol          plain    TTA   benefit  epis_prefix  epis_eval")

    rows = []
    for k, test_ids in enumerate(folds):
        train_ids = [c for c in ids if c not in test_ids]
        model, fs, ls = train_base(cells, train_ids)
        for cid in test_ids:
            cell = cells[cid]
            end_c = cell['end_cycle']
            pre_idx = np.where(end_c <= PREFIX_FRAC * cell['eol'])[0]
            ev_idx = np.where(end_c > PREFIX_FRAC * cell['eol'])[0]
            if len(pre_idx) < 60 or len(ev_idx) < 30:
                continue
            epis_pre = predict_with_uq(model, fs, ls, cell, pre_idx)
            epis_ev = predict_with_uq(model, fs, ls, cell, ev_idx)
            p0 = predict_rul(model, fs, ls, cell, ev_idx)
            r_plain = rmse(cell['y'][ev_idx], p0)

            m_tta, ev2, mse_tta = prefix_adapt(model, fs, cell, train_lstm=True)
            if m_tta is None:
                continue
            p1 = predict_rul(m_tta, fs, ls, cell, ev2)
            r_tta = rmse(cell['y'][ev2], p1)
            bene = r_plain - r_tta
            fam = PROTOCOL.get(cid, '?')
            rows.append((cid, fam, r_plain, r_tta, bene, epis_pre, epis_ev))
            print(f"{cid:6s} {fam:16s} {r_plain:6.1f} {r_tta:6.1f} "
                  f"{bene:+8.1f} {epis_pre:12.4f} {epis_ev:10.4f}", flush=True)

    bene = np.array([r[4] for r in rows])
    ep = np.array([r[6] for r in rows])
    rho, p = spearmanr(ep, bene)
    print(f"\nSpearman(epis_eval, benefit) = {rho:.3f} (p={p:.3f})")
    print(f"plain mean={np.mean([r[2] for r in rows]):.2f}, "
          f"TTA mean={np.mean([r[3] for r in rows]):.2f}, "
          f"oracle-select mean={np.mean([min(r[2], r[3]) for r in rows]):.2f}")
