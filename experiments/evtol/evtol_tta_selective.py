#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Selective test-time adaptation for eVTOL RUL.

Unconditional TTA helps hard cells (VAH11 -73%) but destroys easy ones.
Here the decision is self-supervised: on the test cell's prefix,
  - plain_mse : fit an aux SOH head on FROZEN features
  - tta_mse   : additionally fine-tune the LSTM, then fit the aux head
  - apply TTA only if tta_mse < plain_mse * (1 - margin)
No RUL labels are used for the decision.
"""
import sys, os, copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.evtol.evtol_tta import (
    load_cells, train_base, predict_rul, L, N_FOLDS, SEED,
    PREFIX_FRAC, TTA_EPOCHS, TTA_LR)
from src.evaluate import rmse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MARGIN = 0.10   # require >=10% prefix-SOH MSE improvement to accept TTA


def prefix_adapt(model, fs, cell, train_lstm):
    """Train an aux SOH head (optionally adapting the LSTM) on the prefix."""
    D = cell['X'].shape[-1]
    Xn = fs.transform(cell['X'].reshape(-1, D)).reshape(len(cell['X']), L, D)
    X = torch.tensor(Xn, dtype=torch.float32).to(device)

    end_c = cell['end_cycle']
    prefix_end = PREFIX_FRAC * cell['eol']
    pre_idx = np.where(end_c <= prefix_end)[0]
    ev_idx = np.where(end_c > prefix_end)[0]
    if len(pre_idx) < 60 or len(ev_idx) < 30:
        return None, None, None

    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad = False
    if train_lstm:
        for p in m.lstm.parameters():
            p.requires_grad = True
        m.train()
        m.moe.eval(); m.evidence.eval()
        try:
            m.lstm.flatten_parameters()
        except Exception:
            pass

    aux = nn.Linear(128, 1).to(device)
    soh_pre = cell['soh'][pre_idx]
    mu, sd = soh_pre.mean(), soh_pre.std() + 1e-9
    y = torch.tensor((soh_pre - mu) / sd, dtype=torch.float32).to(device)
    x_pre = X[pre_idx]

    params = list(aux.parameters()) + (list(m.lstm.parameters()) if train_lstm else [])
    opt = torch.optim.Adam(params, lr=TTA_LR if train_lstm else 1e-3)
    for ep in range(TTA_EPOCHS):
        opt.zero_grad()
        _, _, _, _, feats = m(x_pre, return_features=True)
        pred = aux(feats).squeeze(-1)
        loss = F.mse_loss(pred, y)
        loss.backward(); opt.step()

    with torch.no_grad():
        _, _, _, _, feats = m(x_pre, return_features=True)
        mse = F.mse_loss(aux(feats).squeeze(-1), y).item()
    m.eval()
    return m, ev_idx, float(mse)


if __name__ == '__main__':
    cells = load_cells()
    ids = sorted(cells)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ids))
    folds = [[ids[i] for i in perm[k::N_FOLDS]] for k in range(N_FOLDS)]
    print(f"{len(ids)} cells, selective TTA (margin {MARGIN:.0%})")

    plain_all, tta_all, sel_all = [], [], []
    for k, test_ids in enumerate(folds):
        train_ids = [c for c in ids if c not in test_ids]
        model, fs, ls = train_base(cells, train_ids)
        for cid in test_ids:
            cell = cells[cid]
            end_c = cell['end_cycle']
            ev_idx = np.where(end_c > PREFIX_FRAC * cell['eol'])[0]
            if len(ev_idx) < 30:
                continue
            p_plain = predict_rul(model, fs, ls, cell, ev_idx)
            r_plain = rmse(cell['y'][ev_idx], p_plain)

            # plain aux fit (frozen features)
            _, _, mse_plain = prefix_adapt(model, fs, cell, train_lstm=False)
            # adapted model + aux fit
            m_tta, ev2, mse_tta = prefix_adapt(model, fs, cell, train_lstm=True)
            if m_tta is None:
                plain_all.append(r_plain); tta_all.append(r_plain); sel_all.append(r_plain)
                continue
            p_tta = predict_rul(m_tta, fs, ls, cell, ev2)
            r_tta = rmse(cell['y'][ev2], p_tta)

            use_tta = (mse_tta is not None and mse_plain is not None and
                       mse_tta < mse_plain * (1 - MARGIN))
            r_sel = r_tta if use_tta else r_plain
            plain_all.append(r_plain); tta_all.append(r_tta); sel_all.append(r_sel)
            print(f"  {cid}: plain={r_plain:6.1f} tta={r_tta:6.1f} "
                  f"mse_plain={mse_plain:.3f} mse_tta={mse_tta:.3f} "
                  f"-> {'TTA' if use_tta else 'plain'}", flush=True)

    print(f"\nSelective TTA (post-prefix evaluation):")
    print(f"  plain   : {np.mean(plain_all):.2f}+/-{np.std(plain_all):.2f}")
    print(f"  TTA     : {np.mean(tta_all):.2f}+/-{np.std(tta_all):.2f}")
    print(f"  selected: {np.mean(sel_all):.2f}+/-{np.std(sel_all):.2f}")
