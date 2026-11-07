#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test-time adaptation (TTA) for eVTOL RUL.

Protocol: 5-fold cell CV. For each test cell:
  - the first 50% of its life is the adaptation prefix;
  - measured SOH (forward-filled from capacity tests) on the prefix gives
    a self-supervised target;
  - only the LSTM encoder (plus a temporary SOH head) is fine-tuned;
    MoE / evidence / router stay frozen;
  - RUL is then evaluated on the remaining life and compared with the
    unadapted model on the same windows.
"""
import sys, os, copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from src.train import set_seed, get_device, prepare_dataloaders
from src.model import SMoELSTM
from src.loss import EvidentialRegressionLoss
from src.evaluate import rmse

device = get_device('cuda')
L = 30
EOL_THR = 0.85
N_FOLDS = 5
SEED = 42
N_EXPERTS = 10
TOP_K = 4
PREFIX_FRAC = 0.5
TTA_EPOCHS = 30
TTA_LR = 1e-4


def load_cells():
    d = np.load('data/evtol_features_v4.npy', allow_pickle=True).item()
    cells = {}
    for cid, v in d.items():
        soh, cyc = v['soh'], v['cycle']
        tcyc = np.asarray(v['test_cycles'], dtype=float)
        tcap = np.asarray(v['test_cap'], dtype=float)
        below = np.where(soh < EOL_THR)[0]
        if len(below) == 0:
            continue
        i = below[0]
        eol = (cyc[i - 1] + (soh[i - 1] - EOL_THR) /
               (soh[i - 1] - soh[i]) * (cyc[i] - cyc[i - 1])) if i > 0 else cyc[0]
        keep = cyc <= eol
        X, c = v['X'][keep].copy(), cyc[keep]
        N = len(X)
        if N < L + 10:
            continue
        # causal (forward-filled) SOH at each cycle
        soh_ff = np.zeros(N)
        for k in range(N):
            j = np.searchsorted(tcyc, c[k], side='right') - 1
            soh_ff[k] = tcap[max(j, 0)]
        rul = eol - c
        Xw = np.stack([X[j:j + L] for j in range(N - L + 1)], 0)
        yw = np.array([rul[j + L - 1] for j in range(N - L + 1)])
        end_c = np.array([c[j + L - 1] for j in range(N - L + 1)])
        soh_end = np.array([soh_ff[j + L - 1] for j in range(N - L + 1)])
        cells[cid] = {'X': Xw.astype(np.float32), 'y': yw.astype(np.float32),
                      'end_cycle': end_c, 'soh': soh_end, 'eol': eol}
    return cells


class DS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X); self.y = torch.tensor(y).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_base(cells, train_ids, seed=SEED):
    Xtr = np.concatenate([cells[c]['X'] for c in train_ids])
    ytr = np.concatenate([cells[c]['y'] for c in train_ids])
    N, Lw, D = Xtr.shape
    fs = MinMaxScaler().fit(Xtr.reshape(-1, D))
    Xtr_n = fs.transform(Xtr.reshape(-1, D)).reshape(N, Lw, D)
    ls = MinMaxScaler().fit(ytr.reshape(-1, 1))
    ytr_n = ls.transform(ytr.reshape(-1, 1)).ravel()

    set_seed(seed)
    loader, _ = prepare_dataloaders(DS(Xtr_n, ytr_n), 64, 0.0, seed)
    model = SMoELSTM(input_dim=D, hidden_dim=128, num_experts=N_EXPERTS,
                     top_k=TOP_K, lstm_layers=2, evidence_hidden=64).to(device)
    crit = EvidentialRegressionLoss(coeff=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    E = N_EXPERTS
    for epoch in range(100):
        model.train()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X, return_gates=True)
            g, nu, al, be, gi = out[:5]
            loss = crit(y, g, nu, al, be)
            gates, indices, feats, rough = gi
            loss = loss + 0.05 * model.moe.balance_loss(gates, indices, E)
            with torch.no_grad():
                outs = torch.stack([model.moe.experts[e](feats) for e in range(E)], 1)
                ge = torch.stack([model.evidence(outs[:, e])[0].squeeze(-1)
                                  for e in range(E)], 1)
                err = (ge - y).pow(2); best = err.argmin(-1)
            clean = model.moe.gate.clean_logits(torch.cat([feats, rough], -1))
            masked = clean.clone()
            masked.scatter_(-1, best.unsqueeze(-1), float('-inf'))
            top = masked.topk(TOP_K - 1, dim=-1).values.logsumexp(-1)
            hard = F.relu(top - clean.gather(-1, best.unsqueeze(-1)).squeeze(-1))
            loss = loss + 1.0 * hard.mean()
            loss.backward(); opt.step()
    model.eval()
    return model, fs, ls


def tta_adapt(model, fs, cell, prefix_frac=PREFIX_FRAC):
    """Fine-tune the LSTM on the cell's prefix with a SOH self-supervised head."""
    D = cell['X'].shape[-1]
    X = torch.tensor(fs.transform(cell['X'].reshape(-1, D)),
                     dtype=torch.float32).to(device)
    end_c = cell['end_cycle']
    prefix_end = prefix_frac * cell['eol']
    pre_idx = np.where(end_c <= prefix_end)[0]
    ev_idx = np.where(end_c > prefix_end)[0]
    if len(pre_idx) < 60 or len(ev_idx) < 30:
        return None, None, None

    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.lstm.parameters():
        p.requires_grad = True
    aux = nn.Linear(128, 1).to(device)

    soh_pre = cell['soh'][pre_idx]
    mu, sd = soh_pre.mean(), soh_pre.std() + 1e-9
    y_pre = torch.tensor((soh_pre - mu) / sd, dtype=torch.float32).to(device)
    x_pre = X[pre_idx]
    opt = torch.optim.Adam(list(m.lstm.parameters()) + list(aux.parameters()),
                           lr=TTA_LR)
    for ep in range(TTA_EPOCHS):
        opt.zero_grad()
        _, _, _, _, feats = m(x_pre, return_features=True)
        pred = aux(feats).squeeze(-1)
        loss = F.mse_loss(pred, y_pre)
        loss.backward(); opt.step()

    m.eval()
    return m, ev_idx, (mu, sd)


def predict_rul(model, fs, ls, cell, idx):
    D = cell['X'].shape[-1]
    x = torch.tensor(fs.transform(cell['X'][idx].reshape(-1, D)),
                     dtype=torch.float32).to(device)
    with torch.no_grad():
        g, *_ = model(x)
    return ls.inverse_transform(g.cpu().numpy().reshape(-1, 1)).ravel()


if __name__ == '__main__':
    cells = load_cells()
    ids = sorted(cells)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ids))
    folds = [[ids[i] for i in perm[k::N_FOLDS]] for k in range(N_FOLDS)]
    print(f"{len(ids)} cells, prefix {PREFIX_FRAC:.0%}, TTA epochs {TTA_EPOCHS}")

    all_plain, all_tta = [], []
    for k, test_ids in enumerate(folds):
        train_ids = [c for c in ids if c not in test_ids]
        model, fs, ls = train_base(cells, train_ids)
        for cid in test_ids:
            cell = cells[cid]
            end_c = cell['end_cycle']
            prefix_end = PREFIX_FRAC * cell['eol']
            ev_idx = np.where(end_c > prefix_end)[0]
            if len(ev_idx) < 30:
                continue
            p0 = predict_rul(model, fs, ls, cell, ev_idx)
            r0 = rmse(cell['y'][ev_idx], p0)
            m_tta, ev2, _ = tta_adapt(model, fs, cell)
            if m_tta is None:
                print(f"  {cid}: skip TTA (few prefix windows), plain={r0:.1f}")
                all_plain.append(r0); all_tta.append(r0)
                continue
            p1 = predict_rul(m_tta, fs, ls, cell, ev2)
            r1 = rmse(cell['y'][ev2], p1)
            r0b = rmse(cell['y'][ev2], predict_rul(model, fs, ls, cell, ev2))
            all_plain.append(r0b); all_tta.append(r1)
            print(f"  {cid}: eval_RMSE plain={r0b:6.1f} TTA={r1:6.1f} "
                  f"({(r1 - r0b) / max(r0b,1e-6) * 100:+.0f}%)", flush=True)

    print(f"\nTTA summary (post-prefix evaluation):")
    print(f"  plain: {np.mean(all_plain):.2f}+/-{np.std(all_plain):.2f}")
    print(f"  TTA  : {np.mean(all_tta):.2f}+/-{np.std(all_tta):.2f}")
