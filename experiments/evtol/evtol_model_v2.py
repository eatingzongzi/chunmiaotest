#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eVTOL model v2: paper SMoE-LSTM + attention pooling + protocol FiLM
+ multi-task auxiliary heads (future SOH / IR).

Backbone remains the paper architecture (LSTM 2x128, sparse MoE 10/top-4,
evidential NIR head). Evaluation: 5-fold CV on the 18 usable cells.

Baseline v1 (same features, plain SMoELSTM): CV 48.54.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from src.train import set_seed, get_device, prepare_dataloaders
from src.model import SparseMoE, EvidenceLayer
from src.loss import EvidentialRegressionLoss
from src.evaluate import rmse

device = get_device('cuda')
L = 30
EOL_THR = 0.85
N_FOLDS = 5
SEED = 42
AUX_H = (50, 100)
AUX_W = 0.1


# ── Data ────────────────────────────────────────────────────────────
def load_cells():
    d = np.load('data/evtol_features_v4.npy', allow_pickle=True).item()
    cells = {}
    for cid, v in d.items():
        soh_traj, cyc = v['soh'], v['cycle']
        below = np.where(soh_traj < EOL_THR)[0]
        if len(below) == 0:
            continue
        i = below[0]
        eol = (cyc[i - 1] + (soh_traj[i - 1] - EOL_THR) /
               (soh_traj[i - 1] - soh_traj[i]) * (cyc[i] - cyc[i - 1])) if i > 0 else cyc[0]
        keep = cyc <= eol
        X, c = v['X'][keep].copy(), cyc[keep]
        s = np.asarray(soh_traj)[keep]
        if len(X) < L + 10:
            continue
        N = len(X)
        rul = eol - c
        Xw = np.stack([X[j:j + L] for j in range(N - L + 1)], 0)
        yw = np.array([rul[j + L - 1] for j in range(N - L + 1)])

        # protocol vector from the first 30 cycles (causal)
        x0 = X[:30]
        proto = np.array([
            x0[:, 1].mean() / 3000.0,          # charge C-rate proxy
            np.log1p(x0[:, 12].mean()) / 10.0,  # cruise duration
            x0[:, 13].mean() / 2000.0,          # cruise current
            x0[:, 27].mean() / 40.0,            # temperature
        ], dtype=np.float32)

        # auxiliary targets: future SOH (+50/+100) and IR ratio (+50)
        ir_ratio = X[:, 32:36].mean(axis=1)
        aux = []
        for h in AUX_H:
            aux.append(np.array([s[min(j + L - 1 + h, N - 1)]
                                 for j in range(N - L + 1)]))
        aux.append(np.array([ir_ratio[min(j + L - 1 + 50, N - 1)]
                             for j in range(N - L + 1)]))
        aux = np.stack(aux, axis=1).astype(np.float32)
        cells[cid] = {'X': Xw.astype(np.float32), 'y': yw.astype(np.float32),
                      'proto': proto, 'aux': aux}
    return cells


class DS(torch.utils.data.Dataset):
    def __init__(self, X, y, p, a):
        self.X = torch.tensor(X); self.y = torch.tensor(y).unsqueeze(-1)
        self.p = torch.tensor(p); self.a = torch.tensor(a)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.p[i], self.a[i]


# ── Model ───────────────────────────────────────────────────────────
class EVTOLV2(nn.Module):
    def __init__(self, input_dim, hidden=128, num_experts=10, top_k=4,
                 lstm_layers=2, evidence_hidden=64, proto_dim=4, n_aux=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, lstm_layers, batch_first=True,
                            dropout=0.1)
        self.attn = nn.Linear(hidden, 1)
        self.router = nn.Linear(hidden, 1)
        self.film = nn.Sequential(nn.Linear(proto_dim, 32), nn.ReLU(),
                                  nn.Linear(32, 2 * hidden))
        self.moe = SparseMoE(hidden, hidden, num_experts, top_k,
                             dropout=0.1, gate_input_dim=hidden + 1)
        self.evidence = EvidenceLayer(hidden, evidence_hidden)
        self.aux_head = nn.Linear(hidden, n_aux)

    def forward(self, x, proto, return_gates=True):
        lstm_out, _ = self.lstm(x)
        scores = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        h = (lstm_out * scores.unsqueeze(-1)).sum(dim=1)
        film = self.film(proto)
        scale, shift = film.chunk(2, dim=-1)
        h = h * (1.0 + torch.tanh(scale)) + shift
        rough = self.router(h)
        z, gi = self.moe(h, gate_input=torch.cat([h, rough], -1),
                         return_gates=True)
        gamma, nu, alpha, beta = self.evidence(z)
        aux = self.aux_head(h)
        return gamma, nu, alpha, beta, gi, aux


def train_eval(train_ids, test_ids, cells, seed=SEED):
    Xtr = np.concatenate([cells[c]['X'] for c in train_ids])
    ytr = np.concatenate([cells[c]['y'] for c in train_ids])
    ptr = np.stack([np.repeat(cells[c]['proto'][None, :], len(cells[c]['y']), 0)
                    for c in train_ids]).reshape(-1, 4)
    atr = np.concatenate([cells[c]['aux'] for c in train_ids])
    N, Lw, D = Xtr.shape
    fs = MinMaxScaler().fit(Xtr.reshape(-1, D))
    Xtr_n = fs.transform(Xtr.reshape(-1, D)).reshape(N, Lw, D)
    ls = MinMaxScaler().fit(ytr.reshape(-1, 1))
    ytr_n = ls.transform(ytr.reshape(-1, 1)).ravel()
    ps = MinMaxScaler().fit(ptr)
    ptr_n = ps.transform(ptr)
    am, asd = atr.mean(0), atr.std(0) + 1e-9
    atr_n = (atr - am) / asd

    set_seed(seed)
    loader, _ = prepare_dataloaders(DS(Xtr_n, ytr_n, ptr_n, atr_n), 64, 0.0, seed)
    model = EVTOLV2(D).to(device)
    crit = EvidentialRegressionLoss(coeff=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for epoch in range(100):
        model.train()
        for X, y, p, a in loader:
            X, y, p, a = X.to(device), y.to(device), p.to(device), a.to(device)
            opt.zero_grad()
            g, nu, al, be, gi, aux = model(X, p)
            loss = crit(y, g, nu, al, be)
            gates, indices, feats, rough = gi
            loss = loss + 0.05 * model.moe.balance_loss(gates, indices, 10)
            with torch.no_grad():
                outs = torch.stack([model.moe.experts[e](feats) for e in range(10)], 1)
                ge = torch.stack([model.evidence(outs[:, e])[0].squeeze(-1)
                                  for e in range(10)], 1)
                err = (ge - y).pow(2); best = err.argmin(-1)
            clean = model.moe.gate.clean_logits(torch.cat([feats, rough], -1))
            masked = clean.clone()
            masked.scatter_(-1, best.unsqueeze(-1), float('-inf'))
            top3 = masked.topk(3, dim=-1).values.logsumexp(-1)
            hard = F.relu(top3 - clean.gather(-1, best.unsqueeze(-1)).squeeze(-1))
            loss = loss + 1.0 * hard.mean()
            loss = loss + AUX_W * F.mse_loss(aux, a)
            loss.backward(); opt.step()

    model.eval()
    res = {}
    with torch.no_grad():
        for c in test_ids:
            X = cells[c]['X']
            x = torch.tensor(fs.transform(X.reshape(-1, D)).reshape(len(X), Lw, D),
                             dtype=torch.float32).to(device)
            p = torch.tensor(ps.transform(np.repeat(cells[c]['proto'][None, :],
                                                    len(X), 0)),
                             dtype=torch.float32).to(device)
            g, *_ = model(x, p)
            pred = ls.inverse_transform(g.cpu().numpy().reshape(-1, 1)).ravel()
            res[c] = rmse(cells[c]['y'], pred)
    return res


if __name__ == '__main__':
    cells = load_cells()
    ids = sorted(cells)
    print(f"{len(ids)} cells, feature dim {cells[ids[0]]['X'].shape[-1]}")
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ids))
    folds = [[ids[i] for i in perm[k::N_FOLDS]] for k in range(N_FOLDS)]

    per_cell, fold_means = {}, []
    for k, test_ids in enumerate(folds):
        train_ids = [c for c in ids if c not in test_ids]
        res = train_eval(train_ids, test_ids, cells)
        fold_means.append(np.mean(list(res.values())))
        for c, v in res.items():
            per_cell.setdefault(c, []).append(v)
        print(f"  fold {k}: avg={fold_means[-1]:6.1f}  "
              f"{ {c: round(v,1) for c, v in res.items()} }", flush=True)
    allv = [v for vs in per_cell.values() for v in vs]
    print(f"\nv2 CV mean={np.mean(allv):.2f}  "
          f"fold-mean={np.mean(fold_means):.2f}+/-{np.std(fold_means):.2f}")
    print(f"reference v1 (plain SMoE-LSTM, IR40): 48.54")
    worst = sorted(per_cell.items(), key=lambda kv: -np.mean(kv[1]))[:5]
    print("hardest:", [(c, round(float(np.mean(v)), 1)) for c, v in worst])
