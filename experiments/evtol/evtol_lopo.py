#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leave-one-protocol-out (LOPO) evaluation with the paper's SMoE-LSTM.

Each fold holds out an ENTIRE protocol family (not random cells), so the
model is tested on operating conditions it has never seen. Protocol
families are taken from the CMU dataset README.

Usage:
    python experiments/evtol/evtol_lopo.py --features ir   # 39-d
    python experiments/evtol/evtol_lopo.py --features base # 28-d
"""
import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from config import Config
from src.train import set_seed, get_device, prepare_dataloaders
from src.model import SMoELSTM
from src.loss import EvidentialRegressionLoss
from src.evaluate import rmse

device = get_device('cuda')
L = 30
EOL_THR = 0.85
SEED = 42

# Protocol families for the 18 cells that reach 85% SOH
PROTOCOL = {
    'VAH01': 'baseline', 'VAH17': 'baseline', 'VAH27': 'baseline',
    'VAH02': 'ext_cruise_1000', 'VAH22': 'ext_cruise_1000',
    'VAH05': 'power_10', 'VAH28': 'power_10',
    'VAH06': 'charge_C2', 'VAH24': 'charge_C2',
    'VAH09': 'temp_20C', 'VAH25': 'temp_20C',
    'VAH10': 'temp_30C',
    'VAH11': 'power_20',
    'VAH12': 'cruise_400',
    'VAH13': 'cruise_600', 'VAH26': 'cruise_600',
    'VAH20': 'charge_1p5C',
    'VAH30': 'temp_35C',
}


def load_cells():
    d = np.load('data/evtol_features_v3.npy', allow_pickle=True).item()
    cells = {}
    for cid, v in d.items():
        soh, cyc = v['soh'], v['cycle']
        below = np.where(soh < EOL_THR)[0]
        if len(below) == 0:
            continue
        i = below[0]
        eol = (cyc[i - 1] + (soh[i - 1] - EOL_THR) /
               (soh[i - 1] - soh[i]) * (cyc[i] - cyc[i - 1])) if i > 0 else cyc[0]
        keep = cyc <= eol
        X, c = v['X'][keep].copy(), cyc[keep]
        if len(X) < L + 10:
            continue
        rul = eol - c
        Xw = np.stack([X[j:j + L] for j in range(len(X) - L + 1)], 0)
        yw = np.array([rul[j + L - 1] for j in range(len(X) - L + 1)])
        cells[cid] = {'X': Xw.astype(np.float32), 'y': yw.astype(np.float32)}
    return cells


class DS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X); self.y = torch.tensor(y).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_eval(feat_slice, train_ids, test_ids, seed=SEED):
    def cut(X): return X[..., feat_slice]
    Xtr = np.concatenate([cut(cells[c]['X']) for c in train_ids])
    ytr = np.concatenate([cells[c]['y'] for c in train_ids])
    N, Lw, D = Xtr.shape
    fs = MinMaxScaler().fit(Xtr.reshape(-1, D))
    Xtr_n = fs.transform(Xtr.reshape(-1, D)).reshape(N, Lw, D)
    ls = MinMaxScaler().fit(ytr.reshape(-1, 1))
    ytr_n = ls.transform(ytr.reshape(-1, 1)).ravel()

    set_seed(seed)
    loader, _ = prepare_dataloaders(DS(Xtr_n, ytr_n), 64, 0.0, seed)
    model = SMoELSTM(input_dim=D, hidden_dim=128, num_experts=10, top_k=4,
                     lstm_layers=2, evidence_hidden=64).to(device)
    crit = EvidentialRegressionLoss(coeff=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for epoch in range(100):
        model.train()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X, return_gates=True)
            g, nu, al, be, gi = out[:5]
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
            loss.backward(); opt.step()

    model.eval()
    res = {}
    with torch.no_grad():
        for c in test_ids:
            X = cut(cells[c]['X'])
            x = torch.tensor(fs.transform(X.reshape(-1, D)).reshape(len(X), Lw, D),
                             dtype=torch.float32).to(device)
            g, *_ = model(x)
            p = ls.inverse_transform(g.cpu().numpy().reshape(-1, 1)).ravel()
            res[c] = rmse(cells[c]['y'], p)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', choices=['base', 'ir'], default='ir')
    args = ap.parse_args()
    sl = slice(0, 28) if args.features == 'base' else slice(0, 40)

    cells = load_cells()
    ids = [c for c in sorted(cells) if c in PROTOCOL]
    print(f"{len(ids)} cells, {len(set(PROTOCOL[c] for c in ids))} protocol families")

    families = sorted(set(PROTOCOL[c] for c in ids))
    fold_means, per_cell = [], {}
    for fam in families:
        test_ids = [c for c in ids if PROTOCOL[c] == fam]
        train_ids = [c for c in ids if c not in test_ids]
        res = train_eval(sl, train_ids, test_ids)
        fold_means.append(np.mean(list(res.values())))
        for c, v in res.items():
            per_cell[c] = v
        print(f"  hold out {fam:16s} ({test_ids}): "
              f"avg={fold_means[-1]:6.1f}  { {c: round(v,1) for c, v in res.items()} }",
              flush=True)

    print(f"\nLOPO mean={np.mean(fold_means):.2f} +/- {np.std(fold_means):.2f} "
          f"(features={args.features})")
    worst = sorted(per_cell.items(), key=lambda kv: -kv[1])[:5]
    print("hardest cells:", [(c, round(v, 1)) for c, v in worst])
