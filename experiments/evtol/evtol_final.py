#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final eVTOL model: clean paper architecture with only validated parts.

Kept:
  - LSTM 2x128, sparse MoE (6 experts, top-4), evidential NIG head
  - hard routing supervision + load balancing
  - v4 features (46): base28 + IR + missing flag + cumulative + SOH history
    + temperature-normalised IR
Removed (no gain in experiments): attention pooling, FiLM, multi-task
heads, physics fusion, bidirectional LSTM.

Usage:
  python experiments/evtol/evtol_final.py --mode cv
  python experiments/evtol/evtol_final.py --mode lopo
"""
import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
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
N_EXPERTS = 6
TOP_K = 4

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


TEMP_COLS = [5, 11, 17, 23, 27]   # charge/to/cruise/landing Tmax + T_start


def load_cells(drop_temp=False):
    d = np.load('data/evtol_features_v4.npy', allow_pickle=True).item()
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
        if drop_temp:
            keep_cols = [j for j in range(Xw.shape[-1]) if j not in TEMP_COLS]
            Xw = Xw[..., keep_cols]
        cells[cid] = {'X': Xw.astype(np.float32), 'y': yw.astype(np.float32)}
    return cells


class DS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X); self.y = torch.tensor(y).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_eval(cells, train_ids, test_ids, seed=SEED, n_experts=N_EXPERTS):
    Xtr = np.concatenate([cells[c]['X'] for c in train_ids])
    ytr = np.concatenate([cells[c]['y'] for c in train_ids])
    N, Lw, D = Xtr.shape
    fs = MinMaxScaler().fit(Xtr.reshape(-1, D))
    Xtr_n = fs.transform(Xtr.reshape(-1, D)).reshape(N, Lw, D)
    ls = MinMaxScaler().fit(ytr.reshape(-1, 1))
    ytr_n = ls.transform(ytr.reshape(-1, 1)).ravel()

    set_seed(seed)
    loader, _ = prepare_dataloaders(DS(Xtr_n, ytr_n), 64, 0.0, seed)
    model = SMoELSTM(input_dim=D, hidden_dim=128, num_experts=n_experts,
                     top_k=TOP_K, lstm_layers=2, evidence_hidden=64).to(device)
    crit = EvidentialRegressionLoss(coeff=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    E = n_experts
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
    res = {}
    with torch.no_grad():
        for c in test_ids:
            X = cells[c]['X']
            x = torch.tensor(fs.transform(X.reshape(-1, D)).reshape(len(X), Lw, D),
                             dtype=torch.float32).to(device)
            g, *_ = model(x)
            p = ls.inverse_transform(g.cpu().numpy().reshape(-1, 1)).ravel()
            res[c] = rmse(cells[c]['y'], p)
    return res


def run_cv(cells, ids, n_experts=N_EXPERTS):
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ids))
    folds = [[ids[i] for i in perm[k::N_FOLDS]] for k in range(N_FOLDS)]
    fold_means, per_cell = [], {}
    for k, test_ids in enumerate(folds):
        train_ids = [c for c in ids if c not in test_ids]
        res = train_eval(cells, train_ids, test_ids, n_experts=n_experts)
        fold_means.append(np.mean(list(res.values())))
        for c, v in res.items():
            per_cell.setdefault(c, []).append(v)
        print(f"  fold {k}: avg={fold_means[-1]:6.1f}  "
              f"{ {c: round(v,1) for c, v in res.items()} }", flush=True)
    allv = [v for vs in per_cell.values() for v in vs]
    print(f"\nCLEAN model CV mean={np.mean(allv):.2f}  "
          f"fold-mean={np.mean(fold_means):.2f}+/-{np.std(fold_means):.2f}")
    worst = sorted(per_cell.items(), key=lambda kv: -np.mean(kv[1]))[:5]
    print("hardest:", [(c, round(float(np.mean(v)), 1)) for c, v in worst])


def run_lopo(cells, ids, n_experts=N_EXPERTS):
    families = sorted(set(PROTOCOL[c] for c in ids))
    fold_means, per_cell = [], {}
    for fam in families:
        test_ids = [c for c in ids if PROTOCOL[c] == fam]
        train_ids = [c for c in ids if c not in test_ids]
        res = train_eval(cells, train_ids, test_ids, n_experts=n_experts)
        fold_means.append(np.mean(list(res.values())))
        for c, v in res.items():
            per_cell[c] = v
        print(f"  hold out {fam:16s} ({test_ids}): "
              f"avg={fold_means[-1]:6.1f}  "
              f"{ {c: round(v,1) for c, v in res.items()} }", flush=True)
    print(f"\nCLEAN model LOPO mean={np.mean(fold_means):.2f} "
          f"+/- {np.std(fold_means):.2f}")
    worst = sorted(per_cell.items(), key=lambda kv: -kv[1])[:5]
    print("hardest:", [(c, round(v, 1)) for c, v in worst])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['cv', 'lopo'], default='cv')
    ap.add_argument('--experts', type=int, default=N_EXPERTS)
    ap.add_argument('--drop-temp', action='store_true',
                    help='drop absolute temperature features (keep T-normalised IR)')
    args = ap.parse_args()
    cells = load_cells(drop_temp=args.drop_temp)
    ids = [c for c in sorted(cells) if c in PROTOCOL]
    print(f"{len(ids)} cells, features {cells[ids[0]]['X'].shape[-1]}, "
          f"experts {args.experts}")
    if args.mode == 'cv':
        run_cv(cells, ids, args.experts)
    else:
        run_lopo(cells, ids, args.experts)
