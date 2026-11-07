#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pretrain on NASA ALT (12 generic features) -> fine-tune on CMU eVTOL.

Baseline: CMU 12-feature model trained from scratch (same folds).
Pretrained: same model initialised from NASA ALT weights, then fine-tuned.
The 12 features are relative quantities, standardised per dataset.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.train import set_seed, get_device, prepare_dataloaders
from src.model import SMoELSTM
from src.loss import EvidentialRegressionLoss
from src.evaluate import rmse

device = get_device('cuda')
L = 30
N_FOLDS = 5
SEED = 42
N_EXPERTS = 10
TOP_K = 4
NASA_DROP = 9          # drop uninformative charge-Vmax ratio


class DS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X); self.y = torch.tensor(y).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def windows_from_series(X, rul):
    N = len(X)
    if N < L + 5:
        return None, None
    Xw = np.stack([X[j:j + L] for j in range(N - L + 1)], 0)
    yw = np.array([rul[j + L - 1] for j in range(N - L + 1)])
    return Xw.astype(np.float32), yw.astype(np.float32)


def load_nasa():
    d = np.load('data/nasa_alt/nasa_alt_features.npy', allow_pickle=True).item()
    Xs, ys = [], []
    for pack, v in d.items():
        if not np.isfinite(v['eol']):
            continue
        X = np.delete(v['X'], NASA_DROP, axis=1).copy()
        # impute missing charge stats (missions with no preceding charge)
        for j in range(X.shape[1]):
            col = X[:, j]
            ok = np.isfinite(col)
            if not ok.all():
                col[~ok] = np.median(col[ok]) if ok.any() else 0.0
        rul = v['eol'] - v['cycle']
        keep = rul >= 0
        if keep.sum() < L + 5:
            continue
        Xw, yw = windows_from_series(X[keep], rul[keep])
        if Xw is None:
            continue
        Xs.append(Xw); ys.append(yw)
        print(f"  NASA {pack}: {Xw.shape[0]} windows, eol={v['eol']:.0f}")
    return np.concatenate(Xs), np.concatenate(ys)


def load_cmu():
    d = np.load('data/evtol_cmu_generic.npy', allow_pickle=True).item()
    cells = {}
    for cid, v in d.items():
        X = v['X'].copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            ok = np.isfinite(col)
            if not ok.all():
                col[~ok] = np.median(col[ok]) if ok.any() else 0.0
        Xw, yw = windows_from_series(X, v['rul'])
        if Xw is None:
            continue
        cells[cid] = {'X': Xw, 'y': yw}
    return cells


def train_loop(X, y, seed, init_state=None, epochs=100, lr=1e-3, verbose=False):
    N, Lw, D = X.shape
    fs = StandardScaler().fit(X.reshape(-1, D))
    Xn = fs.transform(X.reshape(-1, D)).reshape(N, Lw, D)
    ls = MinMaxScaler().fit(y.reshape(-1, 1))
    yn = ls.transform(y.reshape(-1, 1)).ravel()

    set_seed(seed)
    loader, _ = prepare_dataloaders(DS(Xn, yn), 64, 0.0, seed)
    model = SMoELSTM(input_dim=D, hidden_dim=128, num_experts=N_EXPERTS,
                     top_k=TOP_K, lstm_layers=2, evidence_hidden=64).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    crit = EvidentialRegressionLoss(coeff=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    E = N_EXPERTS
    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(Xb, return_gates=True)
            g, nu, al, be, gi = out[:5]
            loss = crit(yb, g, nu, al, be)
            gates, indices, feats, rough = gi
            loss = loss + 0.05 * model.moe.balance_loss(gates, indices, E)
            with torch.no_grad():
                outs = torch.stack([model.moe.experts[e](feats) for e in range(E)], 1)
                ge = torch.stack([model.evidence(outs[:, e])[0].squeeze(-1)
                                  for e in range(E)], 1)
                err = (ge - yb).pow(2); best = err.argmin(-1)
            clean = model.moe.gate.clean_logits(torch.cat([feats, rough], -1))
            masked = clean.clone()
            masked.scatter_(-1, best.unsqueeze(-1), float('-inf'))
            top = masked.topk(TOP_K - 1, dim=-1).values.logsumexp(-1)
            hard = F.relu(top - clean.gather(-1, best.unsqueeze(-1)).squeeze(-1))
            loss = loss + 1.0 * hard.mean()
            loss.backward(); opt.step()
    model.eval()
    return model, fs, ls


def eval_cells(model, fs, ls, cells, test_ids):
    per_cell = []
    with torch.no_grad():
        for c in test_ids:
            X = cells[c]['X']; D = X.shape[-1]
            x = torch.tensor(fs.transform(X.reshape(-1, D)).reshape(len(X), L, D),
                             dtype=torch.float32).to(device)
            g, *_ = model(x)
            p = ls.inverse_transform(g.cpu().numpy().reshape(-1, 1)).ravel()
            per_cell.append(rmse(cells[c]['y'], p))
    return float(np.mean(per_cell))


if __name__ == '__main__':
    print("Loading NASA ALT...")
    Xn, yn = load_nasa()
    print(f"NASA: {len(Xn)} windows, dim {Xn.shape[-1]}")
    print("Pretraining...")
    pre_model, _, _ = train_loop(Xn, yn, seed=SEED, epochs=100, lr=1e-3)
    pre_state = {k: v.cpu() for k, v in pre_model.state_dict().items()}
    torch.save(pre_state, 'checkpoints/nasa_pretrain_12f.pt')
    print("pretrained weights saved")

    cells = load_cmu()
    ids = sorted(cells)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ids))
    folds = [[ids[i] for i in perm[k::N_FOLDS]] for k in range(N_FOLDS)]

    base_all, pre_all = [], []
    for k, test_ids in enumerate(folds):
        train_ids = [c for c in ids if c not in test_ids]
        Xtr = np.concatenate([cells[c]['X'] for c in train_ids])
        ytr = np.concatenate([cells[c]['y'] for c in train_ids])
        m0, fs0, ls0 = train_loop(Xtr, ytr, seed=SEED, epochs=100, lr=1e-3)
        r0 = eval_cells(m0, fs0, ls0, cells, test_ids)
        m1, fs1, ls1 = train_loop(Xtr, ytr, seed=SEED, init_state=pre_state,
                                  epochs=60, lr=5e-4)
        r1 = eval_cells(m1, fs1, ls1, cells, test_ids)
        base_all.append(r0); pre_all.append(r1)
        print(f"fold {k}: scratch={r0:.1f}  pretrained={r1:.1f}", flush=True)

    print(f"\nCMU 12-feature, 5-fold CV:")
    print(f"  scratch   : {np.mean(base_all):.2f}+/-{np.std(base_all):.2f}")
    print(f"  pretrained: {np.mean(pre_all):.2f}+/-{np.std(pre_all):.2f}")
