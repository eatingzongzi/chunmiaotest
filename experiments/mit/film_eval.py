#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FiLM conditioning experiment (covariate-shift remedy).

Each cell is described by the mean/std of its 16 features over the first
30 cycles (causal, available at test time). A small MLP maps these stats
to per-channel scale/shift that modulate the LSTM representation:
    h' = h * (1 + tanh(scale)) + shift
Baseline: same model without FiLM. Fixed 24-cell split (test
[13,17,24,25,34,29]); baseline clean single 23.81.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from config import Config
from src.data_loader import _extract_cell_features_paper
from src.train import set_seed, get_device, prepare_dataloaders
from src.model import SMoELSTM
from src.loss import EvidentialRegressionLoss
from src.evaluate import rmse
import h5py

device = get_device('cuda')
L = 30
N_CELLS, N_TEST, SEED_SPLIT = 30, 6, 42
STAT_CYCLES = 30


def load_all_cells():
    f = h5py.File('data/mit_stanford/2017-06-30_batchdata_updated_struct_errorcorrect.mat', 'r')
    batch = f['batch']
    cells = []
    for i in range(batch['summary'].shape[0]):
        try:
            summ = f[batch['summary'][i, 0]]
            cyc = f[batch['cycles'][i, 0]]
            cl = float(f[batch['cycle_life'][i, 0]][()].flatten()[0])
            vd = None
            try:
                vd = f[batch['Vdlin'][i, 0]][()].flatten().astype(np.float64)
            except Exception:
                pass
            feats, cn = _extract_cell_features_paper(f, summ, cyc, cl, vd)
            if len(feats) == 0:
                continue
            eol = int(cl)
            valid = cn <= eol
            if np.sum(valid) < L + 10:
                continue
            cells.append({'cell_id': i, 'eol': eol,
                          'features': feats[valid], 'cycle': cn[valid]})
        except Exception:
            continue
    f.close()
    return cells


def build(cell):
    feats = cell['features']
    N = len(feats)
    X = np.stack([feats[i:i + L] for i in range(N - L + 1)], 0).astype(np.float32)
    y = np.array([cell['eol'] - cell['cycle'][i + L - 1]
                  for i in range(N - L + 1)], dtype=np.float32)
    s = feats[:STAT_CYCLES]
    stats = np.concatenate([s.mean(0), s.std(0)]).astype(np.float32)
    return X, y, stats


cells = load_all_cells()
rng = np.random.RandomState(SEED_SPLIT)
sel = rng.permutation(len(cells))[:N_CELLS]
sel = rng.permutation(sel)
test_ids = [cells[i]['cell_id'] for i in sel[:N_TEST]]
train_ids = [cells[i]['cell_id'] for i in sel[N_TEST:]]
print(f"train {len(train_ids)} / test {test_ids}")

data = {c['cell_id']: build(c) for c in cells}
Xtr = np.concatenate([data[c][0] for c in train_ids])
ytr = np.concatenate([data[c][1] for c in train_ids])
Str = np.concatenate([np.repeat(data[c][2][None, :], len(data[c][1]), 0)
                      for c in train_ids], axis=0)

N, Lw, D = Xtr.shape
fs = MinMaxScaler().fit(Xtr.reshape(-1, D))
Xtr_n = fs.transform(Xtr.reshape(-1, D)).reshape(N, Lw, D)
ls = MinMaxScaler().fit(ytr.reshape(-1, 1))
ytr_n = ls.transform(ytr.reshape(-1, 1)).ravel()
ss = MinMaxScaler().fit(Str)
Str_n = ss.transform(Str)

test_pack = {}
for c in test_ids:
    X, y, s = data[c]
    test_pack[c] = {'X': fs.transform(X.reshape(-1, D)).reshape(len(X), Lw, D),
                    'y': y, 's': ss.transform(np.repeat(s[None, :], len(y), 0))}


class DS(torch.utils.data.Dataset):
    def __init__(self, X, y, s):
        self.X = torch.tensor(X); self.y = torch.tensor(y).unsqueeze(-1)
        self.s = torch.tensor(s)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.s[i]


def train_variant(seed, use_film):
    set_seed(seed)
    loader, _ = prepare_dataloaders(DS(Xtr_n, ytr_n, Str_n), 64, 0.0, seed)
    model = SMoELSTM(input_dim=D, hidden_dim=128, num_experts=10, top_k=4,
                     lstm_layers=2, evidence_hidden=64,
                     film_dim=32 if use_film else 0).to(device)
    crit = EvidentialRegressionLoss(coeff=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for epoch in range(100):
        model.train()
        for X, y, s in loader:
            X, y, s = X.to(device), y.to(device), s.to(device)
            opt.zero_grad()
            out = model(X, return_gates=True, film_stats=s if use_film else None)
            g, nu, al, be, gi = out[:5]
            loss = crit(y, g, nu, al, be)
            gates, indices, feats, rough = gi
            loss = loss + 0.05 * model.moe.balance_loss(gates, indices, 10)
            E = 10
            with torch.no_grad():
                outs = torch.stack([model.moe.experts[e](feats) for e in range(E)], 1)
                ge = torch.stack([model.evidence(outs[:, e])[0].squeeze(-1)
                                  for e in range(E)], 1)
                err = (ge - y).pow(2); best = err.argmin(-1)
            clean = model.moe.gate.clean_logits(torch.cat([feats, rough], -1))
            masked = clean.clone()
            masked.scatter_(-1, best.unsqueeze(-1), float('-inf'))
            top3 = masked.topk(3, dim=-1).values.logsumexp(-1)
            hard = F.relu(top3 - clean.gather(-1, best.unsqueeze(-1)).squeeze(-1))
            loss = loss + 1.0 * hard.mean()
            loss.backward(); opt.step()

    model.eval()
    per_cell = []
    with torch.no_grad():
        for c in test_ids:
            x = torch.tensor(test_pack[c]['X'], dtype=torch.float32).to(device)
            s = torch.tensor(test_pack[c]['s'], dtype=torch.float32).to(device)
            g, *_ = model(x, film_stats=s if use_film else None)
            p = ls.inverse_transform(g.cpu().numpy().reshape(-1, 1)).ravel()
            per_cell.append(rmse(test_pack[c]['y'], p))
    return np.mean(per_cell), per_cell


print("\n=== FiLM vs baseline (fixed 24-cell split, 3 seeds) ===")
base, film = [], []
for s in range(3):
    b_avg, b_pc = train_variant(s, False)
    f_avg, f_pc = train_variant(s, True)
    base.append(b_avg); film.append(f_avg)
    print(f"seed {s}: baseline={b_avg:.2f} {[f'{v:.1f}' for v in b_pc]} | "
          f"FiLM={f_avg:.2f} {[f'{v:.1f}' for v in f_pc]}", flush=True)
print(f"\nbaseline: {np.mean(base):.2f}+/-{np.std(base):.2f}")
print(f"FiLM    : {np.mean(film):.2f}+/-{np.std(film):.2f}  "
      f"(clean baseline reference 23.81)")
