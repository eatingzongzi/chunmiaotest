#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Upgrade v1: attention pooling + multi-task SOH head (clean fixed split).

A/B on the SAME leak-free split as mit_clean_eval Protocol A (42/6,
seed 42): baseline SMoE-LSTM vs attention-pooled + auxiliary SOH head.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import torch.nn as nn
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
AUX_H = 30
SEED_SPLIT = 42


# ── Load all cells (same scan as the loader) ────────────────────────
def load_all_cells():
    f = h5py.File('data/mit_stanford/2017-06-30_batchdata_updated_struct_errorcorrect.mat', 'r')
    batch = f['batch']
    cells = []
    for i in range(batch['summary'].shape[0]):
        try:
            summ = f[batch['summary'][i, 0]]
            cyc = f[batch['cycles'][i, 0]]
            cl = float(f[batch['cycle_life'][i, 0]][()].flatten()[0])
            vdlin = None
            try:
                vdlin = f[batch['Vdlin'][i, 0]][()].flatten().astype(np.float64)
            except Exception:
                pass
            feats, cn = _extract_cell_features_paper(f, summ, cyc, cl, vdlin)
            if len(feats) == 0:
                continue
            eol = int(cl)
            if eol <= cn[0]:
                continue
            valid = cn <= eol
            if np.sum(valid) < L + 10:
                continue
            cells.append({'cell_id': i, 'eol': eol,
                          'features': feats[valid], 'cycle': cn[valid]})
        except Exception:
            continue
    f.close()
    return cells


def build_windows(cell):
    X, feats, cyc = cell['features'], cell['features'], cell['cycle']
    N = len(feats)
    q0 = max(feats[0, 15], 1e-6)
    windows, ruls, auxs = [], [], []
    for i in range(N - L + 1):
        windows.append(feats[i:i + L])
        end = i + L - 1
        ruls.append(cell['eol'] - cyc[end])
        j = min(end + AUX_H, N - 1)
        auxs.append(feats[j, 15] / q0)
    return (np.array(windows, dtype=np.float32),
            np.array(ruls, dtype=np.float32),
            np.array(auxs, dtype=np.float32))


cells = load_all_cells()
rng = np.random.RandomState(SEED_SPLIT)
sel = rng.permutation(len(cells))[:48]
sel = rng.permutation(sel)
test_ids = [cells[i]['cell_id'] for i in sel[:6]]
train_ids = [cells[i]['cell_id'] for i in sel[6:]]
print(f"train {len(train_ids)} / test {test_ids}")

data = {c['cell_id']: build_windows(c) for c in cells}
Xtr = np.concatenate([data[c][0] for c in train_ids])
ytr = np.concatenate([data[c][1] for c in train_ids])
atr = np.concatenate([data[c][2] for c in train_ids])
N, Lw, D = Xtr.shape
fs = MinMaxScaler().fit(Xtr.reshape(-1, D))
Xtr_n = fs.transform(Xtr.reshape(-1, D)).reshape(N, Lw, D)
ls = MinMaxScaler().fit(ytr.reshape(-1, 1))
ytr_n = ls.transform(ytr.reshape(-1, 1)).ravel()

test_pack = {}
for c in test_ids:
    X, y, a = data[c]
    test_pack[c] = {'X': fs.transform(X.reshape(-1, D)).reshape(len(X), Lw, D),
                    'y': y, 'aux': a}


class DS(torch.utils.data.Dataset):
    def __init__(self, X, y, aux):
        self.X = torch.tensor(X); self.y = torch.tensor(y).unsqueeze(-1)
        self.aux = torch.tensor(aux)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.aux[i]


def train_variant(name, use_attn, multitask, seed=0):
    set_seed(seed)
    cfg = Config()
    cfg.input_dim = D
    loader, _ = prepare_dataloaders(DS(Xtr_n, ytr_n, atr), 64, 0.0, seed)
    model = SMoELSTM(input_dim=D, hidden_dim=128, num_experts=10, top_k=4,
                     lstm_layers=2, evidence_hidden=64, gate_hidden=0,
                     use_attention=use_attn, multitask=multitask).to(device)
    crit = EvidentialRegressionLoss(coeff=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for epoch in range(100):
        model.train()
        for X, y, a in loader:
            X, y, a = X.to(device), y.to(device), a.to(device)
            opt.zero_grad()
            out = model(X, return_gates=True, return_aux=multitask)
            g, nu, al, be, gi = out[:5]
            gates, indices, feats, rough = gi
            aux_pred = out[5] if multitask else None
            loss = crit(y, g, nu, al, be)
            loss = loss + 0.05 * model.moe.balance_loss(gates, indices, 10)
            # hard routing supervision (same as src/train.py)
            E = 10
            with torch.no_grad():
                outs = torch.stack([model.moe.experts[e](feats) for e in range(E)], 1)
                ge = torch.stack([model.evidence(outs[:, e])[0].squeeze(-1)
                                  for e in range(E)], 1)
                err = (ge - y).pow(2)
                best = err.argmin(-1)
            clean = model.moe.gate.clean_logits(torch.cat([feats, rough], -1))
            masked = clean.clone()
            masked.scatter_(-1, best.unsqueeze(-1), float('-inf'))
            top3 = masked.topk(3, dim=-1).values.logsumexp(-1)
            hard = F.relu(top3 - clean.gather(-1, best.unsqueeze(-1)).squeeze(-1))
            loss = loss + 1.0 * hard.mean()
            if multitask:
                loss = loss + 0.1 * F.mse_loss(aux_pred, a)
            loss.backward(); opt.step()

    model.eval()
    per_cell = []
    with torch.no_grad():
        for c in test_ids:
            x = torch.tensor(test_pack[c]['X'], dtype=torch.float32).to(device)
            g, *_ = model(x)
            p = ls.inverse_transform(g.cpu().numpy().reshape(-1, 1)).ravel()
            per_cell.append(rmse(test_pack[c]['y'], p))
    print(f"{name}: per-cell {[f'{v:.1f}' for v in per_cell]} "
          f"avg={np.mean(per_cell):.2f}", flush=True)
    return np.mean(per_cell)


print("\n=== A/B: fixed split, 3 seeds ===")
base, attn, mt = [], [], []
for s in range(3):
    base.append(train_variant('baseline', False, False, s))
    attn.append(train_variant('attention', True, False, s))
    mt.append(train_variant('attn+aux', True, True, s))
print(f"\nbaseline   : {np.mean(base):.2f}")
print(f"attention  : {np.mean(attn):.2f}")
print(f"attn+aux   : {np.mean(mt):.2f}")
