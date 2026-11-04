#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Physics-informed hybrid (PIH): power-law identification + gated fusion.

For every cycle, causally fit the degradation law
    cap0 - cap(t) = a * t^b
to the observed capacity prefix, producing:
    RUL_phys, fit R2, observation length, local fade rate, validity.
A small gate network learns per-sample trust w:
    RUL = w * RUL_phys + (1 - w) * RUL_NN
trained end-to-end with the evidential loss.

Protocol: 24-cell fixed split (paper protocol), test [13,17,24,25,34,29].
Baseline (clean): single 23.81, 5-model ensemble 22.75.
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
EOL_CAP = 0.88
N_CELLS, N_TEST, SEED_SPLIT = 30, 6, 42
GATE_LAMBDA = 1.0
GATE_TAU = 0.02


# ── Cells ───────────────────────────────────────────────────────────
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


def physics_features(cap, cycles):
    """Causal power-law fit for every cycle; returns (N, 5)."""
    N = len(cap)
    P = np.zeros((N, 5), dtype=np.float32)
    cap0 = cap[0]
    target = cap0 - EOL_CAP
    for k in range(N):
        t = cycles[:k + 1].astype(float)
        loss = cap0 - cap[:k + 1]
        ok = (loss > 1e-4) & (t > 0)
        valid = 0.0
        rul_phys, r2, fade = 0.0, 0.0, 0.0
        if ok.sum() >= 10 and target > 1e-4:
            logt = np.log(t[ok]); logl = np.log(loss[ok])
            try:
                b, loga = np.polyfit(logt, logl, 1)
                if np.isfinite(b) and b > 0.05:
                    pred = loga + b * logt
                    ss_res = np.sum((logl - pred) ** 2)
                    ss_tot = np.sum((logl - logl.mean()) ** 2) + 1e-12
                    r2 = float(max(0.0, 1.0 - ss_res / ss_tot))
                    eol_est = np.exp((np.log(target) - loga) / b)
                    rul_phys = float(np.clip(eol_est - t[-1], 0.0, 3000.0))
                    valid = 1.0
            except Exception:
                pass
        # local fade rate over the last 50 cycles (Ah/cycle)
        j = max(0, k - 50)
        if k > j:
            fade = float((cap[j] - cap[k]) / (k - j))
        P[k] = [rul_phys, r2, float(min(k + 1, 1000) / 1000.0),
                float(np.clip(fade * 1000, -10, 10)), valid]
    return P


def build_cell_data(cell):
    cap = cell['features'][:, 15].astype(float)
    P = physics_features(cap, cell['cycle'])
    feats = cell['features']
    N = len(feats)
    X, y, ph = [], [], []
    for i in range(N - L + 1):
        end = i + L - 1
        X.append(feats[i:i + L])
        y.append(cell['eol'] - cell['cycle'][end])
        ph.append(P[end])
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32),
            np.array(ph, dtype=np.float32))


print("loading cells + physics features...")
all_cells = load_all_cells()
rng = np.random.RandomState(SEED_SPLIT)
sel = rng.permutation(len(all_cells))[:N_CELLS]
sel = rng.permutation(sel)
test_ids = [all_cells[i]['cell_id'] for i in sel[:N_TEST]]
train_ids = [all_cells[i]['cell_id'] for i in sel[N_TEST:]]
print(f"train {len(train_ids)} / test {test_ids}")

data = {}
for c in all_cells:
    data[c['cell_id']] = build_cell_data(c)

Xtr = np.concatenate([data[c][0] for c in train_ids])
ytr = np.concatenate([data[c][1] for c in train_ids])
ptr = np.concatenate([data[c][2] for c in train_ids])
N, Lw, D = Xtr.shape
fs = MinMaxScaler().fit(Xtr.reshape(-1, D))
Xtr_n = fs.transform(Xtr.reshape(-1, D)).reshape(N, Lw, D)
ls = MinMaxScaler().fit(ytr.reshape(-1, 1))
ytr_n = ls.transform(ytr.reshape(-1, 1)).ravel()

# physics features normalised: RUL_phys via label scaler, others fixed scales
def norm_phys(P):
    Q = P.copy()
    Q[:, 0] = ls.transform(Q[:, 0:1]).ravel()
    return Q

ptr_n = norm_phys(ptr)
phys_scaler = MinMaxScaler().fit(ptr_n[:, 1:])
ptr_all = np.concatenate([ptr_n[:, :1], phys_scaler.transform(ptr_n[:, 1:])], 1)

test_pack = {}
for c in test_ids:
    X, y, p = data[c]
    pn = norm_phys(p)
    test_pack[c] = {
        'X': fs.transform(X.reshape(-1, D)).reshape(len(X), Lw, D),
        'y': y,
        'p': np.concatenate([pn[:, :1], phys_scaler.transform(pn[:, 1:])], 1),
    }

print(f"train windows {len(Xtr_n)}, feature dim {D}, "
      f"physics-valid fraction {ptr[:, 4].mean():.2f}")


# ── Physics-fusion model ────────────────────────────────────────────
class PhysicsFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.backbone = SMoELSTM(
            input_dim=input_dim, hidden_dim=128, num_experts=10, top_k=4,
            lstm_layers=2, evidence_hidden=64, gate_hidden=0)
        self.gate = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, p, return_gates=False):
        # p = [rul_phys_norm, r2, n_obs, fade, valid]
        out = self.backbone(x, return_gates=True)
        gamma, nu, alpha, beta = out[:4]
        gi = out[4]
        g_norm = gamma.squeeze(-1)                 # NN prediction (normalised)
        w = torch.sigmoid(self.gate(torch.cat([g_norm.unsqueeze(-1), p], -1)))
        w = w * p[:, 4:5]                          # invalid physics -> pure NN
        phys = p[:, 0] * p[:, 4]
        fused = w.squeeze(-1) * phys + (1 - w.squeeze(-1)) * g_norm
        gamma_f = fused.unsqueeze(-1)
        return gamma_f, nu, alpha, beta, gi, w.squeeze(-1), g_norm


class DS(torch.utils.data.Dataset):
    def __init__(self, X, y, p):
        self.X = torch.tensor(X); self.y = torch.tensor(y).unsqueeze(-1)
        self.p = torch.tensor(p)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.p[i]


def train_pih(seed, use_physics=True):
    set_seed(seed)
    loader, _ = prepare_dataloaders(DS(Xtr_n, ytr_n, ptr_all), 64, 0.0, seed)
    model = PhysicsFusion(D).to(device)
    crit = EvidentialRegressionLoss(coeff=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for epoch in range(100):
        model.train()
        for X, y, p in loader:
            X, y, p = X.to(device), y.to(device), p.to(device)
            if not use_physics:
                p = p.clone(); p[:, 4] = 0.0  # invalidate physics (ablation)
            opt.zero_grad()
            g, nu, al, be, gi, w, g_norm = model(X, p)
            loss = crit(y, g, nu, al, be)
            # Oracle gate supervision: target weight says which predictor is
            # better on this sample (physics vs NN), in normalised RUL units.
            with torch.no_grad():
                phys_norm = p[:, 0] * p[:, 4]
                err_nn = (g_norm - y.squeeze(-1)).pow(2)
                err_phys = (phys_norm - y.squeeze(-1)).pow(2)
                target_w = torch.sigmoid((err_nn - err_phys) / GATE_TAU) * p[:, 4]
            gate_loss = F.binary_cross_entropy(
                w.clamp(1e-6, 1 - 1e-6), target_w.clamp(1e-6, 1 - 1e-6))
            loss = loss + GATE_LAMBDA * gate_loss
            gates, indices, feats, rough = gi
            loss = loss + 0.05 * model.backbone.moe.balance_loss(gates, indices, 10)
            E = 10
            with torch.no_grad():
                outs = torch.stack([model.backbone.moe.experts[e](feats)
                                    for e in range(E)], 1)
                ge = torch.stack([model.backbone.evidence(outs[:, e])[0].squeeze(-1)
                                  for e in range(E)], 1)
                err = (ge - y).pow(2); best = err.argmin(-1)
            clean = model.backbone.moe.gate.clean_logits(torch.cat([feats, rough], -1))
            masked = clean.clone()
            masked.scatter_(-1, best.unsqueeze(-1), float('-inf'))
            top3 = masked.topk(3, dim=-1).values.logsumexp(-1)
            hard = F.relu(top3 - clean.gather(-1, best.unsqueeze(-1)).squeeze(-1))
            loss = loss + 1.0 * hard.mean()
            loss.backward(); opt.step()

    model.eval()
    per_cell, ws, cell_ws = [], [], {}
    with torch.no_grad():
        for c in test_ids:
            x = torch.tensor(test_pack[c]['X'], dtype=torch.float32).to(device)
            p = torch.tensor(test_pack[c]['p'], dtype=torch.float32).to(device)
            g, nu, al, be, gi, w, g_norm = model(x, p)
            pred = ls.inverse_transform(g.cpu().numpy().reshape(-1, 1)).ravel()
            per_cell.append(rmse(test_pack[c]['y'], pred))
            ws.append(float(w.mean()))
            cell_ws[c] = float(w.mean())
    return np.mean(per_cell), per_cell, ws, cell_ws


print("\n=== Physics-informed hybrid vs pure NN (fixed split, 3 seeds) ===")
for use in [True]:
    all_r, all_w = [], []
    for s in range(3):
        avg, per_cell, ws, cell_ws = train_pih(s, use_physics=use)
        all_r.append(avg); all_w.append(np.mean(ws))
        print(f"seed {s}: PIH avg={avg:.2f} per-cell="
              f"{[f'{v:.1f}' for v in per_cell]} mean_gate_w={np.mean(ws):.2f} "
              f"cell_w={{{', '.join(f'{c}:{v:.2f}' for c, v in cell_ws.items())}}}",
              flush=True)
    print(f"\nPIH 3-seed: {np.mean(all_r):.2f}+/-{np.std(all_r):.2f} "
          f"(baseline clean single 23.81)")
