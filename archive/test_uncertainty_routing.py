#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Route by evidential uncertainty instead of a learned gate.

Each expert output goes through the evidence head -> (gamma, nu, alpha,
beta). A good expert is confident (low epistemic / aleatoric) on samples
in its territory. Test routing weights from uncertainty:
  E. soft weighting by exp(-epistemic)          (dense)
  F. soft weighting by exp(-aleatoric)          (dense)
  G. sparse top-4 by lowest epistemic           (keeps sparsity)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import rmse, evaluate_per_cell

device = get_device('cuda')
config = Config()
config.epochs = 100
config.random_seed = 42
config.moe_balance_coeff = 0.05
config.route_sup_coeff = 1.0

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)
model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
model.eval()

def to_rul(g):
    return label_scaler.inverse_transform(g.detach().cpu().numpy().reshape(-1, 1)).ravel()

results = {k: [] for k in ['A_gate', 'E_epis_soft', 'F_alea_soft', 'G_epis_top4', 'O_oracle']}
for cell in test_cells:
    x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
    y_orig = cell['y_orig']
    with torch.no_grad():
        _, (h_n, _) = model.lstm(x)
        h = h_n[-1]
        gammas, nu_s, alpha_s, beta_s = [], [], [], []
        for e in range(10):
            out = model.moe.experts[e](h)
            g, nu, al, be = model.evidence(out)
            gammas.append(g.squeeze(-1))    # (N,)
            nu_s.append(nu.squeeze(-1))
            alpha_s.append(al.squeeze(-1))
            beta_s.append(be.squeeze(-1))
        G = torch.stack(gammas, dim=1)      # (N,10)
        NU = torch.stack(nu_s, dim=1)
        AL = torch.stack(alpha_s, dim=1)
        BE = torch.stack(beta_s, dim=1)
        epis = BE / (NU * (AL - 1.0))       # (N,10)
        alea = BE / (AL - 1.0)

    # A: learned sparse gate (baseline)
    gammaA, *_ = model(x)
    results['A_gate'].append(rmse(y_orig, to_rul(gammaA)))

    # E: soft weights from epistemic uncertainty
    w_e = torch.softmax(-epis, dim=-1)
    predE = (G * w_e).sum(dim=1)
    results['E_epis_soft'].append(rmse(y_orig, to_rul(predE)))

    # F: soft weights from aleatoric uncertainty
    w_f = torch.softmax(-alea, dim=-1)
    predF = (G * w_f).sum(dim=1)
    results['F_alea_soft'].append(rmse(y_orig, to_rul(predF)))

    # G: sparse top-4 by lowest epistemic, equal weights
    top4 = torch.topk(-epis, 4, dim=-1).indices           # (N,4)
    sel = torch.gather(G, 1, top4)
    predG = sel.mean(dim=1)
    results['G_epis_top4'].append(rmse(y_orig, to_rul(predG)))

    # O: oracle
    errs = (G.cpu().numpy() - y_orig[:, None]) ** 2
    results['O_oracle'].append(float(np.sqrt(np.min(errs, axis=1).mean())))

print("\n=== Uncertainty-routing comparison (avg over 6 test cells) ===")
for k, v in results.items():
    print(f"  {k:12s}: {np.mean(v):6.2f}   (per-cell: {[f'{x:.1f}' for x in v]})")
print("\nReference: learned gate 22.77 | paper 16.11")
