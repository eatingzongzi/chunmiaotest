#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ensemble v4: robust variants of epistemic weighting (no retraining).

Loads the saved checkpoints and compares:
  A. epis_weight        (baseline, 26.16)
  B. top3-epis equal    (drop the 2 most uncertain models per sample)
  C. top3-epis weighted (1/epis weights within the top 3)
  D. trimmed-by-epis    (drop highest-epis model, mean of the rest)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import get_device
from src.model import SMoELSTM
from src.evaluate import rmse

device = get_device('cuda')
SEEDS = [42, 43, 44, 45, 46]

models, loaders = [], []
for seed in SEEDS:
    config = Config()
    model = SMoELSTM(
        input_dim=config.input_dim, hidden_dim=config.hidden_dim,
        num_experts=config.num_experts, top_k=config.top_k,
        lstm_layers=config.lstm_layers, evidence_hidden=config.evidence_hidden,
        gate_hidden=config.gate_hidden, bidirectional=config.bidirectional,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join('checkpoints', f'ens_seed{seed}.pt'),
                                     map_location=device))
    model.eval()
    _, test_cells, _, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=seed)
    models.append((model, label_scaler))
    loaders.append(test_cells)
print("5 checkpoints loaded")

summaries = {k: [] for k in ['A', 'B', 'C', 'D']}
for si, test_cells in enumerate(loaders):
    cell_res = {k: [] for k in summaries}
    for cell in test_cells:
        x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
        y_orig = cell['y_orig']
        preds, epis = [], []
        with torch.no_grad():
            for m, ls in models:
                g, nu, al, be = m(x)
                preds.append(ls.inverse_transform(
                    g.detach().cpu().numpy().reshape(-1, 1)).ravel())
                epis.append((be / (nu * (al - 1.0))).detach().cpu().numpy().ravel())
        P = np.array(preds)          # (5, N)
        E = np.array(epis)           # (5, N)
        inv = 1.0 / (E + 1e-6)

        # A: full 1/epis weighting
        wA = inv / inv.sum(axis=0, keepdims=True)
        rA = (P * wA).sum(axis=0)

        # B: top-3 lowest epis, equal weights
        top3 = np.argsort(E, axis=0)[:3]          # (3, N)
        rB = np.array([P[top3[:, j], j].mean() for j in range(P.shape[1])])

        # C: top-3 lowest epis, 1/epis weights
        rC = np.array([
            np.average(P[top3[:, j], j], weights=inv[top3[:, j], j])
            for j in range(P.shape[1])
        ])

        # D: drop the single highest-epis model, mean of the rest
        drop = E.argmax(axis=0)
        keep = np.array([[m for m in range(5) if m != drop[j]] for j in range(P.shape[1])])
        rD = np.array([P[keep[j], j].mean() for j in range(P.shape[1])])

        for k, rv in [('A', rA), ('B', rB), ('C', rC), ('D', rD)]:
            cell_res[k].append(rmse(y_orig, rv))
    for k in summaries:
        summaries[k].append(np.mean(cell_res[k]))
    print(f"  seed{SEEDS[si]}: A={np.mean(cell_res['A']):6.1f} "
          f"B={np.mean(cell_res['B']):6.1f} C={np.mean(cell_res['C']):6.1f} "
          f"D={np.mean(cell_res['D']):6.1f}", flush=True)

print("\n=== 5-seed averages ===")
for k, v in summaries.items():
    print(f"  {k}: {np.mean(v):6.2f}  (per-seed: {[f'{x:.1f}' for x in v]})")
print("\nReference: epis_weight 26.16 | single-model 32.33 | oracle 18.87 | paper 16.11")
