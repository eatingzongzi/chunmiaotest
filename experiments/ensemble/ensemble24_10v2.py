#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""10-member 24-cell ensemble, corrected per-model scalers + epis weights.

Fixes ensemble24_10.py (wrongly used the evaluation seed's scaler for all
models) and applies episodic weighting, which beat median for 24-cell
models (26.16 at 5 members). Evaluates on seeds 42-46 tests to match the
26.16 baseline, and on all 10 seeds' tests for the wider picture.
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
SEEDS = list(range(42, 52))


def make_model(config, device):
    return SMoELSTM(
        input_dim=config.input_dim, hidden_dim=config.hidden_dim,
        num_experts=config.num_experts, top_k=config.top_k,
        lstm_layers=config.lstm_layers, evidence_hidden=config.evidence_hidden,
        gate_hidden=config.gate_hidden, bidirectional=config.bidirectional,
    ).to(device)


models, scalers = [], []
for seed in SEEDS:
    config = Config()
    model = make_model(config, device)
    model.load_state_dict(torch.load(os.path.join('checkpoints', f'ens_seed{seed}.pt'),
                                     map_location=device))
    model.eval()
    # reload data only for the label scaler (fast; no training)
    _, _, _, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=seed)
    models.append(model)
    scalers.append(label_scaler)
print("10 models loaded with own scalers")

for scope, seed_list in [('seeds42-46', SEEDS[:5]), ('all10', SEEDS)]:
    summaries = {k: [] for k in ['epis_weight', 'median', 'mean']}
    for si in seed_list:
        _, test_cells, _, _ = load_mit_stanford_paper(
            'data/mit_stanford', window_length=30, n_cells=30, n_test=6,
            seed=si)
        cell_res = {k: [] for k in summaries}
        for cell in test_cells:
            x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
            y_orig = cell['y_orig']
            preds, epis = [], []
            with torch.no_grad():
                for m, ls in zip(models, scalers):
                    g, nu, al, be = m(x)
                    preds.append(ls.inverse_transform(
                        g.detach().cpu().numpy().reshape(-1, 1)).ravel())
                    epis.append((be / (nu * (al - 1.0))).detach().cpu().numpy().ravel())
            P = np.array(preds)
            E = np.array(epis)
            inv = 1.0 / (E + 1e-6)
            w = inv / inv.sum(axis=0, keepdims=True)
            cell_res['epis_weight'].append(rmse(y_orig, (P * w).sum(axis=0)))
            cell_res['median'].append(rmse(y_orig, np.median(P, axis=0)))
            cell_res['mean'].append(rmse(y_orig, P.mean(axis=0)))
        for k in summaries:
            summaries[k].append(np.mean(cell_res[k]))
    print(f"\n=== {scope} ===")
    for k, v in summaries.items():
        print(f"  {k:12s}: {np.mean(v):6.2f}  (per-seed: {[f'{x:.1f}' for x in v]})")

print("\nReference: 5-member epis 26.16 (seeds42-46) | 42-cell median 16.25 | paper 16.49")
