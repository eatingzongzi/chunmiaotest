#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ensemble v5: aggregation variants and 10-model mixed pool (no retraining).

42-cell mean ensemble reached 20.12. Now check trimmed/median on the 5
42-cell models and a 10-model pool (5 x 24-cell + 5 x 42-cell) — the
24-cell models add diversity from their narrower training coverage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import get_device
from src.model import SMoELSTM
from src.evaluate import rmse

device = get_device('cuda')
SEEDS = [42, 43, 44, 45, 46]


def load_model(ckpt_name, config, device):
    model = SMoELSTM(
        input_dim=config.input_dim, hidden_dim=config.hidden_dim,
        num_experts=config.num_experts, top_k=config.top_k,
        lstm_layers=config.lstm_layers, evidence_hidden=config.evidence_hidden,
        gate_hidden=config.gate_hidden, bidirectional=config.bidirectional,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join('checkpoints', ckpt_name),
                                     map_location=device))
    model.eval()
    return model


config = Config()
ens24, ens42 = [], []
for seed in SEEDS:
    ens24.append(load_model(f'ens_seed{seed}.pt', config, device))
    ens42.append(load_model(f'ens42_seed{seed}.pt', config, device))
print("10 checkpoints loaded")

results = {k: [] for k in ['m42_mean', 'm42_median', 'm42_trimmed',
                           'm10_mean', 'm10_epis']}
for si in range(len(SEEDS)):
    _, test_cells, _, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=SEEDS[si])
    cell_res = {k: [] for k in results}
    for cell in test_cells:
        x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
        y_orig = cell['y_orig']
        preds42, preds10, epis10 = [], [], []
        with torch.no_grad():
            for m, ls in zip(ens42, [label_scaler] * 5):
                g, *_ = m(x)
                preds42.append(ls.inverse_transform(
                    g.detach().cpu().numpy().reshape(-1, 1)).ravel())
            for m, ls in zip(ens24 + ens42, [label_scaler] * 10):
                g, nu, al, be = m(x)
                preds10.append(ls.inverse_transform(
                    g.detach().cpu().numpy().reshape(-1, 1)).ravel())
                epis10.append((be / (nu * (al - 1.0))).detach().cpu().numpy().ravel())
        P42 = np.array(preds42)
        P10 = np.array(preds10)
        E10 = np.array(epis10)
        inv10 = 1.0 / (E10 + 1e-6)
        w10 = inv10 / inv10.sum(axis=0, keepdims=True)

        cell_res['m42_mean'].append(rmse(y_orig, P42.mean(axis=0)))
        cell_res['m42_median'].append(rmse(y_orig, np.median(P42, axis=0)))
        s42 = np.sort(P42, axis=0)
        cell_res['m42_trimmed'].append(rmse(y_orig, s42[1:4].mean(axis=0)))
        cell_res['m10_mean'].append(rmse(y_orig, P10.mean(axis=0)))
        cell_res['m10_epis'].append(rmse(y_orig, (P10 * w10).sum(axis=0)))
    for k in results:
        results[k].append(np.mean(cell_res[k]))
    print(f"  seed{SEEDS[si]}: 42mean={np.mean(cell_res['m42_mean']):6.1f} "
          f"42median={np.mean(cell_res['m42_median']):6.1f} "
          f"42trim={np.mean(cell_res['m42_trimmed']):6.1f} "
          f"10mean={np.mean(cell_res['m10_mean']):6.1f} "
          f"10epis={np.mean(cell_res['m10_epis']):6.1f}", flush=True)

print("\n=== 5-seed averages ===")
for k, v in results.items():
    print(f"  {k:10s}: {np.mean(v):6.2f}  (per-seed: {[f'{x:.1f}' for x in v]})")
print("\nReference: 42-cell mean 20.12 | paper 16.49+/-5.62 | paper 16.11")
