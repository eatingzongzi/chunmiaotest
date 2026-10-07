#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Can epistemic uncertainty flag extreme/anomalous cells? (Route C)

Evaluates the 20 saved 42-cell models on the 10 seed 42-51 test splits:
for each cell compute ensemble RMSE and the mean (ensemble) epistemic
uncertainty, both in original RUL scale. A trustworthy model should be
MORE uncertain on hard/anomalous cells — that is the warning signal.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from scipy.stats import spearmanr

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


config = Config()
models = []
for seed in SEEDS:
    model = make_model(config, device)
    model.load_state_dict(torch.load(os.path.join('checkpoints', f'ens42_seed{seed}.pt'),
                                     map_location=device))
    model.eval()
    _, _, _, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=seed)
    models.append((model, label_scaler))
print("20 checkpoints loaded")

rows = []
for seed in SEEDS:
    _, test_cells, _, _ = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=seed)
    for cell in test_cells:
        x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
        y_orig = cell['y_orig']
        preds, epis = [], []
        with torch.no_grad():
            for m, ls in models:
                g, nu, al, be = m(x)
                p = ls.inverse_transform(
                    g.detach().cpu().numpy().reshape(-1, 1)).ravel()
                # epistemic variance, scaled to original RUL units
                e = (be / (nu * (al - 1.0))).detach().cpu().numpy().ravel()
                e_orig = e * (float(ls.data_range_[0]) ** 2)
                preds.append(p)
                epis.append(e_orig)
        P = np.array(preds)
        cell_rmse = rmse(y_orig, np.median(P, axis=0))
        cell_epis = float(np.median(np.array(epis), axis=0).mean())
        rows.append({
            'cell_id': cell['cell_id'], 'eol': cell['eol'],
            'rmse': cell_rmse, 'epis': cell_epis,
        })

rows.sort(key=lambda r: r['rmse'])
print("\n=== Per-cell ensemble RMSE vs mean epistemic uncertainty ===")
print(f"{'cell':>5} {'eol':>5} {'RMSE':>8} {'epis':>10} {'epis/RMSE^2':>12}")
hard_epis, easy_epis = [], []
for r in rows:
    hard = r['rmse'] > 25
    if hard:
        hard_epis.append(r['epis'])
    else:
        easy_epis.append(r['epis'])
    tag = "HARD" if hard else "    "
    print(f"{r['cell_id']:>5} {r['eol']:>5} {r['rmse']:>8.1f} {r['epis']:>10.1f} "
          f"{r['epis']/(r['rmse']**2+1e-6):>12.3f} [{tag}]")

rmse_arr = np.array([r['rmse'] for r in rows])
epis_arr = np.array([r['epis'] for r in rows])
rho, p = spearmanr(rmse_arr, epis_arr)
print(f"\nSpearman(per-cell RMSE, epistemic): rho={rho:.3f} (p={p:.2e})")
print(f"Mean epistemic: hard cells (RMSE>25) = {np.mean(hard_epis):.1f} "
      f"({len(hard_epis)} cells) | easy cells = {np.mean(easy_epis):.1f} "
      f"({len(easy_epis)} cells) | ratio = "
      f"{np.mean(hard_epis)/max(np.mean(easy_epis),1e-9):.2f}x")
