#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ensemble of 42-cell-training models with epistemic weighting.

Each model trains on 42 of the 48 cells (its own seed's 6-cell test held
out), covering nearly all degradation patterns including long-lived cells.
Epistemic weighting was the best aggregation for 24-cell models (26.16);
this tests whether fuller training coverage pushes the ensemble lower.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import rmse

device = get_device('cuda')
SEEDS = [42, 43, 44, 45, 46]
CKPT = 'checkpoints'
os.makedirs(CKPT, exist_ok=True)

models, loaders = [], []
for seed in SEEDS:
    ckpt_path = os.path.join(CKPT, f'ens42_seed{seed}.pt')
    config = Config()
    config.epochs = 100
    config.random_seed = seed
    config.moe_balance_coeff = 0.05
    config.route_sup_coeff = 1.0
    config.router_mse_coeff = 0.1
    train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=seed)
    if os.path.exists(ckpt_path):
        from src.model import SMoELSTM
        model = SMoELSTM(
            input_dim=config.input_dim, hidden_dim=config.hidden_dim,
            num_experts=config.num_experts, top_k=config.top_k,
            lstm_layers=config.lstm_layers, evidence_hidden=config.evidence_hidden,
            gate_hidden=config.gate_hidden, bidirectional=config.bidirectional,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
    else:
        model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
        model.eval()
        torch.save(model.state_dict(), ckpt_path)
    models.append((model, label_scaler))
    loaders.append(test_cells)
    print(f"seed {seed} ready", flush=True)

summaries = {k: [] for k in ['own', 'mean', 'epis_weight']}
for si, test_cells in enumerate(loaders):
    cell_res = {k: [] for k in summaries}
    for cell in test_cells:
        x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
        y_orig = cell['y_orig']
        preds, epis = [], []
        with torch.no_grad():
            for mi, (m, ls) in enumerate(models):
                g, nu, al, be = m(x)
                p = ls.inverse_transform(
                    g.detach().cpu().numpy().reshape(-1, 1)).ravel()
                preds.append(p)
                epis.append((be / (nu * (al - 1.0))).detach().cpu().numpy().ravel())
                if mi == si:
                    own_p = p
        P = np.array(preds)
        E = np.array(epis)
        inv = 1.0 / (E + 1e-6)
        w = inv / inv.sum(axis=0, keepdims=True)
        cell_res['own'].append(rmse(y_orig, own_p))
        cell_res['mean'].append(rmse(y_orig, P.mean(axis=0)))
        cell_res['epis_weight'].append(rmse(y_orig, (P * w).sum(axis=0)))
    for k in summaries:
        summaries[k].append(np.mean(cell_res[k]))
    print(f"  seed{SEEDS[si]}: own={np.mean(cell_res['own']):6.1f} "
          f"mean={np.mean(cell_res['mean']):6.1f} "
          f"epis_weight={np.mean(cell_res['epis_weight']):6.1f}", flush=True)

print("\n=== 5-seed averages (42-cell training) ===")
for k, v in summaries.items():
    print(f"  {k:12s}: {np.mean(v):6.2f}  (per-seed: {[f'{x:.1f}' for x in v]})")
print("\nReference: 24-cell epis-ens 26.16 | single 32.33 | paper 16.11")
