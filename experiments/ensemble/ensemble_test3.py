#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ensemble v3: epistemic-guided model selection/weighting.

Oracle model choice (best single model per cell) reaches 18.87 — the
question is picking that model at test time. Each model's epistemic
uncertainty is its own confidence measure; test weighting predictions by
1/epistemic and selection by lowest epistemic. Models are SAVED to
checkpoints/ so follow-up aggregation experiments don't retrain.
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
    ckpt_path = os.path.join(CKPT, f'ens_seed{seed}.pt')
    config = Config()
    config.epochs = 100
    config.random_seed = seed
    config.moe_balance_coeff = 0.05
    config.route_sup_coeff = 1.0
    config.router_mse_coeff = 0.1
    train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=seed)
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

print("\n=== Epistemic-guided ensemble ===")
summaries = {k: [] for k in ['trimmed', 'epis_weight', 'epis_select', 'oracle']}
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
        P = np.array(preds)                  # (5, N)
        E = np.array(epis)                   # (5, N)

        sorted_p = np.sort(P, axis=0)
        trimmed = sorted_p[1:4].mean(axis=0)

        # per-sample weighting by 1/epistemic (normalised across models)
        inv = 1.0 / (E + 1e-6)
        w = inv / inv.sum(axis=0, keepdims=True)
        weighted = (P * w).sum(axis=0)

        # per-sample selection of the lowest-epistemic model
        chosen = P[E.argmin(axis=0), np.arange(P.shape[1])]

        oracle = np.array([rmse(y_orig, P[j]) for j in range(5)]).min()
        r = {
            'trimmed': rmse(y_orig, trimmed),
            'epis_weight': rmse(y_orig, weighted),
            'epis_select': rmse(y_orig, chosen),
            'oracle': oracle,
        }
        for k in summaries:
            cell_res[k].append(r[k])
    for k in summaries:
        summaries[k].append(np.mean(cell_res[k]))
    print(f"  seed{SEEDS[si]}: trimmed={np.mean(cell_res['trimmed']):6.1f} "
          f"epis_weight={np.mean(cell_res['epis_weight']):6.1f} "
          f"epis_select={np.mean(cell_res['epis_select']):6.1f} "
          f"oracle={np.mean(cell_res['oracle']):6.1f}", flush=True)

print("\n=== 5-seed averages ===")
for k, v in summaries.items():
    print(f"  {k:12s}: {np.mean(v):6.2f}  (per-seed: {[f'{x:.1f}' for x in v]})")
print("\nReference: trimmed ensemble 28.11 | single-model 32.33 | paper 16.11")
