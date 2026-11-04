#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""20-member 42-cell ensemble (seeds 42-61).

15-member median on the seeds-42-51 tests: 15.66 (< paper 16.11).
More members keep shaving the median; also report the seeds-42-51
same-scope number for direct comparison.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.model import SMoELSTM
from src.evaluate import rmse

device = get_device('cuda')
SEEDS = list(range(42, 62))
CKPT = 'checkpoints'


def make_model(config, device):
    return SMoELSTM(
        input_dim=config.input_dim, hidden_dim=config.hidden_dim,
        num_experts=config.num_experts, top_k=config.top_k,
        lstm_layers=config.lstm_layers, evidence_hidden=config.evidence_hidden,
        gate_hidden=config.gate_hidden, bidirectional=config.bidirectional,
    ).to(device)


models, scalers, test_cellss = [], [], {}
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
        model = make_model(config, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
    else:
        model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
        model.eval()
        torch.save(model.state_dict(), ckpt_path)
    models.append((model, label_scaler))
    test_cellss[seed] = test_cells
    print(f"seed {seed} ready", flush=True)


def evaluate(seed_list):
    summaries = {k: [] for k in ['mean', 'median']}
    for seed in seed_list:
        test_cells = test_cellss[seed]
        cell_res = {k: [] for k in summaries}
        for cell in test_cells:
            x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
            y_orig = cell['y_orig']
            preds = []
            with torch.no_grad():
                for m, ls in models:
                    g, *_ = m(x)
                    preds.append(ls.inverse_transform(
                        g.detach().cpu().numpy().reshape(-1, 1)).ravel())
            P = np.array(preds)
            cell_res['mean'].append(rmse(y_orig, P.mean(axis=0)))
            cell_res['median'].append(rmse(y_orig, np.median(P, axis=0)))
        for k in summaries:
            summaries[k].append(np.mean(cell_res[k]))
    return summaries


print("\n=== 20-member ensemble ===")
for scope, seeds in [('seeds42-51 (same scope)', SEEDS[:10]),
                     ('all20 seeds', SEEDS)]:
    s = evaluate(seeds)
    print(f"  {scope}: mean={np.mean(s['mean']):.2f} median={np.mean(s['median']):.2f}")
    print(f"    per-seed median: {[f'{x:.1f}' for x in s['median']]}")

print("\nReference: 15-member median (42-51) 15.66 | paper Table5 16.11 | Table6 16.49")
