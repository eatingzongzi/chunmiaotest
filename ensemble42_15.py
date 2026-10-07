#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""15-member 42-cell ensemble (seeds 42-56), median/mean aggregation.

10-member median reached 16.25 (vs paper 16.49). More members stabilise
the median further; odd count 15 is good for median aggregation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.model import SMoELSTM
from src.evaluate import rmse

device = get_device('cuda')
SEEDS = list(range(42, 57))
CKPT = 'checkpoints'


def make_model(config, device):
    return SMoELSTM(
        input_dim=config.input_dim, hidden_dim=config.hidden_dim,
        num_experts=config.num_experts, top_k=config.top_k,
        lstm_layers=config.lstm_layers, evidence_hidden=config.evidence_hidden,
        gate_hidden=config.gate_hidden, bidirectional=config.bidirectional,
    ).to(device)


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
        model = make_model(config, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
    else:
        model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
        model.eval()
        torch.save(model.state_dict(), ckpt_path)
    models.append((model, label_scaler))
    loaders.append(test_cells)
    print(f"seed {seed} ready", flush=True)

summaries = {k: [] for k in ['mean', 'median']}
for si, test_cells in enumerate(loaders):
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
    print(f"  seed{SEEDS[si]}: mean={np.mean(cell_res['mean']):6.1f} "
          f"median={np.mean(cell_res['median']):6.1f}", flush=True)

print("\n=== 15-member ensemble, 15-seed averages ===")
for k, v in summaries.items():
    print(f"  {k:8s}: {np.mean(v):6.2f}  (per-seed: {[f'{x:.1f}' for x in v]})")
print("\nReference: 10-member median 16.25 | paper Table5 16.11 | paper Table6 16.49")
