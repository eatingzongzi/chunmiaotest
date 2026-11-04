#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leak-free re-evaluation of the MIT results.

Problem: earlier ensembles (42-cell: 15.53, 24-cell: 26.16) aggregated
models trained on DIFFERENT splits, so for a given test cell most
ensemble members had TRAINED on it. Here each protocol uses ONE fixed
split; test cells are never seen by any model. Diversity comes only from
random initialisation (5 seeds).

Protocols:
  A: 42 train / 6 test (our optimised data protocol)
  B: 24 train / 6 test (paper protocol)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell, rmse

device = get_device('cuda')
N_MODELS = 5


def run_protocol(tag, n_cells, seed_split=42):
    print(f"\n=== Protocol {tag}: {n_cells - 6} train / 6 test (fixed split) ===",
          flush=True)
    train_ds, test_cells, _, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=n_cells, n_test=6,
        seed=seed_split)
    print(f"  test cells: {[c['cell_id'] for c in test_cells]} "
          f"(eol: {[c['eol'] for c in test_cells]})", flush=True)

    models = []
    for k in range(N_MODELS):
        config = Config()
        config.epochs = 100
        config.random_seed = 100  # fixed split; diversity via trial offset
        config.moe_balance_coeff = 0.05
        config.route_sup_coeff = 1.0
        config.router_mse_coeff = 0.1
        m, _ = train_model(config, train_ds, device, trial=k, verbose=False)
        m.eval()
        models.append(m)
        print(f"  model {k+1}/{N_MODELS} trained", flush=True)

    # per-cell predictions in original units
    per_cell = {c['cell_id']: {'y': c['y_orig']} for c in test_cells}
    for c in test_cells:
        x = torch.tensor(c['X'], dtype=torch.float32).to(device)
        preds = []
        with torch.no_grad():
            for m in models:
                g, *_ = m(x)
                preds.append(label_scaler.inverse_transform(
                    g.cpu().numpy().reshape(-1, 1)).ravel())
        per_cell[c['cell_id']]['preds'] = np.array(preds)

    singles, means, medians = [], [], []
    for cid, d in per_cell.items():
        P = d['preds']
        s = np.mean([rmse(d['y'], P[i]) for i in range(len(P))])
        mn = rmse(d['y'], P.mean(0))
        md = rmse(d['y'], np.median(P, 0))
        singles.append(s); means.append(mn); medians.append(md)
        print(f"  cell {cid:2d}: single(avg)={s:6.1f} "
              f"ens_mean={mn:6.1f} ens_median={md:6.1f}", flush=True)
    print(f"  --> {tag}: single={np.mean(singles):.2f} "
          f"ens_mean={np.mean(means):.2f} ens_median={np.mean(medians):.2f}",
          flush=True)
    return {'single': np.mean(singles), 'mean': np.mean(means),
            'median': np.mean(medians)}


if __name__ == '__main__':
    ra = run_protocol('A (42-cell)', 48)
    rb = run_protocol('B (24-cell paper)', 30)
    print("\n=== SUMMARY (leak-free, fixed split, 5-model ensemble) ===")
    print(f"  42-cell: {ra}")
    print(f"  24-cell: {rb}")
    print("  Paper: 16.11 (Table 5) / 16.49+/-5.62 (Table 6)")
