#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Corrected distillation evaluation: each model uses its OWN label scaler.

distill.py's evaluation shared the evaluation seed's scaler across all
models (42-cell training sets differ in RUL range), polluting the numbers.
Student predictions are un-normalised with the seed-42 scaler it trained
with; each teacher model with its own.
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
    m = make_model(config, device)
    m.load_state_dict(torch.load(os.path.join('checkpoints', f'ens42_seed{seed}.pt'),
                                 map_location=device))
    m.eval()
    _, _, _, ls = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=seed)
    models.append((m, ls))

student = make_model(config, device)
student.load_state_dict(torch.load(os.path.join('checkpoints', 'student_distill.pt'),
                                   map_location=device))
student.eval()
_, _, _, ls42 = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=42)

summaries = {'student': [], 'teacher': []}
for seed in SEEDS:
    _, test_cells, _, _ = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=seed)
    cell_s, cell_t = [], []
    for cell in test_cells:
        x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
        y_orig = cell['y_orig']
        with torch.no_grad():
            g, *_ = student(x)
            ps = ls42.inverse_transform(
                g.detach().cpu().numpy().reshape(-1, 1)).ravel()
            preds_t = []
            for m, ls in models:
                gt, *_ = m(x)
                preds_t.append(ls.inverse_transform(
                    gt.detach().cpu().numpy().reshape(-1, 1)).ravel())
            pt = np.median(np.array(preds_t), axis=0)
        cell_s.append(rmse(y_orig, ps))
        cell_t.append(rmse(y_orig, pt))
    summaries['student'].append(np.mean(cell_s))
    summaries['teacher'].append(np.mean(cell_t))
    print(f"  seed{seed}: student={np.mean(cell_s):6.1f} teacher={np.mean(cell_t):6.1f}",
          flush=True)

print("\n=== Distillation (corrected scalers, 10-seed average) ===")
for k, v in summaries.items():
    print(f"  {k:9s}: {np.mean(v):6.2f}  (per-seed: {[f'{x:.1f}' for x in v]})")
print("Reference: 10-model teacher 16.25 | single-model (no distill) ~37.6 | paper 16.11")
