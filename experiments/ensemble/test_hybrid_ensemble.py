#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hybrid ensemble on the paper protocol: distilled student + 10
24-cell models, median aggregation (zero training, checkpoints only).

Student alone: 24.17 (5-seed). 24-cell models alone (10, median): 28.60.
The hybrid should beat both if their errors are complementary.
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
SEEDS24 = list(range(42, 52))


def make_model(config, device):
    return SMoELSTM(
        input_dim=config.input_dim, hidden_dim=config.hidden_dim,
        num_experts=config.num_experts, top_k=config.top_k,
        lstm_layers=config.lstm_layers, evidence_hidden=config.evidence_hidden,
        gate_hidden=config.gate_hidden, bidirectional=config.bidirectional,
    ).to(device)


config = Config()
student = make_model(config, device)
student.load_state_dict(torch.load(os.path.join('checkpoints', 'student_distill2.pt'),
                                   map_location=device))
student.eval()
_, _, _, ls42 = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=42)

models24, scalers24 = [], []
for seed in SEEDS24:
    m = make_model(config, device)
    m.load_state_dict(torch.load(os.path.join('checkpoints', f'ens_seed{seed}.pt'),
                                 map_location=device))
    m.eval()
    _, _, _, ls = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=seed)
    models24.append(m)
    scalers24.append(ls)
print("student + 10 24-cell models loaded")

per_seed = {'student': [], 'hybrid': [], '24only': []}
for seed in [42, 43, 44, 45, 46]:
    _, test_cells, _, _ = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=seed)
    cs, ch, c24 = [], [], []
    for cell in test_cells:
        x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
        y_orig = cell['y_orig']
        with torch.no_grad():
            g, *_ = student(x)
            ps = ls42.inverse_transform(
                g.detach().cpu().numpy().reshape(-1, 1)).ravel()
            p24 = []
            for m, ls in zip(models24, scalers24):
                gm, *_ = m(x)
                p24.append(ls.inverse_transform(
                    gm.detach().cpu().numpy().reshape(-1, 1)).ravel())
        P24 = np.array(p24)
        Phybrid = np.vstack([ps[None, :], P24])
        cs.append(rmse(y_orig, ps))
        ch.append(rmse(y_orig, np.median(Phybrid, axis=0)))
        c24.append(rmse(y_orig, np.median(P24, axis=0)))
    for k, v in [('student', cs), ('hybrid', ch), ('24only', c24)]:
        per_seed[k].append(np.mean(v))
    print(f"seed {seed}: student={np.mean(cs):6.1f} hybrid={np.mean(ch):6.1f} "
          f"24only={np.mean(c24):6.1f}", flush=True)

print("\n=== Paper-protocol 5-seed averages ===")
for k, v in per_seed.items():
    print(f"  {k:9s}: {np.mean(v):6.2f}  (per-seed: {[f'{x:.1f}' for x in v]})")
print("Reference: paper 16.49 | 24-cell 5-model epis 26.16")
