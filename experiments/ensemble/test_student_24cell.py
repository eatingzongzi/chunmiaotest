#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Can the 42-cell distilled student beat 24-cell models on the PAPER
protocol test sets?

The student learned from soft targets over all 48 cells (19万 samples),
so it has seen every degradation pattern — including the hard cells that
sink every 24-cell-trained model. Evaluate it on the paper-protocol
(30-cell pool, 24/6 split) test sets for seeds 42-46, vs the 24-cell
ensemble baseline of 26.16.
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
SEEDS = [42, 43, 44, 45, 46]


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
print("student v2 loaded (trained on all-48-cell soft targets)")

per_seed = []
for seed in SEEDS:
    _, test_cells, _, _ = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=seed)
    cell_rmse = []
    for cell in test_cells:
        x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
        y_orig = cell['y_orig']
        with torch.no_grad():
            g, *_ = student(x)
            ps = ls42.inverse_transform(
                g.detach().cpu().numpy().reshape(-1, 1)).ravel()
        cell_rmse.append(rmse(y_orig, ps))
        print(f"  seed{seed} cell {cell['cell_id']:2d} eol={cell['eol']:4d} "
              f"RMSE={cell_rmse[-1]:6.1f}", flush=True)
    per_seed.append(np.mean(cell_rmse))
    print(f"  seed {seed}: avg={per_seed[-1]:.1f}", flush=True)

print(f"\n=== Distilled student on PAPER protocol (24-cell splits) ===")
print(f"  5-seed average: {np.mean(per_seed):.2f}  "
      f"(per-seed: {[f'{x:.1f}' for x in per_seed]})")
print("Reference: 24-cell models 26.16 (5-model epis ensemble) | "
      "24-cell single 32.33 | paper 16.49")
