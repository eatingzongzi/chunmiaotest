#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose whether the sparse gating actually works.

Paper ablation (Section 4.3): w/o SMoE -> RMSE 26.42, with SMoE -> 16.11.
Our reproduction sits at the w/o-SMoE level, so check two suspects:

  1. Expert collapse: load distribution across the 10 experts.
  2. Noise at eval time: NoisyTopKGate adds random noise unconditionally,
     including during evaluation. Compare noisy vs clean gating RMSE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell

device = get_device('cuda')
config = Config()
config.epochs = 100
config.random_seed = 42

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)
model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
model.eval()

# ── 1. Expert load on the training set (clean logits) ────────────────
loads = np.zeros(config.num_experts)
with torch.no_grad():
    for i in range(0, len(train_ds.X), 512):
        x = train_ds.X[i:i + 512].to(device)
        _, (h_n, _) = model.lstm(x)
        clean = model.moe.gate.w_g(h_n[-1])
        _, idx = torch.topk(clean, config.top_k, dim=-1)
        for k in range(config.top_k):
            np.add.at(loads, idx[:, k].cpu().numpy(), 1)
print("\n=== Expert load share (clean top-k, train set) ===")
print("  " + " ".join(f"E{e}:{loads[e]/loads.sum():.3f}" for e in range(config.num_experts)))

# gate weight norms
wn = [model.moe.gate.w_g.weight[e].norm().item() for e in range(config.num_experts)]
print("  gate row norms:", " ".join(f"{v:.2f}" for v in wn))

# ── 2. Noisy vs clean gating on the test set ─────────────────────────
per_cell_noisy, avg_noisy = evaluate_per_cell(model, test_cells, device, label_scaler)
print(f"\nNoisy gating (current eval): avg RMSE={avg_noisy['rmse']:.2f}")

# Monkey-patch the gate to use clean logits (no noise) at eval time
def clean_forward(self, x):
    clean_logits = self.w_g(x)
    top_k_logits, top_k_indices = torch.topk(clean_logits, self.top_k, dim=-1)
    mask = torch.full_like(clean_logits, float('-inf'))
    mask.scatter_(-1, top_k_indices, top_k_logits)
    gates = F.softmax(mask, dim=-1)
    return gates, top_k_indices

from src.model import NoisyTopKGate
NoisyTopKGate.forward = clean_forward
per_cell_clean, avg_clean = evaluate_per_cell(model, test_cells, device, label_scaler)
print(f"Clean gating (no noise):  avg RMSE={avg_clean['rmse']:.2f}")

print("\nPer-cell comparison:")
for a, b in zip(per_cell_noisy, per_cell_clean):
    print(f"  cell {a['cell_id']:2d} eol={a['eol']:4d}: "
          f"noisy={a['metrics']['rmse']:.2f}  clean={b['metrics']['rmse']:.2f}")
