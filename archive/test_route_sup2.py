#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two-stage routing: stage 1 trains all (balance + route-sup), stage 2
freezes experts/LSTM/evidence and trains ONLY the gate to learn routing.

Oracle upper bound with route-sup experts is 11.52 (vs paper 16.11) —
the gate is the last bottleneck. This isolates its learning.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device, prepare_dataloaders, set_seed
from src.loss import EvidentialRegressionLoss
from src.evaluate import evaluate_per_cell

device = get_device('cuda')
config = Config()
config.epochs = 100
config.random_seed = 42
config.moe_balance_coeff = 0.05
config.route_sup_coeff = 1.0

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)

# ── Stage 1: full training ──────────────────────────────────────────
model, _ = train_model(config, train_ds, device, trial=0, verbose=False)
per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
print(f"Stage 1 (full train): avg RMSE={avg['rmse']:.2f}")

# ── Stage 2: freeze everything except the gate ──────────────────────
for p in model.parameters():
    p.requires_grad = False
for p in model.moe.gate.parameters():
    p.requires_grad = True

set_seed(config.random_seed)
train_loader, _ = prepare_dataloaders(train_ds, config.batch_size,
                                      val_ratio=0.0, seed=config.random_seed)
criterion = EvidentialRegressionLoss(coeff=config.evidential_coeff)
optimizer = torch.optim.Adam(
    [p for p in model.moe.gate.parameters() if p.requires_grad],
    lr=config.learning_rate, weight_decay=0.0)

import torch.nn.functional as F
tau = config.route_sup_tau
for epoch in range(1, 51):
    model.train()
    tot = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        gamma, nu, alpha, beta, gate_info = model(X, return_gates=True)
        loss = criterion(y, gamma, nu, alpha, beta)
        gates, indices, feats = gate_info
        E = model.moe.num_experts
        with torch.no_grad():
            outs = torch.stack([model.moe.experts[e](feats) for e in range(E)], dim=1)
            gamma_e = torch.stack([model.evidence(outs[:, e])[0].squeeze(-1)
                                   for e in range(E)], dim=1)
            err = (gamma_e - y).pow(2)
            target = F.softmax(-err / tau, dim=-1)
        clean = model.moe.gate.w_g(feats)
        gate_prob = F.softmax(clean, dim=-1)
        route = -(target * torch.log(gate_prob + 1e-8)).sum(-1).mean()
        loss = loss + 3.0 * route   # stronger routing supervision in stage 2
        loss.backward()
        optimizer.step()
        tot += loss.item() * X.size(0)
    if epoch % 10 == 0:
        print(f"  stage2 epoch {epoch:2d}: loss={tot/len(train_ds):.4f}", flush=True)

model.eval()
per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
print(f"\nStage 2 (gate-only training): avg RMSE={avg['rmse']:.2f} "
      f"MAE={avg['mae']:.2f} R2={avg['r2']:.4f}")
for i, pc in enumerate(per_cell):
    print(f"  #{i+1} cell_id={pc['cell_id']:2d} eol={pc['eol']:4d} "
          f"RMSE={pc['metrics']['rmse']:.2f}")

# gate agreement
agree, n = 0, 0
for cell in test_cells:
    x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
    y_orig = cell['y_orig']
    with torch.no_grad():
        _, (h_n, _) = model.lstm(x)
        h = h_n[-1]
        errs = []
        for e in range(10):
            out = model.moe.experts[e](h)
            gamma_e, *_ = model.evidence(out)
            pred = label_scaler.inverse_transform(
                gamma_e.detach().cpu().numpy().reshape(-1, 1)).ravel()
            errs.append((pred - y_orig) ** 2)
        errs = np.array(errs).T
        best = errs.argmin(axis=1)
        clean = model.moe.gate.w_g(h)
        _, idx = torch.topk(clean, config.top_k, dim=-1)
        routed = idx.cpu().numpy()
        hit = np.array([best[j] in routed[j] for j in range(len(x))])
        agree += hit.sum(); n += len(x)
print(f"Gate agreement after stage 2: {agree/n:.1%} "
      f"(stage1=32.7%, random=40%)")
print("\nReference: stage1 22.77 | oracle 11.52 | paper 16.11")
