#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune the distilled student on each seed's 24-cell training set.

The student (from all-48-cell soft targets) scored 24.17 on the paper
protocol with ZERO 24-cell training. Fine-tuning on each split's own
24 cells adapts it to the paper protocol; expect <20 (baselines: 24-cell
models 32.33 single / 26.16 ensemble, paper 16.49).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import prepare_dataloaders, set_seed, get_device
from src.model import SMoELSTM
from src.loss import EvidentialRegressionLoss
from src.evaluate import rmse

device = get_device('cuda')
SEEDS = [42, 43, 44, 45, 46]
CKPT = 'checkpoints'


def make_model(config, device):
    return SMoELSTM(
        input_dim=config.input_dim, hidden_dim=config.hidden_dim,
        num_experts=config.num_experts, top_k=config.top_k,
        lstm_layers=config.lstm_layers, evidence_hidden=config.evidence_hidden,
        gate_hidden=config.gate_hidden, bidirectional=config.bidirectional,
    ).to(device)


config = Config()
per_seed = []
for seed in SEEDS:
    student = make_model(config, device)
    student.load_state_dict(torch.load(os.path.join(CKPT, 'student_distill2.pt'),
                                       map_location=device))

    train_ds, test_cells, _, ls = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=seed)
    loader, _ = prepare_dataloaders(train_ds, config.batch_size,
                                    val_ratio=0.0, seed=seed)
    criterion = EvidentialRegressionLoss(coeff=2.0)
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-4, weight_decay=1e-3)

    set_seed(seed)
    for epoch in range(1, 41):
        student.train()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            g, nu, al, be = student(X)
            loss = criterion(y, g, nu, al, be)
            loss.backward()
            optimizer.step()

    student.eval()
    cell_rmse = []
    for cell in test_cells:
        x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
        y_orig = cell['y_orig']
        with torch.no_grad():
            g, *_ = student(x)
            ps = ls.inverse_transform(
                g.detach().cpu().numpy().reshape(-1, 1)).ravel()
        cell_rmse.append(rmse(y_orig, ps))
    per_seed.append(np.mean(cell_rmse))
    print(f"seed {seed}: avg={per_seed[-1]:.1f}  "
          f"per-cell={[f'{c:.1f}' for c in cell_rmse]}", flush=True)
    torch.save(student.state_dict(),
               os.path.join(CKPT, f'student_ft24_seed{seed}.pt'))

print(f"\n=== 24-cell fine-tuned student (paper protocol) ===")
print(f"  5-seed average: {np.mean(per_seed):.2f}  "
      f"(per-seed: {[f'{x:.1f}' for x in per_seed]})")
print("Reference: zero-finetune student 24.17 | 24-cell ensemble 26.16 | paper 16.49")
