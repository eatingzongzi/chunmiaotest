#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Control experiment: LSTM-only (no SMoE) under the same paper protocol.

If LSTM-only lands near our SMoE-LSTM result (~24), the MoE module in our
reproduction adds nothing (dead module / collapse-equivalent). If it is
clearly worse, the MoE contributes and the gap to 16.11 lies elsewhere.
Paper ablation: w/o SMoE = 26.42, with SMoE = 16.11.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import set_seed, get_device, prepare_dataloaders
from src.loss import EvidentialRegressionLoss
from src.model import EvidenceLayer
from src.evaluate import evaluate_per_cell


class LSTMOnly(nn.Module):
    """SMoELSTM minus the SMoE module (same LSTM + evidence head)."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 128,
                 lstm_layers: int = 2, evidence_hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers,
                            batch_first=True, dropout=dropout)
        self.evidence = EvidenceLayer(hidden_dim, evidence_hidden)

        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.evidence(h_n[-1])


device = get_device('cuda')
config = Config()
config.epochs = 100
config.random_seed = 42

train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=42)

set_seed(config.random_seed)
train_loader, _ = prepare_dataloaders(train_ds, config.batch_size,
                                      val_ratio=0.0, seed=config.random_seed)
model = LSTMOnly().to(device)
criterion = EvidentialRegressionLoss(coeff=config.evidential_coeff)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                             weight_decay=config.weight_decay)

for epoch in range(1, config.epochs + 1):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        gamma, nu, alpha, beta = model(X)
        loss = criterion(y, gamma, nu, alpha, beta)
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        print(f"  epoch {epoch}/{config.epochs} loss={loss.item():.4f}", flush=True)

per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
print(f"\nLSTM-only (no SMoE): avg RMSE={avg['rmse']:.2f} MAE={avg['mae']:.2f} "
      f"MAPE={avg['mape']:.2f}% R2={avg['r2']:.4f}")
for i, pc in enumerate(per_cell):
    print(f"  #{i+1} cell_id={pc['cell_id']:2d} eol={pc['eol']:4d} "
          f"RMSE={pc['metrics']['rmse']:.2f}")
print("\nReference: SMoE-LSTM (collapse) seed42 = 23.89 | paper w/o SMoE = 26.42 | paper = 16.11")
