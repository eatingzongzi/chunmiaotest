#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Knowledge distillation: 20-model ensemble -> single model.

Teacher: median of the 20 saved 42-cell models (ensemble 15.53).
Student: one SMoELSTM, trained on soft targets from the teacher using
the evidential NLL + regulariser so it keeps uncertainty quantification.
Expected: ~16-17 RMSE at 1/20 inference cost.
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
SEEDS = list(range(42, 52))
config = Config()
config.random_seed = 42

# ── 1. Generate soft targets on the seed-42 training set ────────────
train_ds, _, _, label_scaler = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=42)
print(f"teacher target generation on {len(train_ds)} samples...")

models = []
for seed in SEEDS:
    m = SMoELSTM(
        input_dim=config.input_dim, hidden_dim=config.hidden_dim,
        num_experts=config.num_experts, top_k=config.top_k,
        lstm_layers=config.lstm_layers, evidence_hidden=config.evidence_hidden,
        gate_hidden=config.gate_hidden, bidirectional=config.bidirectional,
    ).to(device)
    m.load_state_dict(torch.load(os.path.join('checkpoints', f'ens42_seed{seed}.pt'),
                                 map_location=device))
    m.eval()
    models.append(m)

soft_orig = []
X_all = train_ds.X
with torch.no_grad():
    for i in range(0, len(X_all), 512):
        x = X_all[i:i + 512].to(device)
        preds = []
        for m in models:
            g, *_ = m(x)
            preds.append(label_scaler.inverse_transform(
                g.detach().cpu().numpy().reshape(-1, 1)).ravel())
        soft_orig.append(np.median(np.array(preds), axis=0))
soft_orig = np.concatenate(soft_orig)
soft_norm = label_scaler.transform(soft_orig.reshape(-1, 1)).ravel()
print("soft targets ready")

# ── 2. Train the student on soft targets ────────────────────────────
student = SMoELSTM(
    input_dim=config.input_dim, hidden_dim=config.hidden_dim,
    num_experts=config.num_experts, top_k=config.top_k,
    lstm_layers=config.lstm_layers, evidence_hidden=config.evidence_hidden,
    gate_hidden=config.gate_hidden, bidirectional=config.bidirectional,
).to(device)

# replace dataset targets with soft targets
import torch.utils.data as data
class SoftDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

soft_ds = SoftDataset(X_all, soft_norm)
loader, _ = prepare_dataloaders(soft_ds, config.batch_size, val_ratio=0.0, seed=42)
criterion = EvidentialRegressionLoss(coeff=2.0)
optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-3)

set_seed(config.random_seed)
for epoch in range(1, 101):
    student.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        g, nu, al, be = student(X)
        loss = criterion(y, g, nu, al, be)
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        print(f"  distill epoch {epoch}/100 loss={loss.item():.4f}", flush=True)

torch.save(student.state_dict(), os.path.join('checkpoints', 'student_distill.pt'))
print("student saved")

# ── 3. Evaluate student vs teacher on the 10 test splits ────────────
student.eval()
summaries = {'student': [], 'teacher': []}
for seed in SEEDS:
    _, test_cells, _, ls = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=seed)
    cell_s, cell_t = [], []
    for cell in test_cells:
        x = torch.tensor(cell['X'], dtype=torch.float32).to(device)
        y_orig = cell['y_orig']
        with torch.no_grad():
            g, *_ = student(x)
            ps = ls.inverse_transform(
                g.detach().cpu().numpy().reshape(-1, 1)).ravel()
            preds_t = []
            for m in models:
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

print("\n=== Distillation result (10-seed average) ===")
for k, v in summaries.items():
    print(f"  {k:9s}: {np.mean(v):6.2f}  (per-seed: {[f'{x:.1f}' for x in v]})")
print("Reference: teacher 15.53 (paper 16.11) | student cost = 1/20 of ensemble")
