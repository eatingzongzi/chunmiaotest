#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Joint distillation for the 24-cell paper protocol.

Start from the v3 student (all-48-cell soft targets, 15.96 on the 42-cell
protocol / 23.14 zero-finetune on 24-cell). Continue training on a MIX of
  - replay soft targets (42-cell protocol, from the 20-model teacher)
  - hard labels of the ten 24-cell training splits
so the student keeps full-pattern knowledge while adapting to the paper
protocol's label scale. Finetuning alone forgot (25.56); mixed replay
should avoid that.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.utils.data as data

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


def make_model(config, device):
    return SMoELSTM(
        input_dim=config.input_dim, hidden_dim=config.hidden_dim,
        num_experts=config.num_experts, top_k=config.top_k,
        lstm_layers=config.lstm_layers, evidence_hidden=config.evidence_hidden,
        gate_hidden=config.gate_hidden, bidirectional=config.bidirectional,
    ).to(device)


class SoftDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ── Teachers (20 x 42-cell models) with own scalers ─────────────────
teachers, scalers = [], []
for seed in range(42, 62):
    m = make_model(config, device)
    m.load_state_dict(torch.load(os.path.join('checkpoints', f'ens42_seed{seed}.pt'),
                                 map_location=device))
    m.eval()
    _, _, _, ls = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=seed)
    teachers.append(m)
    scalers.append(ls)
print("20 teachers loaded")

# ── Replay soft targets (42-cell protocol, subset for speed) ────────
replay_X, replay_y = [], []
for seed in SEEDS:
    train_ds, _, _, ls = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=seed)
    X_all = train_ds.X
    soft_parts = []
    with torch.no_grad():
        for i in range(0, len(X_all), 512):
            x = X_all[i:i + 512].to(device)
            preds = []
            for m, lsm in zip(teachers, scalers):
                g, *_ = m(x)
                preds.append(lsm.inverse_transform(
                    g.detach().cpu().numpy().reshape(-1, 1)).ravel())
            soft_parts.append(np.median(np.array(preds), axis=0))
    soft = np.concatenate(soft_parts)
    soft_n = ls.transform(soft.reshape(-1, 1)).ravel()
    # keep half as replay
    keep = np.random.RandomState(seed).choice(len(X_all), len(X_all) // 2,
                                              replace=False)
    replay_X.append(X_all[keep])
    replay_y.append(soft_n[keep])
    print(f"  replay soft targets for seed {seed}: {len(keep)} samples", flush=True)

# ── Hard labels of the ten 24-cell training splits ──────────────────
hard_X, hard_y = [], []
for seed in SEEDS:
    train_ds, _, _, ls = load_mit_stanford_paper(
        'data/mit_stanford', window_length=30, n_cells=30, n_test=6, seed=seed)
    X_all = train_ds.X
    y_all = train_ds.y.numpy().ravel()  # normalised hard labels
    hard_X.append(X_all)
    hard_y.append(y_all)
    print(f"  hard labels for seed {seed}: {len(X_all)} samples", flush=True)

X_mix = torch.cat(replay_X + hard_X, dim=0)
y_mix = np.concatenate(replay_y + hard_y)
print(f"mixed training samples: {len(X_mix)}")

# ── Continue training from the v3 student ───────────────────────────
student = make_model(config, device)
student.load_state_dict(torch.load(os.path.join('checkpoints', 'student_distill20.pt'),
                                   map_location=device))
mix_ds = SoftDataset(X_mix, y_mix)
loader, _ = prepare_dataloaders(mix_ds, config.batch_size, val_ratio=0.0, seed=42)
criterion = EvidentialRegressionLoss(coeff=2.0)
optimizer = torch.optim.Adam(student.parameters(), lr=3e-4, weight_decay=1e-3)

set_seed(42)
for epoch in range(1, 31):
    student.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        g, nu, al, be = student(X)
        loss = criterion(y, g, nu, al, be)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"  joint epoch {epoch}/30 loss={loss.item():.4f}", flush=True)

torch.save(student.state_dict(), os.path.join('checkpoints', 'student_joint24.pt'))
print("joint student saved")

# ── Evaluate: 24-cell protocol (primary) ────────────────────────────
student.eval()
_, _, _, ls42 = load_mit_stanford_paper(
    'data/mit_stanford', window_length=30, n_cells=48, n_test=6, seed=42)
per_seed = []
for seed in [42, 43, 44, 45, 46]:
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
    per_seed.append(np.mean(cell_rmse))
    print(f"  24cell seed {seed}: avg={np.mean(cell_rmse):.1f} "
          f"per-cell={[f'{c:.1f}' for c in cell_rmse]}", flush=True)

print(f"\n=== Joint-distilled student on 24-cell protocol ===")
print(f"  5-seed average: {np.mean(per_seed):.2f}  "
      f"(per-seed: {[f'{x:.1f}' for x in per_seed]})")
print("Reference: v3 zero-finetune 23.14 | v2 24.15 | 24-cell ensemble 26.16 | paper 16.49")
