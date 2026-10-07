#!/usr/bin/env python
"""Reproduce the paper's exact experimental protocol (Section 4.1, Table 5).

Differences vs. the original load_mit_stanford / main.py:
  1. Uses ONLY the ``2017-06-30`` batch.
  2. Randomly selects 30 cells -> 24 train / 6 test (#1..#6).
  3. Evaluates each test cell separately, then averages RMSE across cells.

Usage:
    python run_paper_repro.py --trials 1   # quick check
    python run_paper_repro.py --trials 10  # full protocol
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from config import Config
from src.data_loader import load_mit_stanford_paper
from src.train import train_model, get_device
from src.evaluate import evaluate_per_cell


def parse_args():
    p = argparse.ArgumentParser(description="Paper-aligned MIT-Stanford reproduction")
    p.add_argument('--trials', type=int, default=1, help='Number of trials (default: 1)')
    p.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
    p.add_argument('--window_length', type=int, default=30)
    p.add_argument('--n_cells', type=int, default=30)
    p.add_argument('--n_test', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--verbose', action='store_true', help='Print per-epoch training logs')
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    config = Config()
    config.epochs = args.epochs
    config.window_length = args.window_length

    # ── Load paper-aligned data ──────────────────────────────────────
    t0 = time.time()
    train_ds, test_cells, feat_scaler, label_scaler = load_mit_stanford_paper(
        'data/mit_stanford',
        window_length=args.window_length,
        n_cells=args.n_cells,
        n_test=args.n_test,
        seed=args.seed,
    )
    print(f"Data load: {time.time() - t0:.1f}s")
    print(f"Train samples: {len(train_ds)}")
    print(f"Test cells ({len(test_cells)}):")
    for i, c in enumerate(test_cells):
        print(f"  #{i + 1} cell_id={c['cell_id']:2d} eol={c['eol']:4d} "
              f"samples={c['X'].shape[0]:4d}")

    # ── Multiple trials ──────────────────────────────────────────────
    per_trial = []  # list (per trial) of list (per cell) of metrics dicts
    for trial in range(args.trials):
        t0 = time.time()
        print(f"\n{'-' * 50}\nTRIAL {trial + 1}/{args.trials}\n{'-' * 50}")
        model, _ = train_model(config, train_ds, device, trial=trial,
                               verbose=args.verbose)
        per_cell, avg = evaluate_per_cell(model, test_cells, device, label_scaler)
        per_trial.append(per_cell)
        print(f"  [trial {trial + 1}] avg RMSE={avg['rmse']:.2f} "
              f"MAE={avg['mae']:.2f} MAPE={avg['mape']:.2f}% R2={avg['r2']:.4f} "
              f"({time.time() - t0:.0f}s)")

    # ── Aggregate per-cell over trials ───────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Per-cell RMSE (mean over {args.trials} trials)")
    print(f"{'=' * 60}")
    cell_rmses = []
    for i in range(len(test_cells)):
        rmses = [per_trial[t][i]['metrics']['rmse'] for t in range(args.trials)]
        mean_rmse = float(np.mean(rmses))
        std_rmse = float(np.std(rmses))
        cell_rmses.append(mean_rmse)
        print(f"  #{i + 1} (cell_id={test_cells[i]['cell_id']:2d}, "
              f"eol={test_cells[i]['eol']:4d}): RMSE={mean_rmse:6.2f} +/- {std_rmse:5.2f}")

    print(f"\n  Average RMSE across cells: {np.mean(cell_rmses):.2f}")
    print(f"  Paper (Table 5) average:   16.11")
    print(f"  Paper per-cell: #1=10.27 #2=15.19 #3=22.69 #4=18.99 #5=20.87 #6=8.67")


if __name__ == '__main__':
    main()
