#!/usr/bin/env python
"""SMoE-LSTM: Trustworthy Li-ion Battery RUL Prediction with Uncertainty Quantification.

Reproduces the experiments from:
  Zhao et al., "Towards trustworthy lithium-ion battery remaining useful
  life prediction with uncertainty quantification", J. Energy Storage 167 (2026).

Usage:
    # Quick test with synthetic data (no real dataset needed)
    python main.py --synthetic

    # Train on MIT-Stanford dataset
    python main.py --dataset mit --data_dir data/mit_stanford

    # Train on XJTU dataset
    python main.py --dataset xjtu --xjtu_batch batch1 --data_dir data/xjtu

    # Evaluate only (load checkpoint)
    python main.py --dataset mit --evaluate --checkpoint checkpoints/model.pt
"""

import os
import sys
import argparse
import json
import warnings
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.model import SMoELSTM
from src.loss import EvidentialRegressionLoss
from src.data_loader import (
    BatteryDataset,
    load_mit_stanford,
    load_xjtu,
    generate_synthetic_data,
)
from src.train import (
    set_seed,
    get_device,
    prepare_dataloaders,
    train_model,
)
from src.evaluate import (
    evaluate_model,
    print_evaluation_report,
    print_multi_trial_summary,
    multi_trial_summary,
    measure_inference_time,
)

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description="SMoE-LSTM: Trustworthy Battery RUL Prediction with UQ"
    )

    # Data
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing the pipeline')
    parser.add_argument('--dataset', type=str, default='mit', choices=['mit', 'xjtu'],
                        help='Dataset to use (default: mit)')
    parser.add_argument('--data_dir', type=str, default='data/mit_stanford',
                        help='Path to dataset root')
    parser.add_argument('--xjtu_batch', type=str, default='batch1',
                        choices=['batch1', 'batch2'],
                        help='XJTU batch to use (default: batch1)')

    # Model
    parser.add_argument('--window_length', type=int, default=30,
                        help='Sliding window length (default: 30)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--num_experts', type=int, default=10,
                        help='Number of SMoE experts (default: 10)')
    parser.add_argument('--top_k', type=int, default=4,
                        help='Top-K experts to activate (default: 4)')
    parser.add_argument('--lstm_layers', type=int, default=2,
                        help='Number of LSTM layers (default: 2)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay (default: 1e-3)')
    parser.add_argument('--evidential_coeff', type=float, default=2.0,
                        help='Evidential loss coefficient lambda (default: 2.0)')
    parser.add_argument('--num_trials', type=int, default=10,
                        help='Number of independent trials (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')

    # Misc
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only (skip training, requires --checkpoint)')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory for saving model checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--single_trial', action='store_true',
                        help='Run only 1 trial (for quick testing)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose training output')

    return parser.parse_args()


def build_config(args) -> Config:
    """Merge CLI args into Config."""
    cfg = Config()
    cfg.window_length = args.window_length
    cfg.hidden_dim = args.hidden_dim
    cfg.num_experts = args.num_experts
    cfg.top_k = args.top_k
    cfg.lstm_layers = args.lstm_layers
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.evidential_coeff = args.evidential_coeff
    cfg.num_trials = 1 if args.single_trial else args.num_trials
    cfg.random_seed = args.seed
    cfg.save_dir = args.save_dir
    cfg.device = args.device
    cfg.data_mit = args.data_dir if args.dataset == 'mit' else cfg.data_mit
    cfg.data_xjtu = args.data_dir if args.dataset == 'xjtu' else cfg.data_xjtu
    return cfg


def main():
    args = parse_args()
    config = build_config(args)
    device = get_device(config.device)

    print(f"{'='*60}")
    print(f"  SMoE-LSTM: Trustworthy Battery RUL Prediction with UQ")
    print(f"  Device: {device}")
    print(f"  Config: {config}")
    print(f"{'='*60}\n")

    # ── Load data ──────────────────────────────────────────────────
    if args.synthetic:
        print("[INFO] Generating synthetic data for pipeline testing...")
        train_dataset, test_dataset, feat_scaler, label_scaler = \
            generate_synthetic_data(
                n_batteries=30,
                cycles_per_battery=150,
                window_length=config.window_length,
            )
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples:  {len(test_dataset)}")
    elif args.dataset == 'mit':
        print(f"[INFO] Loading MIT-Stanford dataset from {config.data_mit}...")
        train_dataset, test_dataset, feat_scaler, label_scaler = \
            load_mit_stanford(config.data_mit, window_length=config.window_length)
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples:  {len(test_dataset)}")
    elif args.dataset == 'xjtu':
        print(f"[INFO] Loading XJTU {args.xjtu_batch} dataset from {config.data_xjtu}...")
        train_dataset, test_dataset, feat_scaler, label_scaler = \
            load_xjtu(config.data_xjtu, batch=args.xjtu_batch,
                      window_length=config.window_length)
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples:  {len(test_dataset)}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, drop_last=False)

    # ── Evaluate only? ─────────────────────────────────────────────
    if args.evaluate:
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            print("[ERROR] --evaluate requires a valid --checkpoint path")
            sys.exit(1)

        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        model = SMoELSTM(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            lstm_layers=config.lstm_layers,
            evidence_hidden=config.evidence_hidden,
        ).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                         weights_only=True))

        result = evaluate_model(model, test_loader, device, label_scaler)
        print_evaluation_report(result)

        # Inference time
        timing = measure_inference_time(model, test_loader, device)
        print(f"  Inference time: {timing['mean_per_sample']*1000:.2f} ± "
              f"{timing['std_per_sample']*1000:.2f} ms/sample")
        return

    # ── Multi-trial training ───────────────────────────────────────
    os.makedirs(config.save_dir, exist_ok=True)

    all_trial_results = []
    for trial in range(config.num_trials):
        print(f"\n{'─'*60}")
        print(f"  TRIAL {trial + 1} / {config.num_trials}")
        print(f"{'─'*60}")

        # Train
        model, history = train_model(
            config, train_dataset, device, trial=trial,
            verbose=args.verbose or args.single_trial,
        )

        # Save checkpoint
        ckpt_path = os.path.join(config.save_dir, f"model_trial{trial+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # Evaluate
        result = evaluate_model(model, test_loader, device, label_scaler)

        # Timing
        timing = measure_inference_time(model, test_loader, device)
        result['inference_time_ms'] = timing['mean_per_sample'] * 1000

        all_trial_results.append(result)

        if args.single_trial or args.synthetic:
            print_evaluation_report(result, trial + 1)

    # ── Multi-trial summary ────────────────────────────────────────
    if len(all_trial_results) > 1:
        summary = multi_trial_summary(all_trial_results)
        print_multi_trial_summary(summary)

        # Save summary
        summary_path = os.path.join(config.save_dir, 'results_summary.json')
        serialisable = {
            'config': {k: str(v) if not isinstance(v, (int, float, bool, type(None)))
                       else v for k, v in config.__dict__.items()},
            'summary': {k: {'mean': float(v['mean']), 'std': float(v['std'])}
                        for k, v in summary.items()},
            'timestamp': datetime.now().isoformat(),
        }
        with open(summary_path, 'w') as f:
            json.dump(serialisable, f, indent=2)
        print(f"Results saved to {summary_path}")

    # ── Compare with paper results ─────────────────────────────────
    if not args.synthetic and len(all_trial_results) > 0:
        print(f"\n{'='*60}")
        print(f"  Comparison with Paper Results")
        print(f"{'='*60}")
        paper_metrics = {
            'RMSE': 16.11,
            'MAE': 12.77,
            'MAPE': 7.42,
            'R2': 0.9821,
        }
        best_result = min(all_trial_results,
                          key=lambda r: r['metrics']['rmse'])
        for key, paper_val in paper_metrics.items():
            our_val = best_result['metrics'][key.lower()]
            print(f"  {key:8s}: Paper = {paper_val:.4f}, "
                  f"Ours = {our_val:.4f}")


if __name__ == '__main__':
    main()
