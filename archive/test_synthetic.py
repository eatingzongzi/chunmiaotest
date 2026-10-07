"""Validate SMoE-LSTM can learn well on synthetic data with known patterns."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data_loader import generate_synthetic_data
from src.model import SMoELSTM
from src.train import train_model, get_device
from src.evaluate import evaluate_model, print_evaluation_report
from config import Config

print("=== Synthetic Data Validation ===", flush=True)

# Generate synthetic data with strong feature-RUL correlation
train_ds, test_ds, _, label_scaler = generate_synthetic_data(
    n_batteries=30, cycles_per_battery=150, window_length=30
)
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)

config = Config()
config.epochs = 100
config.batch_size = 64
config.learning_rate = 1e-3
config.early_stop_patience = 30
device = get_device('cuda')

# Train
t0 = time.time()
model, history = train_model(config, train_ds, device, trial=0, verbose=True)
print(f"Training: {time.time()-t0:.1f}s", flush=True)

# Evaluate
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
result = evaluate_model(model, test_loader, device, label_scaler)
print_evaluation_report(result, 1)

m = result['metrics']
print(f"Synthetic Result:", flush=True)
print(f"  RMSE: {m['rmse']:.2f}", flush=True)
print(f"  R2:   {m['r2']:.4f}", flush=True)
print(f"  If RMSE < 10 and R2 > 0.95: model can learn well -> issue is in real features", flush=True)
print(f"  If RMSE > 50 or R2 < 0.8: model has bugs -> need to fix architecture/loss", flush=True)
