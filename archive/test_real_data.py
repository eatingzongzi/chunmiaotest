"""Full training on MIT-Stanford dataset (100 epochs, GPU)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data_loader import load_mit_stanford
from src.model import SMoELSTM
from src.train import train_model, get_device
from src.evaluate import evaluate_model, print_evaluation_report
from config import Config

print("=== SMoE-LSTM Training on MIT-Stanford (100 epochs) ===", flush=True)

# ── Load data ────────────────────────────────────────────────
data_dir = 'data/mit_stanford'
t0 = time.time()
train_ds, test_ds, feat_scaler, label_scaler = load_mit_stanford(data_dir, window_length=30)
print(f"Data loading: {time.time()-t0:.1f}s, Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)

# ── Train ────────────────────────────────────────────────────
config = Config()
config.epochs = 50
config.batch_size = 64
device = get_device('cuda')

t0 = time.time()
model, history = train_model(config, train_ds, device, trial=0, verbose=True)
print(f"Training: {time.time()-t0:.1f}s", flush=True)

# ── Evaluate ─────────────────────────────────────────────────
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
result = evaluate_model(model, test_loader, device, label_scaler)
print_evaluation_report(result, 1)

m = result['metrics']
print(f"Paper Target vs Ours:", flush=True)
print(f"  RMSE : Paper=16.11,  Ours={m['rmse']:.2f}", flush=True)
print(f"  MAE  : Paper=12.77,  Ours={m['mae']:.2f}", flush=True)
print(f"  MAPE : Paper=7.42%,  Ours={m['mape']:.2f}%", flush=True)
print(f"  R2   : Paper=0.9821, Ours={m['r2']:.4f}", flush=True)
print(f"  Spearman(epi): Paper≈0.49, Ours={result['spearman_epistemic']:.4f}", flush=True)

os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/model_mit_trial1.pt')
print("Model saved to checkpoints/model_mit_trial1.pt", flush=True)
