"""Ablation study: compare LSTM-only vs SMoE-LSTM on synthetic data."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data_loader import generate_synthetic_data
from src.loss import EvidentialRegressionLoss
from src.train import set_seed, get_device
from src.evaluate import compute_all_metrics
from config import Config

class LSTMDirect(nn.Module):
    """LSTM -> FC -> Evidence (NO SMoE, just like the paper's extractor without MoE)."""
    def __init__(self, input_dim=16, hidden_dim=128, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.evidence = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        feats = h_n[-1]
        out = self.evidence(feats)
        gamma, nu, alpha, beta = out.chunk(4, dim=-1)
        gamma = gamma
        nu = F.softplus(nu) + 1e-6
        alpha = F.softplus(alpha) + 1.0 + 1e-6
        beta = F.softplus(beta) + 1e-6
        return gamma, nu, alpha, beta


class MLPBaseline(nn.Module):
    """Simple MLP (no LSTM, just last time step features)."""
    def __init__(self, input_dim=16, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x):
        # x: (B, L, D) -> use last time step only
        feats = x[:, -1, :]
        out = self.net(feats)
        gamma, nu, alpha, beta = out.chunk(4, dim=-1)
        gamma = gamma
        nu = F.softplus(nu) + 1e-6
        alpha = F.softplus(alpha) + 1.0 + 1e-6
        beta = F.softplus(beta) + 1e-6
        return gamma, nu, alpha, beta


def train_and_eval(model_cls, name, train_ds, test_ds, label_scaler, device, epochs=100):
    set_seed(42)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = model_cls().to(device)
    criterion = EvidentialRegressionLoss(coeff=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_val_loss = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            gamma, nu, alpha, beta = model(X)
            loss = criterion(y, gamma, nu, alpha, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                gamma, nu, alpha, beta = model(X)
                loss = criterion(y, gamma, nu, alpha, beta)
                val_loss += loss.item() * X.size(0)
        val_loss /= len(test_ds)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"  {name:12s} Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    all_y, all_preds = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            gamma, nu, alpha, beta = model(X)
            all_y.append(y.numpy())
            all_preds.append(gamma.cpu().numpy())
    y_true = np.concatenate(all_y).ravel()
    y_pred = np.concatenate(all_preds).ravel()

    y_true_orig = label_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_orig = label_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    metrics = compute_all_metrics(y_true_orig, y_pred_orig)

    print(f"  {name:12s}: RMSE={metrics['rmse']:.1f}, R2={metrics['r2']:.4f}, "
          f"Time={time.time()-t0:.1f}s")
    return metrics


print("=== Model Ablation Study (Synthetic Data) ===", flush=True)

train_ds, test_ds, _, label_scaler = generate_synthetic_data(
    n_batteries=30, cycles_per_battery=150, window_length=30
)
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)
device = get_device('cuda')

from src.model import SMoELSTM

results = {}
for cls, name in [
    (lambda: SMoELSTM(input_dim=16, hidden_dim=128, num_experts=10, top_k=4,
                       lstm_layers=2, dropout=0.1, evidence_hidden=64), "SMoE-LSTM"),
    (lambda: LSTMDirect(input_dim=16, hidden_dim=128, lstm_layers=2, dropout=0.1), "LSTM-only"),
    (lambda: MLPBaseline(input_dim=16, hidden_dim=128), "MLP (no LSTM)"),
]:
    m = train_and_eval(cls, name, train_ds, test_ds, label_scaler, device, epochs=100)
    results[name] = m

print("\n=== Summary ===")
print(f"{'Model':<15} {'RMSE':>8} {'R2':>8}")
print("-" * 33)
for name, m in results.items():
    print(f"{name:<15} {m['rmse']:>8.1f} {m['r2']:>8.4f}")
