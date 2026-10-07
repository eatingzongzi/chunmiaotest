"""Compare MSE vs Evidential loss on synthetic data to isolate loss function issues."""
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

class LSTMSimple(nn.Module):
    """LSTM -> FC -> direct RUL prediction (MSE)."""
    def __init__(self, input_dim=16, hidden_dim=128, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
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
        return self.head(feats)


class LSTMEvidence(nn.Module):
    """LSTM -> FC -> NIG evidence (same as ablation LSTM-only)."""
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
        return gamma, F.softplus(nu)+1e-6, F.softplus(alpha)+1.0+1e-6, F.softplus(beta)+1e-6


def train_mse(model, train_ds, test_ds, device, epochs=100):
    set_seed(42)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_val_loss = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += criterion(pred, y).item() * X.size(0)
        val_loss /= len(test_ds)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"  MSE-LSTM   Epoch {epoch:3d}: val_loss(norm)={val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    all_y, all_preds = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred = model(X)
            all_y.append(y.numpy())
            all_preds.append(pred.cpu().numpy())
    y_true = np.concatenate(all_y).ravel()
    y_pred = np.concatenate(all_preds).ravel()

    return y_true, y_pred, time.time() - t0


print("=== Loss Function Comparison (Synthetic Data) ===", flush=True)

train_ds, test_ds, _, label_scaler = generate_synthetic_data(
    n_batteries=30, cycles_per_battery=150, window_length=30
)
device = get_device('cuda')

# 1. MSE loss (LSTM -> direct prediction)
print("\n--- MSE Loss ---")
y_true, y_pred_mse, t_mse = train_mse(LSTMSimple(), train_ds, test_ds, device, epochs=100)
y_true_orig = label_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
y_pred_mse_orig = label_scaler.inverse_transform(y_pred_mse.reshape(-1, 1)).ravel()
m_mse = compute_all_metrics(y_true_orig, y_pred_mse_orig)
print(f"  MSE-LSTM:  RMSE={m_mse['rmse']:.1f}, R2={m_mse['r2']:.4f}, Time={t_mse:.1f}s")

# 2. Evidential loss (same architecture)
print("\n--- Evidential Loss ---")
from src.model import SMoELSTM
# Use our LSTM-only evidence model from ablation
set_seed(42)
model_ev = LSTMEvidence().to(device)
criterion_ev = EvidentialRegressionLoss(coeff=2.0)
optimizer_ev = torch.optim.Adam(model_ev.parameters(), lr=1e-3, weight_decay=1e-3)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

best_val = float('inf')
best_state = None
t0 = time.time()

for epoch in range(1, 101):
    model_ev.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer_ev.zero_grad()
        gamma, nu, alpha, beta = model_ev(X)
        loss = criterion_ev(y, gamma, nu, alpha, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_ev.parameters(), 1.0)
        optimizer_ev.step()

    model_ev.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            gamma, nu, alpha, beta = model_ev(X)
            val_loss += criterion_ev(y, gamma, nu, alpha, beta).item() * X.size(0)
    val_loss /= len(test_ds)

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model_ev.state_dict().items()}

    if epoch % 20 == 0 or epoch == 1:
        print(f"  EVI-LSTM   Epoch {epoch:3d}: val_loss={val_loss:.4f}")

if best_state:
    model_ev.load_state_dict(best_state)

model_ev.eval()
all_y_ev, all_preds_ev = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        gamma, nu, alpha, beta = model_ev(X)
        all_y_ev.append(y.numpy())
        all_preds_ev.append(gamma.cpu().numpy())
y_true_ev = np.concatenate(all_y_ev).ravel()
y_pred_ev = np.concatenate(all_preds_ev).ravel()
y_true_ev_orig = label_scaler.inverse_transform(y_true_ev.reshape(-1, 1)).ravel()
y_pred_ev_orig = label_scaler.inverse_transform(y_pred_ev.reshape(-1, 1)).ravel()
m_ev = compute_all_metrics(y_true_ev_orig, y_pred_ev_orig)
print(f"  EVI-LSTM:  RMSE={m_ev['rmse']:.1f}, R2={m_ev['r2']:.4f}, Time={time.time()-t0:.1f}s")

# 3. Linear regression baseline
print("\n--- Linear Regression ---")
from sklearn.linear_model import LinearRegression
# Flatten the last time step for linear regression
X_test_last = test_ds.X[:, -1, :].numpy()
y_test = test_ds.y.numpy().ravel()
X_train_last = train_ds.X[:, -1, :].numpy()
y_train = train_ds.y.numpy().ravel()

lr = LinearRegression()
lr.fit(X_train_last, y_train)
y_pred_lr = lr.predict(X_test_last)
y_true_lr_orig = label_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
y_pred_lr_orig = label_scaler.inverse_transform(y_pred_lr.reshape(-1, 1)).ravel()
m_lr = compute_all_metrics(y_true_lr_orig, y_pred_lr_orig)
print(f"  LinearReg: RMSE={m_lr['rmse']:.1f}, R2={m_lr['r2']:.4f}")

print(f"\n=== Summary ===")
print(f"{'Model':<15} {'RMSE':>8} {'R2':>8}")
print("-" * 33)
print(f"{'MSE-LSTM':<15} {m_mse['rmse']:>8.1f} {m_mse['r2']:>8.4f}")
print(f"{'EVI-LSTM':<15} {m_ev['rmse']:>8.1f} {m_ev['r2']:>8.4f}")
print(f"{'LinearReg':<15} {m_lr['rmse']:>8.1f} {m_lr['r2']:>8.4f}")
print(f"\n  Synthetic data RUL range: [{label_scaler.data_min_[0]:.0f}, {label_scaler.data_max_[0]:.0f}]")
