"""Optimized training: LSTM-only (no SMoE), 200 epochs, MSE loss, batch 1 only."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import h5py

from src.train import set_seed, get_device
from src.evaluate import compute_all_metrics, compute_spearman_correlation

# ═══════════════════════════════════════════════════════════════════════════
# Simple LSTM model
# ═══════════════════════════════════════════════════════════════════════════
class LSTMRUL(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


# ═══════════════════════════════════════════════════════════════════════════
# Load batch 1 only
# ═══════════════════════════════════════════════════════════════════════════
def load_batch1(data_dir, window_length=30):
    from src.data_loader import _extract_cell_features, sliding_window_sampling, BatteryDataset

    fpath = os.path.join(data_dir, '2017-05-12_batchdata_updated_struct_errorcorrect.mat')
    f = h5py.File(fpath, 'r')
    batch = f['batch']

    all_cell_data = []
    for i in range(batch['summary'].shape[0]):
        try:
            summ_ref = batch['summary'][i, 0]
            cyc_ref = batch['cycles'][i, 0]
            cl_ref = batch['cycle_life'][i, 0]
            summ = f[summ_ref]
            cyc = f[cyc_ref]
            cycle_life = float(f[cl_ref][()].flatten()[0])

            features, cycle_nums = _extract_cell_features(f, summ, cyc, cycle_life)
            if len(features) == 0:
                continue

            eol = int(cycle_life)
            if eol <= cycle_nums[0]:
                continue

            valid = cycle_nums <= eol
            if np.sum(valid) < 30:
                if len(cycle_nums) >= 30:
                    eol = int(cycle_nums[-1]) + 1
                else:
                    continue
            else:
                features = features[valid]
                cycle_nums = cycle_nums[valid]

            all_cell_data.append({'features': features, 'cycle_nums': cycle_nums, 'eol': eol})
        except Exception as e:
            print(f"  Cell {i} error: {e}")
            continue
    f.close()

    # Split: 80/20 cell-based
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_cell_data))
    n_train = int(0.8 * len(all_cell_data))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    def build(cell_indices):
        all_X, all_y = [], []
        for idx in cell_indices:
            cell = all_cell_data[idx]
            X, y = sliding_window_sampling(cell['features'], cell['cycle_nums'],
                                           cell['eol'], window_length)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        return np.concatenate(all_X), np.concatenate(all_y)

    X_train, y_train = build(train_idx)
    X_test, y_test = build(test_idx)

    # Normalize
    N, L, D = X_train.shape
    feat_scaler = MinMaxScaler().fit(X_train.reshape(-1, D))
    X_train = feat_scaler.transform(X_train.reshape(-1, D)).reshape(N, L, D)
    X_test = feat_scaler.transform(X_test.reshape(-1, D)).reshape(len(X_test), L, D)

    label_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))
    y_train = label_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_test = label_scaler.transform(y_test.reshape(-1, 1)).ravel()

    print(f"  Batch1: {len(all_cell_data)} cells, Train: {len(train_idx)}/{len(X_train)} samples, "
          f"Test: {len(test_idx)}/{len(X_test)} samples")
    return BatteryDataset(X_train, y_train), BatteryDataset(X_test, y_test), label_scaler, all_cell_data


print("=== LSTM-Only Training on Batch1 (200 epochs) ===", flush=True)
device = get_device('cuda')
data_dir = 'data/mit_stanford'

t0 = time.time()
train_ds, test_ds, label_scaler, cell_data = load_batch1(data_dir, window_length=30)
print(f"Load time: {time.time()-t0:.1f}s", flush=True)

# Train
set_seed(42)
model = LSTMRUL().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

best_val = float('inf')
best_state = None

t0 = time.time()
for epoch in range(1, 201):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            val_loss += criterion(model(X), y).item() * X.size(0)
    val_loss /= len(test_ds)

    if val_loss < best_val - 1e-6:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}: val_loss(norm)={val_loss:.6f}, best={best_val:.6f}")

if best_state:
    model.load_state_dict(best_state)

print(f"Training time: {time.time()-t0:.1f}s", flush=True)

# Evaluate
model.eval()
all_y, all_preds = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        all_y.append(y.numpy())
        all_preds.append(model(X).cpu().numpy())

y_true = np.concatenate(all_y).ravel()
y_pred = np.concatenate(all_preds).ravel()
y_true_orig = label_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
y_pred_orig = label_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

m = compute_all_metrics(y_true_orig, y_pred_orig)
print(f"\n=== Results ===")
print(f"  RMSE: {m['rmse']:.1f}")
print(f"  MAE:  {m['mae']:.1f}")
print(f"  MAPE: {m['mape']:.1f}%")
print(f"  R2:   {m['r2']:.4f}")
print(f"  RUL range: [{label_scaler.data_min_[0]:.0f}, {label_scaler.data_max_[0]:.0f}]")

# Compare with linear regression
from sklearn.linear_model import LinearRegression
X_last = test_ds.X[:, -1, :].numpy()
y_test_np = test_ds.y.numpy().ravel()
X_train_last = train_ds.X[:, -1, :].numpy()
y_train_np = train_ds.y.numpy().ravel()

lr = LinearRegression().fit(X_train_last, y_train_np)
y_pred_lr = lr.predict(X_last)
y_pred_lr_orig = label_scaler.inverse_transform(y_pred_lr.reshape(-1, 1)).ravel()
m_lr = compute_all_metrics(y_true_orig, y_pred_lr_orig)
print(f"\n  LinearReg baseline: RMSE={m_lr['rmse']:.1f}, R2={m_lr['r2']:.4f}")
