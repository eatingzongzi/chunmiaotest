"""Test hypothesis: does paper evaluate only the LAST window per test cell?"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import CubicSpline
import h5py

from src.train import set_seed, get_device
from src.evaluate import compute_all_metrics


class AttnLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=256, lstm_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.001)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = F.softmax(self.attn(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        context = self.bn(context)
        return self.head(context)


def extract_features_v2(f, summ, cyc, cycle_life):
    """Same improved features as before, using top 8 features."""
    n_cycles = int(summ['cycle'].shape[1])
    ir_all = summ['IR'][()].flatten()
    qd_all = summ['QDischarge'][()].flatten()
    cycle_nums = summ['cycle'][()].flatten().astype(np.int32)
    v_refs = cyc['V'][()].flatten()
    i_refs = cyc['I'][()].flatten()
    dqdv_refs = cyc['discharge_dQdV'][()].flatten()

    features_list = []
    for i in range(n_cycles):
        feats = np.zeros(16, dtype=np.float32)
        v_raw = np.zeros(1000); i_raw = np.zeros(1000); dqdv_raw = np.zeros(1000)
        try: v_raw = f[v_refs[i]][()].flatten().astype(np.float64)
        except: pass
        try: i_raw = f[i_refs[i]][()].flatten().astype(np.float64)
        except: pass
        try: dqdv_raw = f[dqdv_refs[i]][()].flatten().astype(np.float64)
        except: pass

        v = v_raw[v_raw > 0.1]
        i_charge_mask = i_raw[v_raw > 0.1] > 0.01
        if np.sum(i_charge_mask) >= 5:
            v_charge = v[i_charge_mask]
            i_charge = i_raw[v_raw > 0.1][i_charge_mask]
            i_max = np.max(i_charge)
            cc_mask = (i_charge >= 0.85 * i_max) & (v_charge >= 3.0) & (v_charge <= 3.65)
            if np.sum(cc_mask) >= 5:
                v_cc = v_charge[cc_mask]
                x = np.linspace(0, 1, len(v_cc))
                try: feats[0:6] = CubicSpline(x, v_cc)(np.linspace(0, 1, 6))
                except: feats[0:6] = np.interp(np.linspace(0, 1, 6), x, v_cc)

            v_max = np.max(v_charge)
            cv_mask = (v_charge >= v_max - 0.03) & (i_charge < 0.7 * i_max)
            if np.sum(cv_mask) >= 5:
                i_cv = i_charge[cv_mask]
                x = np.linspace(0, 1, len(i_cv))
                try: feats[6:12] = CubicSpline(x, i_cv)(np.linspace(0, 1, 6))
                except: feats[6:12] = np.interp(np.linspace(0, 1, 6), x, i_cv)

        dqdv = dqdv_raw[np.abs(dqdv_raw) > 1e-12]
        if len(dqdv) >= 10:
            peak_idx = np.argmin(dqdv)
            feats[12] = float(dqdv[peak_idx])
            feats[13] = float(peak_idx) / len(dqdv)

        feats[14] = float(ir_all[i]) if i < len(ir_all) else 0.0
        feats[15] = float(qd_all[i]) if i < len(qd_all) else 0.0
        features_list.append(feats)

    features = np.array(features_list, dtype=np.float32)
    cycles = np.array(cycle_nums, dtype=np.int32)

    # Differential features
    if len(features) >= 15:
        baseline = features[5:15].mean(axis=0, keepdims=True)
        ir_vals = features[:, 14].copy()
        features = features - baseline
        features[:, 14] = ir_vals

    # Top 8 features: CC-V4,5,6 + CV-I1,5,6 + IC-peak + Max-Q
    selected = [3, 4, 5, 6, 10, 11, 12, 15]
    features = features[:, selected]

    return features, cycles


class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def sliding_window(features, cycle_indices, eol, window_length):
    N = len(features)
    if N <= window_length:
        return np.array([]), np.array([]), np.array([])
    X_list, y_list, cell_cycle_list = [], [], []
    for i in range(N - window_length):
        window = features[i:i + window_length]
        current_cycle = cycle_indices[i + window_length - 1]
        rul = eol - current_cycle
        X_list.append(window)
        y_list.append(rul)
        cell_cycle_list.append(current_cycle)
    if not X_list:
        return np.array([]), np.array([]), np.array([])
    return np.stack(X_list, axis=0), np.array(y_list), np.array(cell_cycle_list)


print("=== Evaluation Method Comparison ===", flush=True)
device = get_device('cuda')

# Load data
fpath = 'data/mit_stanford/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
f = h5py.File(fpath, 'r')
batch = f['batch']

cell_data_by_idx = {}
for i in range(batch['summary'].shape[0]):
    try:
        summ_ref = batch['summary'][i, 0]
        cyc_ref = batch['cycles'][i, 0]
        cl_ref = batch['cycle_life'][i, 0]
        summ = f[summ_ref]; cyc = f[cyc_ref]
        cycle_life = float(f[cl_ref][()].flatten()[0])
        features, cycle_nums = extract_features_v2(f, summ, cyc, cycle_life)
        if len(features) == 0: continue
        eol = int(cycle_life)
        if eol <= cycle_nums[0]: continue
        valid = cycle_nums <= eol
        if np.sum(valid) < 30: continue
        features = features[valid]; cycle_nums = cycle_nums[valid]
        # Per-cell z-score
        feat_mean = features.mean(axis=0, keepdims=True)
        feat_std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - feat_mean) / feat_std
        cell_data_by_idx[i] = {'features': features, 'cycles': cycle_nums, 'eol': eol}
    except: continue
f.close()

# Split
rng = np.random.RandomState(42)
ids = list(cell_data_by_idx.keys())
rng.shuffle(ids)
n_train = int(0.8 * len(ids))
train_ids = ids[:n_train]; test_ids = ids[n_train:]

# Build training data
X_train_list, y_train_list = [], []
for cid in train_ids:
    c = cell_data_by_idx[cid]
    X, y, _ = sliding_window(c['features'], c['cycles'], c['eol'], 30)
    if len(X) > 0: X_train_list.append(X); y_train_list.append(y)
X_train = np.concatenate(X_train_list); y_train = np.concatenate(y_train_list)

# Normalize
N, L, D = X_train.shape
feat_scaler = MinMaxScaler().fit(X_train.reshape(-1, D))
X_train = feat_scaler.transform(X_train.reshape(-1, D)).reshape(N, L, D)
label_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))
y_train_norm = label_scaler.transform(y_train.reshape(-1, 1)).ravel()

# Build test data PER CELL (keep cell structure for per-cell eval)
test_cell_data = []
for cid in test_ids:
    c = cell_data_by_idx[cid]
    X, y, cycles = sliding_window(c['features'], c['cycles'], c['eol'], 30)
    if len(X) > 0:
        X_norm = feat_scaler.transform(X.reshape(-1, D)).reshape(len(X), L, D)
        test_cell_data.append({'X': X_norm, 'y': y, 'eol': c['eol'], 'cid': cid})

print(f"Train: {len(train_ids)} cells, {len(X_train)} samples")
print(f"Test:  {len(test_ids)} cells, {sum(len(c['y']) for c in test_cell_data)} samples total")
for c in test_cell_data:
    print(f"  Cell {c['cid']}: {len(c['y'])} samples, RUL=[{c['y'].min():.0f}, {c['y'].max():.0f}], EOL={c['eol']}")

# Train model
train_ds = BatteryDataset(X_train, y_train_norm)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)

set_seed(42)
model = AttnLSTM(input_dim=8).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)

# Simple validation on a held-out subset of train cells
val_ids = train_ids[-5:]
val_X_list, val_y_list = [], []
for cid in val_ids:
    c = cell_data_by_idx[cid]
    X, y, _ = sliding_window(c['features'], c['cycles'], c['eol'], 30)
    if len(X) > 0:
        X_norm = feat_scaler.transform(X.reshape(-1, D)).reshape(len(X), L, D)
        val_X_list.append(X_norm); val_y_list.append(y)
val_X = np.concatenate(val_X_list); val_y = np.concatenate(val_y_list)
val_y_norm = label_scaler.transform(val_y.reshape(-1, 1)).ravel()
val_ds = BatteryDataset(val_X, val_y_norm)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

best_val = float('inf')
best_state = None
patience = 0
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

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            val_loss += criterion(model(X), y).item() * X.size(0)
    val_loss /= len(val_ds)
    scheduler.step(val_loss)

    if val_loss < best_val - 1e-7:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1
    if patience >= 40: break

if best_state: model.load_state_dict(best_state)
print(f"Training: {time.time()-t0:.1f}s, best val_loss={best_val:.6f}")

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation: TWO methods
# ═══════════════════════════════════════════════════════════════════════════
model.eval()

# Method A: ALL sliding window samples (standard per-sample evaluation)
all_y_true, all_y_pred = [], []
for c in test_cell_data:
    X_tensor = torch.tensor(c['X'], dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().ravel()
    all_y_true.append(c['y'])
    all_y_pred.append(preds)

y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)
y_true_all_orig = label_scaler.inverse_transform(y_true_all.reshape(-1, 1)).ravel()
y_pred_all_orig = label_scaler.inverse_transform(y_pred_all.reshape(-1, 1)).ravel()
m_all = compute_all_metrics(y_true_all_orig, y_pred_all_orig)

# Method B: LAST window only per test cell (cell-level evaluation)
y_true_last = np.array([c['y'][-1] for c in test_cell_data])
y_pred_last = np.array([all_y_pred[i][-1] for i in range(len(test_cell_data))])
y_true_last_orig = label_scaler.inverse_transform(y_true_last.reshape(-1, 1)).ravel()
y_pred_last_orig = label_scaler.inverse_transform(y_pred_last.reshape(-1, 1)).ravel()
m_last = compute_all_metrics(y_true_last_orig, y_pred_last_orig)

# Method C: Mean prediction over last 10 windows per cell
y_true_last10 = []
y_pred_last10 = []
for i, c in enumerate(test_cell_data):
    n_last = min(10, len(c['y']))
    y_true_last10.append(c['y'][-n_last:].mean())
    y_pred_last10.append(all_y_pred[i][-n_last:].mean())
y_true_last10_orig = label_scaler.inverse_transform(np.array(y_true_last10).reshape(-1, 1)).ravel()
y_pred_last10_orig = label_scaler.inverse_transform(np.array(y_pred_last10).reshape(-1, 1)).ravel()
m_last10 = compute_all_metrics(y_true_last10_orig, y_pred_last10_orig)

print(f"\n=== Results ===")
print(f"Method A (ALL {len(y_true_all)} samples): RMSE={m_all['rmse']:.1f}, R2={m_all['r2']:.4f}")
print(f"Method B (LAST window, {len(y_true_last)} samples): RMSE={m_last['rmse']:.1f}, R2={m_last['r2']:.4f}")
print(f"Method C (Avg last 10, {len(y_true_last10)} samples): RMSE={m_last10['rmse']:.1f}, R2={m_last10['r2']:.4f}")

# Per-cell breakdown
print(f"\nPer-cell RUL predictions (Method B - last window):")
for i, c in enumerate(test_cell_data):
    true_rul = y_true_last_orig[i]
    pred_rul = y_pred_last_orig[i]
    print(f"  Cell {c['cid']} (EOL={c['eol']}): true RUL={true_rul:.0f}, pred RUL={pred_rul:.0f}, error={abs(true_rul-pred_rul):.0f}")

print(f"\nPaper target: RMSE=16.11, R2=0.9821")
print(f"If Method B/C is close to paper, the evaluation method matters.")
print(f"If not, the features/model still need improvement.")
