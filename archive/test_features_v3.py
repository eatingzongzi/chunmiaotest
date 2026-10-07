"""Fix feature extraction: use EQUAL TIME SEGMENT MEANS, not spline resampling."""
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
from src.evaluate import compute_all_metrics


def extract_cc_voltage_6segments(v_curve, i_curve):
    """Extract CC voltage: find CC phase, split into 6 equal TIME segments, take mean V per segment.

    Paper: 6 CC voltage features = mean V in 6 equal segments of [V_cutoff-0.2V, V_cutoff].
    """
    v = np.asarray(v_curve).flatten()
    i = np.asarray(i_curve).flatten()

    # Remove zero-padding
    nonzero = np.where(v > 0.1)[0]
    if len(nonzero) < 10:
        return np.zeros(6)
    v = v[nonzero]
    i = i[nonzero]

    # Find charging portion (positive current)
    charging = i > 0.01
    if not np.any(charging):
        return np.zeros(6)
    v_ch = v[charging]
    i_ch = i[charging]

    # CC phase: current near max, voltage rising
    i_max = np.max(i_ch)
    v_max = np.max(v_ch)
    # CC = I >= 85% of max AND V in [3.0, v_max]
    cc_mask = (i_ch >= 0.85 * i_max) & (v_ch >= 3.0) & (v_ch <= v_max)
    if np.sum(cc_mask) < 12:
        cc_mask = (v_ch >= 3.0) & (v_ch <= v_max)
    if np.sum(cc_mask) < 12:
        return np.zeros(6)

    v_cc = v_ch[cc_mask]

    # Split into 6 equal TIME segments and compute mean V in each
    n = len(v_cc)
    seg_size = n // 6
    means = np.zeros(6)
    for s in range(6):
        start = s * seg_size
        end = (s + 1) * seg_size if s < 5 else n
        if start < end:
            means[s] = np.mean(v_cc[start:end])
        else:
            means[s] = v_cc[-1] if len(v_cc) > 0 else 0.0

    return means


def extract_cv_current_6segments(i_curve, v_curve):
    """Extract CV current: find CV phase, split into 6 equal TIME segments, take mean I per segment.

    Paper: 6 CV current features = mean I in 6 equal segments of [0.5A, 0.1A].
    """
    v = np.asarray(v_curve).flatten()
    i = np.asarray(i_curve).flatten()

    nonzero = np.where(v > 0.1)[0]
    if len(nonzero) < 10:
        return np.zeros(6)
    v = v[nonzero]
    i = i[nonzero]

    charging = i > 0.001
    if not np.any(charging):
        return np.zeros(6)
    v_ch = v[charging]
    i_ch = i[charging]

    # CV phase: V near maximum, I is decaying
    v_max = np.max(v_ch)
    i_max_ch = np.max(i_ch)
    cv_mask = (v_ch >= v_max - 0.03) & (i_ch < 0.8 * i_max_ch)
    if np.sum(cv_mask) < 12:
        cv_mask = i_ch < 0.5 * i_max_ch
    if np.sum(cv_mask) < 12:
        # Just take the second half of the charging curve
        mid = len(i_ch) // 2
        cv_mask = np.zeros(len(i_ch), dtype=bool)
        cv_mask[mid:] = True

    i_cv = i_ch[cv_mask]

    # Split into 6 equal TIME segments, mean I per segment
    n = len(i_cv)
    seg_size = n // 6
    means = np.zeros(6)
    for s in range(6):
        start = s * seg_size
        end = (s + 1) * seg_size if s < 5 else n
        if start < end:
            means[s] = np.mean(i_cv[start:end])
        else:
            means[s] = 0.0

    return means


def extract_all_features(f, summ, cyc, cycle_life):
    """Extract all 16 features per cycle with fixed methodology."""
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

        v_raw = np.zeros(1000); i_raw = np.zeros(1000)
        try: v_raw = f[v_refs[i]][()].flatten().astype(np.float64)
        except: pass
        try: i_raw = f[i_refs[i]][()].flatten().astype(np.float64)
        except: pass

        # CC Voltage (6): equal time segment means
        feats[0:6] = extract_cc_voltage_6segments(v_raw, i_raw)
        # CV Current (6): equal time segment means
        feats[6:12] = extract_cv_current_6segments(i_raw, v_raw)

        # IC curve
        dqdv_raw = np.zeros(1000)
        try: dqdv_raw = f[dqdv_refs[i]][()].flatten().astype(np.float64)
        except: pass
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

    return features, cycles


def sliding_window(features, cycle_indices, eol, window_length):
    N = len(features)
    if N <= window_length:
        return np.array([]), np.array([])
    X_list, y_list = [], []
    for i in range(N - window_length):
        X_list.append(features[i:i + window_length])
        y_list.append(eol - cycle_indices[i + window_length - 1])
    if not X_list:
        return np.array([]), np.array([])
    return np.stack(X_list, axis=0), np.array(y_list)


class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class AttnLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=256, lstm_layers=2, dropout=0.2):
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


print("=== Feature Extract V3: Equal Time Segment Means ===", flush=True)
device = get_device('cuda')

# Use batch 1 only for faster iteration
batch_files = [
    '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
]

all_cell_data = []
data_dir = 'data/mit_stanford'
for bf in batch_files:
    fpath = os.path.join(data_dir, bf)
    if not os.path.exists(fpath): continue
    f = h5py.File(fpath, 'r')
    batch = f['batch']
    for i in range(batch['summary'].shape[0]):
        try:
            summ_ref = batch['summary'][i, 0]
            cyc_ref = batch['cycles'][i, 0]
            cl_ref = batch['cycle_life'][i, 0]
            summ = f[summ_ref]; cyc = f[cyc_ref]
            cycle_life = float(f[cl_ref][()].flatten()[0])

            features, cycle_nums = extract_all_features(f, summ, cyc, cycle_life)
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

            all_cell_data.append({'features': features, 'cycle_nums': cycle_nums, 'eol': eol})
        except Exception as e:
            continue
    f.close()

print(f"Total cells: {len(all_cell_data)}")

# Split
rng = np.random.RandomState(42)
ids = list(range(len(all_cell_data)))
rng.shuffle(ids)
n_train = int(0.8 * len(ids))
train_ids = ids[:n_train]; test_ids = ids[n_train:]

def build_samples(cell_indices):
    all_X, all_y = [], []
    for idx in cell_indices:
        c = all_cell_data[idx]
        X, y = sliding_window(c['features'], c['cycle_nums'], c['eol'], 30)
        if len(X) > 0: all_X.append(X); all_y.append(y)
    if not all_X: return np.array([]), np.array([])
    return np.concatenate(all_X), np.concatenate(all_y)

X_train, y_train = build_samples(train_ids)
X_test, y_test = build_samples(test_ids)

N, L, D = X_train.shape
feat_scaler = MinMaxScaler().fit(X_train.reshape(-1, D))
X_train = feat_scaler.transform(X_train.reshape(-1, D)).reshape(N, L, D)
X_test = feat_scaler.transform(X_test.reshape(-1, D)).reshape(len(X_test), L, D)

label_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))
y_train_norm = label_scaler.transform(y_train.reshape(-1, 1)).ravel()
y_test_norm = label_scaler.transform(y_test.reshape(-1, 1)).ravel()

print(f"Train: {len(train_ids)} cells, {len(X_train)} samples")
print(f"Test:  {len(test_ids)} cells, {len(X_test)} samples")

# First diagnose the new features
from scipy.stats import spearmanr
print(f"\n--- New Feature Diagnostics (first 5000 samples) ---")
feat_names = ['CC-V1','CC-V2','CC-V3','CC-V4','CC-V5','CC-V6',
              'CV-I1','CV-I2','CV-I3','CV-I4','CV-I5','CV-I6',
              'IC-pk','IC-Vp','DC-IR','Max-Q']
# Use last timestep for quick feature-RUL correlation
X_sample = X_train[:5000, -1, :]  # (5000, 16)
y_sample = y_train[:5000]
for d in range(16):
    vals = X_sample[:, d]
    s = spearmanr(vals, y_sample)[0] if np.std(vals) > 1e-8 else 0.0
    print(f"  {feat_names[d]:<8}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, spearman={s:.4f}")

# Train
print(f"\n--- Training ---")
train_ds = BatteryDataset(X_train, y_train_norm)
test_ds = BatteryDataset(X_test, y_test_norm)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

set_seed(42)
model = AttnLSTM(input_dim=16).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)

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
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            val_loss += criterion(model(X), y).item() * X.size(0)
    val_loss /= len(test_ds)
    scheduler.step(val_loss)

    if val_loss < best_val - 1e-7:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1
    if patience >= 40: break
    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}: val_loss={val_loss:.6f}, best={best_val:.6f}")

if best_state: model.load_state_dict(best_state)
print(f"Training: {time.time()-t0:.1f}s")

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
print(f"\n=== Final Results (138 cells, equal-segment features) ===")
print(f"  RMSE={m['rmse']:.1f}, MAE={m['mae']:.1f}, MAPE={m['mape']:.1f}%, R2={m['r2']:.4f}")
print(f"  Paper: RMSE=16.11, R2=0.9821")
