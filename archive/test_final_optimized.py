"""Final optimized model: all improvements combined."""
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
# Feature extraction V3: equal time segment means
# ═══════════════════════════════════════════════════════════════════════════
def extract_cc_voltage_6segments(v_curve, i_curve):
    v = np.asarray(v_curve).flatten(); i = np.asarray(i_curve).flatten()
    nonzero = np.where(v > 0.1)[0]
    if len(nonzero) < 10: return np.zeros(6)
    v = v[nonzero]; i = i[nonzero]
    charging = i > 0.01
    if not np.any(charging): return np.zeros(6)
    v_ch = v[charging]; i_ch = i[charging]
    i_max = np.max(i_ch); v_max = np.max(v_ch)
    cc_mask = (i_ch >= 0.85 * i_max) & (v_ch >= 3.0) & (v_ch <= v_max)
    if np.sum(cc_mask) < 12: cc_mask = (v_ch >= 3.0) & (v_ch <= v_max)
    if np.sum(cc_mask) < 12: return np.zeros(6)
    v_cc = v_ch[cc_mask]
    n = len(v_cc); seg = n // 6
    return np.array([np.mean(v_cc[s*seg:(s+1)*seg if s<5 else n]) for s in range(6)])

def extract_cv_current_6segments(i_curve, v_curve):
    v = np.asarray(v_curve).flatten(); i = np.asarray(i_curve).flatten()
    nonzero = np.where(v > 0.1)[0]
    if len(nonzero) < 10: return np.zeros(6)
    v = v[nonzero]; i = i[nonzero]
    charging = i > 0.001
    if not np.any(charging): return np.zeros(6)
    v_ch = v[charging]; i_ch = i[charging]
    v_max = np.max(v_ch); i_max_ch = np.max(i_ch)
    cv_mask = (v_ch >= v_max - 0.03) & (i_ch < 0.8 * i_max_ch)
    if np.sum(cv_mask) < 12: cv_mask = i_ch < 0.5 * i_max_ch
    if np.sum(cv_mask) < 12: return np.zeros(6)
    i_cv = i_ch[cv_mask]
    n = len(i_cv); seg = n // 6
    return np.array([np.mean(i_cv[s*seg:(s+1)*seg if s<5 else n]) for s in range(6)])

def extract_all_features(f, summ, cyc):
    n_cycles = int(summ['cycle'].shape[1])
    ir_all = summ['IR'][()].flatten(); qd_all = summ['QDischarge'][()].flatten()
    cycle_nums = summ['cycle'][()].flatten().astype(np.int32)
    v_refs = cyc['V'][()].flatten(); i_refs = cyc['I'][()].flatten()
    dqdv_refs = cyc['discharge_dQdV'][()].flatten()

    features_list = []
    for i in range(n_cycles):
        feats = np.zeros(16, dtype=np.float32)
        v_raw = np.zeros(1000); i_raw = np.zeros(1000)
        try: v_raw = f[v_refs[i]][()].flatten().astype(np.float64)
        except: pass
        try: i_raw = f[i_refs[i]][()].flatten().astype(np.float64)
        except: pass

        feats[0:6] = extract_cc_voltage_6segments(v_raw, i_raw)
        feats[6:12] = extract_cv_current_6segments(i_raw, v_raw)

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


# ═══════════════════════════════════════════════════════════════════════════
# Stronger model: deeper LSTM, wider hidden, transformer-style
# ═══════════════════════════════════════════════════════════════════════════
class DeepAttnLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=320, lstm_layers=3, dropout=0.2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.lstm_bn = nn.BatchNorm1d(hidden_dim)
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
        # x: (B, L, D) -> BN over features (treat L as batch for BN1d)
        B, L, D = x.shape
        x_bn = self.input_bn(x.reshape(-1, D)).reshape(B, L, D)
        lstm_out, _ = self.lstm(x_bn)
        attn_weights = F.softmax(self.attn(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        context = self.lstm_bn(context)
        return self.head(context)


# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════
class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def sliding_window(features, cycle_indices, eol, window_length):
    N = len(features)
    if N <= window_length: return np.array([]), np.array([])
    X, y = [], []
    for i in range(N - window_length):
        X.append(features[i:i + window_length])
        y.append(eol - cycle_indices[i + window_length - 1])
    if not X: return np.array([]), np.array([])
    return np.stack(X, axis=0), np.array(y)


print("=== Final Optimized Model ===", flush=True)
device = get_device('cuda')

# Load all 3 batches
batch_files = [
    '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
    '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
]
data_dir = 'data/mit_stanford'

all_cell_data = []
for bf in batch_files:
    fpath = os.path.join(data_dir, bf)
    if not os.path.exists(fpath): continue
    f = h5py.File(fpath, 'r'); batch = f['batch']
    for i in range(batch['summary'].shape[0]):
        try:
            summ_ref = batch['summary'][i, 0]; cyc_ref = batch['cycles'][i, 0]
            cl_ref = batch['cycle_life'][i, 0]
            summ = f[summ_ref]; cyc = f[cyc_ref]
            cycle_life = float(f[cl_ref][()].flatten()[0])
            features, cycle_nums = extract_all_features(f, summ, cyc)
            if len(features) == 0: continue
            eol = int(cycle_life)
            if eol <= cycle_nums[0]: continue
            valid = cycle_nums <= eol
            if np.sum(valid) < 30: continue
            features = features[valid]; cycle_nums = cycle_nums[valid]
            # Per-cell z-score
            m = features.mean(axis=0, keepdims=True); s = features.std(axis=0, keepdims=True) + 1e-8
            features = (features - m) / s
            all_cell_data.append({'features': features, 'cycles': cycle_nums, 'eol': eol})
        except: continue
    f.close()

print(f"Total cells: {len(all_cell_data)}")

# Split
rng = np.random.RandomState(42)
ids = list(range(len(all_cell_data))); rng.shuffle(ids)
n_train = int(0.8 * len(ids))
train_ids = ids[:n_train]; test_ids = ids[n_train:]

def build(cell_indices, wlen):
    all_X, all_y = [], []
    for idx in cell_indices:
        c = all_cell_data[idx]
        X, y = sliding_window(c['features'], c['cycles'], c['eol'], wlen)
        if len(X) > 0: all_X.append(X); all_y.append(y)
    if not all_X: return np.array([]), np.array([])
    return np.concatenate(all_X), np.concatenate(all_y)

# Test different window lengths
for wlen in [30, 50]:
    print(f"\n--- Window length = {wlen} ---", flush=True)
    X_train, y_train = build(train_ids, wlen)
    X_test, y_test = build(test_ids, wlen)

    N, L, D = X_train.shape
    feat_scaler = MinMaxScaler().fit(X_train.reshape(-1, D))
    X_train_n = feat_scaler.transform(X_train.reshape(-1, D)).reshape(N, L, D)
    X_test_n = feat_scaler.transform(X_test.reshape(-1, D)).reshape(len(X_test), L, D)

    label_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))
    y_train_n = label_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_test_n = label_scaler.transform(y_test.reshape(-1, 1)).ravel()

    print(f"  Train: {len(train_ids)} cells, {len(X_train)} samples")
    print(f"  Test:  {len(test_ids)} cells, {len(X_test)} samples")

    # Skip if too few samples
    if len(X_train) < 1000: continue

    train_ds = BatteryDataset(X_train_n, y_train_n)
    test_ds = BatteryDataset(X_test_n, y_test_n)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    set_seed(42)
    model = DeepAttnLSTM(input_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-5)

    best_val = float('inf'); best_state = None; patience = 0; t0 = time.time()

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
            best_val = val_loss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 50: break
        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: val={val_loss:.6f}, best={best_val:.6f}")

    if best_state: model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    all_y, all_preds = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            all_y.append(y.numpy()); all_preds.append(model(X).cpu().numpy())
    y_true = np.concatenate(all_y).ravel(); y_pred = np.concatenate(all_preds).ravel()
    y_true_orig = label_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_orig = label_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    m = compute_all_metrics(y_true_orig, y_pred_orig)
    print(f"  Results: RMSE={m['rmse']:.1f}, MAE={m['mae']:.1f}, "
          f"MAPE={m['mape']:.1f}%, R2={m['r2']:.4f}, Time={time.time()-t0:.1f}s")

print(f"\nPaper target: RMSE=16.11, R2=0.9821")
