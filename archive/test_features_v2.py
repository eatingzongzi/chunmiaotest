"""Improved feature engineering: differential features + feature selection."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Improved feature extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_better_features_v2(f, summ, cyc, cycle_life):
    """Extract features with differential normalization and better CC/CV detection."""
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

        # Get raw data
        v_raw = np.zeros(1000); i_raw = np.zeros(1000); dqdv_raw = np.zeros(1000)
        try:
            v_raw = f[v_refs[i]][()].flatten().astype(np.float64)
        except: pass
        try:
            i_raw = f[i_refs[i]][()].flatten().astype(np.float64)
        except: pass
        try:
            dqdv_raw = f[dqdv_refs[i]][()].flatten().astype(np.float64)
        except: pass

        # ── CC Voltage features: use voltage during charging, broader range ──
        v = v_raw[v_raw > 0.1]
        i_charge_mask = i_raw[v_raw > 0.1] > 0.01

        if np.sum(i_charge_mask) >= 5:
            v_charge = v[i_charge_mask]
            i_charge = i_raw[v_raw > 0.1][i_charge_mask]
            i_max = np.max(i_charge)
            # CC phase: I near max, V in 3.2-3.6 range
            cc_mask = (i_charge >= 0.85 * i_max) & (v_charge >= 3.0) & (v_charge <= 3.65)
            if np.sum(cc_mask) >= 5:
                v_cc_segment = v_charge[cc_mask]
                # Resample to 6 points using cubic spline
                x_orig = np.linspace(0, 1, len(v_cc_segment))
                try:
                    cs = CubicSpline(x_orig, v_cc_segment)
                    feats[0:6] = cs(np.linspace(0, 1, 6))
                except:
                    feats[0:6] = np.interp(np.linspace(0, 1, 6), x_orig, v_cc_segment)

        # ── CV Current features: use current decay at max voltage ──
        if np.sum(i_charge_mask) >= 5:
            v_max = np.max(v_charge)
            # CV phase: V near max, I decaying
            cv_mask = (v_charge >= v_max - 0.03) & (i_charge < 0.7 * i_max)
            if np.sum(cv_mask) >= 5:
                i_cv_segment = i_charge[cv_mask]
                x_orig = np.linspace(0, 1, len(i_cv_segment))
                try:
                    cs = CubicSpline(x_orig, i_cv_segment)
                    feats[6:12] = cs(np.linspace(0, 1, 6))
                except:
                    feats[6:12] = np.interp(np.linspace(0, 1, 6), x_orig, i_cv_segment)

        # ── IC curve features ──
        dqdv = dqdv_raw[np.abs(dqdv_raw) > 1e-12]
        if len(dqdv) >= 10:
            peak_idx = np.argmin(dqdv)
            feats[12] = float(dqdv[peak_idx])  # peak value
            feats[13] = float(peak_idx) / len(dqdv)  # normalized peak position

        # ── Summary features ──
        feats[14] = float(ir_all[i]) if i < len(ir_all) else 0.0
        feats[15] = float(qd_all[i]) if i < len(qd_all) else 0.0

        features_list.append(feats)

    features = np.array(features_list, dtype=np.float32)
    cycles = np.array(cycle_nums, dtype=np.int32)

    # ── Make features differential (change from baseline cycles 5-15) ──
    if len(features) >= 15:
        baseline = features[5:15].mean(axis=0, keepdims=True)
        # For DC-IR (index 14), keep absolute value (not differential)
        ir_vals = features[:, 14].copy()
        features = features - baseline
        features[:, 14] = ir_vals  # Restore absolute IR

    return features, cycles


def sliding_window_sampling(features, cycle_indices, eol, window_length):
    N = len(features)
    if N <= window_length:
        return np.array([]), np.array([])
    X_list, y_list = [], []
    for i in range(N - window_length):
        window = features[i:i + window_length]
        current_cycle = cycle_indices[i + window_length - 1]
        rul = eol - current_cycle
        X_list.append(window)
        y_list.append(rul)
    if not X_list:
        return np.array([]), np.array([])
    return np.stack(X_list, axis=0), np.array(y_list)


class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Model (same as improved)
# ═══════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════
# Load batch1 with improved features
# ═══════════════════════════════════════════════════════════════════════════
def load_with_improved_features(data_dir, window_length=30, feature_selection=None):
    """feature_selection: list of feature indices to use (None=all 16)."""
    from src.data_loader import sliding_window_sampling as sws

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

            features, cycle_nums = extract_better_features_v2(f, summ, cyc, cycle_life)
            if len(features) == 0:
                continue

            eol = int(cycle_life)
            if eol <= cycle_nums[0]:
                continue

            valid = cycle_nums <= eol
            if np.sum(valid) < 30:
                continue
            features = features[valid]
            cycle_nums = cycle_nums[valid]

            # Per-cell z-score norm
            feat_mean = features.mean(axis=0, keepdims=True)
            feat_std = features.std(axis=0, keepdims=True)
            feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
            features = (features - feat_mean) / feat_std

            # Feature selection
            if feature_selection is not None:
                features = features[:, feature_selection]

            all_cell_data.append({
                'features': features, 'cycle_nums': cycle_nums, 'eol': eol
            })
        except Exception as e:
            continue
    f.close()

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
        if not all_X:
            return np.array([]), np.array([])
        return np.concatenate(all_X), np.concatenate(all_y)

    X_train, y_train = build(train_idx)
    X_test, y_test = build(test_idx)

    N, L, D = X_train.shape
    feat_scaler = MinMaxScaler().fit(X_train.reshape(-1, D))
    X_train = feat_scaler.transform(X_train.reshape(-1, D)).reshape(N, L, D)
    X_test = feat_scaler.transform(X_test.reshape(-1, D)).reshape(len(X_test), L, D)

    label_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))
    y_train = label_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_test = label_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return (BatteryDataset(X_train, y_train), BatteryDataset(X_test, y_test),
            label_scaler, len(all_cell_data))


def train_eval(train_ds, test_ds, label_scaler, device, input_dim):
    set_seed(42)
    model = AttnLSTM(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

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
        if patience >= 40:
            break

    if best_state:
        model.load_state_dict(best_state)

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
    return compute_all_metrics(y_true_orig, y_pred_orig), time.time() - t0


print("=== Feature Engineering Experiment ===", flush=True)
device = get_device('cuda')

# Test different feature sets
feature_names = [
    'CC-V1', 'CC-V2', 'CC-V3', 'CC-V4', 'CC-V5', 'CC-V6',
    'CV-I1', 'CV-I2', 'CV-I3', 'CV-I4', 'CV-I5', 'CV-I6',
    'IC-peak', 'IC-Vpos', 'DC-IR', 'Max-Q',
]

configs = [
    ("All 16 features", None, 16),
    # Keep best 8: CC-V4,5,6 + CV-I1,5,6 + IC-peak + Max-Q
    ("Top 8 features", [3, 4, 5, 6, 10, 11, 12, 15], 8),
    # Keep only features with |Spearman| > 0.5
    ("Best 6 (Spearman>0.5)", [3, 4, 5, 6, 10, 15], 6),
]

for name, feat_sel, input_dim in configs:
    print(f"\n--- {name} (input_dim={input_dim}) ---", flush=True)
    train_ds, test_ds, label_scaler, n_cells = load_with_improved_features(
        'data/mit_stanford', window_length=30, feature_selection=feat_sel)
    print(f"  {n_cells} cells, train={len(train_ds)}, test={len(test_ds)} samples", flush=True)

    m, elapsed = train_eval(train_ds, test_ds, label_scaler, device, input_dim)
    print(f"  RMSE={m['rmse']:.1f}, MAE={m['mae']:.1f}, MAPE={m['mape']:.1f}%, "
          f"R2={m['r2']:.4f}, Time={elapsed:.1f}s", flush=True)

    if feat_sel is not None:
        selected = [feature_names[i] for i in feat_sel]
        print(f"  Selected features: {selected}")
