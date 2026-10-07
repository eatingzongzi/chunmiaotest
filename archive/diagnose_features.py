"""Diagnose which features correlate with RUL labels."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import h5py
from scipy.stats import pearsonr, spearmanr

# Load one cell and check feature-RUL correlation
path = 'data/mit_stanford/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
f = h5py.File(path, 'r')
batch = f['batch']

# Get first 5 cells' data
feature_names = [
    'CC-V1', 'CC-V2', 'CC-V3', 'CC-V4', 'CC-V5', 'CC-V6',
    'CV-I1', 'CV-I2', 'CV-I3', 'CV-I4', 'CV-I5', 'CV-I6',
    'IC-peak', 'IC-Vpos', 'DC-IR', 'Max-Q',
]

from src.data_loader import _extract_cell_features

all_features = []
all_ruls = []

for ci in range(min(20, batch['summary'].shape[0])):
    try:
        summ_ref = batch['summary'][ci, 0]
        cyc_ref = batch['cycles'][ci, 0]
        cl_ref = batch['cycle_life'][ci, 0]

        summ = f[summ_ref]
        cyc = f[cyc_ref]
        cycle_life = float(f[cl_ref][()].flatten()[0])

        features, cycle_nums = _extract_cell_features(f, summ, cyc, cycle_life)

        if len(features) < 30:
            continue

        # RUL for each cycle
        eol = int(cycle_life)
        # Clip to valid cycles (RUL > 0)
        valid = cycle_nums <= eol
        features = features[valid]
        cycle_nums = cycle_nums[valid]
        ruls = eol - cycle_nums

        all_features.append(features)
        all_ruls.append(ruls)
    except Exception as e:
        print(f"  Cell {ci}: {e}")
        continue

f.close()

# Combine all cells
X = np.concatenate(all_features, axis=0)
y = np.concatenate(all_ruls, axis=0)
print(f"Total samples: {len(X)}, RUL range: [{y.min():.0f}, {y.max():.0f}]")

# Per-feature correlation with RUL
print(f"\n{'Feature':<12} {'Pearson r':>10} {'Spearman ρ':>10} {'Std':>10} {'Mean':>10}")
print("-" * 55)
for d in range(16):
    feat_vals = X[:, d]
    if np.std(feat_vals) > 1e-8:
        r, _ = pearsonr(feat_vals, y)
        s, _ = spearmanr(feat_vals, y)
    else:
        r, s = 0.0, 0.0
    print(f"{feature_names[d]:<12} {r:>10.4f} {s:>10.4f} {np.std(feat_vals):>10.4f} {np.mean(feat_vals):>10.4f}")

# Compare: using only summary features (IR + Max-Q) vs all 16
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

print(f"\n=== Linear Regression Baselines ===")
for name, dims in [("Summary only (14-15)", [14, 15]),
                    ("IC only (12-13)", [12, 13]),
                    ("CV-I only (6-11)", list(range(6, 12))),
                    ("CC-V only (0-5)", list(range(0, 6))),
                    ("All 16", list(range(16)))]:
    X_sub = X[:, dims]
    lr = LinearRegression()
    scores = cross_val_score(lr, X_sub, y, cv=5, scoring='r2')
    lr.fit(X_sub, y)
    y_pred = lr.predict(X_sub)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print(f"  {name:<20s}: RMSE={rmse:.1f}, R2={r2:.4f}, CV R2={scores.mean():.4f}+/-{scores.std():.4f}")

# Also check: what's the R² using only the last cycle's QDischarge (a naive baseline)?
last_qd = X[:, 15]  # Max-Q feature
print(f"\n  Naive (mean RUL): RMSE={np.std(y):.1f}")
print(f"  Using only Max-Q:  RMSE={np.sqrt(mean_squared_error(y, last_qd * 1000)):.1f}")
