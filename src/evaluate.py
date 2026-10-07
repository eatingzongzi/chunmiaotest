"""Evaluation metrics and uncertainty analysis.

Computes:
  - RMSE, MAE, MAPE, R²  (Eqs. 25-28)
  - Uncertainty decomposition (Eqs. 19-21)
  - Spearman correlation (uncertainty vs error)
  - Confidence-interval RMSE (Fig. 15)
  - Inference timing
"""

import time
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader

from src.model import SMoELSTM
from sklearn.preprocessing import MinMaxScaler


# ═══════════════════════════════════════════════════════════════════════
#  Point-prediction metrics (Eqs. 25-28)
# ═══════════════════════════════════════════════════════════════════════

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error (Eq. 25)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (Eq. 27)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (Eq. 26)."""
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (Eq. 28)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all four point-prediction metrics."""
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Uncertainty analysis
# ═══════════════════════════════════════════════════════════════════════

def compute_spearman_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
) -> float:
    """Spearman rank correlation between uncertainty and absolute error.

    Higher positive value → better calibration (Section 4.6, Table 8).
    """
    errors = np.abs(y_true - y_pred)
    corr, _ = spearmanr(uncertainty, errors)
    return float(corr)


def confidence_interval_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    quantiles: List[float] = None,
) -> Dict[float, float]:
    """RMSE within expanding confidence intervals (Fig. 15).

    Sorts samples by uncertainty (ascending = higher confidence),
    then computes RMSE for top x% most confident samples.

    Args:
        y_true, y_pred: ground truth and predictions
        uncertainty:     epistemic uncertainty per sample
        quantiles:       confidence thresholds (e.g. [0.05, 0.25, 0.5, 0.75, 0.95])

    Returns:
        {quantile: rmse} dict
    """
    if quantiles is None:
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    # Sort by uncertainty (low uncertainty = high confidence)
    sort_idx = np.argsort(uncertainty)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    n = len(y_true)

    results = {}
    for q in quantiles:
        n_select = max(1, int(n * q))
        results[q] = rmse(y_true_sorted[:n_select], y_pred_sorted[:n_select])

    return results


@torch.no_grad()
def measure_inference_time(
    model: SMoELSTM,
    loader: DataLoader,
    device: torch.device,
    n_warmup: int = 10,
) -> Dict[str, float]:
    """Measure per-sample inference time (Table 7).

    Returns mean and std of per-batch inference time in seconds.
    """
    model.eval()
    times = []

    # Warmup
    for i, (X, _) in enumerate(loader):
        if i >= n_warmup:
            break
        X = X.to(device)
        _ = model(X)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    for X, _ in loader:
        X = X.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _ = model(X)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append(t1 - t0)

    times = np.array(times)
    per_sample = times / loader.batch_size

    return {
        'mean_per_sample': float(np.mean(per_sample)),
        'std_per_sample': float(np.std(per_sample)),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Full evaluation report
# ═══════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: SMoELSTM,
    test_loader: DataLoader,
    device: torch.device,
    label_scaler: MinMaxScaler = None,
) -> Dict:
    """Complete evaluation of a trained SMoE-LSTM model.

    Returns a dict with all metrics and decomposed uncertainties.
    """
    model.eval()
    all_y, all_preds = [], []
    all_alea, all_epis = [], []

    for X, y in test_loader:
        X = X.to(device)
        gamma, nu, alpha, beta = model(X)
        result = SMoELSTM.predict(gamma, nu, alpha, beta)

        all_y.append(y.numpy())
        all_preds.append(result['pred'].cpu().numpy())
        all_alea.append(result['aleatoric'].cpu().numpy())
        all_epis.append(result['epistemic'].cpu().numpy())

    y_true = np.concatenate(all_y).ravel()
    y_pred = np.concatenate(all_preds).ravel()
    alea = np.concatenate(all_alea).ravel()
    epis = np.concatenate(all_epis).ravel()

    # Inverse-transform if label scaler is provided
    if label_scaler is not None:
        y_true_orig = label_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
        y_pred_orig = label_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        # Uncertainties also need inverse scaling (squared scale)
        scale_factor = (label_scaler.data_max_[0] - label_scaler.data_min_[0]) ** 2
        alea_orig = alea * scale_factor
        epis_orig = epis * scale_factor
    else:
        y_true_orig, y_pred_orig = y_true, y_pred
        alea_orig, epis_orig = alea, epis

    # Point metrics
    metrics = compute_all_metrics(y_true_orig, y_pred_orig)

    # Uncertainty quality
    spearman_epis = compute_spearman_correlation(y_true_orig, y_pred_orig, epis_orig)
    spearman_alea = compute_spearman_correlation(y_true_orig, y_pred_orig, alea_orig)
    ci_rmse = confidence_interval_rmse(y_true_orig, y_pred_orig, epis_orig)

    return {
        'metrics': metrics,
        'spearman_epistemic': spearman_epis,
        'spearman_aleatoric': spearman_alea,
        'confidence_interval_rmse': ci_rmse,
        'predictions': {
            'y_true': y_true_orig,
            'y_pred': y_pred_orig,
            'aleatoric': alea_orig,
            'epistemic': epis_orig,
        },
    }


@torch.no_grad()
def evaluate_per_cell(
    model: SMoELSTM,
    test_cells: List[Dict],
    device: torch.device,
    label_scaler: MinMaxScaler = None,
) -> Tuple[List[Dict], Dict]:
    """Per-cell evaluation matching the paper's protocol (Table 5).

    The paper evaluates each of the 6 held-out test cells independently and
    reports the average RMSE across cells (16.11), rather than pooling all
    sliding-window samples into one global metric.

    Args:
        model:         trained SMoE-LSTM model
        test_cells:    list of dicts from load_mit_stanford_paper
                       (each has 'cell_id', 'eol', 'X', 'y' normalised)
        device:        torch device
        label_scaler:  RUL label scaler (for inverse transform)

    Returns:
        (per_cell, avg) where per_cell is a list of per-cell result dicts
        and avg is a dict of mean metrics across cells.
    """
    model.eval()
    per_cell = []

    for cell in test_cells:
        X = torch.tensor(cell['X'], dtype=torch.float32).to(device)
        gamma, nu, alpha, beta = model(X)
        pred = SMoELSTM.predict(gamma, nu, alpha, beta)['pred'].cpu().numpy().ravel()
        y_true = cell['y']

        if label_scaler is not None:
            y_true_orig = label_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
            pred_orig = label_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
        else:
            y_true_orig, pred_orig = y_true, pred

        metrics = compute_all_metrics(y_true_orig, pred_orig)
        per_cell.append({
            'cell_id': cell['cell_id'],
            'eol': cell['eol'],
            'n_samples': len(y_true_orig),
            'metrics': metrics,
        })

    avg = {
        key: float(np.mean([c['metrics'][key] for c in per_cell]))
        for key in ['rmse', 'mae', 'mape', 'r2']
    }
    return per_cell, avg


def multi_trial_summary(trial_results: List[Dict]) -> Dict:
    """Aggregate results across multiple trials (mean ± std).

    Args:
        trial_results: list of result dicts from evaluate_model()

    Returns:
        dict with 'mean' and 'std' for each metric
    """
    all_metrics = defaultdict(list)

    for result in trial_results:
        for key, val in result['metrics'].items():
            all_metrics[key].append(val)
        all_metrics['spearman_epistemic'].append(result['spearman_epistemic'])
        all_metrics['spearman_aleatoric'].append(result['spearman_aleatoric'])

    summary = {}
    for key, vals in all_metrics.items():
        arr = np.array(vals)
        summary[key] = {'mean': float(np.mean(arr)), 'std': float(np.std(arr))}

    return summary


def print_evaluation_report(result: Dict, trial: int = 0):
    """Pretty-print evaluation results."""
    m = result['metrics']
    header = f"Trial {trial}" if trial > 0 else "Evaluation"
    print(f"\n{'='*60}")
    print(f"  {header} Results")
    print(f"{'='*60}")
    print(f"  RMSE : {m['rmse']:.4f}")
    print(f"  MAE  : {m['mae']:.4f}")
    print(f"  MAPE : {m['mape']:.2f}%")
    print(f"  R2   : {m['r2']:.4f}")
    print(f"  Spearman (epistemic) : {result['spearman_epistemic']:.4f}")
    print(f"  Spearman (aleatoric) : {result['spearman_aleatoric']:.4f}")
    print(f"  Avg epistemic uncertainty: {np.mean(result['predictions']['epistemic']):.4f}")
    print(f"  Avg aleatoric uncertainty: {np.mean(result['predictions']['aleatoric']):.4f}")
    print(f"{'='*60}\n")


def print_multi_trial_summary(summary: Dict):
    """Pretty-print multi-trial summary."""
    print(f"\n{'='*60}")
    print(f"  Multi-Trial Summary (mean ± std)")
    print(f"{'='*60}")
    for key in ['rmse', 'mae', 'mape', 'r2', 'spearman_epistemic']:
        if key in summary:
            s = summary[key]
            print(f"  {key:25s}: {s['mean']:.4f} ± {s['std']:.4f}")
    print(f"{'='*60}\n")
