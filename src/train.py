"""Training loop with early stopping and checkpointing.

Implements the training procedure described in Section 4.2:
  - Adam optimizer, Min-Max normalised inputs & labels
  - Evidential loss with coefficient λ = 2.0
  - 10 independent trials per test cell
"""

import os
import time
import numpy as np
from typing import Optional, Dict, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

from config import Config
from src.model import SMoELSTM
from src.loss import EvidentialRegressionLoss


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda") -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def prepare_dataloaders(
    train_dataset: Dataset,
    batch_size: int,
    val_ratio: float = 0.0,
    seed: int = 42,
) -> tuple:
    """Create train/val dataloaders from a training dataset.

    Args:
        train_dataset: full training dataset
        batch_size:    batch size
        val_ratio:     fraction reserved for validation
        seed:          random seed for split

    Returns:
        (train_loader, val_loader); val_loader is None when val_ratio <= 0
        (paper protocol: the whole 24-cell training set is used).
    """
    use_pin = torch.cuda.is_available()

    if val_ratio <= 0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True, num_workers=0,
                                  pin_memory=use_pin)
        return train_loader, None

    n = len(train_dataset)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val

    train_sub, val_sub = torch.utils.data.random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0, pin_memory=use_pin)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=0, pin_memory=use_pin)
    return train_loader, val_loader


def train_epoch(
    model: SMoELSTM,
    loader: DataLoader,
    criterion: EvidentialRegressionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_norm: float = 0.0,
    balance_coeff: float = 0.0,
    route_sup_coeff: float = 0.0,
    route_sup_tau: float = 0.02,
    router_mse_coeff: float = 0.1,
    aug_scale: float = 0.0,
) -> Dict[str, float]:
    """Run one training epoch.

    Returns:
        dict with 'loss', 'rmse', 'mae', 'aux_loss' averaged over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_aux = 0.0
    total_route = 0.0
    all_preds, all_targets = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # Feature augmentation: randomly perturb the cell-level summary
        # channels (F1 IC peak, F2 peak voltage, F4 max capacity) so the
        # model learns that feature offsets are not lifetime offsets.
        # Diagnosis: anomalous cells (e.g. cell 29, higher capacity and
        # shifted IC peak) are systematically misread as younger batteries.
        if aug_scale > 0:
            X = X.clone()
            for ch in (12, 13, 15):
                eps = torch.randn(X.shape[0], 1, device=X.device) * aug_scale
                X[:, :, ch] = X[:, :, ch] * (1.0 + eps)

        optimizer.zero_grad()
        gamma, nu, alpha, beta, gate_info = model(X, return_gates=True)
        loss = criterion(y, gamma, nu, alpha, beta)

        # Auxiliary load-balancing loss for the sparse gate (Shazeer 2017)
        aux = 0.0
        if balance_coeff > 0:
            gates, indices, feats, rough = gate_info
            aux = model.moe.balance_loss(gates, indices, model.moe.num_experts)
            loss = loss + balance_coeff * aux

        # Routing supervision: teach the gate to predict which expert is
        # actually best per sample, and supervise the rough-RUL head that
        # feeds the gate. Two complementary terms:
        #  - hard term: penalise samples whose best expert is NOT in the
        #    gate's top-k (aligned with the hard top-k selection used at
        #    inference; diagnosis showed the best expert is a stable signal,
        #    so a soft full-distribution target was the wrong objective)
        #  - soft term (optional): align the full expert distribution.
        route = 0.0
        if route_sup_coeff > 0:
            gates, indices, feats, rough = gate_info
            E = model.moe.num_experts
            # rough RUL head supervision (normalised scale, weak coefficient)
            loss = loss + router_mse_coeff * F.mse_loss(rough, y)
            with torch.no_grad():
                # per-expert prediction through the shared evidence head
                outs = torch.stack([model.moe.experts[e](feats) for e in range(E)],
                                   dim=1)                              # (B, E, H)
                gamma_e = torch.stack(
                    [model.evidence(outs[:, e])[0].squeeze(-1) for e in range(E)],
                    dim=1)                                            # (B, E)
                err = (gamma_e - y).pow(2)                            # (B, E)
                best = err.argmin(dim=-1)                             # (B,)
            # gate's routing preference over ALL experts (clean logits)
            clean = model.moe.gate.clean_logits(torch.cat([feats, rough], dim=-1))
            # hard term: logsumexp of the best-3 others must not beat the best
            masked = clean.clone()
            masked.scatter_(-1, best.unsqueeze(-1), float('-inf'))
            top3_lse = masked.topk(model.moe.top_k - 1, dim=-1).values.logsumexp(-1)
            hard = F.relu(top3_lse - clean.gather(-1, best.unsqueeze(-1)).squeeze(-1))
            route = hard.mean()
            if route_sup_tau > 0:
                target = F.softmax(-err / route_sup_tau, dim=-1)      # soft oracle
                gate_prob = F.softmax(clean, dim=-1)
                soft = -(target * torch.log(gate_prob + 1e-8)).sum(-1).mean()
                route = route + 0.1 * soft
            loss = loss + route_sup_coeff * route

        loss.backward()

        # Gradient clipping for stability (disabled when clip_norm <= 0)
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_aux += (aux if isinstance(aux, float) else aux.item()) * X.size(0)
        total_route += (route if isinstance(route, float) else route.item()) * X.size(0)
        all_preds.append(gamma.detach().cpu())
        all_targets.append(y.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    n = len(all_targets)

    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))

    return {
        'loss': total_loss / n,
        'rmse': rmse,
        'mae': mae,
        'aux_loss': total_aux / n,
        'route_loss': total_route / n,
    }


@torch.no_grad()
def validate_epoch(
    model: SMoELSTM,
    loader: DataLoader,
    criterion: EvidentialRegressionLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Run one validation epoch."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        gamma, nu, alpha, beta = model(X)
        loss = criterion(y, gamma, nu, alpha, beta)

        total_loss += loss.item() * X.size(0)
        all_preds.append(gamma.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    n = len(all_targets)

    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))

    return {
        'loss': total_loss / n,
        'rmse': rmse,
        'mae': mae,
    }


def train_model(
    config: Config,
    train_dataset: Dataset,
    device: Optional[torch.device] = None,
    trial: int = 0,
    verbose: bool = True,
) -> tuple:
    """Full training routine for the SMoE-LSTM model.

    Args:
        config:         hyperparameter configuration
        train_dataset:  training data
        device:         torch device
        trial:          trial index (sets random seed offset)
        verbose:        whether to print progress

    Returns:
        (model, train_history_dict)
    """
    if device is None:
        device = get_device(config.device)

    seed = config.random_seed + trial
    set_seed(seed)

    # ── Data loaders ──────────────────────────────────────────────
    train_loader, val_loader = prepare_dataloaders(
        train_dataset, config.batch_size, val_ratio=config.val_ratio, seed=seed,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = SMoELSTM(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_experts=config.num_experts,
        top_k=config.top_k,
        lstm_layers=config.lstm_layers,
        evidence_hidden=config.evidence_hidden,
        gate_hidden=config.gate_hidden,
        bidirectional=config.bidirectional,
    ).to(device)

    criterion = EvidentialRegressionLoss(coeff=config.evidential_coeff)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = None
    if config.lr_scheduler_step > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_scheduler_step, gamma=config.lr_scheduler_gamma,
        )

    # ── Training loop ─────────────────────────────────────────────
    use_early_stop = (val_loader is not None) and (config.early_stop_patience > 0)
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = defaultdict(list)

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer,
                                    device, clip_norm=config.clip_grad_norm,
                                    balance_coeff=config.moe_balance_coeff,
                                    route_sup_coeff=config.route_sup_coeff,
                                    route_sup_tau=config.route_sup_tau,
                                    router_mse_coeff=config.router_mse_coeff,
                                    aug_scale=config.aug_scale)

        if scheduler:
            scheduler.step()

        history['train_loss'].append(train_metrics['loss'])
        history['train_rmse'].append(train_metrics['rmse'])

        # Validation + early stopping (skipped unless enabled; the paper
        # trains a fixed 100 epochs without a validation split)
        if use_early_stop:
            val_metrics = validate_epoch(model, val_loader, criterion, device)
            history['val_loss'].append(val_metrics['loss'])
            history['val_rmse'].append(val_metrics['rmse'])

            # Early stopping
            if val_metrics['loss'] < best_val_loss - config.early_stop_min_delta:
                best_val_loss = val_metrics['loss']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.early_stop_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        if verbose and (epoch % 10 == 0 or epoch == 1):
            line = (f"  Epoch {epoch:3d}/{config.epochs} | "
                    f"train loss: {train_metrics['loss']:.4f}")
            if use_early_stop:
                line += (f" | val loss: {history['val_loss'][-1]:.4f} | "
                         f"val RMSE: {history['val_rmse'][-1]:.4f}")
            print(line)

    # Restore best state (only meaningful with early stopping enabled)
    if use_early_stop and best_state is not None:
        model.load_state_dict(best_state)

    return model, dict(history)


@torch.no_grad()
def predict(
    model: SMoELSTM,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run inference and return predictions + uncertainties.

    Returns:
        dict with 'pred', 'aleatoric', 'epistemic' as numpy arrays
    """
    model.eval()
    all_preds, all_alea, all_epis = [], [], []

    for X, _ in loader:
        X = X.to(device)
        gamma, nu, alpha, beta = model(X)
        result = SMoELSTM.predict(gamma, nu, alpha, beta)

        all_preds.append(result['pred'].cpu().numpy())
        all_alea.append(result['aleatoric'].cpu().numpy())
        all_epis.append(result['epistemic'].cpu().numpy())

    return {
        'pred': np.concatenate(all_preds).ravel(),
        'aleatoric': np.concatenate(all_alea).ravel(),
        'epistemic': np.concatenate(all_epis).ravel(),
    }
