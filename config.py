"""Configuration and hyperparameters for SMoE-LSTM model.

All values correspond to Table 3 and Section 4.2 of the paper:
  Zhao et al., "Towards trustworthy lithium-ion battery remaining useful
  life prediction with uncertainty quantification", J. Energy Storage 167 (2026).
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ── Network architecture (Table 3) ──────────────────────────────
    input_dim: int = 16               # 16 extracted features per cycle
    hidden_dim: int = 128             # LSTM hidden / SMoE hidden dimension
    output_dim: int = 1               # single RUL scalar
    num_experts: int = 10             # total expert networks in SMoE
    top_k: int = 4                    # experts activated per inference step
    lstm_layers: int = 2              # stacked LSTM layers

    # ── Evidence layer ─────────────────────────────────────────────
    evidence_hidden: int = 64         # intermediate FC dimension before NIG head

    # ── Training parameters (Table 3) ──────────────────────────────
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    evidential_coeff: float = 2.0     # lambda — balances NLL and regularisation

    # ── Sliding window ─────────────────────────────────────────────
    window_length: int = 30           # fixed-length sliding window (cycles)

    # ── Reproducibility & data ─────────────────────────────────────
    random_seed: int = 42
    num_trials: int = 10              # independent runs per test cell

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer: str = "adam"

    # ── Early stopping ─────────────────────────────────────────────
    # Paper protocol: fixed 100 epochs, no early stopping (0 = disabled)
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-4

    # ── Scheduler ──────────────────────────────────────────────────
    # Paper protocol: fixed lr 1e-3 across all epochs (0 = disabled)
    lr_scheduler_step: int = 0
    lr_scheduler_gamma: float = 0.5

    # ── Validation split ───────────────────────────────────────────
    # Paper protocol: all 24 training cells used for training (0 = no split)
    val_ratio: float = 0.0

    # ── Gradient clipping ──────────────────────────────────────────
    # Not in the paper's Table 3 (0 = disabled)
    clip_grad_norm: float = 0.0

    # ── MoE load balancing ─────────────────────────────────────────
    # Auxiliary loss coefficient of the Shazeer-2017 noisy top-k gating
    # (0 = disabled). Without it the gate collapses to a fixed subset of
    # experts (diagnosed: 4/10 experts idle). The paper cites [57] for the
    # gating mechanism but omits the coefficient from Table 3.
    moe_balance_coeff: float = 0.01

    # ── Routing supervision (optimisation beyond the paper) ────────
    # Teaches the gate to predict which expert is actually best for each
    # sample: soft oracle targets = softmax(-per-expert squared error / tau).
    # Diagnosis showed the trained gate routes worse than random
    # (25% vs 40% agreement); this loss directly supervises routing.
    route_sup_coeff: float = 0.0
    route_sup_tau: float = 0.02
    router_mse_coeff: float = 0.1   # supervision of the rough-RUL router head

    # ── Gate capacity ──────────────────────────────────────────────
    gate_hidden: int = 0              # hidden layer width in the gate (0 = linear)

    # ── LSTM direction (optimisation beyond the paper) ─────────────
    bidirectional: bool = False       # bidirectional LSTM (2x hidden -> projected)

    # ── Feature augmentation (optimisation beyond the paper) ───────
    # Random multiplicative perturbation of F1/F2/F4 during training so
    # anomalous cells (capacity/IC-peak offsets) generalise better.
    aug_scale: float = 0.0            # std of the perturbation (0 = disabled)

    # ── Paths ──────────────────────────────────────────────────────
    data_mit: str = "data/mit_stanford"
    data_xjtu: str = "data/xjtu"
    save_dir: str = "checkpoints"

    # ── Device ─────────────────────────────────────────────────────
    device: str = "cuda"  # set to "cpu" if no GPU available


# Pre-built configs matching the paper
default_config = Config()
