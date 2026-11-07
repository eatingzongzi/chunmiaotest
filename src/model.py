"""SMoE-LSTM model for trustworthy battery RUL prediction with UQ.

Architecture (Figure 9 of the paper):
  ┌──────────────────────┐
  │   Battery features   │  (batch, seq_len, 16)
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │  Feature Extractor   │
  │  ┌─────────────────┐ │
  │  │  LSTM (2 layers) │ │
  │  └────────┬────────┘ │
  │           ▼          │
  │  ┌─────────────────┐ │
  │  │  Sparse MoE      │ │  10 experts, Top-4 sparse gating
  │  └────────┬────────┘ │
  └───────────┼──────────┘
              ▼
  ┌──────────────────────┐
  │   Evidence Layer     │  FC → ReLU → FC → (γ, ν, α, β)
  └──────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════
#  Noisy Top-K Gating (Eqs. 16-18)
# ═══════════════════════════════════════════════════════════════════════

class NoisyTopKGate(nn.Module):
    """Sparse gating with additive noise and Top-K selection.

    H(x)  = Softmax(KeepTopK(W_g x + ε · Softplus(W_noise x), k))   (Eq.16-18)

    Args:
        input_dim:   dimension of input features
        num_experts: total number of experts
        top_k:       number of experts to activate per sample
        hidden_dim:  optional hidden layer before the gating linears
                     (0 = plain linear gate)
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int,
                 hidden_dim: int = 0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        if hidden_dim > 0:
            self.pre = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            g_dim = hidden_dim
        else:
            self.pre = nn.Identity()
            g_dim = input_dim

        self.w_g = nn.Linear(g_dim, num_experts, bias=False)
        self.w_noise = nn.Linear(g_dim, num_experts, bias=False)

    def clean_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Noise-free gate logits (used for routing supervision & eval)."""
        return self.w_g(self.pre(x))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sparse gating weights.

        Args:
            x: input features, shape (batch, input_dim)

        Returns:
            gates:  sparse softmax weights, shape (batch, num_experts)
            indices: expert indices with top-k activations, shape (batch, top_k)
        """
        clean_logits = self.clean_logits(x)                           # (B, E)

        # Eq. 17: additive noise scaled by Softplus-learned magnitude
        noise_std = F.softplus(self.w_noise(self.pre(x)))
        noise = torch.randn_like(clean_logits) * noise_std
        noisy_logits = clean_logits + noise                           # (B, E)

        # Eq. 18: KeepTopK — retain top-k, set rest to -inf
        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)

        mask = torch.full_like(noisy_logits, float('-inf'))
        mask.scatter_(-1, top_k_indices, top_k_logits)

        # Softmax — -inf entries become 0
        gates = F.softmax(mask, dim=-1)                               # (B, E)

        return gates, top_k_indices


# ═══════════════════════════════════════════════════════════════════════
#  Expert Network
# ═══════════════════════════════════════════════════════════════════════

class Expert(nn.Module):
    """Single expert: 2-layer MLP with ReLU.

    Each expert specialises in a degradation sub-pattern.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════
#  Sparse Mixture of Experts
# ═══════════════════════════════════════════════════════════════════════

class SparseMoE(nn.Module):
    """Sparsely-gated Mixture of Experts layer (Section 3.3.1).

    y = Σ G(x)_i · E_i(x)                                    (Eq. 14)

    where G is the sparse Top-K gate and E_i are expert FFNs.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int,
                 top_k: int, dropout: float = 0.1,
                 gate_input_dim: int = None, gate_hidden: int = 0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        if gate_input_dim is None:
            gate_input_dim = input_dim
        self.gate = NoisyTopKGate(gate_input_dim, num_experts, top_k,
                                  hidden_dim=gate_hidden)
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, input_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor,
                gate_input: torch.Tensor = None,
                return_gates: bool = False):
        """Route input through sparse experts (fully vectorised).

        All expert weights are stacked into 3-D tensors so every expert
        runs as one batched matmul — no per-expert Python loop. The
        per-expert parameter layout is unchanged (checkpoint compatible).

        Args:
            x:            shape (B, input_dim) — expert input
            gate_input:   optional gate input (defaults to x); may carry
                          extra routing signals such as a rough RUL estimate
            return_gates: also return (gates, indices) for the
                          auxiliary load-balancing loss

        Returns:
            weighted expert mixture, shape (B, input_dim);
            plus (gates, indices) when return_gates is True
        """
        if gate_input is None:
            gate_input = x
        gates, indices = self.gate(gate_input)                 # (B, E), (B, K)
        B, D = x.shape

        # Stack all expert weights (cheap view-concatenation) — expert i is
        # Linear(D->H) -> ReLU -> Dropout -> Linear(H->D) inside experts[i].
        W1 = torch.stack([ex.net[0].weight for ex in self.experts], dim=0)  # (E,H,D)
        b1 = torch.stack([ex.net[0].bias for ex in self.experts], dim=0)    # (E,H)
        W2 = torch.stack([ex.net[3].weight for ex in self.experts], dim=0)  # (E,D,H)
        b2 = torch.stack([ex.net[3].bias for ex in self.experts], dim=0)    # (E,D)

        # One batched forward through all E experts
        h = torch.einsum('bd,ehd->beh', x, W1) + b1                       # (B,E,H)
        h = F.relu(h)
        h = F.dropout(h, p=self.experts[0].net[2].p, training=self.training)
        out = torch.einsum('beh,edh->bed', h, W2) + b2                    # (B,E,D)

        # Weighted combination of the top-k routed experts
        sel_w = gates.gather(1, indices).unsqueeze(-1)                    # (B,K,1)
        sel_out = torch.gather(out, 1,
                               indices.unsqueeze(-1).expand(-1, -1, D))   # (B,K,D)
        output = (sel_out * sel_w).sum(dim=1)                             # (B,D)

        if return_gates:
            return output, (gates, indices)
        return output

    @staticmethod
    def balance_loss(gates: torch.Tensor, indices: torch.Tensor,
                     num_experts: int) -> torch.Tensor:
        """Auxiliary load-balancing loss (Shazeer 2017 / Fedus 2021).

        L = E * Σ_i f_i * P_i

        where f_i is the fraction of routed samples assigned to expert i
        and P_i is the batch-mean gate probability of expert i. Minimised
        when both load and probability are uniform across experts; without
        it the top-k gate collapses to a fixed expert subset.

        Args:
            gates:   sparse softmax gate weights, shape (B, E)
            indices: routed expert indices,      shape (B, K)
            num_experts: total number of experts

        Returns:
            scalar balance loss
        """
        B = gates.shape[0]
        device = gates.device

        # Load fraction per expert (a sample routed to k experts counts k times)
        f = torch.zeros(num_experts, device=device, dtype=gates.dtype)
        ones = torch.ones(indices.numel(), device=device, dtype=gates.dtype)
        f.scatter_add_(0, indices.reshape(-1), ones)
        f = f / B

        # Mean gate probability per expert
        P = gates.mean(dim=0)

        return float(num_experts) * torch.sum(f * P)


# ═══════════════════════════════════════════════════════════════════════
#  Evidence Layer
# ═══════════════════════════════════════════════════════════════════════

class EvidenceLayer(nn.Module):
    """Maps feature representation to NIG hyperparameters (γ, ν, α, β).

    Output activation constraints (Section 3.4):
      γ ∈ ℝ        → linear
      ν > 0        → Softplus
      α > 1        → Softplus + 1
      β > 0        → Softplus
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden_dim, 4)  # (γ, ν, α, β)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        out = self.head(h)                            # (B, 4)

        gamma, nu, alpha, beta = out.chunk(4, dim=-1)

        # Enforce parameter constraints
        gamma = gamma                                  # unbounded (Eq. 19)
        nu = F.softplus(nu) + 1e-6                    # ν > 0
        alpha = F.softplus(alpha) + 1.0 + 1e-6        # α > 1
        beta = F.softplus(beta) + 1e-6                 # β > 0

        return gamma, nu, alpha, beta


# ═══════════════════════════════════════════════════════════════════════
#  SMoE-LSTM — Full Model
# ═══════════════════════════════════════════════════════════════════════

class SMoELSTM(nn.Module):
    """SMoE-LSTM model for trustworthy RUL prediction.

    Combines:
      1. LSTM for temporal feature encoding (Section 3.1-3.2)
      2. Sparse MoE for multi-expert degradation routing (Section 3.3.1)
      3. Evidence layer for joint prediction & UQ (Section 3.3.2)
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 128,
                 num_experts: int = 10, top_k: int = 4,
                 lstm_layers: int = 2, dropout: float = 0.1,
                 evidence_hidden: int = 64, gate_hidden: int = 0,
                 bidirectional: bool = False, use_attention: bool = False,
                 multitask: bool = False, film_dim: int = 0):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.multitask = multitask
        self.film_dim = film_dim
        if film_dim > 0:
            self.film_net = nn.Linear(film_dim, 2 * hidden_dim)

        # ── LSTM encoder ──────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Project the bidirectional hidden state (2*H) back to H
        if bidirectional:
            self.lstm_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.lstm_proj = nn.Identity()

        # Attention pooling over the LSTM time axis (optional)
        if use_attention:
            attn_in = 2 * hidden_dim if bidirectional else hidden_dim
            self.attn = nn.Linear(attn_in, 1)

        # Multi-task auxiliary head: predicts future capacity (SOH), optional
        if multitask:
            self.aux_head = nn.Linear(hidden_dim, 1)

        # ── Rough RUL head: gives the gate an explicit degradation-stage
        #    signal (experts specialise by RUL stage, so routing benefits
        #    from knowing the stage directly).
        self.router = nn.Linear(hidden_dim, 1)

        # ── Sparse Mixture of Experts ─────────────────────────────
        self.moe = SparseMoE(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            gate_input_dim=hidden_dim + 1,   # [features, rough RUL]
            gate_hidden=gate_hidden,
        )

        # ── Evidence layer ────────────────────────────────────────
        self.evidence = EvidenceLayer(
            input_dim=hidden_dim,
            hidden_dim=evidence_hidden,
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor,
                return_gates: bool = False,
                return_aux: bool = False,
                film_stats: torch.Tensor = None) -> Tuple:
        """Forward pass.

        Args:
            x:            battery feature sequence, shape (batch, seq_len, input_dim)
            return_gates: also return the MoE (gates, indices) pair for the
                          auxiliary load-balancing loss
            return_aux:   also return the multi-task SOH prediction

        Returns:
            gamma:  predictive mean  (Eq. 19)
            nu:     pseudo-count
            alpha:  DoF parameter
            beta:   scale parameter
            plus (gates, indices) when return_gates is True
            plus aux when return_aux is True
        """
        # ── LSTM encoding ─────────────────────────────────────────
        lstm_out, (h_n, _) = self.lstm(x)               # h_n: (L, B, H) or (2L, B, H)

        if self.use_attention:
            # Learn a scalar importance score per time step, then pool
            scores = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)  # (B,L)
            pooled = (lstm_out * scores.unsqueeze(-1)).sum(dim=1)          # (B,H or 2H)
            features = self.lstm_proj(pooled) if self.bidirectional else pooled
        elif self.bidirectional:
            # h_n order: [l0_fwd, l0_bwd, l1_fwd, l1_bwd, ...]
            fwd, bwd = h_n[-2], h_n[-1]
            features = self.lstm_proj(torch.cat([fwd, bwd], dim=-1))   # (B, H)
        else:
            features = h_n[-1]                           # (B, H)

        # FiLM conditioning: modulate by cell-level statistics
        if self.film_dim > 0 and film_stats is not None:
            film = self.film_net(film_stats)             # (B, 2H)
            scale, shift = film.chunk(2, dim=-1)
            features = features * (1.0 + torch.tanh(scale)) + shift

        aux = None
        if self.multitask:
            aux = self.aux_head(features).squeeze(-1)    # (B,) SOH-scale output

        # ── SMoE routing ──────────────────────────────────────────
        # Gate sees [features, rough RUL] so it can route by degradation
        # stage; experts still consume the LSTM features alone.
        rough = self.router(features)                    # (B, 1), normalised scale
        gate_in = torch.cat([features, rough], dim=-1)   # (B, H+1)
        if return_gates:
            features, gate_info = self.moe(features, gate_input=gate_in,
                                           return_gates=True)
        else:
            features = self.moe(features, gate_input=gate_in)   # (B, H)

        # ── Evidence layer ────────────────────────────────────────
        gamma, nu, alpha, beta = self.evidence(features)

        if return_gates:
            # gate_info = (gates, indices); append pre-MoE features and the
            # rough RUL so the training loop can compute oracle targets and
            # supervise the router head.
            out = (gamma, nu, alpha, beta, gate_info + (features, rough))
            return out + (aux,) if return_aux else out
        if return_aux:
            return gamma, nu, alpha, beta, aux
        return gamma, nu, alpha, beta

    @staticmethod
    def predict(gamma: torch.Tensor, nu: torch.Tensor,
                alpha: torch.Tensor, beta: torch.Tensor) -> dict:
        """Compute point prediction and decomposed uncertainties.

        Predictive mean:           ŷ  = γ                    (Eq. 19)
        Aleatoric uncertainty:     σ²a = β / (α − 1)        (Eq. 20)
        Epistemic uncertainty:     σ²e = β / (ν (α − 1))    (Eq. 21)

        Returns dict with keys: 'pred', 'aleatoric', 'epistemic'
        """
        with torch.no_grad():
            pred = gamma.detach()
            aleatoric = (beta / (alpha - 1.0)).detach()
            epistemic = (beta / (nu * (alpha - 1.0))).detach()

        return {
            'pred': pred,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
        }
