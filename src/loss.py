"""Evidential regression loss based on the Normal-Inverse-Gamma (NIG) prior.

Implements Equations 22-24 from the paper:
  L(θ) = NLL(θ) + λ · R(θ)

where:
  - NLL is the negative log marginal likelihood (Student-t evidence)
  - R   is the evidence regulariser that penalises high evidence on wrong predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def nig_nll(y: torch.Tensor, gamma: torch.Tensor, nu: torch.Tensor,
            alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Negative log marginal likelihood of the NIG prior (Eq. 23).

    The marginal likelihood is a Student-t distribution:
        p(y | γ, ν, α, β) = St(y; γ, β(1+ν)/(ν α), 2α)

    Args:
        y:      ground-truth targets,   shape (batch, 1)
        gamma:  NIG mean parameter,     shape (batch, 1)
        nu:     pseudo-observation count, shape (batch, 1)
        alpha:  degrees-of-freedom param, shape (batch, 1)
        beta:   scale parameter,         shape (batch, 1)

    Returns:
        Scalar NLL loss averaged over the batch.
    """
    # Ω = 2β(1 + ν)  — the scale term in the Student-t
    omega = 2.0 * beta * (1.0 + nu)

    # Student-t negative log-likelihood
    # log St(y; γ, σ²_st, 2α) with σ²_st = β(1+ν) / (ν α)
    # = ½ log(π/ν) − α log(Ω) + (α+½) log( (y−γ)²ν + Ω ) + log Γ(α) − log Γ(α+½)
    term1 = 0.5 * (torch.log(torch.pi / nu))
    term2 = -alpha * torch.log(omega)
    term3 = (alpha + 0.5) * torch.log((y - gamma).pow(2) * nu + omega)
    term4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    nll = term1 + term2 + term3 + term4
    return nll.mean()


def evidence_regulariser(y: torch.Tensor, gamma: torch.Tensor,
                         nu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Evidence regularisation loss (Eq. 24).

    Penalises total evidence Φ = 2ν + α when prediction error is large.

    R(θ) = |y − γ| · (2ν + α)

    Args:
        y:      ground-truth targets, shape (batch, 1)
        gamma:  predicted mean,       shape (batch, 1)
        nu:     pseudo-count,         shape (batch, 1)
        alpha:  DoF parameter,        shape (batch, 1)

    Returns:
        Scalar regularisation loss averaged over the batch.
    """
    error = torch.abs(y - gamma)
    total_evidence = 2.0 * nu + alpha
    reg = error * total_evidence
    return reg.mean()


class EvidentialRegressionLoss(nn.Module):
    """Joint evidential loss: NLL + λ · regulariser (Eq. 22).

    Usage:
        criterion = EvidentialRegressionLoss(coeff=2.0)
        loss = criterion(y_true, gamma, nu, alpha, beta)
    """

    def __init__(self, coeff: float = 2.0):
        super().__init__()
        self.coeff = coeff

    def forward(self, y: torch.Tensor, gamma: torch.Tensor, nu: torch.Tensor,
                alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        nll = nig_nll(y, gamma, nu, alpha, beta)
        reg = evidence_regulariser(y, gamma, nu, alpha)
        return nll + self.coeff * reg
