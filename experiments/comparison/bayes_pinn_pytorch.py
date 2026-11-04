#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PyTorch port of the NASA hybrid Bayesian PINN's per-battery model.

Original (TF): variational MLP, input = cumulative discharge energy (kWh),
output = Normal(mean q_max, std). Mean branch DenseVariational 4->2->1
(elu), std branch 1. Prior N(0, 2^2), kl_weight = 1/batch_size, Adam
lr=0.04, NLL loss.

Comparison protocol (per-battery): train on the first 60% of a battery's
energy-capacity curve, predict the WHOLE curve, score the last 40%.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PRIOR_SCALE = 2.0


class BayesLinear(nn.Module):
    """Variational dense layer: weight = mu + sigma*eps, KL to N(0, s^2)."""

    def __init__(self, in_features, out_features, prior_scale=PRIOR_SCALE,
                 activation=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.w_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.w_rho = nn.Parameter(-3.0 * torch.ones(out_features, in_features))
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(-3.0 * torch.ones(out_features))
        self.prior_scale = prior_scale

    def kl_divergence(self):
        # KL(q=N(mu, softplus(rho)) || p=N(0, s^2)) summed over weights
        w_std = F.softplus(self.w_rho)
        b_std = F.softplus(self.b_rho)
        p2 = self.prior_scale ** 2
        kl_w = (torch.log(p2 / (w_std ** 2 + 1e-12)) - 1 +
                (w_std ** 2 + self.w_mu ** 2) / p2) / 2
        kl_b = (torch.log(p2 / (b_std ** 2 + 1e-12)) - 1 +
                (b_std ** 2 + self.b_mu ** 2) / p2) / 2
        return kl_w.sum() + kl_b.sum()

    def forward(self, x):
        w_std = F.softplus(self.w_rho)
        b_std = F.softplus(self.b_rho)
        w = self.w_mu + w_std * torch.randn_like(w_std)
        b = self.b_mu + b_std * torch.randn_like(b_std)
        out = F.linear(x, w, b)
        if self.activation is not None:
            out = self.activation(out)
        return out


class BayesPINNPerBattery(nn.Module):
    """MeanStdVar + Normal head, matching the NASA repo structure."""

    def __init__(self, batch_size):
        super().__init__()
        self.mean_branch = nn.Sequential(
            BayesLinear(1, 4, activation=F.elu),
            BayesLinear(4, 2, activation=F.elu),
            BayesLinear(2, 1),
        )
        self.std_branch = BayesLinear(1, 1)
        self.batch_size = batch_size

    def kl_divergence(self):
        kl = 0.0
        for m in self.modules():
            if isinstance(m, BayesLinear):
                kl = kl + m.kl_divergence()
        return kl / self.batch_size

    def forward(self, x):
        mean = self.mean_branch(x)
        std_logit = self.std_branch(x)
        std = F.softplus(std_logit) + 1e-9
        return mean, std


def fit_per_battery(cum_kwh, q_max, frac_train=0.6, epochs=2000, seed=0,
                    lr=0.04, verbose=False):
    """Train on the first frac_train of the curve; return predicted curve
    mean/std (original units) and the fitted model."""
    torch.manual_seed(seed)
    X = torch.tensor(cum_kwh, dtype=torch.float32).reshape(-1, 1)
    Y = torch.tensor(q_max, dtype=torch.float32).reshape(-1, 1)

    x_max = X.max().item()
    y_max = Y.max().item()
    Xn, Yn = X / x_max, Y / y_max - 1.0

    n = len(X)
    n_tr = max(2, int(n * frac_train))
    Xtr, Ytr = Xn[:n_tr], Yn[:n_tr]

    model = BayesPINNPerBattery(batch_size=n_tr)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        mean, std = model(Xtr)
        nll = -torch.distributions.Normal(mean, std).log_prob(Ytr).mean()
        kl = model.kl_divergence()
        loss = nll + kl
        loss.backward()
        opt.step()
        if verbose and ep % 500 == 0:
            print(f"    ep {ep}: nll={nll.item():.4f} kl={kl.item():.4f}")

    model.eval()
    with torch.no_grad():
        mean_all, std_all = model(Xn)
    mean_orig = (mean_all + 1.0) * y_max
    std_orig = std_all * y_max
    return mean_orig.squeeze(-1).numpy(), std_orig.squeeze(-1).numpy(), model


if __name__ == '__main__':
    data = np.load('mit_energy_curves.npy', allow_pickle=True).item()
    rows = []
    for cid in sorted(data):
        c = data[cid]
        m, s, _ = fit_per_battery(c['cum_kwh'], c['q_max'], frac_train=0.6)
        n = len(c['q_max'])
        n_tr = max(2, int(n * 0.6))
        y_tail = c['q_max'][n_tr:]
        m_tail = m[n_tr:]
        rmse = float(np.sqrt(np.mean((m_tail - y_tail) ** 2)))
        mape = float(np.mean(np.abs((m_tail - y_tail) / y_tail)) * 100)
        rows.append((cid, rmse, mape))
        print(f"cell {cid:2d}: tail RMSE={rmse:.4f} Ah  MAPE={mape:.2f}%", flush=True)

    rmse_all = np.mean([r[1] for r in rows])
    mape_all = np.mean([r[2] for r in rows])
    print(f"\nNASA-model per-battery protocol on MIT data (48 cells, "
          f"60% train / 40% predict):")
    print(f"  mean RMSE={rmse_all:.4f} Ah  mean MAPE={mape_all:.2f}%")
    np.save('bayes_pinn_results.npy', np.array(rows), allow_pickle=True)
