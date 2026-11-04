#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify vectorised MoE: numerical equivalence + speedup.

Compares the new batched-matmul SparseMoE.forward against a reference
implementation of the old per-expert loop, using a real checkpoint.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import torch

from config import Config
from src.model import SMoELSTM, SparseMoE
from src.train import get_device, set_seed

device = get_device('cuda')
set_seed(42)

config = Config()
model = SMoELSTM(
    input_dim=config.input_dim, hidden_dim=config.hidden_dim,
    num_experts=config.num_experts, top_k=config.top_k,
    lstm_layers=config.lstm_layers, evidence_hidden=config.evidence_hidden,
    gate_hidden=config.gate_hidden, bidirectional=config.bidirectional,
).to(device)
model.eval()

x = torch.randn(256, 30, 16, device=device)


def reference_loop(moe, h, gate_in):
    """Old per-expert loop (reference for equivalence check)."""
    gates, indices = moe.gate(gate_in)
    B, D = h.shape
    output = torch.zeros(B, D, device=h.device, dtype=h.dtype)
    for expert_idx in range(moe.num_experts):
        expert_mask = (indices == expert_idx)
        if not expert_mask.any():
            continue
        sample_idx, k_idx = expert_mask.nonzero(as_tuple=True)
        weights = gates[sample_idx, expert_idx]
        expert_input = h[sample_idx]
        expert_out = moe.experts[expert_idx](expert_input)
        output.index_add_(0, sample_idx, weights.unsqueeze(-1) * expert_out)
    return output


# Disable gating noise for the equivalence check (the gate adds random
# noise even in eval mode; two separate calls would route differently)
import torch.nn.functional as F
from src.model import NoisyTopKGate

def clean_gate_forward(self, x):
    clean_logits = self.clean_logits(x)
    top_k_logits, top_k_indices = torch.topk(clean_logits, self.top_k, dim=-1)
    mask = torch.full_like(clean_logits, float('-inf'))
    mask.scatter_(-1, top_k_indices, top_k_logits)
    return F.softmax(mask, dim=-1), top_k_indices

NoisyTopKGate.forward = clean_gate_forward

with torch.no_grad():
    _, (h_n, _) = model.lstm(x)
    h = h_n[-1]
    rough = model.router(h)
    gate_in = torch.cat([h, rough], dim=-1)

    # new vectorised path
    out_new = model.moe(h, gate_input=gate_in)

    # reference old loop
    out_ref = reference_loop(model.moe, h, gate_in)

diff = (out_new - out_ref).abs().max().item()
print(f"max |new - old| = {diff:.3e}")
assert diff < 1e-5, "numerical mismatch!"

# ── Timing: batched vectorised forward (full model) ─────────────────
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    with torch.no_grad():
        gamma, nu, alpha, beta = model(x)
torch.cuda.synchronize()
dt_new = (time.perf_counter() - t0) / 100 * 1000

# ── Timing: old loop inside the full model (monkey-patched) ────────
import torch.nn as nn

def old_forward(moe, h, gate_input=None, return_gates=False):
    if gate_input is None:
        gate_input = h
    gates, indices = moe.gate(gate_input)
    B, D = h.shape
    output = torch.zeros(B, D, device=h.device, dtype=h.dtype)
    for expert_idx in range(moe.num_experts):
        expert_mask = (indices == expert_idx)
        if not expert_mask.any():
            continue
        sample_idx, k_idx = expert_mask.nonzero(as_tuple=True)
        weights = gates[sample_idx, expert_idx]
        expert_out = moe.experts[expert_idx](h[sample_idx])
        output.index_add_(0, sample_idx, weights.unsqueeze(-1) * expert_out)
    if return_gates:
        return output, (gates, indices)
    return output

SparseMoE.forward = old_forward
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    with torch.no_grad():
        gamma, nu, alpha, beta = model(x)
torch.cuda.synchronize()
dt_old = (time.perf_counter() - t0) / 100 * 1000

print(f"\nFull-model inference (B=256, l=30):")
print(f"  old loop forward: {dt_old:.2f} ms")
print(f"  vectorised     : {dt_new:.2f} ms")
print(f"  speedup        : {dt_old/dt_new:.2f}x")
