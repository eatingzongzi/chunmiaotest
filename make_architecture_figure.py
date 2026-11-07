#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Draw the current SMoE-LSTM architecture for the project report."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

C_IN = '#DCEBF7'; C_ENC = '#D8EAD3'; C_MOE = '#FCE8C8'
C_EVI = '#E7DDF2'; C_OUT = '#F8D7DA'; C_OPT = '#F2F2F2'


def box(ax, x, y, w, h, text, color, fs=11, dashed=False, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                       fc=color, ec='#444444', lw=1.2,
                       ls='--' if dashed else '-', zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, zorder=3, fontweight='bold' if bold else 'normal')


def arrow(ax, p1, p2, style='-|>', color='#333333', lw=1.4, ls='-'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls, zorder=1))


fig, ax = plt.subplots(figsize=(18, 8.2))
ax.set_xlim(0, 19)
ax.set_ylim(0, 8.6)
ax.axis('off')

# ── Main pipeline ───────────────────────────────────────────────────
box(ax, 0.3, 3.6, 2.0, 1.5,
    '输入窗口\n30 循环 × 16 特征\nV/I · IC · IR · 容量', C_IN, 9.5)
arrow(ax, (2.3, 4.35), (2.9, 4.35))

box(ax, 2.9, 3.75, 2.0, 1.2, 'LSTM 编码器\n2 层 × 128', C_ENC, 10.5)
arrow(ax, (4.9, 4.35), (5.4, 4.35))

box(ax, 5.4, 3.8, 1.7, 1.1, '时序表示 h\n(注意力池化*)', C_ENC, 9.5)

# rough RUL branch
arrow(ax, (6.25, 4.9), (6.25, 5.9))
box(ax, 5.3, 5.9, 1.9, 0.75, 'RUL 粗估计头', C_MOE, 9.5)
arrow(ax, (7.2, 6.27), (7.9, 6.27))
box(ax, 7.9, 5.95, 2.1, 0.65, '稀疏门控\nNoisy Top-K', C_MOE, 9.5)

# gate -> experts
arrow(ax, (8.95, 5.95), (8.95, 5.55))

# experts
box(ax, 7.9, 2.2, 2.1, 3.3, '', C_MOE)
ax.text(8.95, 5.42, '10 个专家网络 (2 层 MLP)', ha='center', va='top',
        fontsize=8.5, zorder=3)
for i in range(5):
    box(ax, 8.08 + (i % 5) * 0.38, 4.15, 0.30, 0.85, '', 'white', dashed=False)
    box(ax, 8.08 + (i % 5) * 0.38, 3.05, 0.30, 0.85, '', 'white', dashed=False)
ax.text(8.95, 2.42, 'top-4 激活', ha='center', va='center', fontsize=8.5,
        color='#8a4b08')
arrow(ax, (6.25, 4.35), (7.9, 4.35))

# weighted sum
ax.add_patch(Circle((10.35, 4.35), 0.22, fc='white', ec='#444', lw=1.2, zorder=3))
ax.text(10.35, 4.35, 'Σ', ha='center', va='center', fontsize=12, zorder=4)
arrow(ax, (10.0, 4.35), (10.13, 4.35))
arrow(ax, (10.57, 4.35), (10.95, 4.35))

box(ax, 10.95, 3.75, 2.0, 1.2, '证据层\nFC→ReLU→FC', C_EVI, 10)
arrow(ax, (12.95, 4.35), (13.35, 4.35))

box(ax, 13.35, 3.55, 1.6, 1.6, '(γ, ν, α, β)\nNIG 分布', C_EVI, 10)

# outputs: gamma goes through the optional physics fusion node
ax.add_patch(Circle((15.55, 5.05), 0.18, fc='white', ec='#444', lw=1.2, zorder=3))
ax.text(15.55, 5.05, '⊕', ha='center', va='center', fontsize=12, zorder=4)
arrow(ax, (14.95, 5.05), (15.37, 5.05))
arrow(ax, (15.73, 5.05), (16.1, 5.05))
arrow(ax, (14.95, 4.55), (16.1, 4.15))
arrow(ax, (14.95, 4.15), (16.1, 3.45))
box(ax, 16.1, 4.7, 2.7, 0.7, 'RUL 点预测  γ', C_OUT, 10)
box(ax, 16.1, 3.8, 2.7, 0.7, '偶然不确定性  β/(α−1)', C_OUT, 9)
box(ax, 16.1, 3.1, 2.7, 0.7, '认知不确定性  β/[ν(α−1)]', C_OUT, 9)

# ── Optional modules (dashed) ──────────────────────────────────────
box(ax, 5.75, 6.9, 3.0, 0.65,
    '* 注意力池化（可选）', C_OPT, 8.5, dashed=True)
arrow(ax, (6.5, 6.9), (6.5, 4.9), ls='--', color='#999')
box(ax, 13.0, 6.2, 4.6, 0.65,
    '物理先验（可选）：幂律 RUL + 门控融合', C_OPT, 8.5, dashed=True)
arrow(ax, (15.55, 6.2), (15.55, 5.23), ls='--', color='#999')

# ── Training losses ────────────────────────────────────────────────
box(ax, 3.2, 1.25, 4.2, 0.85,
    '证据回归损失\nNLL + λ·|y−γ|·(2ν+α)', C_OPT, 9, dashed=True)
box(ax, 7.8, 1.25, 3.4, 0.85,
    '硬路由监督\nrelu(LSE(其余top-3)−最优logit)', C_OPT, 8.5, dashed=True)
box(ax, 11.6, 1.25, 3.0, 0.85,
    '负载平衡损失\nE · Σ f·P', C_OPT, 9, dashed=True)
arrow(ax, (8.95, 2.6), (8.95, 2.1), ls='--', color='#999')

ax.text(9.5, 0.45,
        '工程优化：MoE 向量化（批矩阵运算）— 训练 1.9×、推理 1.76×，数值一致',
        ha='center', va='center', fontsize=10, style='italic', color='#333')

plt.tight_layout()
out_png = 'architecture.png'
out_svg = 'architecture.svg'
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(out_svg, bbox_inches='tight', facecolor='white')
print(f'saved {out_png} and {out_svg}')
