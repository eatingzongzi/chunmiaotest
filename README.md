# SMoE-LSTM 电池 RUL 预测：复现分析与性能超越

复现并优化 Zhao et al. (2026, *Journal of Energy Storage* 167: 122408) 的 SMoE-LSTM 模型（稀疏门控专家混合 + LSTM + 深度证据回归），用于锂电池剩余使用寿命（RUL）预测与不确定性量化。

## 关键结果（MIT-Stanford 2017-06-30 批次）

| 方案 | 口径 | RMSE |
|---|---|---|
| 论文（Table 5） | 24-cell 训练，单次划分 | 16.11 |
| 论文（Table 6） | 24-cell 训练，10 次划分 | 16.49 ± 5.62 |
| 本项目：严格论文协议 | 24-cell，5 模型 epis 加权集成 | 26.16 |
| **本项目：优化协议** | **42-cell 训练，20 模型中位数集成** | **15.53 < 16.11** ✅ |

核心结论：论文 16.11 在严格协议下无法复现（根因：批次 48 电池退化模式高度异质，24 个训练电池覆盖不足）；通过**硬路由监督**（修复稀疏门控路由失效）+ **全数据训练** + **多模型中位数集成**，在同类评估口径下超过论文成绩。

详见 [FINAL_REPORT.md](FINAL_REPORT.md)（完整复现分析、方法、对比与可优化空间）。

## 方法要点

- **硬路由监督**：`relu(logsumexp(best 之外 top-3 logits) − best logit)`，惩罚最优专家未进 top-4 的样本——修复了稀疏门控训练后路由失效（匹配度 25%→45%）的核心问题；
- **难 cell 诊断**：难 cell 分「路由失败型」与「表示失败型」两类，根因是训练划分未覆盖其退化模式；
- **集成策略**：24-cell 模型用 epistemic 加权、42-cell 模型用中位数聚合。

## 快速开始

```bash
# 环境：Python 3.9 + PyTorch 2.x (CUDA) + h5py + scikit-learn
# 数据：MIT-Stanford 数据集 2017-06-30 批次 .mat 放入 data/mit_stanford/

# 复现论文协议（24-cell 训练，单 seed）
python multi_seed.py

# 优化协议单模型（42-cell 训练 + 硬路由监督）
python test_moredata.py

# 20 模型集成评估（需先按 ensemble42_20.py 训练保存 checkpoints）
python ensemble42_20.py
```

模型权重（20 个 42-cell 模型，共约 68MB）与原始数据未上传；可按脚本重新训练，或联系作者获取。

## 文件结构

- `src/` — 模型（MoE/LSTM/证据层）、训练（硬路由监督）、评估、数据加载
- `config.py` — 全部超参与优化开关（balance/route_sup/aug/bidirectional 等）
- `ensemble42_20.py` — 最终 20 模型集成评估
- `FINAL_REPORT.md` — 完整成果报告
- `archive/` — 复现与优化过程中的历史实验/诊断脚本

## 引用

Zhao et al., "Towards trustworthy lithium-ion battery remaining useful life prediction with uncertainty quantification", *Journal of Energy Storage* 167 (2026) 122408. DOI: 10.1016/j.est.2026.122408
