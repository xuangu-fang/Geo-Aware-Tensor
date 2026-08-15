# Geo-Aware Tensor POC

> 2026-08-15 更新：当前的四方向定位、优先级、共享接口与两个新 POC 结果，
> 统一记录在 [`papers/four_tracks/README.md`](papers/four_tracks/README.md)。
> 原 Paper A/B 均保留；新增 domain-kernel GP functional Tucker 与
> SDF-conditioned neural functional Tucker 两条线。

## 中文研究文档

建议先读最新的[故事线与进展总报告](papers/zh/最新版本故事线与进展总报告.md)
（[PDF 版](papers/zh/最新版本故事线与进展总报告.pdf)）。它用一份短报告说明两篇论文当前的
核心方法、与初始 proposal 的重大变化、正负证据、应删除的支线和最小下一步。
最新一轮冻结实验和明确的 GO/NO-GO 决策见
[第七轮发表导向迭代报告](papers/zh/第七轮发表导向迭代报告.md)。

中文核心技术报告与完整迭代记录见 [`papers/zh/`](papers/zh/README.md)；这些材料是证据账本，
不再作为项目的首要阅读入口。机器可读统计与图片见
[`papers/longterm_results/`](papers/longterm_results/TABLES.md)。

下一轮的项目管理、数据准入、论文实验和审稿风险分别见
[`PROJECT_MANAGEMENT.md`](papers/PROJECT_MANAGEMENT.md)、
[`DATASET_SELECTION.md`](papers/DATASET_SELECTION.md)、
[`NEXT_ROUND_PAPER_PLAN.md`](papers/NEXT_ROUND_PAPER_PLAN.md) 和
[`REVIEW_RISK_REGISTER.md`](papers/REVIEW_RISK_REGISTER.md)。
GitHub 首批 milestones/issues 的可验收 backlog 见
[`GITHUB_BACKLOG.md`](papers/GITHUB_BACKLOG.md)。

## Mature tensor-refocused two-paper deliverable

The original POC has been extended into two papers with one shared core:
traditional CP/Tucker factors become geometry-aware while their multilinear
decoder stays explicit. Each track has a full iteration log, frozen multi-seed
confirmation, paired statistics, figures, negative results, and manifests:

- [Paper A: Operator Geometry-Aware Bayesian Tucker](papers/paper_a/DRAFT_TUCKER.md)
  places mode operators, an explicit Tucker core, and conditional Bayesian
  uncertainty inside classical tensor decomposition; its main frozen result
  uses 2% observations.
- [Paper B: Geometry-Conditioned Phase Tensor Factorization](papers/paper_b/DRAFT.md)
  uses explicit phase-paired CP factors for 1%-observed fields on unseen
  geometries and a higher query resolution. Its The Well external stress test
  is rejected because all methods have NRMSE approximately one; only the
  controlled mechanism evidence is currently positive.
- [Final delivery overview](papers/FINAL_DELIVERY.md) summarizes the supported
  claims, limitations, evidence, and exact entry points.
- [Tensor-core contract](papers/TENSOR_CORE_REFOCUS.md) and [iteration ledger](papers/TENSOR_REFOCUS_PROGRESS.md)
  record what counts as a genuine tensor contribution. The
  [frozen shared evaluation protocol](papers/EVALUATION_PROTOCOL.md) records the
  anti-leakage, fairness, and seed-level statistical rules.

The sections below document the earlier broad POC and remain useful as a quick
demo and external-data baseline suite; the dense operator GP and monolithic
IP-NF results are now precursor baselines, not the main tensor claims.

This repository turns the two original proposals into a reproducible masked-field
reconstruction benchmark. It implements both directions:

1. **Operator-Spectral Bayesian Tensor**: an exact Bayesian posterior over a
   low-joint-energy Tucker spectral core (`bayesian_spectral_tensor`), plus the
   original mean-field Bayesian CP research variant (`bayesian_spectral_cp`).
2. **Geo-NFT**: nonlinear eigenfeature tensor factors with explicit
   operator-energy control (`geo_nft`).

All methods receive exactly the same noisy observations. Metrics are computed on
unobserved entries only. The main sweeps use 0.5%, 1%, and 5% observations.

## What is implemented

- Analytic Laplacian eigenbases for periodic, Dirichlet, and Neumann modes.
- Geometry-correct random, periodic-gap, block, and fixed-sensor masks.
- Classical discrete CP, SIREN, Fourier-INR, raw-coordinate neural CP,
  deterministic spectral CP, wrong-geometry spectral CP, both proposed models.
- Exact empirical-Bayes core inference, predictive variance, 95% coverage, NLL,
  uncertainty/error rank correlation, spectral energy diagnostics.
- Local adapters for The Well Active Matter and RealPDEBench cylinder PIV.
- Multi-seed JSON artifacts, reconstruction figures, CSV/Markdown aggregation,
  and observation-scaling plots.

## Environment

```bash
cd /home/ubuntu/project/Geo-Aware-Tensor
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
```

The external-data gates need the optional range-reading dependencies:

```bash
.venv/bin/pip install -e '.[dev,data]'
.venv/bin/python experiments/build_independent_wave_dataset.py
.venv/bin/python experiments/gate_the_well_acoustic.py
.venv/bin/python experiments/extract_the_well_pilot.py
```

Generated fields stay under ignored `data/`; compact audits, pinned split
manifests, checksums, and visual gates are committed under `papers/dataset_gates/`
and `experiments/dataset_splits/`.

The completed runs used the already provisioned CUDA environment at
`/home/ubuntu/project/yanjiu/.venv` (PyTorch 2.11 + CUDA 12.8 on an A100). To
reuse it without installing anything:

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python -m pytest -q
```

## Quick demo

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python experiments/run_poc.py \
  --dataset synthetic_boundary \
  --models cp,inr,neural_cp,spectral_cp,bayesian_spectral_tensor,geo_nft \
  --ratios 0.01 --masks random --seeds 0 --steps 1600 \
  --rank 4 --reg-weight 0.05 --output runs/demo
```

Real local data:

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python experiments/run_poc.py \
  --dataset active_matter --ratios 0.005,0.01,0.05 --seeds 0,1,2 \
  --steps 2200 --rank 8 --reg-weight 0.01 --output runs/active_matter
```

Aggregate any set of runs:

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/aggregate_results.py runs/demo runs/active_matter \
  --output reports/my_results
```

## Model definitions

For mode-specific eigenfeatures `Phi_m(x_m)` and rank `R`, deterministic spectral
CP is

```text
u(x_1,...,x_M) = sum_r c_r product_m [Phi_m(x_m)^T w_{m,r}].
```

The Bayesian core model selects the lowest joint product-operator energies,
`lambda_k = sum_m lambda_{m,k_m}`, and uses

```text
w_k ~ Normal(0, [alpha (1 + lambda_k)^p]^-1).
```

Its Gaussian posterior and evidence are computed exactly in feature space. Geo-NFT
replaces each linear factor by a small eigenfeature adapter and penalizes the
energy of the *complete* nonlinear factor. CP columns are scale-normalized so the
component amplitude cannot evade the geometric regularizer.

## Results and scope

See [the detailed POC report](reports/POC_REPORT.md), [aggregated tables](reports/results/RESULTS.md),
and [literature/data audit](reports/LITERATURE_AND_DATA.md). Raw run directories
are intentionally git-ignored but remain locally reproducible from the commands
recorded in the report.

This is a reconstruction POC, not yet evidence for forecasting or operator
learning across unseen trajectories. Bayesian coverage is reported honestly and
is currently under-calibrated on the harder 3-D fields.
