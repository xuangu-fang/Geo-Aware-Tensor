# 几何感知张量研究项目管理

> 2026-08-15 口径更新：项目现在保留四条可独立成稿的研究线。本文下方的
> A/B 里程碑是原两条线的历史记录，不作删除；新的优先级、方向 3/4
> POC 与 repo 拆分规则以 [`four_tracks/README.md`](four_tracks/README.md)
> 为当前单一事实源。

## 1. 项目目标与边界

本项目保留原来两篇独立论文，并增加两个快速 POC 方向：

- **Paper A — Operator Geometry-Aware Tensor Decomposition**：研究 mode operator、显式 CP/Tucker core 与部分贝叶斯后验如何共同提高极低观测率下的张量重建和不确定性质量。
- **Paper B — Geometry/Phase-Aligned Neural Tensor Factorization**：研究内禀传播坐标和显式 phase-paired tensor factors 如何实现跨几何、跨分辨率的稀疏场重建。
- **Direction 3 — Domain-kernel Bayesian Functional Tucker**：用不规则域 GP kernel 定义连续 Tucker 因子。
- **Direction 4 — Geometry-conditioned Neural Functional Tucker**：用 SDF 条件的神经函数因子实现跨网格、跨形状的显式 Tucker。

四条线共享：数据对象协议、mask taxonomy、mismatch × observation-ratio phase diagram、baseline fairness、seed 纪律和结果注册格式。四条线不共享必须不同的主方法声明，也不为了统一而合并优化器或推断框架。

## 2. 研究工作流与状态机

每个实验只能按下列状态推进：

```text
IDEA → SMOKE → PILOT → SELECTED → FROZEN → CONFIRM → PAPER
                     ↘ REJECTED
```

| 状态 | 允许用途 | 最低要求 |
|---|---|---|
| IDEA | 设计讨论 | 明确假设、对照和失败判据 |
| SMOKE | 实现正确性 | 单 seed、小数据、测试通过，不形成性能结论 |
| PILOT | 方向筛选 | 3 个 selection seeds；允许修改方法 |
| SELECTED | 候选配置 | 相对主要 baseline 有稳定方向性收益 |
| FROZEN | 冻结配置 | 固定数据 split、超参数范围、primary metric |
| CONFIRM | 最终确认 | 至少 10 个未参与选择的新 seed；不 optional stopping |
| PAPER | 论文证据 | 统计、资源、负结果、复现入口齐全 |
| REJECTED | 不再投入 | 记录失败原因和恢复条件 |

任何 `CONFIRM` 之后的配置改动必须产生新的 experiment ID，原确认结果不可覆盖。

## 3. Experiment ID 与目录规范

统一命名：

```text
{paper}-{workstream}-{round}-{dataset}-{ratio}-{mask}-{variant}
```

例如：

- `A-METHOD-R4-OPTUCKER-01-RANDOM-BLOCKPOST`
- `B-DATA-R4-WELLMAZE-005-SENSORS-PAIRED`
- `SHARED-PHASE-R2-MIXED-02-RANDOM-ALL`

运行产物写入 `runs/<experiment_id>/`，只包含配置、逐 seed 指标和轻量日志；论文级聚合写入 `papers/<paper>/results/`。大型数据、checkpoint 和项目内 Python 包不得提交 Git。

## 4. 当前里程碑

| 里程碑 | 状态 | 完成定义 |
|---|---|---|
| M0 POC 与三轮方法迭代 | DONE | A/B 实现、10-seed A 确认、B envelope 去留、共享 phase diagram |
| M1 数据证据链 | DONE | 独立 solver + The Well 112-case block-mean pilot；全量 source/hash/leakage audit 与一 seed harness |
| M2 Paper A 推断与消融 | CONTROLLED CONFIRMED | 随机 2%、中心 block 缺失、30% 噪声均完成十 seed；core-IV 主动采样拒绝；外部证据仍缺 |
| M3 Paper B 外部泛化 | REJECTED | The Well early-40 所有方法 NRMSE≈1；paired 只解释约 1.6% 方差，微小 paired 差异不构成有效重建 |
| M4 论文冻结 | BLOCKED-B / WRITING-A | A 进入写作；B 缺少通过绝对效果门槛的外部数据，不可按现有 evidence 投稿 |

## 5. 下一轮工作包

### WP-A1：Paper A 外部有效性或机制论文定位

- 输入：独立 wave/Helmholtz solver 和 The Well acoustic subset。
- 输出：只选择具有真实 tensor modes 和可定义 mode operators 的数据；否则明确定位为 controlled mechanism paper。
- 晋级条件：至少一个非同源数据上，Geo-Tucker 在 10 个 seed 中稳定优于 flat operator GP 与 Geo-CP；wrong geometry 明显退化。
- 失败条件：正确/错误 operator 无差别，或收益只来自更大 core。

### WP-A2：conditional Bayesian 表述与负结果固定

- 输入：当前 operator Tucker。
- 输出：统一表述为“operator-regularized MAP factors + conditional Bayesian core”；core-IV 失败进入 limitation。
- 禁止：为了主动采样正结果临时加入 random/IV 混合策略；不得宣称 fully Bayesian。

### WP-B1：独立多几何波传播

- 输入：多障碍、多源、多材料 wave/Helmholtz 数据。
- 输出：unseen geometry × unseen resolution × sparse sensors 主表。
- 晋级条件：paired tensor 在低 mismatch/aligned 子集稳定优于 IP-NF/GINO/TFNO，并用 phase diagram 明确失效边界。
- **先验绝对门槛（The Well early-40 被拒后冻结）：**外部数据 macro NRMSE ≤0.8，且相对最强 trivial baseline 的 MSE skill ≥20%；未通过时禁止讨论小幅 paired p-value。

### WP-B2：几何误差和 source 不确定性

- 输入：source shift、边界扰动、material coefficient error。
- 状态：6% 相关路径误差三 seed 已完成，原 paired 平滑退化且仍优于验证 U-Net/wrong path。
- 决策：路径后验边缘化收益不足，不升级；若继续，只允许预先声明的 source-shift failure gallery。

### WP-S1：统一 baseline harness

- 统一 mask、noise、train/validation/test、参数量和 compute 预算。
- 固定外部实现版本与 commit SHA。
- 每个 baseline 必须注明它控制的因果问题。

## 6. 决策责任与检查点

| 检查点 | 需要审阅的内容 | 默认决策 |
|---|---|---|
| Dataset gate | 字段、license、大小、geometry 可恢复性 | 不满足 geometry metadata 就只做 stress test |
| Pilot gate | 3 seed paired differences、失败案例 | 无稳定方向性收益则 REJECTED |
| Freeze gate | 主指标、超参数、seed 区间 | 冻结后禁止用 test 调参 |
| Paper gate | 主表、消融、复杂度、负结果 | 任何核心因果链缺失则不写强 claim |

## 7. GitHub 项目管理映射

仓库使用 issue labels：

- `paper-a`, `paper-b`, `shared`
- `data`, `method`, `baseline`, `evaluation`, `documentation`
- `idea`, `pilot`, `frozen`, `confirmation`, `blocked`
- `priority:P0`, `priority:P1`, `priority:P2`

Milestones：

1. `M1 Dataset evidence chain`
2. `M2 Paper A submission evidence`
3. `M3 Paper B submission evidence`
4. `M4 Reproducible paper freeze`

远端项目管理已同步到 `xuangu-fang/Geo-Aware-Tensor`：draft PR #1、四个
milestone 和按数据/方法/baseline/消融/确认划分的 issues 已建立。Issue #2
（独立波动 smoke）完成后关闭；Issue #3（The Well 数据门禁）由固定 revision、
LFS 校验和与 64/16/32 split manifest 验收。后续实验提交需引用 experiment ID
与对应 issue，避免在 PR 评论里形成不可检索的隐式实验历史。
