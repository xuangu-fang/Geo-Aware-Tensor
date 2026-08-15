# Physics-Informed Tensor Learning：中心 research hub

更新时间：2026-08-15。

## 1. 中心仓库职责

本仓库不再同时承载三条活跃方法线的日常实现。它只维护：

1. 三个独立方向的最新结论、commit 与 GO/NO-GO；
2. 共享的数据协议、mask 定义、指标、baseline naming 和泄漏审计；
3. 跨方向 phase diagram 与论文定位，避免三个仓库重复造 baseline；
4. 已完成的历史代码和原始结果，作为迁移前 provenance；
5. 哪些发现可以共享，哪些结论只能属于某一个任务。

新的模型实现、方向特定实验和逐轮研究日志必须进入对应独立仓库。

## 2. 活跃仓库 registry

| Track | Local folder | GitHub | 初始迁移 commit | 当前状态 |
|---|---|---|---|---|
| 1. Operator-prior tensor | `/home/ubuntu/project/operator-prior-tensor` | <https://github.com/xuangu-fang/operator-prior-tensor> | `5580586` | calibrated mismatch phase 已为正；推进 PDE/operator perturbation |
| 3. Operator-spectral FunBaT | `/home/ubuntu/project/operator-spectral-funbat` | <https://github.com/xuangu-fang/operator-spectral-funbat> | `be71080` | global dictionary 为 Stage-0；高级 mode-wise spectral POC 开始 |
| 4. Functional operator completion | `/home/ubuntu/project/functional-operator-completion` | <https://github.com/xuangu-fang/functional-operator-completion> | `adf3056` | 从 geometry conditioning 转向 incomplete simulation campaign |

迁移规则：中心仓库保留迁移前快照，不从子仓库机械反向复制全部代码。Hub 只引用已推送 commit、摘要表和必要的共享协议修订。

## 3. 统一研究问题

三个方向共享“物理信息如何进入低秩连续张量”这一总问题，但入口不同：

| Track | 物理信息入口 | 主要统计收益 | 首要 failure mode |
|---|---|---|---|
| 1 | 已知 operator basis 与 spectral shrinkage | 极稀疏时降低 factor variance | basis mismatch 产生不可消除 bias |
| 3 | GP covariance / operator-induced spectrum | mode-wise function prior 与 UQ | kernel 不可识别或分离近似失真 |
| 4 | 函数输入 encoder + 不完整组合的 Tucker interaction | 跨 simulation combinations 共享统计强度 | 退化为已有 MIONet 或只会 random-entry interpolation |

方向 2 的 phase-factorized wave tensor 已 STOP/DOWNGRADE，保留在 `papers/four_tracks/`，不迁移为活跃仓库。

## 4. 共享实验纪律

### 4.1 稀疏率

- 早期实验优先 2%、5%、10%；方向 3/4 的机制 sanity 可额外使用 1%。
- 必须区分 training-label fraction、test-domain few-shot sensors 和 simulation-combination coverage。
- 不允许把“只运行 10% 组合”和“每个场只观测 10% 坐标”合并成一个 observation ratio。

### 4.2 预算

- 早筛 3 seeds、300--500 updates；过 gate 后才扩展。
- 所有 paired models 共享 split、mask、noise realization、normalization 和 selection protocol。
- validation 只用于评估；learned kernel routing、rank 或 checkpoint 不能读取冻结 test target。

### 4.3 指标

共同主指标是 held-out NRMSE、RMSE、MAE。Bayesian 方法额外报告 NLL、95% coverage、interval width 和 error--uncertainty correlation。NRMSE 接近或高于 1 时，不讨论小幅相对显著性。

### 4.4 共享 baseline families

| Family | Track 1 | Track 3 | Track 4 |
|---|---|---|---|
| CP/Tucker | operator CP/Tucker、discrete CP/Tucker | functional GP CP/Tucker | functional CP/Tucker |
| Neural continuous | neural functional CP/Tucker、SIREN | neural mean + GP residual、FunBaT | joint INR、concat DeepONet |
| Operator learning | 只作外部压力测试 | 可作 neural mean | MIONet 为第一强 baseline，FNO/GINO 次之 |
| Kernel/GP | flat product GP | global dictionary、per-mode、oracle/swap | GP/RBF 只作插值或 few-shot control |

## 5. 当前跨方向优先级

1. **Track 3 高级 POC**：先验证 planted mode--kernel recovery，再做 operator-derived nonnegative spectral separation；数学漂亮不等于有效，必须允许 NO-GO。
2. **Track 1 外部化 phase boundary**：把 synthetic subspace rotation 横轴替换为 PDE coefficient/operator perturbation 与实际 projection residual。
3. **Track 4 新任务验证**：先证明 whole-combination holdout 上相对 MIONet 的收益，再考虑 operator encoder 或 Bayesian adaptation。

## 6. Hub 更新模板

每个子仓库完成一个里程碑后，本文件只记录：

- repository + commit；
- 本轮唯一 hypothesis；
- protocol 和最关键数字；
- 正信号、负信号与是否改变主张；
- 下一轮 GO/NO-GO。

详细公式、实现与 raw artifacts 留在子仓库。

## 7. 共享历史材料

- 四方向统一地图：[`papers/four_tracks/README.md`](papers/four_tracks/README.md)
- 完整迭代日志：[`papers/four_tracks/ITERATIONS.md`](papers/four_tracks/ITERATIONS.md)
- 共享审计协议：[`papers/four_tracks/tech_reports/SHARED_AUDIT_PROTOCOL.md`](papers/four_tracks/tech_reports/SHARED_AUDIT_PROTOCOL.md)
- R7 路线复核：[`papers/four_tracks/tech_reports/ROUND4_STRATEGY_REVIEW.md`](papers/four_tracks/tech_reports/ROUND4_STRATEGY_REVIEW.md)

