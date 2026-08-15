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
| 1. Operator-prior tensor | `/home/ubuntu/project/operator-prior-tensor` | <https://github.com/xuangu-fang/operator-prior-tensor> | `07de48d` | **条件 GO**：10% 稳定，2%--5% 尚不稳定 |
| 3. Operator-spectral FunBaT | `/home/ubuntu/project/operator-spectral-funbat` | <https://github.com/xuangu-fang/operator-spectral-funbat> | `08bc6db` | **条件 GO**：算子谱原子 + generic escape atoms |
| 4. Functional operator completion | `/home/ubuntu/project/functional-operator-completion` | <https://github.com/xuangu-fang/functional-operator-completion> | `e265748` | **NO-GO**：公平增强后 MIONet 显著领先 Tucker |

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

## 5. 本轮决策与跨方向优先级

### Track 3：条件 GO，优先继续

- 1%/2%/5% planted 数据上，per-mode/rank NRMSE 为 `0.482/0.072/0.033`，global dictionary 为 `0.593/0.088/0.045`；5% 已追平 oracle。
- diffusion/advection/wave 联合谱的 rank-4 非负分离误差分别为 `0.0028/0.0325/0.1079`，说明扩散和输运更适合，波动是当前困难区。
- kernel atom top-1 恢复率只有 22%--33%，所以不得宣称发现了真实 kernel 标签。
- 删除真实高频 support 后，operator-only NRMSE 恶化到 `0.631`；加入 generic atoms 后恢复到 `0.068`。下一步主线应是 **operator-centered robust kernel bank**，而不是不受约束的字典学习。

### Track 4：NO-GO，停止堆模型

- 60% simulation-combination coverage、10% output-coordinate coverage 时，原始 Tucker/MIONet NRMSE 为 `0.576/0.536`。
- 给双方完全相同的 Fourier coordinate lifting 后，Tucker/MIONet 变为 `0.317/0.098`；高 coefficient contrast 下为 `0.346/0.141`。
- 原先接近来自双方共同的弱 coordinate trunk，不是 functional tensor 的独立优势。继续添加 operator encoder 会逐渐退化成已有 MIONet/NO-CTR。
- 保留 incomplete-campaign protocol、代码和负结果；除非提出 MIONet 未利用的新统计结构，否则不再作为主会论文推进。

### Track 1：条件 GO，先做冻结确认

- 新增变系数 Neumann diffusion Green tensor；coefficient contrast 由 0 增至 2 时，实测 projection residual 从 `0.0459` 增至 `0.0965`。
- contrast=1、10% 观测时，Operator Tucker/Neural Functional Tucker NRMSE 为 `0.158/0.189`；2% 时为 `0.273/0.262`，说明正信号目前只在 10% 较稳定。
- basis cutoff 5/8/12 的 residual 为 `0.1645/0.0699/0.0253`，但 2% NRMSE 为 `0.293/0.273/0.331`。这证伪了“投影误差越低，恢复必然越好”，主线应明确为 bias--variance phase boundary。
- 下一门槛：冻结 cutoff/rank 后做 5 个 fresh seeds 和 structured source/receiver fibers；至少 4/5 seeds 获胜且绝对 NRMSE 显著低于 1，才升级为论文主线。

## 6. 工程边界

- 三个仓库均可用各自的 `PYTHONPATH=src python -m pytest -q` 独立复现。
- Track 1 与 Track 3 为保留迁移历史，当前都使用顶层 Python 包名 `geoaware`。不要在同一个虚拟环境中连续 editable-install 两者，否则后安装者可能覆盖 import 路径。后续若长期并行维护，应分别改成唯一包名或使用每仓独立环境。

## 7. Hub 更新模板

每个子仓库完成一个里程碑后，本文件只记录：

- repository + commit；
- 本轮唯一 hypothesis；
- protocol 和最关键数字；
- 正信号、负信号与是否改变主张；
- 下一轮 GO/NO-GO。

详细公式、实现与 raw artifacts 留在子仓库。

## 8. 共享历史材料

- 四方向统一地图：[`papers/four_tracks/README.md`](papers/four_tracks/README.md)
- 完整迭代日志：[`papers/four_tracks/ITERATIONS.md`](papers/four_tracks/ITERATIONS.md)
- 共享审计协议：[`papers/four_tracks/tech_reports/SHARED_AUDIT_PROTOCOL.md`](papers/four_tracks/tech_reports/SHARED_AUDIT_PROTOCOL.md)
- R7 路线复核：[`papers/four_tracks/tech_reports/ROUND4_STRATEGY_REVIEW.md`](papers/four_tracks/tech_reports/ROUND4_STRATEGY_REVIEW.md)
