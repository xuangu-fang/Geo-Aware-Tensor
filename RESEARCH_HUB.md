# Physics-Informed Tensor Learning：中心 research hub

更新时间：2026-08-19。

## 1. 中心仓库职责

本仓库不再同时承载各方向的日常实现。它只维护：

1. 各独立方向的最新结论、commit 与 GO/NO-GO；
2. 共享的数据协议、mask 定义、指标、baseline naming 和泄漏审计；
3. 跨方向 phase diagram 与论文定位，避免各仓库重复造 baseline；
4. 已完成的历史代码和原始结果，作为迁移前 provenance；
5. 哪些发现可以共享，哪些结论只能属于某一个任务。

共享大型数据、runs 和 cache 的机器级布局见 [`DATA_STORAGE.md`](DATA_STORAGE.md)；数据准入、现有本机资源、官方 benchmark 和分方向优先级见 [`SHARED_DATASETS.md`](SHARED_DATASETS.md)。仓库内只跟踪数据 manifest、split、checksum、小型汇总和最终图。

新的模型实现、方向特定实验和逐轮研究日志必须进入对应独立仓库。

## 2. 活跃仓库 registry

| Track | Local folder | GitHub | 当前冻结 commit | 当前状态 |
|---|---|---|---|---|
| 1. Operator-prior tensor | `/home/ubuntu/project/operator-prior-tensor` | <https://github.com/xuangu-fang/operator-prior-tensor> | `277b9d5` | **条件 GO**：Green confirmation 保留；group-wise joint-operator 扩展待执行 |
| 2. Wavefield low-rank representation | `/home/ubuntu/project/wavefield-low-rank-representation` | <https://github.com/xuangu-fang/wavefield-low-rank-representation> | `3412b00` | **长期探索重启**：先研究复杂复波场表征；旧 phase-CP 负结果保留，不设近期投稿 gate |
| 3. Operator-spectral FunBaT | `/home/ubuntu/project/operator-spectral-funbat` | <https://github.com/xuangu-fang/operator-spectral-funbat> | `8a8b184` | **条件 GO**：各向异性扩散主线 + fixed generic support floor |
| 4. Functional operator completion | `/home/ubuntu/project/functional-operator-completion` | <https://github.com/xuangu-fang/functional-operator-completion> | `13614f0` | **暂停**：Domain-Heat 条件正信号保留；domain-transport 与 active campaign completion 进入未来探索池 |

迁移规则：中心仓库保留迁移前快照，不从子仓库机械反向复制全部代码。Hub 只引用已推送 commit、摘要表和必要的共享协议修订。

## 3. 统一研究问题

四个方向都关心“物理信息如何进入可压缩/连续表示”，但入口和成熟度不同：

| Track | 物理信息入口 | 主要统计收益 | 首要 failure mode |
|---|---|---|---|
| 1 | 已知 operator basis 与 spectral shrinkage | 极稀疏时降低 factor variance | basis mismatch 产生不可消除 bias |
| 2 | phase carrier、travel time、multipath/plane-wave atoms | 对齐快速振荡后暴露慢变或低秩 residual | 单路径 prior 无法解释反射/散射，稀疏跨几何过拟合 |
| 3 | GP covariance / operator-induced spectrum | mode-wise function prior 与 UQ | kernel 不可识别或分离近似失真 |
| 4 | MIONet + 域内多尺度 heat transport features | 未见孔洞拓扑下改善稀疏场恢复 | 只对当前 diffusion 生成器有效，或 wave 任务完全失败 |

旧方向 2 的 phase-factorized wave tensor 仍为 STOP/DOWNGRADE；它的结果保留在 `papers/four_tracks/`。新方向 2 不是撤销该负结论，而是把问题上移到“复杂复数波场如何表征”：三角载波只是底层原语，未来允许 multipath、局部 wave atoms、生成式 residual 和跨任务预训练。该方向暂不与 Track 1/3 竞争投稿优先级。

### Track 2：长期 representation-first 重启

- 一句话目标：先显式消除可解释的快速相位传播，再让低秩或生成模型学习慢变、多路径和不确定残差。
- APEX (`arXiv:2605.26732`) 的幅度/相位非对称性与简化 phase-prior + flow-matching 结果构成直接动机。
- 初始仓库只提供 phase embedding、paired carriers、complex demodulation 和 rank diagnostics；oracle demodulation 仅作表征上限，不称预测结果。
- 第一阶段应做 `frequency × path complexity × phase estimator` 的 representation phase diagram，再选择 sparse completion、cross-frequency 或 operator task 中的一个进入模型比较。
- 大规模数据继续放在 `/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/`；本机 OpenFWI 子集位于共享 `ai-physical-dynamics/datasets/openfwi_curvefault_a/`。

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

| Family | Track 1 | Track 2 | Track 3 | Track 4 |
|---|---|---|---|---|
| CP/Tucker | operator CP/Tucker、discrete CP/Tucker | raw complex、幅相分离、解调 residual CP/Tucker/TT | functional GP CP/Tucker | functional CP/Tucker |
| Neural continuous | neural functional CP/Tucker、SIREN | SIREN/joint INR、plane-wave/local wave atoms | neural mean + GP residual、FunBaT | joint INR、concat DeepONet |
| Operator learning | 只作外部压力测试 | 按任务使用 FNO/MIONet/GINO；不与 sparse completion 混表 | 可作 neural mean | MIONet 为第一强 baseline，FNO/GINO 次之 |
| Kernel/GP / classic ROM | flat product GP | SVD/POD、shifted/transported POD、wrong/oracle carrier | global dictionary、per-mode、oracle/swap | GP/RBF 只作插值或 few-shot control |

## 5. 本轮决策与跨方向优先级

### Track 3：条件 GO，优先进入外部 PDE gate

- fresh seeds 201--205、2% 观测下，各向异性扩散的 operator per-mode/rank NRMSE 为 `0.1183±0.0582`，generic per-mode/rank 为 `0.1567±0.0990`，5/5 paired wins，并匹配 oracle 均值。
- 相对 operator-global 只有 3/5 wins 和约 2.4% 均值提升，所以 per-rank routing 不单独作为贡献。
- fixed 25% generic support floor 在 reference/shifted/anisotropic strict mismatch 中分别把 `0.672/0.632/0.615` 降到 `0.040/0.085/0.130`，全部 5/5；matched anisotropic 代价约 `+0.012`。
- full signed rank-4 分离误差在 advection 约为 `0.18`，anisotropic diffusion 为 `0.0043`。主线限定为 even/axis-separable operator spectra；输运降为 limitation。
- 下一门槛：不由相同 finite atoms 生成的 PDE solutions、structured sensors 和 FunBaT/RR-FBTC/neural functional tensor 公平对比。

### Track 4：保留证据，当前暂停

- 旧 Functional Tucker replacement 结果不变：公平增强 coordinate trunk 后 MIONet 明显领先，停止堆叠 Tucker 组件。
- 新任务保留强 Spectral MIONet，并加入 source-conditioned multiscale domain heat features，处理未见双孔域和每场 1%--10% 输出标签。
- fresh seeds 下 1% 标签的 Spectral/SDF/Domain-Heat MIONet NRMSE 为 `0.219/0.231/0.146`；孔边界误差为 `0.362/—/0.188`。2%/5%/10% 时 Domain-Heat 为 `0.118/0.125/0.112`。
- geodesic-wave stress 中所有方法约 `1.02--1.09`，因此只定位于 screened/diffusive operator，不宣称通用优势。
- 两个未来候选已写入独立仓库但不启动：其一是 sparse neural operator 的 domain-transport coordinates；其二是 neural operator mean + Bayesian low-rank residual 的 active simulation-campaign completion。
- 若未来重启，二者只能选择一个主故事；前者以跨拓扑传播坐标为核心，后者以节约 simulator calls 为核心。

### Track 1：冻结确认后条件 GO

- cutoff 8、rank `(4,5,5)`、400 steps、fresh seeds 101--105。
- 10% random 下 Operator/宽 Neural Tucker 为 `0.1645/0.2065`，4/5 wins；receiver-fiber 为 `0.2165/0.2695`，4/5 wins。
- source-fiber 为 `0.2937/0.2562`，只有 3/5；2% structured masks 明显失败。
- 参数匹配控制为 212 对 210 参数，wrong operator 约 `0.94--0.96`。论文主线是可测量的 bias--variance phase boundary，不是无条件极稀疏优势。
- 方法扩展：operator 绑定其真实 coordinate group；联合 $(x,y)$ operator 不再默认拆成每轴一条 PDE，未知 groups 使用 neural factors。
- 下一门槛：先做规则二维 joint operator vs per-axis approximation，并以 operator separability/subspace residual 形成相图；通过后再进入不规则边界/孔洞，最后才消费外部 PDE。

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
