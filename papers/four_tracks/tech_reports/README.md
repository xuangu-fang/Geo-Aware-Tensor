# 四方向共享技术报告目录

本目录是四条研究线的技术单一事实源。每份报告必须回答同一组问题，避免“方法名越来越多，但无法判断实现、数据和 baseline 是否正确”。

## 报告索引

| 方向 | 技术报告 | 当前责任边界 |
|---|---|---|
| 1. Operator-informed Bayesian Tucker | [`TRACK1_OPERATOR_TUCKER.md`](TRACK1_OPERATOR_TUCKER.md) | 固定离散张量、算子谱先验、conditional Bayesian core |
| 2. Phase-factorized Wave Tensor | [`TRACK2_PHASE_WAVE.md`](TRACK2_PHASE_WAVE.md) | 波场相位恒等式、传播距离、专项 wave benchmark |
| 3. Domain-kernel GP Functional Tucker | [`TRACK3_DOMAIN_KERNEL_GP.md`](TRACK3_DOMAIN_KERNEL_GP.md) | 非欧域 kernel、有限特征 POC、未来 GP posterior |
| 4. Geometry-conditioned Neural Functional Tucker | [`TRACK4_NEURAL_FUNCTIONAL_TUCKER.md`](TRACK4_NEURAL_FUNCTIONAL_TUCKER.md) | SDF/边界距离条件的连续因子、显式 Tucker/CP |

共享实验审计规则见 [`SHARED_AUDIT_PROTOCOL.md`](SHARED_AUDIT_PROTOCOL.md)。四份报告可以有不同方法，但不得使用不同的指标定义、数据泄漏标准或 baseline 命名口径。

## 每份报告的强制结构

1. 一句话研究问题与可证伪 claim；
2. 完整 formulation、符号和几何进入位置；
3. 数学对象到代码类/函数的逐项映射；
4. optimization 与 inference：哪些是 MAP、哪些有 posterior、哪些只是启发式；
5. dataset cards：生成方程、几何族、样本数、split、mask、能验证什么、不能验证什么；
6. baseline cards：每个 baseline 回答的因果问题、输入预算、参数/计算公平性；
7. tests：单元测试、数据审计、机制消融、统计确认；
8. 已有正负证据，明确 exploratory/confirmatory 状态；
9. 最小下一轮实验矩阵、GO/NO-GO 门槛；
10. 论文定位与不能使用的宣传口径。

## 当前最重要的共享结论

- “1% observation”必须说明是**训练域完整张量的 entry subsampling**，还是**测试域 few-shot observations**。当前新方向 POC 是前者，并且对测试域零观测；它更接近稀疏监督下的跨几何 surrogate，而不是传统 transductive tensor completion。
- 当前 irregular elliptic 数据只有 6 个手工几何族、其中 1 个 validation、1 个 hole test。它足够做机制 POC，不足以证明 geometry generalization。
- 当前 `boundary_distance` 是活动域内部到最近外边界或孔洞边界的正距离；它不是含域外符号的完整 SDF。报告和模型名应区分“interior boundary distance”与真正 SDF。
- 方向 3 当前是 kernel-feature neural POC，不是完整 GP，也不是严格 GP-MAP；只有显式 GP prior/coefficients 和 posterior 后才能使用 Bayesian GP 主张。
- 任何方法若 NRMSE 接近 1，即使 paired p-value 显著也视为任务整体无效。外部任务先过绝对效果门槛，再讨论相对提升。

## 你最关心的“数据和 baseline 到底选对了没有”

| 方向 | 当前数据判断 | 当前 baseline 判断 | 结论 |
|---|---|---|---|
| 1 | aligned operator Tucker 只适合 sanity；irregular elliptic/The Well 没有外部正信号 | operator CP/flat GP/wrong operator 正确；原先缺 neural functional CP/Tucker，且其收敛预算被低估 | 方向可保留，但旧主表是 inverse-crime 机制证据，必须重跑修正后先验并增加公开数据 |
| 2 | band-aligned harmonic 只是 sanity；independent wave 和 The Well 都明确失败 | ordinary CP/joint INR/wrong phase/trivial gate 已配齐；WaveBench 官方 FNO/U-Net 只能在 operator protocol 中比 | 当前暂停；不应继续扩大波场应用或堆 envelope |
| 3 | irregular elliptic 适合 intrinsic-vs-Euclidean 机制测试，但只有 1 validation shape，hole test 已读 | 新 intrinsic/Euclidean section 对照正确；旧 `topology_erased` 命名过强；真正 KRR/GP、FunBaT 尚缺 | 只证明 intrinsic feature 有机制信号，尚未证明 GP/Bayesian 方法 |
| 4 | 6-shape elliptic 适合实现 POC，不足以学 topology distribution；边界距离又是 generator 显式变量 | same-input CP 和 joint INR 选得对；`coordinate` 仍读全局几何，不是 no-geometry；下一级应先加 F-INR/CORAL，再加 GINO | Tucker 尚未超过 CP/INR；先扩形状和修正 geometry ablation，不可先加复杂 encoder |
