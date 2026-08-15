# 四方向共享技术报告目录

本目录是四条研究线的技术单一事实源。每份报告必须回答同一组问题，避免“方法名越来越多，但无法判断实现、数据和 baseline 是否正确”。

## 报告索引

| 方向 | 技术报告 | 当前责任边界 |
|---|---|---|
| 1. Operator-informed Bayesian Tensor | [`TRACK1_OPERATOR_TUCKER.md`](TRACK1_OPERATOR_TUCKER.md) | operator factor space、ratio×mismatch phase diagram、CP/Tucker decoder |
| 2. Phase-factorized Wave Tensor | [`TRACK2_PHASE_WAVE.md`](TRACK2_PHASE_WAVE.md) | 波场相位恒等式、传播距离、专项 wave benchmark |
| 3. Domain-kernel GP Functional Tensor | [`TRACK3_DOMAIN_KERNEL_GP.md`](TRACK3_DOMAIN_KERNEL_GP.md) | geometry-kernel dictionary、ELBO evidence selection、finite GP |
| 4. Geometry-coordinate Functional CP | [`TRACK4_NEURAL_FUNCTIONAL_TUCKER.md`](TRACK4_NEURAL_FUNCTIONAL_TUCKER.md) | coordinate/SDF/source continuous CP；NO/boundary modules 仅负消融 |

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
- 旧 irregular elliptic 数据只有 6 个手工几何族、其中 1 个 validation、1 个已读 hole test。方向 4 新协议已扩为 48 train、8 ID validation、8 双孔 topology-OOD validation，并冻结 8 个未读 test specs；这足够做早期多形状 POC，但仍不是公开外部证据。
- 当前 `boundary_distance` 是活动域内部到最近外边界或孔洞边界的正距离；它不是含域外符号的完整 SDF。报告和模型名应区分“interior boundary distance”与真正 SDF。
- 方向 3 已实现有限特征 variational GP：显式 Gaussian prior、full-covariance variational posterior、KL、Gaussian likelihood、mini-batch ELBO 与 posterior variance。准确名称是 finite-feature GP hybrid，尚不是完整 Bayesian functional Tucker。
- 任何方法若 NRMSE 接近 1，即使 paired p-value 显著也视为任务整体无效。外部任务先过绝对效果门槛，再讨论相对提升。

## 你最关心的“数据和 baseline 到底选对了没有”

| 方向 | 当前数据判断 | 当前 baseline 判断 | 结论 |
|---|---|---|---|
| 1 | aligned、35% mixed 与强 non-aligned 三层 truth 已形成 bias control，但仍是自建 tensor family | functional CP/Tucker、operator CP/Tucker 与 SIREN 共享 2/5/10% masks 和 500 steps | operator factor space 有清楚适用区；decoder 不应绑死 Tucker，强 mismatch 下 neural CP 胜 |
| 2 | clean traveling harmonic、independent wave、The Well 均未过绝对门槛 | correct/wrong travel time、ordinary CP、joint INR、zero gate 已齐 | **STOP / DOWNGRADE**；不再进入 WaveBench 或继续堆 phase 组件 |
| 3 | 新协议含 3 train、2 unseen validation、1 frozen hole test；matched/near-matched/elliptic 三层数据明确区分 sanity 与 mismatch | Matérn、heat、geodesic、Euclidean 和可学习 PSD mixture 均用相同 ELBO/q(u) | kernel evidence selection 在 sanity/near-match 跑通；elliptic neural residual 中性，不宣称通用胜利 |
| 4 | 48/8/8 随机多孔洞协议与未读 test 保留 | FNO、boundary integral、pooled/wrong-hole、DeepSets rank gate、descriptor gate 均已对照 | 所有 fancy geometry operator 均未稳定胜 local CP；正式收口 geometry-coordinate/SDF CP |
