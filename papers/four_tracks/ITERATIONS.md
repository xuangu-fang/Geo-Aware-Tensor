# 新方向快速 POC 迭代记录

## R0：接口与训练 smoke

- 设置：1% entry observations，4 个 r24 训练形状，r32 slanted-channel validation，200 steps。
- 结果：domain-kernel Tucker 0.339，SDF-Tucker 0.285，joint INR 0.261。
- 判断：实现可运行，但所有方法明显未收敛；本轮只作代码检查，不作方法结论。

原始记录：`results/new_tracks_validation_smoke_seed0.json`。

## R1：纯 domain-kernel section

- 改动：训练增至预设的 900 steps；方向 3 只使用 \(k_\Omega(x,s)\) 多尺度 kernel sections。
- 结果：domain-kernel Tucker 0.262，topology-erased kernel Tucker 0.255；正确几何反而更差。SDF-Tucker、SDF-CP、joint INR 分别为 0.176、0.171、0.159。
- 反思：单独的 source-to-query 平滑 covariance 丢掉局部坐标、边界距离与短程变化。这个负结果不是“GP 无效”，而是当前 kernel specification 不完整。
- 决策：采用标准 additive/composite kernel 思路；让正确核和 topology-erased 消融共享完全相同的坐标/SDF局部部分，只替换域内核部分。

原始记录：`results/new_tracks_validation_seed0.json`。

## R2：局部核 + 域内核的 composite POC

- 改动：kernel factor 输入改为 `[domain Matérn sections, coordinates, SDF, source, Euclidean distance]`；Tucker core 与训练预算不变。
- seed 0 validation：domain-kernel Tucker 从 0.262 降到 0.179，且优于 topology-erased 的 0.188；边界为 0.184 vs 0.207。
- 冻结：在读取 hole test 前固定 ranks、kernel scales、steps 与所有 baseline。
- 3-seed validation：correct kernel 在 3/3 seeds 优于 topology-erased；全域 0.184±0.012 vs 0.196±0.019，边界 0.185±0.012 vs 0.209±0.009。
- 3-seed hole test：domain-kernel Tucker 全域 0.153±0.015、边界 0.167±0.017，是当前方法中平均最好；topology-erased 分别为 0.173±0.014、0.200±0.019。

原始记录：`results/new_tracks_validation_composite_seed{0,1,2}.json` 与 `results/new_tracks_hole_test_seed{0,1,2}.json`。

## 对方向 4 的同步判断

- SDF neural functional Tucker 已经能在未见网格与未见形状上重建，NRMSE 远低于无效门槛 1。
- 在 hole test 上，SDF-Tucker 平均优于 coordinate-Tucker，尤其是边界带；说明 SDF 几何输入有效。
- 但 SDF-Tucker 目前平均不如 SDF-CP 与 joint INR，不能把“非对角 Tucker core”写成已验证贡献。
- 下一轮只尝试一个最小修正：用 CP-diagonal core 初始化 Tucker，使 Tucker 严格包含一个已可训练的 CP 起点；不能在已读取的 hole case 上选择该改动，需新增 validation/test geometries。

## 当前状态

| 方向 | 状态 | 可以说什么 | 不能说什么 |
|---|---|---|---|
| 3. Domain-kernel Functional Tucker | PILOT positive | 域核在孔洞/边界上提供稳定增益 | 已完成 full Bayesian GP、已具论文级统计 |
| 4. Geometry-conditioned Neural Functional Tucker | POC mixed | SDF 条件因子能使用孔洞几何 | Tucker 已优于 CP/INR、已具主会结果 |

## R3：四方向独立技术审计（2026-08-15）

本轮不把旧 hole 结果继续升格，而是重新审计任务、优化、baseline 和命名。详细结果见 `tech_reports/`。

### 共享协议修正

- 旧 runner 用单个随机 minibatch loss 选 checkpoint；现改为定期在全部 fixed observed entries 上评估。
- 当前 1% 是训练域 target-label fraction，验证域没有 target context；任务是 sparse-supervision zero-shot surrogate，不是 test-domain 1% completion。
- 当前 `boundary_distance` 是域内无符号正距离，不是完整 ambient SDF。
- 旧 hole case 已被多次读取，之后只能作 development evidence。

### 方向 1

- 修复 unit-RMS forward factor 与 raw-coefficient spectral penalty 不一致的 scale loophole。
- 修正后 2% smoke 仍为正：operator Tucker `0.155`，operator CP `0.410`，flat GP `0.597`，wrong `1.489`。
- neural functional Tucker 需要更长优化：500-step `0.523`，2000-step `0.307`。旧确认表需重跑。

### 方向 2

- independent-wave locked validation 使用 5 train geometries×2 sources@24、1 unseen validation geometry@32、2222 个训练标签，test 未读。
- mean `1.0002`，joint INR `1.4820`，ordinary CP `3.2679`，wrong phase `2.8920`，paired phase `3.4732`。所有 gate 失败，且 wrong 在 3/3 seeds 超过 correct。
- 决策：PAUSE / NARROW GO；不读 locked test，不晋级 WaveBench。

### 方向 3

- 当前是 kernel-section neural input，没有 GP prior/posterior，不得称 GP-MAP。
- 参数匹配的 intrinsic-only 为 `0.2602±0.0055`，Euclidean-RBF-only `0.3320±0.0212`；intrinsic+local 为 `0.1905±0.0219`，Euclidean+local `0.2031±0.0297`。
- 机制信号为正，但下一个里程碑是显式 KRR/GP 与 FunBaT baseline，不是直接加变分组件。

### 方向 4

- 完整-observed checkpoint 修复后，CP-shaped-core Tucker `0.1922±0.0305`，functional CP `0.1958±0.0312`，joint INR `0.1723±0.0175`。
- Tucker 只赢 CP 1/3 seeds；CP-shaped core 只是代数初值，不是 trained-CP warm start。
- 决策：新生成 80–200 shapes 与冻结多孔洞 test；只允许真实 CP warm start + off-diagonal residual 这一个最小方法修正。

## R4：方向 1 固定预算公平主表（2026-08-15）

- 按新协议锁为 3 个 validation seeds、500 steps、随机冷启动；所有方法共享 observation mask、噪声与 observed-only normalization。
- 比较 Operator Tucker/CP、Neural Functional Tucker/CP 与 SIREN；参数量依次为 247、320、8,178、8,872、19,105。
- 2% random：Operator Tucker `0.2582±0.0718`，明显优于 Neural F-Tucker `0.5704±0.0625` 与 SIREN `0.8358±0.0080`。
- 2% periodic gap：Operator Tucker `0.5975±0.1298`，优势存在但明显缩小；Neural F-CP 为 `0.6889±0.0507`。
- 1% random：Operator Tucker `1.2147±0.3165`，弱于 Neural F-Tucker `0.8984±0.1035`；当前不能宣称观测越稀疏优势越大。
- flat-GP/HOSVD 初始化把 Operator Tucker 的 1% random 改善到 `0.8587±0.1118`，证明旧正结果包含不可忽略的初始化收益。它被保留为消融，不混入 cold-start 主表。
- 在 35% dense-Tucker + 65% CP/local-residual 的部分失配 truth 上，Operator Tucker `0.4517±0.0461`，Neural F-Tucker `0.4562±0.0198`；只赢 2/3 seeds，平均差约 1%，优势不显著。
- 决策：方向 1 继续，但论文故事暂时收窄为“operator factor space 在达到可识别阈值后降低样本复杂度”；下一轮优先换 non-aligned truth，而不是继续在 exact spectral truth 上调参。

原始记录：`results/track1_fixed_budget_validation_r2/results.json` 与
`results/track1_initializer_ablation_validation_r2/results.json`、
`results/track1_mixed_validation_r2/results.json`。

## R5：3--5 seeds / 300--500 steps 早筛轮（2026-08-15）

### 方向 2：clean traveling harmonic 最终判定

- learner-free generator 使用真正的 `travel time - time` characteristic，频率不进入模型初始化。
- 3 seeds×500 steps：zero `1.0000`，joint INR `1.2753±0.0203`，correct paired phase `1.4859±0.0806`，ordinary CP `2.4778±0.1679`，wrong Euclidean phase `2.5336±0.0849`。
- correct geometry 虽优于 wrong geometry，但仍未学到绝对有效预测；决策为 **STOP / DOWNGRADE**。

### 方向 3：ELBO+SGD 与 GP residual

- 实现 full-covariance finite-feature variational GP：显式 prior、`q(u)`、KL、Gaussian ELL、mini-batch ELBO 和 posterior variance。
- pure GP 没有几何优势：intrinsic `0.3282±0.0101`，Euclidean `0.3216±0.0148`。
- 共享 neural CP mean 后，mean-only `0.2036±0.0169`，intrinsic GP residual `0.1765±0.0205`，Euclidean residual `0.2280±0.0148`；intrinsic 在 3/3 seeds 赢 mean-only。
- coverage95 为 `0.9329`，且只有一个未见 validation geometry；因此是条件 GO，不是最终确认。

### 方向 4：Geometry-NO × CP

- 新数据为 48 train 0/1-hole、8 ID validation、8 two-hole OOD validation；8 个 test specs 冻结未读。
- 3 seeds×400 steps、1% labels、所有模型共享同一 case schedule：coordinate/SDF CP `0.2480±0.0047 / 0.2553±0.0049`（ID/OOD）；最佳 unmasked geometry-NO-CP `0.2840±0.0065 / 0.2958±0.0029`；masked geometry-NO-CP `0.3437±0.0186 / 0.3669±0.0130`；同 encoder dense head `0.7296±0.0300 / 0.7738±0.0218`。
- NO observed loss 更低但 validation 更差，属于明确稀疏过拟合；一次小门控 NO residual 修正仍未超过 CP，停止继续加深。
- 当前可保留的结论是 geometry-conditioned low-rank CP 有效；FNO 融合只是可复现实验接口和负信号。

### 方向 1：2%--10% observation-ratio phase curve

- 锁定 3 seeds、500 steps、random cold start，只使用 2%/5%/10%；五个模型共享 mask、noise、observed-only normalization 和预算。
- aligned truth：Operator Tucker 从 `0.2582±0.0718`（2%）降至 `0.0765±0.0142`（10%），作为实现 sanity 为正。
- 35% format/local mismatch：2% 时 Operator Tucker `0.4517±0.0461` 与 Neural F-Tucker `0.4546±0.0183` 基本持平；5%/10% 最佳改为 Operator CP `0.2723±0.0254 / 0.2160±0.0115`。实用识别阈值约为 5%，贡献应落在 operator factor space，不应绑死 Tucker core。
- 强 non-aligned failure control：5%/10% Neural F-CP `0.6711±0.0406 / 0.4889±0.0239`，明显优于 Operator Tucker `0.9715±0.0549 / 0.8224±0.0216`。这确认 operator approximation bias 的边界，不包装成正结果。
- SIREN 在 mixed truth 上 observed NRMSE 近零但 held-out 仍为 `0.872/0.705/0.526`，而 non-aligned Operator Tucker 连 observed 都无法拟合；前者是 sparse memorization，后者是 basis misspecification。
- 下一步不再争论“越稀疏越强”，而做 operator mismatch strength × observation ratio 的二维 phase diagram。

## R6：kernel dictionary 与方向 4 收口（2026-08-15）

### 方向 3：geometry-kernel dictionary + ELBO selection

- 统一 1% observations、3 seeds、400 steps、3 train geometries、2 unseen validation geometries；1 个 hole test 冻结未读。
- kernel dictionary 含 Matérn/resolvent、heat/diffusion、graph-geodesic RBF 与 Euclidean RBF。PSD mixture 使用 `[sqrt(w_q) Phi_q]`，非负权重与 full-covariance `q(u)` 由 mini-batch ELBO+SGD 联合学习。
- matched heat-GP sanity：mixture `0.0725±0.0046`，单 heat `0.0741`，learned heat weight 平均最高 `0.519`。
- perturbed near-match：mixture `0.1432±0.0096`，优于单 heat `0.1914`，说明 evidence selection 不只在 exact match 下有效。
- screened elliptic mismatch：pure mixture `0.3116±0.0055`，Euclidean `0.3251`；neural mean-only `0.2073±0.0105`，+heat residual `0.1985±0.0267`，+mixture residual `0.2054±0.0275`。hybrid 收益跨 seeds 不稳定。
- 决策：保留“几何 kernel dictionary + variational evidence selection”作为方法主线；matched/near-match 明确标 sanity，不能声称任意 PDE 上超过 neural baseline。

### 方向 4：boundary operator 与 rank modulation 均 NO-GO

- boundary-integral CP seed0：local CP ID/OOD `0.2538/0.2614`，正确 integral `0.2577/0.2546`；但去掉 hole tokens `0.2474/0.2545`、反转 boundary type `0.2569/0.2525`，几何因果不成立。no-SDF sanity 再次失败。
- fallback Boundary-DeepSets rank gate 在 3 seeds 下为 ID/OOD `0.2533±0.0086 / 0.2550±0.0020`；local CP 为 `0.2480±0.0058 / 0.2553±0.0060`。DeepSets 仅 1/3 seeds 胜 local，正确 boundary 仅 2/3 胜 wrong boundary。
- Boundary-Augmented NO、BI-GreenNet 等已有工作也意味着“使用边界积分”本身不是新意。
- 决策：方向 4 正式收口为 geometry-coordinate/SDF functional CP；FNO、boundary integral 和 rank modulation 只保留为可复现负消融，不再增加组件。
