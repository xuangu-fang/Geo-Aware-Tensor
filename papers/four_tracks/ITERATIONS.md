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
