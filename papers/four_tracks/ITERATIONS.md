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

