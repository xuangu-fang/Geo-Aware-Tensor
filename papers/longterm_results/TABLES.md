# 三轮持续迭代统计

## Paper A 新 seed 确认

| 观测率 | Geo-BTucker | Geo-CP | Flat GP | Wrong BTucker | Discrete BTucker |
|---:|---:|---:|---:|---:|---:|
| 1% | 0.676±0.072 | 0.809±0.072 | 0.752±0.030 | 1.462±0.097 | — |
| 2% | 0.125±0.015 | 0.367±0.055 | 0.612±0.028 | 1.712±0.201 | 1.976±0.131 |

## Paper B phase-envelope selection

| 版本 | Envelope | Paired CP | IP-NF | 结论 |
|---|---:|---:|---:|---|
| learned Q=2 | 0.644±0.118 | 0.897±0.033 | 0.624±0.027 | 不进入主方法 |
| fixed-RBF Tucker | 0.684±0.088 | 0.897±0.033 | 0.624±0.027 | 不进入主方法 |

## Paper A phase diagram（两 seed exploratory）

| observation | mismatch | Geo-BTucker | Geo-CP | Flat GP |
|---:|---:|---:|---:|---:|
| 0.01 | 0 | 0.389 | 0.505 | 0.540 |
| 0.01 | 0.25 | 0.648 | 0.540 | 0.586 |
| 0.01 | 0.5 | 0.831 | 0.667 | 0.691 |
| 0.01 | 0.75 | 0.922 | 0.768 | 0.750 |
| 0.01 | 1 | 0.650 | 0.803 | 0.755 |
| 0.02 | 0 | 0.175 | 0.234 | 0.389 |
| 0.02 | 0.25 | 0.437 | 0.347 | 0.423 |
| 0.02 | 0.5 | 0.459 | 0.467 | 0.525 |
| 0.02 | 0.75 | 0.386 | 0.445 | 0.592 |
| 0.02 | 1 | 0.126 | 0.353 | 0.604 |

## Paper B phase diagram（两 seed exploratory）

| observation | mismatch | Paired CP | Envelope CP | Geo-Tucker | IP-NF |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 0 | 0.147 | 0.145 | 0.137 | 0.248 |
| 0.005 | 0.5 | 0.868 | 0.867 | 0.827 | 0.631 |
| 0.005 | 1 | 1.177 | 1.091 | 1.048 | 0.878 |
| 0.01 | 0 | 0.131 | 0.130 | 0.111 | 0.203 |
| 0.01 | 0.5 | 0.824 | 0.822 | 0.759 | 0.560 |
| 0.01 | 1 | 1.183 | 1.010 | 0.909 | 0.815 |
| 0.02 | 0 | 0.082 | 0.084 | 0.087 | 0.124 |
| 0.02 | 0.5 | 0.686 | 0.442 | 0.513 | 0.462 |
| 0.02 | 1 | 0.835 | 0.574 | 0.696 | 0.609 |

完整 paired statistics 与逐 seed 数值见 `summary.json`。

## Paper B 官方 FNO baseline（The Well early-40 / 1%）

| 方法 | 32 test geometry macro NRMSE | 胜场 | 参数量 |
|---|---:|---:|---:|
| Paired phase CP | **0.99175±0.00610** | 8/10 | 23,040 |
| Official NeuralOperator 2.0 FNO | 0.99808±0.00301 | 2/10 | 357,473 |

paired 相对改善 `0.63%`；单侧 paired Wilcoxon `p=0.01367`。FNO 架构先在
validation 上从 FNO/TFNO 中选择，再固定到已有 confirmation seeds 10--19；逐 seed
结果、compute 与图见 `the_well_official_fno_confirmation.json/.png`。

## 不规则边界分支最终 NO-GO

| Gate | Proposed/operator 方法 | 最强简单 baseline | 决策 |
|---|---:|---:|---|
| Paper A elliptic 1% | operator CP 0.668 | coordinate CP **0.180** | 附录 |
| Paper B unseen-domain elliptic | operator-spectral CP 0.278 | SDF CP **0.181** | 附录 |

correct operator 均优于 wrong/bbox operator，但没有优于普通 coordinate/SDF functional
CP，因此不再投入 boundary-specific method tuning。
