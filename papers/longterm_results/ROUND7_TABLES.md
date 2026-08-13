# 第七轮：发表导向证据表

## Paper A：固定方法的压力测试

| Setting | Geo-BTucker | Geo-CP | Flat GP | Wrong Tucker |
|---|---:|---:|---:|---:|
| 2% + center block missing | 0.174±0.074 | 0.515±0.103 | 0.631±0.030 | 1.640±0.119 |
| 2% + 30% obs. noise | 0.487±0.091 | 0.969±0.134 | 0.661±0.025 | 1.631±0.111 |

## Paper A：主动采样负结果

| Correct core-IV | Wrong core-IV | Random |
|---:|---:|---:|
| 0.206±0.014 | 0.330±0.256 | 0.137±0.010 |

## Paper B：The Well early-40 绝对有效性门禁（REJECTED）

| Method | Test macro NRMSE | Paired wins |
|---|---:|---:|
| Paired phase CP | 0.99175±0.00610 | — |
| Official FNO 2.0 | 0.99808±0.00301 | 8/10 |
| The Well U-Net | 1.00174±0.00208 | 9/10 |
| time-scaled persistence | 1.00160±0.00276 | descriptive |
| zero | 1.00640 | descriptive |

paired approximate explained variance: 1.64%.
MSE skill vs zero: 2.89%; vs persistence: 1.96%.
These absolute effects fail the paper gate; pairwise p-values are not treated as positive evidence.

完整逐 seed 数值和 paired tests 见 `round7_summary.json`。
