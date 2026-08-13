# The Well 64/16/32 pilot 提取报告

## 结果

固定 subset 已完整提取并晋级为 **PILOT 数据**：64 train、16 validation、
32 test，共 112 个 trajectory。每个文件包含 202×64×64 pressure、静态 density、
speed of sound、坐标和时间，合计 374,781,856 bytes（约 357 MiB）。数据 revision、
原 shard 与 trajectory index 均随文件保存；逐文件 SHA-256 和集合 manifest hash
记录在 `the_well_pilot_extraction_summary.json`。
从第一个 case 原子落盘到最后一个 case 的文件时间跨度约 598 秒；完整复核无需
重新下载，约 7 秒完成。

## Loader 与泄漏边界

`load_the_well_case` 返回两个物理分离对象：inputs 只含 density、speed of sound、
`pressure(t=0)`、坐标和未来 query time；targets 只含 `pressure(t>0)`。固定随机 mask
测试确认 1% observation ratio 精确且 seed 可复现，metric 只在 held-out entry 计算。
三个 split 严格对应三个官方 shard，未跨 split 复用 trajectory。

## 单 seed sanity（不是模型选择）

在 32 个 test trajectory、seed 0、1% random observation 上：

- zero predictor NRMSE：1.000；
- persistence（复制 `t=0`）NRMSE：2.186；
- observed-value mean NRMSE：0.992；
- 实际 observation ratio：0.01000005。

这些结果只验证 metric 方向和任务非平凡性。它们不能作为 Paper A/B 的性能
baseline，也不能用于挑选方法超参数。

## 数据分布信号

trajectory pressure std 跨度较大：train 约 0.037–0.241，validation
0.071–0.166，test 0.042–0.251。后续必须用 train 统计做归一化，并同时报告
macro（逐 trajectory 平均）和 global NRMSE，避免高能量 case 支配结论。

## 下一门禁

仅允许一个 seed 检查 CP/INR/geometry-aware loader 与 GPU 显存；通过后使用 3 个
selection seed 在 1%、2%、5% observation ratio 上进入 Paper A/B pilot。test split
在配置冻结前不得参与选择。
