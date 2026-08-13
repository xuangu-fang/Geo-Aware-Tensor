# The Well 64/16/32 pilot 提取报告

## 结果

固定 subset 已完整提取并晋级为 **PILOT 数据**：64 train、16 validation、
32 test，共 112 个 trajectory。每个文件包含 202×64×64 pressure、静态 density、
speed of sound、坐标和时间，合计 374,816,800 bytes（约 357 MiB）。空间使用
4×4 block mean 抗混叠下采样。数据 revision、
原 shard 与 trajectory index 均随文件保存；逐文件 SHA-256 和集合 manifest hash
记录在 `the_well_pilot_extraction_summary.json`。
修正版从第一个 case 原子落盘到最后一个 case 的文件时间跨度约 418 秒；完整复核无需
重新下载，约 7 秒完成。

## Loader 与泄漏边界

`load_the_well_case` 返回两个物理分离对象：inputs 只含 density、speed of sound、
`pressure(t=0)`、坐标和未来 query time；targets 只含 `pressure(t>0)`。固定随机 mask
测试确认 1% observation ratio 精确且 seed 可复现，metric 只在 held-out entry 计算。
三个 split 严格对应三个官方 shard，未跨 split 复用 trajectory。

## 单 seed sanity（不是模型选择）

在 32 个 test trajectory、seed 0、1% random observation 上：

- zero predictor NRMSE：1.000；
- persistence（复制 `t=0`）NRMSE：1.989；
- observed-value mean NRMSE：0.991；
- 实际 observation ratio：0.01000005。

这些结果只验证 metric 方向和任务非平凡性。它们不能作为 Paper A/B 的性能
baseline，也不能用于挑选方法超参数。

## 数据分布信号

trajectory pressure std 跨度较大：train 约 0.031–0.223，validation
0.063–0.152，test 0.035–0.242。后续必须用 train 统计做归一化，并同时报告
macro（逐 trajectory 平均）和 global NRMSE，避免高能量 case 支配结论。

## 下采样修正记录

首个完整提取曾采用 stride-4 点采样。Paper B 全量 train harness 随后发现
trajectory 55 的高压环恰好落在采样点之间，导致 64×64 的 `pressure(t=0)` 全零。
因此重开数据 issue，并在方法训练前把所有连续 field 统一改为 4×4 block mean。
修正后 112/112 initial pressure 均非零，abs max 范围 2.58–7.74；全部 case、
统计与 SHA-256 已重新生成。这个记录保留为门禁失败与修复证据。

## 下一门禁

仅允许一个 seed 检查 CP/INR/geometry-aware loader 与 GPU 显存；通过后使用 3 个
selection seed 在 1%、2%、5% observation ratio 上进入 Paper A/B pilot。test split
在配置冻结前不得参与选择。
