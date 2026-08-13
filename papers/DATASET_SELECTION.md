# 下一轮数据集选择与准入标准

## 1. 数据准入标准

主实验数据必须满足至少四项：

1. 原始 field 能组织成有物理意义的多阶 tensor，而不是任意 reshape；
2. geometry/material/boundary/source 与 target 分开保存，可构造 correct/wrong/no-geometry 对照；
3. 能建立未见 geometry 或未见 mesh/resolution split；
4. license、下载方式和版本可复现；
5. target 不是由 learner 直接共享的解析公式生成；
6. 数据量能在本机完成 10-seed confirmation，或可定义公开的固定 subset。

不满足 2–3 的公共数据只能作为普通 completion stress test，不能支撑 geometry-aware claim。

## 2. 推荐数据证据链

### Tier 1：自建独立 wave/Helmholtz solver（两篇共享，P0）

建议建立二维多障碍传播数据：

- geometry：wall/door、圆/椭圆、多障碍、窄通道；
- material：均匀、两相、平滑随机介质；
- source：位置、频率、相位可变；
- output：time-domain pressure，或 multi-frequency complex field；
- resolution：至少 32²/48²/64²；
- metadata：mesh、mass/stiffness/operator、BC、source、material、distance、完整 field。

推荐 tensor 组织：

- Paper A：`source × frequency/time × spatial-mode-1 × spatial-mode-2`，或对空间做 geometry operator basis 后得到 `instance × time/frequency × spectral mode`；
- Paper B：`geometry × time/frequency × irregular spatial point`，保留 query coordinates。

生成器与 learner 独立模块，数据落盘后训练代码不得 import solver 内部函数。首版约 64 geometries × 4 sources × 32 times，优先保证可重复而非追求 TB 规模。

### Tier 2：The Well acoustic scattering / maze（P0 公共数据）

The Well 提供统一 HDF5/metadata API；acoustic scattering maze 包含 pressure、velocity、material density 和 sound speed。官方文字数据卡记为 201 个 256×256 时间步；我们固定 revision 的实际 HDF5 shape 为 202，2000 trajectories，Hub LFS 实际总量 319.6GB。它与本项目的“障碍/材料几何 + 波传播”最匹配。[官方 maze 数据卡](https://polymathic-ai.org/the_well/datasets/acoustic_scattering_maze/)，[数据集总览](https://polymathic-ai.org/the_well/datasets_overview/)。

首轮不下载全量：

- 固定 64 train / 16 validation / 32 test trajectories；
- 时间 stride 4，空间先用 64²，再做 128² evaluation；
- geometry/material mask 从 density/sound-speed field 构造；
- source 若不能从 metadata 唯一恢复，则 B 的 source-aligned 模型增加 source-estimation preprocessing，不能使用人工 oracle 标签。

准入 gate：随机抽取 8 trajectories，验证 geometry 确实变化、source/initial condition 可恢复、HDF5 字段和 license 足以发布 split manifest。若 maze geometry 在轨迹间不变化，则改用 `acoustic_scattering` inclusions 或将其降为跨初值而非跨几何实验。

当前 gate 已通过：固定 revision `8df383a...` 上的 8 个材料场 hash 全部不同，
初始压力可恢复 3–5 个源环组件；Hub Dataset Viewer 不支持该 HDF5，但 HTTP
range-read 可用，因此不需要先下载 47.9GB 的三个完整 shard。详细证据见
`papers/dataset_gates/THE_WELL_ACOUSTIC_GATE.md`。

### Tier 3：CFDBench geometry subsets（P1）

CFDBench 的 cavity/tube/dam/cylinder 都提供 boundary condition、geometry、physical-property 三类变化；插值版约 13.4GB，统一为 64×64，适合快速验证 unseen-geometry completion。[官方仓库](https://github.com/luo-yining/CFDBench)。

优点：下载和 baseline 适配成本低。局限：插值到规则网格会弱化 mesh geometry，且流动机制与 Paper B 的传播相位不完全匹配。因此：

- Paper A 可用于 operator construction 和 mask robustness；
- Paper B 只作为“不含明显 traveling-phase 对齐”的 scope/negative test；
- 不应以 CFDBench 的成功或失败单独决定 B 的主 claim。

### Tier 4：WaveBench（P1/P2）

WaveBench 是波传播 PDE 数据集合，公开压缩包约 75.6GB，并提供生成代码。[官方数据记录](https://zenodo.org/records/8015145)。它适合验证波传播 baseline，但在投入下载前必须审计每个子数据的 geometry/source metadata 和 license。

### Tier 5：PDEBench 与 RealPDEBench（补充）

- PDEBench 提供标准 PDE 数据、生成代码及 FNO/U-Net/PINN baseline，适合验证 pipeline 与标准 neural-operator baseline，但多数任务不是 varying geometry 主问题。[官方仓库](https://github.com/pdebench/PDEBench)。
- RealPDEBench 提供真实/模拟配对物理数据，适合作为现实域偏移压力测试；若具体 scenario 不提供可变化 geometry，就不用于核心 geometry claim。[官方仓库](https://github.com/AI4Science-WestlakeU/RealPDEBench)。

## 3. 最终推荐

下一轮先并行准备但顺序执行：

1. **自建独立 wave/Helmholtz 小数据**：最快建立完全可控且无同源争议的证据；
2. **The Well acoustic maze subset gate**：最强公共主数据候选；
3. **CFDBench geometry subset**：低成本外部 stress test；
4. WaveBench/RealPDEBench 仅在前两者不足时扩展。

数据适配完成前不得启动大规模模型 sweep。先用一个 seed 验证字段、mask、operator 和 metric，再进入 3-seed pilot。
