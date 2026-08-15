# 方向 2 技术报告：Phase-factorized Wave Tensor

> 审计日期：2026-08-15  
> 当前结论：**STOP / DOWNGRADE。** independent reflected-wave R1 失败后，唯一允许的 clean R2 在真正 traveling harmonic 上仍未通过绝对 gate。保留三角恒等式作为受控机制或学生项目，不再占用主论文实验预算。

## 1. 一页结论

这条线真正简洁的想法只有一句话：波场如果主要由传播相位

\[
u(g,t,x)\approx A(g,t,x)\cos\{k[d_g(x,s)-ct]+\varphi\}
\]

决定，那么三角恒等式可以把联合的空间—时间相位拆成少量空间因子和时间因子的乘积：

\[
\cos(kd-kct)=\cos(kd)\cos(kct)+\sin(kd)\sin(kct).
\]

因此，一条看似沿着空间移动的波，在合适的内禀传播距离 `d_g` 下可能只有很低的 CP rank。几何只负责把普通欧氏距离换成遵守障碍物或介质速度的 travel time / geodesic distance；CP 只负责显式组合空间相位和时间相位。

目前有三个必须同时说清楚的事实：

1. 代码中的四项 carrier 的确忠实实现了角差恒等式，这次新增了精确单元测试；
2. 最强的 `0.0952±0.0144` 正结果来自**与模型频率字典对齐的驻波型合成数据**，不是对真正移动波包或多重散射的充分证明；
3. independent wave、真实不规则边界波和 The Well 都没有给出可发表的正证据；进一步的 clean R2 即使把数据收窄为严格的 `τ-t` traveling harmonic，paired phase 仍为 `1.486±0.081`，高于 zero baseline 的 `1.000`，三 seed 全部拒绝。

所以当前适合的论文故事是：**物理相位可分性给波场 functional CP 提供结构归纳偏置，以及这个归纳偏置何时成立、何时因移动包络和多路径散射失效。**

## 2. 任务到底是什么

### 2.1 当前代码实际解决的任务

当前主实验不是标准“给新样本少量传感器，再补全该样本”的 tensor completion，也不是标准“从完整输入场映射到完整输出场”的 neural operator。它实际是：

- 在若干训练几何、训练时间上，只读取每个目标场约 1%–2% 的标注点；
- 用这些稀疏标签训练一个跨几何共享的函数；
- 对没有参与训练的新几何和新网格做 zero-shot 场预测；
- 测试任务中生成的 `observed` mask 没有进入 phase CP 的前向或适配过程，只用其补集计算指标。

因此，报告中更准确的术语应为 **extreme sparse supervision under cross-geometry transfer**，而不是默认称为“测试场 1% observation completion”。如果以后要声称 few-shot completion，测试传感器必须显式进入 inference，并且所有 baseline 也得到同一批传感器。

### 2.2 合法的应用边界

它适合：声学、地震波、超声、Helmholtz、浅水线性波，以及具有单到少量主要传播到达路径的场。它不自动适合扩散、反应、一般流体或任意 PDE。即使都叫“物理场”，只有存在稳定相位坐标时，三角恒等式才提供结构优势。

## 3. Formulation

### 3.1 几何传播坐标

设几何/介质实例为 `g`，源位置为 `s`。理想坐标是最短 travel time：

\[
d_g(x,s)=\inf_{\gamma:s\to x}\int_\gamma \frac{1}{c_g(z)}\,\mathrm dz.
\]

在常速障碍域中，它退化为绕开障碍的最短路长度；在异质介质中，边权应为 `length / speed`。代码目前有两种实现：

- 受控障碍数据 `source_geodesic_distance` 使用流体网格上的无权 Dijkstra，即假设流体内常速；
- The Well adapter 使用局部声速倒数构造加权图，再做多源 Dijkstra。

这比欧氏距离更符合首达波，但它只表示**一个最短到达时间**，不表示反射、绕射、多个源的相干叠加或多条到达路径。

### 3.2 显式 paired CP

对频带 `k_b` 和候选速度 `c_j`，代码构造四个 CP 项：

\[
\begin{aligned}
C_1 &= \cos(k_bd)\cos(k_bc_jt),\\
C_2 &= \sin(k_bd)\sin(k_bc_jt),\\
C_3 &= \cos(k_bd)\sin(k_bc_jt),\\
C_4 &= \sin(k_bd)\cos(k_bc_jt).
\end{aligned}
\]

其中 `C1+C2` 精确等于 `cos(k_b(d-c_jt))`，`C4-C3` 精确等于 `sin(k_b(d-c_jt))`。四项而不是两项，是为了允许任意相位偏移和正弦/余弦组合。

完整模型可写为

\[
\hat u(g,t,x)=\sum_{b,j,q=1}^4
w_{bjq}\,G_{bjq}(e_g)\,
T_{bjq}(t)\,\tau_{bjq}(t)\,
X_{bjq}(x,d_g,\mathrm{SDF})\,\xi_{bjq}(d_g),
\]

其中 `ξ`、`τ` 是上面的固定空间/时间 carrier，`G,T,X` 是三个彼此独立的 MLP amplitude factor。没有联合 `(d,t)` residual 绕过 CP 收缩，因此它确实是显式 functional CP。

当前固定 `5 bands × 3 speeds × 4 trig terms = 60` 个分量。这个 rank 不是从数据自动选择，也没有 Bayesian inference。

### 3.3 Phase-envelope

移动波包还包含不可分的包络 `A(d-ct)`。当前两种小扩展分别是：

\[
A_b(d,t)=1+\sum_{r=1}^Q a_{br}E_r(d)H_r(t)
\]

和固定 RBF 距离/时间基配合小 Tucker core。它们保持显式张量结构，但 learned rank-2 envelope 虽将 moving-envelope NRMSE 从约 `0.897` 改善到 `0.644`，仍没有超过联合 IP-NF 的 `0.624`；固定 RBF Tucker 也为负。因此 envelope 可作为失效边界和小 ablation，不应升级为主模型。

## 4. 实现与优化审计

核心实现位于 [`src/geoaware/neural_tensor.py`](../../../src/geoaware/neural_tensor.py)：

- `SpeedAlignedPhaseCP`：主模型；
- `PhaseEnvelopeCP`：learned separable envelope；
- `PhaseEnvelopeTucker`：固定 RBF 包络基和小 core；
- `GeometryNeuralCP/Tucker`：普通 geometry-conditioned tensor baseline。

训练入口为 [`experiments/paper_b_tensor_run.py`](../../../experiments/paper_b_tensor_run.py)。训练是确定性经验风险最小化：

\[
\min_\theta \frac{1}{|\Omega_{train}|}\sum_{i\in\Omega_{train}}
\left(\frac{\hat u_i-u_i^{obs}}{s_{obs}}\right)^2
+10^{-7}\|\theta\|^2,
\]

使用 AdamW、gradient clipping，`s_obs` 只由观测训练值计算。这里的 “inference” 只是训练后前向查询，不输出 posterior 或置信区间。

本次审计确认的忠实部分：

- carrier reshape 顺序是 `(band, speed, trig-term)`；
- 四项能精确重建 traveling cosine 和 sine；
- correct/wrong phase 模型参数量一致；
- envelope 的距离网和时间网分离，没有隐藏 joint residual；
- 各模型构造前重置 seed，加入/删除无关 baseline 不改变其他模型初始化。

必须修正或至少披露的部分：

1. `geodesic_harmonic_field` 的主要信号是 `exp(-a t) cos(kd+φ)`，即驻波空间 harmonic 乘时间衰减；它没有 `d-ct` 的移动主相位。paired CP 在这里的优势可能主要来自固定 Fourier 空间字典，而不是 speed-aligned identity。
2. generator 使用 `[7,13,19]`，模型字典使用 `[7,13,19,27,37]`，存在明确 simulator–model band alignment。它适合 sanity check，不适合唯一 headline 数据。
3. 文档曾称 mask 为 “shared masks”，实际 `make_tasks` 根据 geometry/time 产生不同随机 mask。应统一改称 independent task-wise random masks。
4. 两套 phase carrier 分别存在于核心模型和 The Well harness，后者仍是复制实现，后续要合并以避免协议漂移。
5. 受控代码把源位置硬编码为 `(-0.72,-0.38)`；新 independent dataset 有两个源，当前主模型接口还未真正支持 source-conditioned transfer。

本次最小实现改进：新增纯函数 `paired_phase_carriers` 和精确恒等式测试；新增 [`phase_wave_protocol.py`](../../../src/geoaware/phase_wave_protocol.py) 将外部绝对有效性 gate 固化为代码；新增机器可读协议 [`track2_phase_wave_protocol.json`](../../../experiments/track2_phase_wave_protocol.json)。

## 5. 数据集审计

| 数据 | 是否独立于模型 | 几何/物理匹配 | 已有证据 | 审计结论 |
|---|---|---|---|---|
| eikonal harmonic control | generator 独立写出，但 band 与模型显式对齐 | 有 geodesic，主信号不是 traveling wave | 10 seeds，paired `0.0952` | 机制 sanity 为正；不能单独支撑论文 |
| moving-envelope control | 是 | 更接近 `A(d-ct)cos(k(d-ct))` | envelope `0.644`，IP-NF `0.624` | 诚实负边界；联合 INR 更合适 |
| independent wave smoke | 独立 finite-difference solver；不 import learner | 变速、反射、8 几何、2 源、24→32 | locked validation 3 seeds 已完成；所有 learned model 失败 | 绝对有效性和 correct/wrong attribution 均失败；test 保持未读 |
| clean traveling harmonic R2 | learner-free analytic generator；生成频率不进入模型 | 严格 `τ-t`、两个非对齐无理频率、2 源、24→32 | 3 seeds×500 steps；paired `1.486±0.081` | correct 明显优于 wrong，但仍比 zero 差；触发 STOP/DOWNGRADE |
| irregular outer-boundary wave | 独立 solver | 真正不规则外边界和孔洞/通道 | 一 seed：paired `1.091`，wrong `1.037`，joint `1.037` | 绝对无效，且 correct 不优于 wrong |
| The Well acoustic maze | Clawpack 外部数据 | 多源、强反射、有限速度高密墙 | paired `0.9918`，几乎所有方法为 1 | 当前 single-arrival phase 模型不匹配；结果拒绝 |
| WaveBench | 公开 TMLR benchmark | 线性波很匹配，但主要是 regular-grid operator tasks | 尚未接入 | 可做公开应用，但必须先改成 operator protocol |
| The Well Helmholtz staircase | 外部高精度解 | 固定楼梯边界、源和频率变化；时间解析 | 尚未接入 | 可测频率/source factor，但不能证明跨几何 |

### 5.1 Independent wave 为什么优先

[`independent_wave_solver.py`](../../../src/geoaware/independent_wave_solver.py) 解的是

\[
u_{tt}+\eta u_t+L_cu=f(t),
\]

包含变量波速、反射边界、两个源、8 种障碍拓扑和 24→32 网格。数据生成器不 import 学习模型，且 operator 对称性、半正定性、晚期散射能量均有 gate。它比 harmonic control 更接近真实波。

### 5.2 Independent-wave R1 locked validation（新增）

新增入口为 [`run_independent_wave_phase_r1.py`](../../../experiments/run_independent_wave_phase_r1.py)。协议在运行前固定为：

- train：5 个 geometry、两个源、resolution 24；
- validation：未见 `circle_offset`、两个源、resolution 32；
- test：`wall_right_door_high` 和 `double_diagonal` 文件完全不读取；
- 1% task-wise random training target labels，共 2222 个；
- validation 没有传感器 context，属于 source-conditioned zero-shot；
- checkpoint 只按**全部 2222 个 observed train labels**上的最低 loss 选择；
- source `(s_x,s_y)` 显式进入 geometry factor，相对源坐标进入 spatial factor；
- correct carrier 使用 `∫ds/c(x)` 的物理 travel time，wrong control 只将它替换为 Euclidean-distance / mean-speed。

三 seeds 的 validation 结果为：

| 模型 | 参数量 | global NRMSE | late-field NRMSE | boundary NRMSE | absolute gate |
|---|---:|---:|---:|---:|---|
| train observed mean | 0 | `1.0002±0.0001` | `1.0032` | `1.0025` | trivial |
| zero | 0 | `1.0068±0.0000` | `1.0176` | `1.0024` | trivial |
| joint INR | 11,713 | `1.4820±0.1839` | `1.5013` | `1.1066` | FAIL |
| ordinary functional CP | 21,200 | `3.2679±0.5370` | `3.8816` | `1.2315` | FAIL |
| wrong Euclidean phase | 21,200 | `2.8920±0.3071` | `3.7841` | `1.2155` | FAIL |
| paired travel-time phase | 21,200 | `3.4732±0.5044` | `4.3877` | `1.5850` | FAIL |

这不是轻微负信号。paired phase 在 observed train labels 上的 normalized MSE 已低至 `0.0021–0.0051`，却在 validation 达到 `3.01–4.01` NRMSE，说明固定相位字典严重拟合稀疏标签而没有跨几何泛化。wrong Euclidean phase 在三个 seed 上都优于 correct travel-time phase，因此几何归因也失败。按照预先写入代码的 gate，本轮不读取 test，也不晋级 WaveBench/The Well。

机器可读汇总见 [`track2_independent_wave_r1_summary.json`](../results/track2_independent_wave_r1_summary.json)，逐 seed 文件保留全部训练轨迹、checkpoint step、case metrics 和 `test_files_read=[]`。

### 5.3 Clean traveling-harmonic R2（完成，最终判定）

[`run_traveling_harmonic_phase_r2.py`](../../../experiments/run_traveling_harmonic_phase_r2.py) 是 R1 失败后预先允许的唯一 formulation correction。它复用同一 train/validation/test 几何 split、1% train-label mask、零 validation context 和全 observed-train checkpoint，但把目标换成独立模块 [`traveling_harmonic_generator.py`](../../../src/geoaware/traveling_harmonic_generator.py) 生成的

\[
u_s(t,x)=a_s(x)\left[\cos\{2\pi\sqrt{13}(\tau_s(x)-t)+\phi_s\}
+0.42\sin\{2\pi\sqrt{41}(\tau_s(x)-t)-0.37+\phi_s\}\right].
\]

这里没有 standing-wave 主项或 moving envelope：`a_s(x)` 只依赖空间，且新增单元测试验证同时平移 `τ` 和 `t` 时 carrier 严格不变。生成器不 import learner；模型也不读取 `√13,√41`，而从预先固定的 `[1.25,2.75,4.25,5.75,7.25,8.75] Hz` 宽频字典开始学习。因此这是 identity 本身而非同频泄漏的干净测试。

三 seeds、每个模型 500 steps 的 validation 结果为：

| 模型 | 参数量 | global NRMSE | late NRMSE | boundary NRMSE | gate pass |
|---|---:|---:|---:|---:|---:|
| zero | 0 | `1.0000±0.0000` | `1.0000` | `0.9997` | trivial |
| train observed mean | 0 | `1.0002±0.0001` | `1.0002` | `1.0000` | trivial |
| joint INR | 11,713 | `1.2753±0.0203` | `1.2602` | `1.2066` | 0/3 |
| paired travel-time phase | 22,926 | `1.4859±0.0806` | `1.5484` | `1.3242` | 0/3 |
| ordinary functional CP | 21,200 | `2.4778±0.1679` | `2.4213` | `1.7681` | 0/3 |
| wrong Euclidean phase | 22,926 | `2.5336±0.0849` | `2.5312` | `2.5720` | 0/3 |

correct travel time 相比 wrong Euclidean 有稳定而显著的结构优势，说明几何坐标方向本身不是毫无信息；但 absolute gate 先于 pairwise 排名，`1.486` 仍比零预测差得多，不能包装成成功。paired 模型 observed-train normalized MSE 已为 `0.0135–0.0253`，而 validation 为 `1.44–1.58`，主要问题是 1% sparse-label 下的跨几何过拟合，不是训练集未拟合。learned bands 也只在初始化附近小幅移动。

机器可读汇总为 [`track2_traveling_harmonic_r2_summary.json`](../results/track2_traveling_harmonic_r2_summary.json)，三个逐 seed JSON 均记录 `test_files_read=[]`。根据预先承诺，本结果直接触发 STOP/DOWNGRADE，不再读取锁定 test，也不执行 independent-wave R3 或 WaveBench 扩张。

### 5.4 The Well 不是“坏数据”，而是当前任务/方法不匹配

The Well 官方 acoustic-maze 数据是 2000 条、201×256×256 的 Clawpack 模拟，材料密度跨多个数量级，且每条轨迹有 1–6 个初始压力环。官方完整监督 benchmark 上 U-Net/ConvNeXt U-Net 的 VRMSE 可到 `0.0351/0.0153`，说明数据本身可学习；我们的 early-40 是另一个只有 1% target labels 的极低监督协议，不能拿官方数字直接横比。[官方数据卡](https://huggingface.co/datasets/polymathic-ai/acoustic_scattering_maze)

当前 PairedPhaseCP 又把多个源压成一张 “到最近源” 的 travel-time map，无法表示不同源、反射路径和相干叠加。所以这里的负结果应解释为 **single-arrival formulation + extreme sparse labels 的共同失败**，而不是宣称 phase prior 对真实声学普遍无效。

### 5.5 WaveBench 的正确用法

WaveBench 有 24 个线性波数据集，包含 time-harmonic forward、reverse time continuation 和 inverse source，并提供 FNO/U-Net 代码与 checkpoint。[TMLR 论文](https://openreview.net/forum?id=6wpInwnzs8)、[官方代码](https://github.com/wavebench/wavebench)、[Zenodo 数据](https://zenodo.org/records/8015145)

但 WaveBench 的 time-harmonic 任务是 `wavespeed field → pressure field`，RTC/IS 又是特定 inverse operator；它们都不是当前的稀疏场回归。接入时必须定义一个 phase-conditioned **operator**，让它和 FNO/U-Net 接收相同的完整输入场，并分别报告：

- 官方 full-supervision operator setting；
- 额外的 1%/5% target-label setting。

不能把官方 checkpoint 的完整监督结果和我们的 1% 标签结果放在同一张公平比较表里。

## 6. Baseline 审计与最低要求

### 6.1 所有任务都必须有

| Baseline | 作用 | 公平性要求 |
|---|---|---|
| zero / train mean | 判断是否只是接近零均值场 | 不读取测试 target |
| causal persistence / analytic trivial | 波时间任务的强简单基线 | 只用允许的初始帧；Helmholtz 要包含解析 `e^{-iωt}` baseline |
| Euclidean/wrong phase | 唯一改变几何坐标 | 参数、初始化、优化完全匹配 |
| ordinary neural CP | 判断 fixed phase pairing 是否有效 | 同样 geometry/time/space 输入和相近参数 |
| ordinary neural Tucker | 判断 CP 对角配对是否过硬 | 同上；报告 core/rank |
| joint INR | 判断显式低秩是否值得 | 同输入；同时给 parameter-matched 和强容量版本 |
| raw F-INR-style CP/Tucker | 对标 functional tensor prior art | 不用 geodesic/phase，只用合法 raw coordinate/geometry features |

F-INR 已经系统支持 CP、TT、Tucker 与多种 INR backbone，所以本方向不能把“用神经函数做 tensor factors”本身写成创新。[WACV 2026 论文](https://openaccess.thecvf.com/content/WACV2026/papers/Vemuri_F-INR_Functional_Tensor_Decomposition_for_Implicit_Neural_Representations_WACV_2026_paper.pdf)

### 6.2 什么时候必须加 FNO/TFNO/U-Net/GINO

- **FNO/TFNO/U-Net**：regular-grid operator 任务必须加入。FNO 是 function-to-function baseline；TFNO 是把 FNO 的权重 Tucker 化，不等同于我们的 field functional Tucker。应使用官方 `neuraloperator` 实现并分别做小/中容量验证选择。[FNO/TFNO 官方文档](https://neuraloperator.github.io/dev/theory_guide/fno.html)
- **GINO**：如果输入/输出为 irregular mesh、且任务是 geometry-conditioned operator learning，则必须加入。GINO 使用 point cloud/SDF 和 graph-to-latent-grid operator，正是变量几何强基线。[NeurIPS 2023 论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/70518ea42831f02afc3a2828993935ad-Paper-Conference.pdf)
- 对当前纯 sparse field regression，FNO/GINO 并非天然同类；若强行加入，必须给它们相同的输入场和相同的稀疏 target labels，并明确这是 weak-supervision operator experiment。

最低公平协议：相同 train/validation/test 几何 split；相同 target-label mask；相同合法输入；validation-only 超参选择；至少两档容量；相同训练 wall-clock 或充分收敛两种视角；每个模型逐 seed 重置初始化；指标先按 trajectory/case 聚合，再按 seed 推断。

还建议增加 POD/DMD 或低秩时空 SVD 作为经典波场低秩 baseline。否则审稿人会质疑优势只是来自任何低秩时空表示，而不是 phase pairing。

## 7. 指标和测试审计

主指标应为 held-out NRMSE/VRMSE，并额外报告：

- 波前/高频 band error；
- 边界带和 obstacle shadow error；
- 相位误差与幅值误差分解；
- 参数量、训练/推理时间；
- full-field MSE skill relative to strongest trivial predictor。

永久外部 gate：

\[
\mathrm{NRMSE}\le 0.8,
\qquad
1-\frac{\mathrm{MSE}_{model}}{\mathrm{MSE}_{trivial}}\ge20\%.
\]

只有先通过这两项，才检查 learned-model 之间的 paired p-value。新增 [`absolute_wave_gate`](../../../src/geoaware/phase_wave_protocol.py) 已把规则固化；The Well `0.99175` 相对 persistence `1.00160` 会被测试明确拒绝。

当前自动测试包括：

- exact traveling cosine/sine identity；
- carrier 输入 shape 约束；
- near-null 外部结果 gate 拒绝；
- useful result gate 接受；
- CP/Tucker factor contraction、envelope 分离和独立 wave solver PSD/finite tests。

R1 与 R2 均已完成三-seed run；R2 另有严格 traveling-characteristic 单元测试。由于该方向已触发 STOP，source-conditioned loader、POD 和 FNO/GINO 集成不再列为本方向必须补齐的开发项。

## 8. 现有正负证据应该怎样表述

### 正信号

- 在 band-aligned harmonic control 上，paired CP `0.0952±0.0144`，普通 geometry CP `0.1113±0.0213`，joint IP-NF `0.1825±0.0173`；说明固定 phase dictionary 在极低监督下可以有效降低方差。
- correct geodesic 与 Euclidean wrong control 差异极大，说明受控 narrow-wall 几何确实改变了有用坐标。
- exact identity 在实现级别成立，故事本身非常简洁且可解释。

### 负信号

- 正数据频率与模型频率重合，且主信号不是 traveling phase，因果归因弱于此前文档的表述。
- moving envelope 上 joint INR 仍更强，说明显式低秩的适用条件有限。
- independent reflected wave 现在已有 locked 三-seed 负结果：paired phase 比 trivial mean 差得多，correct travel time 也不如 wrong Euclidean control。
- clean traveling harmonic R2 中 correct travel time 虽优于 wrong control，但自身 `1.486±0.081` 仍未恢复信号；这排除了“只要改成真正 traveling phase 就会 work”的解释。
- phase-envelope 增加容量但没有赢过 joint INR，不应继续堆叠包络模块。

## 9. 下一轮最小实验矩阵

| 轮次 | 唯一问题 | 数据 | 模型 | Gate | 决策 |
|---|---|---|---|---|---|
| R1（完成） | 当前 phase CP 能否处理独立 reflected wave | independent wave；train 5 geometry@24，val 1@32，test 2 锁定 | zero/mean、joint INR、ordinary CP、paired、wrong | 全部 FAIL；correct 也不优于 wrong | test 未读；暂停外部扩张 |
| R2（完成） | 优势来自 identity 还是频率泄漏 | learner-free 真 traveling harmonic；两个 off-grid 无理频率 | trainable-frequency paired、ordinary CP、joint INR、wrong | paired 0/3 通过；correct 优于 wrong 但仍比 zero 差 | STOP/DOWNGRADE |
| R3（取消） | 修正后能否处理 independent wave | 原计划同 R1 | 原计划加入 POD | R2 未成功，因此不启动 | 取消 |
| R4（取消） | 能否成为公开 operator application | 原计划 WaveBench | official U-Net/FNO、TFNO、phase operator | R3 未启动 | 取消 |

R1 和唯一允许的 R2 都已执行且失败。source conditioning、物理 `τ-t`、生成器/模型频率解耦和 500-step 公平预算均已落实，因此本方向不再通过 attention、Bayesian uncertainty、learned ray tracing 或大 Tucker core 继续搜索。

## 10. GO / NO-GO

### 当前判定：STOP / DOWNGRADE（仅保留受控学生项目）

保留理由：idea 简洁、代码忠实、受控 aligned 结果很强，并且对“为什么 wave tensor 可能低秩”提供了明确物理解释。但 R1 和 clean R2 都触发绝对无效条件，不能继续推进为主论文或公开应用论文。

### 升级为完整论文的必要条件

1. independent solver 或一个公开 wave benchmark 通过绝对有效性 gate；
2. 至少一个实验去掉 generator/model 同频泄漏；
3. paired phase 在相同输入和训练标签下稳定超过 ordinary CP/Tucker 与 joint INR；
4. operator setting 中至少对标 U-Net/FNO，irregular mesh 时对标 GINO；
5. 明确展示由低 mismatch 到 moving-envelope/multipath 失效的 phase diagram。

### 立即 NO-GO 的情况

- independent wave 上 correct phase 不优于 Euclidean wrong phase；
- learned models 仍集中在 NRMSE≈1；
- 只有与模型频率完全重合的合成数据为正；
- 为赢得 The Well 必须加入复杂到难以辨认三角恒等式主线的通用网络。

在这些条件下，应把本方向收束为一篇清晰的波场结构小论文或教学型项目，而不是继续包装成通用 geometry-aware PDE 主线。
