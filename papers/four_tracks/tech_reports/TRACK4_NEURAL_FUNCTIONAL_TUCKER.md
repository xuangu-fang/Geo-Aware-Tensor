# 方向 4 技术报告：Geometry-conditioned Neural Functional Tucker

更新时间：2026-08-15  
状态：**POC 可运行；论文主张尚未通过**  
当前正确名称：**边界距离条件的神经函数 Tucker**  
不应使用的名称：完整 SDF-Tucker、跨域 few-shot completion、已验证的 geometry-general operator

---

## 0. 给决策者的结论

这条路线想回答一个简洁的问题：

> 能否把不规则域的几何作为连续条件，放进一个显式低秩的函数 Tucker，而不是把所有输入直接交给一个黑盒 MLP？

目前 POC 已证明三件事：

1. 模型可以在 24 分辨率的四个训练形状上只读取 1% 标签，然后直接查询一个未见形状的 32 分辨率网格；其 NRMSE 约为 0.19，显著好于常数预测约 1.0。
2. 加入点到最近边界的距离后，平均优于去掉该局部距离的 Tucker；因此这个局部几何量确实有用。
3. 把 core 初始化为对角 CP 后，Tucker 平均略优于参数量接近的 functional CP，但只在 1/3 seeds 上获胜，并且仍明显输给相同输入的 joint INR。

目前没有证明：

- 非对角 Tucker core 稳定优于 CP；
- 方法优于条件神经场或强 neural operator；
- 模型能从四个训练形状学会对任意孔洞拓扑泛化；
- 测试域给 1% 观测时的 few-shot completion；
- 当前 `boundary_distance` 是完整 signed distance field。

因此当前决定是：**允许继续做一个严格的新一轮 POC，但不能按 AI 主会完整方法稿来写结果**。最近的 GO 条件是：在新的、多形状、多孔洞 validation 上稳定超过同输入 CP 和 joint INR，并在足够训练几何下加入 CORAL/GINO 等强基线。

---

## 1. 任务必须先说清楚

### 1.1 当前真正做的是 Task O-sparse

当前实验中，模型在四个训练域 
\(\Omega_1,\ldots,\Omega_4\) 上读取稀疏标签，然后在一个未见域
\(\Omega_*\) 上零目标观测预测：

\[
\{(\Omega_i,p,s,x,u_i(p,s,x)): (p,s,x)\in\mathcal I_i^{\rm train}\}_{i=1}^4
\longrightarrow
\widehat u_*(p,s,x),
\qquad
|\mathcal I_i^{\rm train}|/|\mathcal T_i|=1\%.
\]

测试域没有提供任何 \(u_*\)。所以 `ratio=0.01` 应叫：

> **training-label fraction / 训练场标签比例**

而不应叫“测试域 observation ratio”。这是一个**稀疏监督的跨几何 surrogate/operator learning** POC。

### 1.2 Task C：测试域 completion/adaptation

真正的低观测补全应写成：

\[
\mathcal O_*=\{(p,s,x,u_*(p,s,x)):(p,s,x)\in\mathcal I_*^{\rm obs}\},
\qquad
\widehat u_*|_{\mathcal I_*^{\rm miss}}
=\operatorname{Adapt}(\theta,\Omega_*,\mathcal O_*).
\]

这里测试域明确提供 0.1%、1%、5% 的目标观测，所有方法必须看到同一批 entries/sensors。建议再分：

- **C-instance**：只使用当前域观测，从头拟合；检验几何正则是否改善单张量补全。
- **C-meta**：先在其他域训练，再只适配 latent code、core 或少量参数；检验跨域先验是否降低测试域样本复杂度。

Task C 的合理 baseline 是 graph harmonic、RBF/GP interpolation、离散 CP/Tucker、per-instance INR、functional CP/Tucker。GINO/DAFNO 不是 C-instance 的自然 baseline，除非给它们同样的预训练与适配协议。

### 1.3 Task O：新域零目标观测

Task O 必须有较多 geometry-solution pairs：

\[
(\Omega,a_\Omega)\mapsto u_\Omega,
\qquad u_{\Omega_*}^{\rm target}\text{ 在推理时不可见}.
\]

这里才适合对标 joint conditional field、DeepONet、CORAL、GINO、DAFNO、Geo-FNO、Transolver。训练标签可以人为抽稀，但应叫 1%/5%/100% **label budget**，不能与 Task C 的 test observation ratio 混表。

### 1.4 两个任务的冻结表

| 项目 | Task C | Task O |
|---|---|---|
| 测试域 target observation | 有，固定 mask | 无 |
| 主要问题 | completion / adaptation | surrogate generalization |
| “1%”含义 | 测试域观察比例 | 训练标签比例 |
| 核心 baseline | 插值、GP、CP/Tucker、per-instance INR | conditional field、DeepONet、CORAL、neural operator |
| 当前已实现 | 否 | 是，但只有 4 个训练形状 |
| 结果是否可混表 | 否 | 否 |

---

## 2. 当前模型的准确 formulation

### 2.1 数据对象

对一个域 \(\Omega\)，当前输出张量为

\[
U_\Omega\in\mathbb R^{N_s\times N_p\times N_\Omega},
\]

三个离散索引分别是 source、diffusivity 和 active-domain node。当前数据中
\(N_s=4\)、\(N_p=14\)，而 \(N_\Omega\) 随形状和网格改变。

### 2.2 几何全局描述符

代码没有使用几何 encoder，而只使用七个统计量：

\[
q_\Omega=
[\rho_\Omega,\mu_x,\mu_y,\sigma_x,\sigma_y,
Q_{.25}(d_\partial),Q_{.75}(d_\partial)].
\]

其中 \(\rho_\Omega\) 是 active nodes 在背景方格中的比例，
\(d_\partial\) 是 active node 到最近边界的正距离。

这个描述符只有 7 维，而且目前只在 4 个训练形状上学习。它不显式包含：

- boundary components 数量；
- hole 数量或 Euler characteristic；
- 边界点云/曲率/法向；
- 两点间是否被孔洞遮挡；
- geodesic distance 或连通瓶颈。

因此现在不能说它已编码拓扑。不同拓扑完全可能共享近似相同的七个统计量。

### 2.3 当前 “SDF” 实际是什么

生成器计算的是

\[
d_\partial(x)=\operatorname{dist}(x,\partial\Omega),\qquad x\in\Omega,
\]

并且只在 active nodes 上存储，所以所有值非负。加载时还按每个域的 95% 分位数归一化：

\[
\widetilde d_\partial(x)=
d_\partial(x)/Q_{.95}(d_\partial).
\]

完整 signed distance field 应在一个共享 ambient domain 上定义，并在域内、域外具有相反符号。当前特征：

- 没有域外值；
- 没有符号变化；
- 不区分外边界与 hole boundary；
- 丢掉一部分绝对长度尺度。

所以报告应使用 **normalized interior boundary distance**，不能继续把它无条件简称为完整 SDF。对只查询域内点的模型，这个正距离仍然是有意义的局部几何输入，但能力弱于 GINO 所用的 ambient SDF。

### 2.4 三组函数因子

当前模型是

\[
\widehat u_\Omega(p,s,x)
=\sum_{a=1}^{R_g}\sum_{b=1}^{R_p}\sum_{c=1}^{R_z}
G_{abc}\,g_a(q_\Omega)\,h_b(\log p,p)\,r_c(z_\Omega(x,s)),
\]

其中

\[
z_\Omega(x,s)=
[x_1,x_2,\widetilde d_\partial(x),s_1,s_2,\|x-s\|_2,1].
\]

三个 factor 都是两层隐藏层、GELU 激活的 MLP。当前配置为

\[
(R_g,R_p,R_z)=(12,12,12),\qquad hidden=64.
\]

### 2.5 一个容易误解的结构事实

虽然原始张量索引是 `source × parameter × node`，当前模型并没有把 source 和 node 分成两个 Tucker modes。它将 \((s,x)\) 共同放进一个 spatial/query factor \(r_c\)。

所以更准确的说法是：

> 对 geometry、physical parameter、joint source-query coordinate 做三因子 functional Tucker。

这并非错误。椭圆 Green's function 本来强烈依赖 \((x,s)\) 的相对关系，联合 factor 可能比强行分开更合理。但论文不能把它写成标准 `source × parameter × node` Tucker 而不解释这一点。

### 2.6 CP restriction

方法匹配 CP 为

\[
\widehat u^{\rm CP}_\Omega(p,s,x)
=\sum_{r=1}^{R}w_r g_r(q_\Omega)h_r(p)r_r(z_\Omega(x,s)),
\qquad R=24.
\]

它和 proposed Tucker 使用相同输入、相同类型的 factor MLP。两者参数量接近：

- Tucker：17,764；
- CP：18,400。

但这只是**参数量接近**，不是完全同 rank、同计算图。Tucker 每个 factor 输出 12 维并拥有 \(12^3\) core；CP 每个 factor 输出 24 维。

### 2.7 joint INR

joint INR 直接学习

\[
\widehat u_\Omega(p,s,x)=
F_\theta([q_\Omega,\log p,p,z_\Omega(x,s)]),
\]

不施加低秩分离。其参数量为 20,353，与 Tucker 接近，是当前最重要的黑盒对照。

---

## 3. 代码映射

| 技术组件 | 位置 | 审计结论 |
|---|---|---|
| query feature | `src/geoaware/functional_tucker.py:19` | 名为 `sdf_query_features`，实际读取正的 interior boundary distance |
| parameter feature | `src/geoaware/functional_tucker.py:34` | 使用 \((\log p,p)\)，适合正 diffusivity |
| Tucker contraction | `src/geoaware/functional_tucker.py:41` | 显式小 core，`einsum` 公式正确 |
| direction-4 model | `src/geoaware/functional_tucker.py:71` | 三个连续 factor；geometry descriptor 每个 query 重复 |
| CP baseline | `src/geoaware/functional_tucker.py:87` | 输入匹配、参数量接近，是有效核心 baseline |
| global descriptor | `experiments/run_irregular_elliptic_paper_b.py:32` | 仅七个统计量，不能可靠表示 topology |
| distance normalization | `experiments/run_irregular_elliptic_paper_b.py:56` | 每域 Q95 normalization，需在报告中显式声明 |
| joint INR | `experiments/run_four_track_fast_poc.py:33` | 相同 16 维输入，比较合理 |
| mask | `experiments/run_irregular_elliptic_paper_b.py:25` | uniform random entries，无空间传感器结构 |
| checkpoint | `experiments/run_four_track_fast_poc.py:63` | 已修为全 observed set MSE，而不是偶然 minibatch loss |
| training loop | `experiments/run_four_track_fast_poc.py:156` | AdamW、梯度裁剪、固定预算；未使用 validation early stop |
| metric | `experiments/run_four_track_fast_poc.py:187` | global 与 boundary NRMSE；boundary 使用自己的 subset std |
| geometry distance 生成 | `src/geoaware/irregular_domain_solver.py:109` | EDT 正距离，代码注释与实际一致 |
| PDE generator | `src/geoaware/irregular_domain_solver.py:191` | screened elliptic；reaction 与 forcing 都显式依赖 boundary distance |

---

## 4. Optimization 与 inference 审计

### 4.1 当前训练过程

每个训练域先固定抽取 1% entries。四个域的实际标签数是：

| 训练形状 | tensor shape | 完整 entries | 1% 标签 |
|---|---:|---:|---:|
| L-shape | \(4\times14\times363\) | 20,328 | 203 |
| U-notch | \(4\times14\times412\) | 23,072 | 231 |
| wavy-3-lobe | \(4\times14\times284\) | 15,904 | 159 |
| dumbbell | \(4\times14\times236\) | 13,216 | 132 |

总共约 725 个不同标签。训练每步先均匀选择一个域，再从其 observed entries 中有放回抽取 2,048 项，所以一个 minibatch 会大量重复同一批标签。这不是数据错误，但 `batch_size=2048` 不代表有 2,048 个独立 observations。

目标使用所有训练 observed values 的全局均值和标准差归一化。优化器为 AdamW：

- learning rate：\(2\times10^{-3}\)；
- weight decay：\(2\times10^{-5}\)；
- steps：900；
- gradient clipping：5；
- 每 45 steps 在完整 observed set 上计算一次训练 MSE 并保存最佳 checkpoint。

### 4.2 protocol fix 是否正确

旧实现按当前随机 minibatch loss 保存“最佳模型”，不同 checkpoint 比较的不是同一批数据，方差很大。新 `observed_mse` 对所有固定 observed entries 求同一尺度的 MSE，再选择 checkpoint。公式、设备移动和 target normalization 均正确。

这一修复避免了 checkpoint noise，但它仍是 training-set selection，不是 validation early stopping。优点是不会偷看 held-out geometry；缺点是无法防止对约 725 个训练标签过拟合。下一轮应在训练几何内部另留 calibration entries 作为 early-stop set，同时不碰 geometry-level validation。

### 4.3 `cp_diagonal` 初始化是否正确

当前 core 初始化为

\[
G_{abc}^{(0)}=
\begin{cases}
1/\sqrt R,&a=b=c\le R,\\
0,&\text{otherwise}.
\end{cases}
\]

因此初始 contraction 精确等价于 rank-\(R\) CP：

\[
\widehat u^{(0)}=\frac1{\sqrt R}\sum_{r=1}^R g_rh_rr_r.
\]

代码是正确的，并新增了 exact contraction unit test。但必须注意：三个 factor MLP 仍是随机初始化，它**不是从一个已训练 CP 的权重 warm start**。off-diagonal core 从第一步起也可自由更新。因此这只能叫“CP-shaped core initialization”。

### 4.4 当前 Tucker 的优化风险

主要风险不是 core 公式，而是参数化：

1. Tucker 存在 factor/core scale gauge：一个 factor 放大、core 对应缩小，预测不变，优化条件数可能变差。
2. 1,728 个 core 参数相对于约 725 个独立观测并不小；dense interaction 很容易吸收噪声。
3. CP baseline 是 rank 24，而 Tucker 初始嵌套 CP 只有 rank 12；二者只匹配参数量，不匹配 initial functional rank。
4. 三个 seeds 同时改变 mask、初始化与 minibatch 顺序，无法定位方差来源。

最小修正顺序应是：

1. 先训练 rank-12 CP；
2. 把其 factor 权重和 diagonal weights 真正复制进 Tucker；
3. 写成 \(u=u_{CP}+\alpha u_{offdiag}\)，令 \(\alpha=0\) 起步；
4. 先只训 CP，再逐渐解冻 off-diagonal residual；
5. 分开报告 data-mask seed 与 optimizer seed。

这比继续增加 encoder、attention 或更大 rank 更值得优先验证。

### 4.5 当前 inference

推理是完全 deterministic zero-shot：给定未见域的 descriptor、节点坐标、边界距离、source 坐标和 parameter，直接逐 query 前向。它具备：

- 节点数量可变；
- query order 不敏感；
- 24 训练、32 推理的跨分辨率查询。

它不具备：

- posterior uncertainty；
- 测试域 latent/core adaptation；
- PDE-constrained refinement；
- 对域外 SDF 的查询；
- 对任意 topology 的理论保证。

---

## 5. 当前数据集审计

### 5.1 自建 irregular-boundary screened elliptic

PDE 为

\[
[\operatorname{diag}(r_\Omega(x,p))+pL_{\Omega,c}]u=f_{\Omega,s,p},
\]

所有外边界和孔洞边界采用 reflecting/zero-flux 条件。linear solve 最大相对残差约 \(3\times10^{-14}\)，数值解本身是可靠的。

当前 shape split：

| split | shape families | resolution | 备注 |
|---|---|---:|---|
| train | L-shape、U-notch、wavy、dumbbell | 24 | 仅 4 个 geometry instances |
| validation | slanted channel | 32 | 仅 1 个 instance |
| historical test | wavy-with-hole | 32 | 已经被以前的实验读取，不能再当 pristine confirmation test |

优点：

- active domain 真正不规则，节点数变化；
- 包含凹边界与一个孔洞；
- train/evaluation geometry 分组，24→32 不存在同 geometry 跨 split 泄漏；
- source、diffusivity、space 语义明确；
- PDE solver 与 learner 分离。

关键缺陷：

1. 只有 6 个 geometry instances。三个 seed 只是三次训练，不是三个独立未见几何样本。
2. 只有一个带孔洞几何，而且已经被查看；无法做 topology confirmation。
3. reaction 和 forcing 都显式包含 \(\exp[-d_\partial(x)/c]\)。因此 boundary distance 是 simulator 直接使用的因果变量。用它获得正信号是合理的，但数据明显 method-matched，不能单独作为外部证据。
4. uniform random entry mask 混合 source、parameter、space，且几乎覆盖所有 mode；这是最容易的 sparse-label setting。
5. Neumann 平滑 elliptic field 对 coordinate MLP 很友好，尚未测试 discontinuity、thin channel、long-range occlusion 或多孔洞。
6. 现有 validation 只有一个 slanted shape，“macro NRMSE”实际上就是单 case NRMSE。

结论：它适合机制 POC，不适合形成主表。

### 5.2 下一版自建数据：必须扩而不复杂化

建议保持同一个 screened elliptic 方程，只扩大 geometry family，避免同时改 PDE 与模型：

- 训练：80–200 个参数化形状；
- validation：20 个新参数实例，其中至少 5 个带孔洞；
- test：30 个完全冻结实例；
- topology-held-out：train 只见 0/1 hole，test 见 2/3 holes；
- family-held-out：star/wavy/notch/channel 按 family 分组；
- resolution：24 训练，32 validation，48 test；
- 保留真实 ambient signed distance、interior distance、boundary component id 三份独立字段。

必须重新生成一个从未读取的新 test manifest 和 hash；旧 `wavy_with_hole` 只作为 development case。

### 5.3 AirfRANS

[AirfRANS](https://arxiv.org/abs/2212.07564) 包含 1,000 个二维亚声速 airfoil RANS simulations，并给出 full/scarce/Reynolds/AoA generalization tasks、surface-force metrics 与官方数据接口。它适合 Task O：fluid domain 具有 airfoil 内边界，mesh/point cloud 随几何变化，输入中已有到 airfoil 的距离与 normals。

正确用法：

- 首先复现官方 corrected split；官方文档特别说明 ML 结果应以更新后的 arXiv 版本为准；
- 用 scarce split 做 label-scarce geometry surrogate，而不是把它改写成 tensor completion；
- 输出分别报告 velocity、pressure、turbulent viscosity，以及 force coefficient；
- 与官方 PointNet/Graph baseline 和至少一个 general-geometry operator 同表；
- SDF/distance、normal、inlet condition 的输入预算对所有方法一致。

风险：AirfRANS 是稳态 CFD，物理与当前平滑 elliptic POC 差别很大；如果这里失败，只能说明方法尚未扩到 CFD，不能回头修改 test。

### 5.4 Geo-FNO / CORAL 的 elasticity 与 NACA-Euler 数据

[Geo-FNO](https://arxiv.org/abs/2207.05209) 接受 point cloud、mesh 或 design parameters，并在 Elasticity、Plasticity、Euler/Navier–Stokes 等一般几何上验证；[CORAL](https://papers.nips.cc/paper_files/paper/2023/hash/df54302388bbc145aacaa1a54a4a5933-Abstract-Conference.html) 也包含 general-geometry operator 与 geometric design 任务。优先复用它们的官方 split 可以同时减少 dataset selection 与 baseline implementation 的争议。

建议先选一个小的 2D elasticity/NACA case，不要一开始同时搬四套数据。选择标准是：geometry identities 可分组、query mesh 可变、官方代码能复现实验、许可证允许再分发或下载。

### 5.5 GINO Car-CFD 与 DAFNO fracture

[GINO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html) 用 SDF、GNO 和 latent-grid FNO 学习变化的 3D car geometry；[DAFNO](https://arxiv.org/abs/2305.00478) 用平滑 characteristic function 处理 irregular/evolving domains，并展示 topology-changing fracture。

这两套更适合最终强验证，不适合下一周的第一个 POC：3D CFD 计算重，fracture 的输入输出语义也与当前 source–diffusivity tensor 不同。先在 AirfRANS/2D elasticity 确认方法有竞争力，再升级。

---

## 6. Baseline cards：当前哪些对，哪些不够

### 6.1 当前 baseline 审计

| baseline | 任务角色 | 输入是否公平 | 当前状态 | 结论 |
|---|---|---|---|---|
| observed global mean | absolute skill | 只用 train observed mean | 已实现 | 必须保留，但太弱，不能作为唯一 trivial baseline |
| coordinate neural Tucker | pointwise distance ablation | 仍读取 7 维 geometry descriptor | 已实现 | 名称“coordinate-only”不准确；只消融 local boundary-distance channel |
| boundary-distance neural CP | core-structure baseline | 与 Tucker 输入一致，参数量接近 | 已实现 | 当前最重要的方法匹配 baseline |
| joint boundary-distance INR | low-rank-vs-black-box | 完全相同 16 维输入，参数量接近 | 已实现 | 当前最强，必须正视 |
| topology-erased kernel Tucker | direction-3 kernel ablation | 不属于 direction 4 | 已实现 | 不应作为 direction-4 的 geometry-free baseline |

一个隐蔽问题是：`coordinate_neural_functional_tucker(use_sdf=False)` 仍然通过 \(q_\Omega\) 读取 fluid fraction 和 boundary-distance quantiles。因此它不是“无几何”模型。下一轮需要三层消融：

1. coordinates/source/parameter only；
2. + global 7-stat descriptor；
3. + pointwise boundary distance；
4. + full ambient SDF/boundary encoder。

### 6.2 Task C 必加 baseline

| baseline | 为什么必须有 |
|---|---|
| nearest / Euclidean RBF | 检验收益是否只是局部平滑 |
| graph harmonic / Laplacian interpolation | 检验不规则域 graph geometry 是否已足够 |
| per-case GP | 提供平滑 kernel 与 UQ 对照 |
| discrete CP/Tucker | 检验 functional factor 是否必要 |
| per-instance coordinate INR | 检验低秩限制是否抗过拟合 |
| functional CP/Tucker | proposed 的直接结构对照 |

所有方法必须读取相同测试域 mask；不能让 functional model 使用其他域预训练，而让 CP 从头开始，除非单独列为 C-meta。

### 6.3 Task O 必加 baseline

**最近、最重要的工作不是 GINO，而是以下三类：**

1. [F-INR](https://arxiv.org/abs/2503.21507)：把高维 INR 分成 axis-specific subnetworks，并支持 CP/TT/Tucker。它与本方向的“functional tensor decomposition”极近。我们的独特点必须是 variable-domain geometry conditioning、跨拓扑与稀疏标签效率，不能只说“把 INR 做 Tucker”。
2. [CORAL](https://papers.nips.cc/paper_files/paper/2023/hash/df54302388bbc145aacaa1a54a4a5933-Abstract-Conference.html)：用 modulated coordinate-based neural fields 在 general geometries 上做 operator learning，可跨 sampling grid/resolution。它是最匹配的 conditional neural field baseline。
3. [DeepONet](https://www.nature.com/articles/s42256-021-00302-5)：branch/trunk 的乘积求和本身就是低秩 operator representation。必须解释 Tucker 比 branch–trunk decomposition 多了什么。

在训练几何数量足够后，再加入：

- **GINO**：SDF + GNO 映射 irregular points 与 latent regular grid，具 discretization invariance；这是完整 ambient SDF 的强基线。
- **DAFNO**：smoothed characteristic function 直接进入 FNO integral layer，适合 topology changes。
- **Geo-FNO**：学习 physical-to-latent deformation，适合 point cloud/mesh/design parameter。
- **Transolver**：Physics-Attention 把 mesh points 聚成 learned slices，适合 general geometries；官方工作为 [ICML 2024 Spotlight](https://icml.cc/virtual/2024/poster/33751)。

只有 4 个训练 geometries 时直接跑这些大模型并把失败当作 proposed 优势是不公平的。应做 geometry-count phase diagram：20、50、100、200、500 shapes；所有方法同时看 1%、5%、20%、100% training labels。

### 6.4 公平性记录

每个表至少附：

- input fields；
- trainable parameter count；
- 实际不同标签数，而不是 minibatch size；
- train wall-clock、peak GPU memory、inference/query throughput；
- 是否预训练、是否读取 test observations；
- rank/width 选择使用哪个 validation split；
- 相同 geometry split 与相同 label mask hash。

---

## 7. 测试与统计审计

### 7.1 已有测试

现有测试验证：

- functional Tucker core shape 正确且前向有限；
- CP-shaped core 的 diagonal/off-diagonal 初始化正确；
- 新增 exact contraction test：`cp_diagonal` 与 CP 代数完全一致；
- 新增 boundary-distance feature test：代码读取的是 stored nonnegative metadata，`use_sdf=False` 只把该通道置零。

### 7.2 缺失的 unit/integration tests

下一轮必须增加：

1. node permutation equivariance：同时重排 coords/distance/query node index 后预测对应重排；
2. full SDF sign convention：域内/域外/边界的符号和单位梯度 sanity；
3. outer/hole boundary component 标签；
4. split audit：同一 geometry family/parameter instance 不跨 train/test；
5. mask audit：entry、sensor、fiber、region 的实际 count 与 coverage；
6. checkpoint regression：完整 observed MSE 的最佳 state 可复现；
7. CP warm-start equivalence：复制训练 CP 后，Tucker 解冻前预测逐点一致；
8. cross-resolution data audit：相同物理解在 24/32/48 上离散收敛。

### 7.3 指标修正

当前 NRMSE 为

\[
\operatorname{NRMSE}=\frac{\sqrt{\operatorname{mean}(\widehat u-u)^2}}
{\operatorname{std}(u)}.
\]

boundary NRMSE 使用 boundary subset 自己的 std。这能描述区域内相对难度，但不同 shape 的 boundary std 不同，容易放大或缩小差异。下一轮同时报告：

- global relative \(L_2\)；
- global NRMSE；
- boundary RMSE / **global** target std；
- boundary relative \(L_2\)；
- hole-shadow、thin-channel、far-source regions；
- PDE residual 或 conservation metric（仅作为辅助，不替代 field error）。

### 7.4 seed 设计

当前 seed 同时控制 mask、model initialization 与 minibatch sampling。下一轮采用二级设计：

- 5 个固定 data-mask seeds；
- 每个 mask 3 个 optimizer seeds；
- geometry instances 为主要统计单位，而不是 optimizer run；
- 最终 test 只运行冻结配置和预注册的 10 confirmation seeds。

---

## 8. 当前证据：只使用 validation

本审计没有重新读取 hole test，也不依据 hole test 调参。由于旧结果文件已经存在，旧 `wavy_with_hole` 已不能作为从未查看的 confirmation set。

在修复 checkpoint protocol 并采用 CP-shaped core initialization 后，slanted-channel validation 的 3 seeds 为：

| 模型 | 参数量 | NRMSE mean ± sample std | boundary NRMSE |
|---|---:|---:|---:|
| boundary-distance Tucker | 17,764 | 0.1922±0.0305 | 0.2208±0.0147 |
| descriptor+coordinate Tucker | 17,764 | 0.2058±0.0335 | 0.2265±0.0202 |
| boundary-distance CP | 18,400 | 0.1958±0.0312 | 0.2245±0.0198 |
| joint boundary-distance INR | 20,353 | **0.1723±0.0175** | **0.2054±0.0255** |
| observed global mean | 0 | 1.0009±0.0008 | 1.0003±0.0007 |

可下的结论：

- 方法绝对有效，不是 NRMSE≈1 的空任务。
- pointwise boundary distance 相对 descriptor+coordinate Tucker 平均降低约 6.6% NRMSE，但 boundary 指标只降低约 2.6%；这不是稳定拓扑证据。
- Tucker 相对 CP 平均只改善约 1.9%，且逐 seed 只赢 1/3。不能称为稳定胜出。
- joint INR 相对 Tucker 的 NRMSE 低约 10.3%。显式 Tucker 的抗过拟合优势尚未出现。
- 一个 validation geometry 加三个训练随机性不足以做显著性检验。

关于 CP initialization：与 protocol-fixed random-core 结果相比，平均 NRMSE 几乎不变；boundary 平均略改善。说明单独把 core 设为 diagonal 并没有解决根本优化问题。

---

## 9. 下一轮实验矩阵

### 9.1 Round 1：先修故事与数据，不加大模型

目的：判断 geometry conditioning 与 Tucker core 是否真实存在正信号。

| 轴 | 配置 |
|---|---|
| task | O-sparse |
| train geometries | 20 / 50 / 100 procedural shapes |
| validation | 20 个未见 instances，包含多孔洞 |
| label fraction | 1% / 5% / 20% / 100% |
| geometry inputs | coord only；+7 stats；+interior distance；+ambient SDF |
| model | CP；random Tucker；CP-warm-start residual Tucker；joint INR |
| masks | entry-random；fixed spatial sensors；source fiber missing；region missing |
| resolution | 24→32，冻结后 24→48 |

晋级门槛：

- geometry feature 的 correct-vs-erased effect 在至少 80% held-out geometries 同方向；
- Tucker 对参数匹配 CP 的 MSE skill ≥10%；
- Tucker 对 joint INR 的 MSE skill ≥10%，或在相同 error 下参数/时间至少省 2 倍；
- 不是只在 entry-random 成立。

### 9.2 Round 2：Task C completion/adaptation

目的：直接验证用户最初关心的低观测抗过拟合。

| 轴 | 配置 |
|---|---|
| test observation | 0.1% / 0.5% / 1% / 5% |
| adaptation | from scratch；只调 latent；只调 core；全模型 |
| masks | random entry；fixed sensor；hole shadow；parameter/source fibers |
| baselines | RBF、graph harmonic、GP、discrete CP/Tucker、per-instance INR、functional CP |
| metrics | missing-only global/boundary error、NLL/UQ（如适用）、适配时间 |

这里 geometry-conditioned Tucker 最合理的简化是：共享 factor networks，测试域只估一个小 geometry code 或 core residual。否则给定单个测试域时，常数 global descriptor factor 不产生可识别优势。

### 9.3 Round 3：外部 Task O

优先级：

1. AirfRANS scarce/corrected split；
2. CORAL/Geo-FNO 的 2D elasticity 或 NACA-Euler；
3. 通过后再做 GINO Car-CFD 或 DAFNO fracture。

第一批强基线只选：joint INR、F-INR、CORAL、一个 neural operator。不要一次实现四个重模型。若数据是 ambient grid + mask，优先 DAFNO；若是 point cloud/unstructured query，优先 GINO；若使用官方 CORAL 数据，先复现 CORAL。

### 9.4 推荐的最小方法改进

按收益/复杂度排序：

1. **真实 CP warm start + off-diagonal residual core**；
2. 把 interior distance 改为明确的 ambient signed distance，并区分 outer/hole component；
3. 新增极小 boundary-point DeepSets encoder，与 7-stat descriptor 做正面对照；
4. 如前三项为正，再尝试 4-way Tucker，比较 joint \((s,x)\) factor 与 separate source/query factors；
5. 暂不加入 attention、learned geodesic、PDE loss 或多个复杂 geometry tokens。

---

## 10. GO / NO-GO 与论文口径

### 10.1 当前判定

| 主张 | 判定 | 原因 |
|---|---|---|
| 连续 functional Tucker 可跨节点数查询 | GO | 代码和 24→32 POC 已验证 |
| boundary distance 是有用输入 | weak GO | 平均为正，但只有一个 validation geometry |
| 完整 SDF/topology generalization | NO-GO | 当前不是 signed field；训练形状和孔洞样本不足 |
| Tucker 优于 CP | NO-GO | 平均仅约 1.9%，只赢 1/3 seeds |
| Tucker 优于 black-box INR | NO-GO | 当前明显落后 |
| 主会完整 paper | NO-GO | 数据、强 baseline、独立 test 均不足 |
| 继续快速 POC | GO | 问题清楚，最小修正成本可控 |

### 10.2 如果下一轮为正，推荐故事

可能标题：

> Geometry-conditioned Functional Tucker for Label-scarce PDE Surrogates on Varying Domains

核心贡献必须保持三点，不能扩写成十个组件：

1. 一个对 varying-domain continuous fields 的显式 functional Tucker formulation；
2. 一个简单但可验证的 geometry conditioning，使孔洞/边界信息进入 factors；
3. 在 sparse training labels 与 new-domain adaptation 下，相对 F-INR/CP/conditional field/operator 的 sample-efficiency phase diagram。

论文卖点不是“我们也能在不规则网格上预测”，GINO、CORAL、Geo-FNO 已经做到。真正可能成立的差异是：

> 在 geometry-solution pairs 很少、每个训练场标签也极稀疏时，显式多 mode 低秩结构是否比 monolithic conditional field 和大 neural operator 更省样本。

### 10.3 如果下一轮仍为负

若真实 CP warm start、更多形状和结构化 masks 后，Tucker 仍不超过 CP/INR，则停止把 dense Tucker core 当贡献。可降级为：

- 共享库中的 method-matched baseline；
- 学生项目：比较 geometry input 与 factorization 的 phase diagram；
- 将 boundary-distance CP 保留为方向 3 的 neural baseline；
- 不再为 Tucker 额外设计 encoder 或 inference。

这不是整个 geometry-aware program 失败，只表示“非对角 core”没有带来可发表价值。

---

## 11. 一手资料

- F-INR, Functional Tensor Decomposition for Implicit Neural Representations: <https://arxiv.org/abs/2503.21507>
- CORAL, Operator Learning with Neural Fields: <https://papers.nips.cc/paper_files/paper/2023/hash/df54302388bbc145aacaa1a54a4a5933-Abstract-Conference.html>
- DeepONet: <https://www.nature.com/articles/s42256-021-00302-5>
- GINO: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html>
- NeuralOperator 官方 GINO 实现: <https://github.com/neuraloperator/neuraloperator>
- DAFNO: <https://arxiv.org/abs/2305.00478>
- DAFNO 官方实现: <https://github.com/ningliu-iga/DAFNO>
- Geo-FNO: <https://arxiv.org/abs/2207.05209>
- Geo-FNO 官方实现: <https://github.com/neuraloperator/Geo-FNO>
- Transolver (ICML 2024): <https://icml.cc/virtual/2024/poster/33751>
- Transolver 官方实现: <https://github.com/thuml/Transolver>
- AirfRANS: <https://arxiv.org/abs/2212.07564>
- AirfRANS 官方文档: <https://airfrans.readthedocs.io/>

---

## 12. 可复现入口

当前 validation POC：

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/python \
  experiments/run_four_track_fast_poc.py \
  --evaluation-split validation \
  --ratio 0.01 \
  --seed 0 \
  --steps 900 \
  --output papers/four_tracks/results/cp_init_validation_seed0.json
```

方向 4 专属测试：

```bash
PYTHONPATH=src /home/ubuntu/project/yanjiu/.venv/bin/pytest -q \
  tests/test_track4_neural_functional.py
```

结果文件只应被解释为 `FAST_POC_NOT_FINAL_EVIDENCE`。任何新模型尺寸、rank 或 geometry input 都只能在新 validation cases 上选择；旧 hole case 不再承担 confirmation test 的职责。
