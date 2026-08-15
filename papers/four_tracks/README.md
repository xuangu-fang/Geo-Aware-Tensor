# 四条研究线的统一地图（当前单一事实源）

更新时间：2026-08-15。

快速 POC 的逐轮负信号、修正与冻结记录见
[`ITERATIONS.md`](ITERATIONS.md)。

## 一句话总纲

这个项目只研究一个核心问题：**当物理场定义在不规则边界、孔洞或非均匀网格上时，怎样把这种几何放进低秩函数张量，而不是只把坐标交给一个黑盒网络。**

四条线不是互相替换，而是四种不同的几何入口：

| 方向 | 几何放在哪里 | 最短故事 | 当前优先级 | 合适定位 |
|---|---|---|---|---|
| 1. Operator-informed Bayesian Tucker | 因子的算子谱先验 | 用已知物理/几何算子压缩贝叶斯 Tucker | P1，保留并补强 baseline | 学生项目或中等规模方法论文；外部数据与 UQ 做强后可升级 |
| 2. Phase-factorized Wave Tensor | 波的传播相位 | 三角恒等式把行波变成显式低秩 CP | P2，应用导向 | 波动、声学、浅水等专项论文，不宣称通用几何学习 |
| 3. Domain-kernel Bayesian Functional Tucker | 域上的 GP covariance | 用不规则域 Matérn/heat kernel 定义连续 Tucker 因子 | P0/P1，高风险高回报 | 若完成可扩展后验与跨域 UQ，可冲概率 ML/AI 主会；否则是完整学生论文 |
| 4. Geometry-conditioned Neural Functional Tucker | SDF 条件的神经函数因子 | 在变形网格上共享显式 neural Tucker，而非 monolithic INR | P0，最快形成强主线 | 最有希望先发展成 AI 主会完整稿；必须对标 GINO/DAFNO/Geo-FNO/Transolver |

这里的“优先级”是投入顺序，不是价值排序。方向 1 和 2 的代码、结果与论文草稿均保留。

## 方向 1：Operator-informed Bayesian Tucker

### 最简公式

\[
\mathcal X=\mathcal G\times_1(\Phi_1W_1)\times_2(\Phi_2W_2)\times_3(\Phi_3W_3),
\qquad
p(W_m)\propto\exp\{-\tfrac12\Vert(1+\Lambda_m)^{p/2}W_m\Vert_F^2\}.
\]

\(\Phi_m,\Lambda_m\) 来自该 mode 的几何或物理算子。它可以被理解为完整 GP 因子的一个**有限谱、低秩、计算便宜的近似**，而不是要被方向 3 删除的旧版本。

### 独特贡献

1. 对固定张量做极低观测率补全，明确保留小 Tucker core。
2. 几何通过 mode operator 进入先验，而不是通过增加网络容量进入。
3. 条件于因子后，对 core 做解析高斯后验，给出轻量 UQ。

### 必须对标

- discrete/flat Bayesian Tucker；
- wrong operator 与 topology-erased operator；
- functional CP、functional Tucker、neural CP/Tucker；
- flat operator GP；
- 参数量和计算量匹配的 INR。

这条线不能只对标传统 Bayesian Tucker。现有 controlled tensor 结果是正信号，但不规则椭圆场上 operator Tucker 弱于 SDF functional CP，说明它目前更适合作为“算子先验何时降低样本复杂度”的论文，而不是通用不规则域 SOTA。

## 方向 2：Phase-factorized Wave Tensor

### 最简公式

对传播相位 \(d_\Omega(x,s)-ct\)，

\[
\cos(k[d_\Omega-ct])
=\cos(kd_\Omega)\cos(kct)+\sin(kd_\Omega)\sin(kct).
\]

因此空间与时间不需要进入同一个 MLP，行波可由少量显式 CP 项表示。

### 独特贡献

1. 低秩不是经验假设，而是由波的三角恒等式导出。
2. correct/wrong travel distance 是直接的几何因果消融。
3. 模型可检查每个频带、速度和包络秩。

### 论文边界

这是一条**特殊场结构**路线。主数据应使用 WaveBench、自建 Helmholtz/声学或浅水波，不再把它包装成任意 PDE 的通用 geometry model。现有受控 harmonic 数据为正；moving envelope 和 The Well 为负。下一步只保留一个很小的 phase-envelope 扩展，并首先通过“绝对有效”门槛。

## 方向 3：Domain-kernel Bayesian Functional Tucker

### 最简公式

在不规则域 \(\Omega\) 上定义 Matérn 型核：

\[
k_\Omega(x,x')=\sum_j
\phi_j(x)\phi_j(x')(1+\ell^2\lambda_j)^{-\nu},
\qquad f_{mr}\sim\mathcal{GP}(0,k_{\Omega_m}).
\]

再把这些连续因子放入 Tucker：

\[
u(z_1,z_2,z_3)=\sum_{a,b,c}G_{abc}
f_{1a}(z_1)f_{2b}(z_2)f_{3c}(z_3).
\]

当前快速 POC 用多个 \(k_\Omega(x,s)\) kernel section 做有限特征，并以二次正则做 MAP；这等价于核/GP 后验均值的有限基近似。**目前还不能称为完整 Bayesian GP 方法**。只有 POC 过门槛后，才实现 inducing-point variational posterior、核超参数后验和校准 UQ。

### 与方向 1 的本质区别

- 方向 1 的对象是一个固定离散张量，算子主要是 mode table 的正则器。
- 方向 3 的对象是连续因子函数，kernel 定义函数空间；跨网格预测是方法语义的一部分。
- 方向 1 强调“便宜、显式、条件 core 后验”；方向 3 强调“非欧域 GP、少样本适配和不确定性”。

### 快速 POC 结果

训练使用 4 个不规则域的 24 网格、每个完整张量仅 1% entries；评估在未见形状的 32 网格上。模型和超参数先在 slanted-channel validation 冻结，再读取带孔洞的 test geometry。下表为 3 个独立训练 seed 的 mean ± sample standard deviation。

| 模型 | validation NRMSE | validation 边界 | hole test NRMSE | hole test 边界 |
|---|---:|---:|---:|---:|
| domain-kernel functional Tucker | 0.1839±0.0122 | **0.1851±0.0124** | **0.1526±0.0148** | **0.1667±0.0169** |
| topology-erased kernel Tucker | 0.1962±0.0185 | 0.2091±0.0094 | 0.1731±0.0140 | 0.1998±0.0185 |
| SDF neural functional Tucker | 0.1931±0.0181 | 0.2192±0.0255 | 0.1756±0.0151 | 0.2053±0.0089 |
| coordinate neural functional Tucker | 0.2027±0.0243 | 0.2213±0.0214 | 0.1854±0.0142 | 0.2354±0.0187 |
| SDF neural functional CP | 0.1972±0.0253 | 0.2211±0.0159 | 0.1587±0.0152 | 0.1971±0.0135 |
| joint SDF INR | **0.1775±0.0219** | 0.2093±0.0298 | 0.1592±0.0078 | 0.1960±0.0057 |
| observed global mean | 1.0009±0.0008 | 1.0003±0.0007 | 1.0006±0.0010 | 1.0012±0.0014 |

当前可下的结论只有：

- 在普通未见形状上，domain kernel 尚未超过 joint INR；不能宣称整体胜出。但 correct kernel 在 3/3 seeds 均优于 topology-erased kernel，平均边界误差降低约 11.5%。
- 在带孔洞的未见域上，正确 domain kernel 相对 topology-erased kernel 平均降低约 11.8% 全域误差、16.6% 边界误差。
- 在 hole test 上，它也平均优于 joint INR/SDF-CP 的全域误差约 4%，边界误差约 15%。这是方向 3 值得继续的第一条稳定正信号。
- 以上仍只有两个 held-out geometries；3 seeds 只够做 POC，不是论文统计证据。

## 方向 4：Geometry-conditioned Neural Functional Tucker

### 最简公式

\[
u(\Omega,p,x,s)=\sum_{a,b,c}G_{abc}\,
g_a(q_\Omega)\,h_b(p)\,r_c(x,s,\operatorname{SDF}_\Omega(x)).
\]

三个 mode factor 都是连续函数，但交互只能经过一个小 Tucker core。最小版本只用坐标、source、物理参数和 SDF；暂不加入复杂几何 encoder、attention 或 learned metric。

### 独特贡献

1. 可在不同节点数与不同网格上查询，不要求规则矩形张量。
2. SDF 让内部孔洞和外边界以连续条件进入 factor。
3. 与 joint INR 的区别不是“也用了 SDF”，而是显式低秩 core、可控 mode ranks 与可解释的 factor sharing。

### 当前证据边界

- hole test 中，SDF Tucker 相对 coordinate Tucker 平均降低约 5.3% 全域误差和 12.8% 边界误差，说明 SDF 确实在使用孔洞几何。
- SDF Tucker 当前平均不如更简单的 SDF CP。因此现阶段只支持“geometry-conditioned functional factorization 能跑通”，不支持“Tucker core 已经带来性能贡献”。
- joint SDF INR 在 validation 上平均最好。方向 4 若要成为主会论文，必须先解决 Tucker 优化/秩选择，再在多形状、多孔洞、cross-resolution 和 structured missing 下稳定超过它，并与 GINO/DAFNO/Geo-FNO/Transolver 对比。

## 四条线共享什么，不共享什么

共享：

- `DomainCase`：coordinates、mesh/graph operator、boundary mask、SDF、source、physical parameters、target field；
- `ObservationSplit`：entry-random、fixed sensors、fiber missing、region missing，且精确记录 ratio；
- 模型协议：`fit(train_cases, observations)` 与 `predict(case, query_indices)`；
- 指标：global NRMSE、boundary/hole-band NRMSE、absolute skill、参数量、耗时；
- 数据门禁：train/validation/test 按 geometry 分组，跨分辨率不泄漏；
- baseline discipline：mean、discrete CP/Tucker、functional/neural CP/Tucker、joint INR、wrong geometry、强 neural operator。

不共享：

- 方向 1 的后验实现；
- 方向 2 的 phase carrier；
- 方向 3 的 GP/kernel inference；
- 方向 4 的 neural factor architecture。

## 当前库结构决策

现阶段继续使用一个 monorepo，不立即拆四个仓库。建议逐步形成：

```text
src/geoaware/
  common/                       # schema, masks, metrics, evaluation
  tracks/operator_tucker/       # 方向 1
  tracks/phase_wave/            # 方向 2
  tracks/domain_kernel_tucker/   # 方向 3
  tracks/neural_functional/      # 方向 4
experiments/
  shared/
  track1_operator_tucker/
  track2_phase_wave/
  track3_domain_kernel/
  track4_neural_functional/
papers/
  four_tracks/                   # 本路线图与共享 POC
  paper_a/                       # 现有方向 1，保留
  paper_b/                       # 现有方向 2，保留
```

现在不做大规模搬家，避免路径 churn；新代码先按新模块落地，旧脚本通过 adapter 逐步迁移。出现以下任一条件时再拆 repo：

1. 某方向通过 3+ datasets、10 seeds 和强 baseline 的论文门槛；
2. 有独立学生长期负责，release cadence 已分离；
3. GP 与 neural-operator 依赖明显冲突；
4. WaveBench/CFD 数据与求解器使应用代码体积独立膨胀。

届时保留一个轻量 `geoaware-core` 包，四个 paper repo 依赖同一 schema/metrics；不使用复制粘贴的四套评估代码。

## 接下来三步

1. **先加 seed，不加新组件。** 对方向 3/4 在 validation 跑 3 个 selection seeds；若 correct-vs-erased 与 SDF-vs-coordinate 不稳定，立即降级故事。
2. **再扩数据。** 首选 AirfRANS scarce split 验证 SDF/孔洞边界，WaveBench 只服务方向 2；同时自建多孔洞 Poisson/Helmholtz phase diagram。
3. **最后加重 baseline。** 方向 4 接 GINO/DAFNO/Transolver；方向 3 接 Euclidean RBF GP、geometry-only GP、functional Bayesian tensor。只有此时仍有优势，才投入完整变分 GP。

## 相关一手工作与代码

- Domain Agnostic FNO: <https://arxiv.org/abs/2305.00478>
- Geometry-Informed Neural Operator: <https://arxiv.org/abs/2309.00583>
- NeuralOperator 官方实现（含 GINO）: <https://github.com/neuraloperator/neuraloperator>
- Geo-FNO 官方实现: <https://github.com/neuraloperator/Geo-FNO>
- AirfRANS 官方数据接口: <https://github.com/Extrality/airfrans_lib>
- WaveBench 官方仓库: <https://github.com/wavebench/wavebench>
- Functional tensor-train: <https://arxiv.org/abs/1510.09088>
- GP nonparametric tensor estimator: <https://proceedings.mlr.press/v48/kanagawa16.html>
- Nonparametric decomposition of sparse tensors: <https://proceedings.mlr.press/v139/tillinghast21a.html>
- 近期 multioutput-GP functional Bayesian tensor work: <https://arxiv.org/abs/2512.21486>
