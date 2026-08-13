# Paper B 中文迭代记录：几何条件化 Phase Tensor Factorization

本记录对应英文 [ITERATIONS.md](../paper_b/ITERATIONS.md)。早期大量 graph smoothing、spectral adapter 与 monolithic IP-NF 实验被保留，因为它们解释了最终为什么必须把几何放进 tensor factor，以及该做法在哪些物理结构上会失败。

## B0：旧 IP-NF 为什么不能作为张量论文结论

旧 Intrinsic Phase Neural Field 把以下特征全部拼接进单个 MLP：

- space coordinate；
- time；
- geometry descriptor；
- signed boundary distance；
- shortest-path distance；
- 多频率 traveling phase。

它在 28→40 cross-resolution 上取得强结果，但没有 mode factors、CP/Tucker core 或 multilinear contraction。因此它证明了 intrinsic phase coordinate 有用，却没有证明 tensor factorization 有用。Refocus 后 IP-NF 被保留为最关键的 geometry-aware no-tensor baseline。

## 前置失败：从 graph spectral adapter 到 intrinsic phase

### 对角 spectral transfer 失败

早期模型学习 \(h(\lambda,t)\)，用 graph eigenvalue 对各 spatial mode 做独立 transfer。在 heterogeneous nonlinear wave 上，unseen NRMSE 约 1.400，RFF 为 1.127，wrong geometry 甚至为 1.195。

诊断：spatially varying speed 和 nonlinear dynamics 会耦合 Laplacian modes；对角 transfer 无法表达 mode mixing。

### intrinsic kernel bank 与 gated residual 失败

尝试多尺度 heat/wave kernels，再接 local neural adapter：

- graph kernel bank：1.439；
- RFF：1.127；
- wrong geometry：1.281。

随后用 zero-gated geometry correction，graph 变为 1.227，但 correct/wrong geometry 几乎相同。模型捕获了 generic smoothing，而没有识别 topology effect。

### few-shot graph context 仍无法隔离 geometry

在 0.5%–2% context、elliptic boundary layer、closed wall 与 narrow door 等 setting 中，graph correction 有时改善 point error，但 correct graph 与 rectangle/wrong graph 基本相同。这说明结果来自普通 context interpolation，不足以证明几何。

### pivot 到 eikonal phase

Near-disconnecting wall-with-door geometry 使 Euclidean distance 与 shortest-path distance 有明确因果差异。独立 Dijkstra/eikonal wavepacket generator 产生沿门传播的相位前沿。Monolithic IP-NF 在旧十 seed 28→40 结果中为 0.6025，明显优于 RFF、SIREN、Neural-CP 与 Euclidean phase。

这个结果确认了“正确传播坐标是 shortest-path phase”，但 tensor refocus 仍需证明多线性分解本身。

## T1：显式 conditional neural CP/Tucker

### 模型

CP：

\[
\widehat u(g,t,x)=\sum_rw_rG_r(e_g)T_r(t)X_r(x;G_g).
\]

Tucker：

\[
\widehat u=\langle\mathcal C,G(e_g)\otimes T(t)\otimes X(x;G_g)\rangle.
\]

其中空间 factor 输入 geodesic distance、SDF、raw coordinate 与 phase bank。它依赖 geometry instance，因此是 conditional spatial factor。

### 工程失败与修复

第一版 runner 每个 optimization step 对 24 个 tasks 分别执行 tiny forward。8 个模型 × 1000 steps 产生近 19 万次小调用，超过八分钟仍未完成。该 run 被中止，不产生科学结论。

修复只是把 observed point 的 geometry/time/space mode features 合并成 batch；公式不变。300-step pilot 降到约 20.7 秒。

### seed-100 pilot

2% observation、24×24、unseen geometry：

| 模型 | NRMSE | high-band |
|---|---:|---:|
| Conditional Tucker | 0.328 | 0.0096 |
| Conditional CP | 0.364 | 0.0118 |
| Diagonal Tucker | 0.469 | 0.0141 |
| IP-NF | 0.843 | 0.0125 |
| Wrong geometry tensor | 1.368 | 0.0529 |
| No-phase tensor | 1.730 | 0.0323 |
| Raw Neural-CP | 1.191 | 0.0135 |
| SIREN | 1.508 | 0.0539 |

单 seed 同时出现 geometry、phase、dense core 与 tensor-vs-IPNF 的正信号。Tucker 只有 16.8k 参数，CP 为 21.0k，因此进入 T2。

## T2：band-gated Tucker——机制无效

### 唯一改动

对五个 phase bands 增加共享 sigmoid gate，并加 `2e-4` mean-gate penalty，希望产生频带选择；factor、rank 与 core 不变。同时加入 parameter-comparable raw F-INR-style Tucker。

### seed-101 结果

- gated Tucker：0.428；
- CP：0.469；
- diagonal Tucker：0.781；
- IP-NF：0.838；
- wrong tensor：1.502；
- raw F-INR Tucker：3.720；
- no-phase：2.851。

但五个 gates 全部收敛在 0.819–0.837，既没有变稀疏，也没有形成有解释力的 band selection。不同 seed 下的 T1/T2 数字不能用来声称 gate 改善。

### 决策

删除 band gate，不继续调 penalty。Plain Tucker 进入 frozen multi-seed cross-resolution test。

## T3：plain Tucker 的 cross-resolution 确认——核心 contract 失败

### protocol

- seeds：100–104；
- train：6 个 24×24 narrow-door geometries；
- test：3 个 unseen 32×32 geometries；
- observation：2%；
- steps：400；
- 主数据：two-packet moving-envelope eikonal field。

### 结果

| 模型 | unseen NRMSE | high-band |
|---|---:|---:|
| Plain conditional Tucker | 0.713±0.024 | 0.01154±0.00200 |
| Conditional CP | 0.721±0.046 | 0.01200±0.00109 |
| Diagonal Tucker | 0.796±0.062 | 0.01458±0.00374 |
| Wrong tensor | 1.414±0.051 | 0.04671±0.00689 |
| No-phase tensor | 1.841±0.564 | 0.02904±0.01022 |
| Raw F-INR Tucker | 1.963±1.019 | 0.03555±0.02635 |
| IP-NF | **0.615±0.029** | **0.01093±0.00271** |
| Neural-CP | 1.212±0.121 | 0.01308±0.00041 |
| SIREN | 1.404±0.153 | 0.04539±0.00426 |

### 诊断

正确几何、phase 与非对角 core 都有贡献，但 IP-NF 比 Tucker 好 13.8%。Shared contract 的 tensor-vs-geometry-aware-flat 条件失败。

根本原因是 moving phase \(d-ct\) 和 Gaussian envelope 都存在 time–space coupling。IP-NF 直接接收 joint phase；Tucker 只能让有限 core 间接合成。

### 决策

不能靠扩大 hidden/core 修数字。下一轮只用精确 trigonometric identity 修正 phase 的数学错位，并保持显式 tensor structure。

## T4：speed-aligned paired-phase CP——在困难数据上仍失败

### 模型变化

对五个 band 和三个 candidate speed 建立四类独立 space/time carriers：

```text
cos(kd) cos(kct)
sin(kd) sin(kct)
cos(kd) sin(kct)
sin(kd) cos(kct)
```

每个 carrier 是一个 CP component。没有网络同时接收 distance 与 time。

### seed-100 falsification

- paired CP：0.8183；
- wrong paired：1.8976；
- plain Tucker：0.6953；
- ordinary CP：0.7067；
- IP-NF：0.6123。

正确与错误几何差异仍巨大，说明 geometry effect 存在；但 paired CP 不如 learned Tucker，也没有追平 IP-NF。

### 诊断

Trigonometric pairing 能精确表达 carrier phase，却不能紧凑表达移动 Gaussian envelope 的 amplitude coupling。预声明 pilot 失败，因此没有扩 seed。

### 故事调整

前四轮都集中在最难的 moving-envelope setting，不应据此断言 tensor idea 不工作。用户目标允许选择更简单、与低秩物理结构相符的 regime。T5 保持同一个 paired CP，改变数据故事，而不是继续加模型模块。

## T5：moderate-rank eikonal harmonics——最终正结果

### 独立生成器

\[
u_g(x,t)=\sum_{b=1}^{3}A_b(e_g)e^{-\nu_bt}
\cos(k_bd_g(x,s)+\varphi_b)+0.06r_g(x,t).
\]

主结构是三个 geometry-dependent standing/damped harmonics；额外 6% moving residual 不属于 dominant rank-3 structure。生成器不调用 neural decoder，也不是 fitted model 的抽样。

### exploratory selection

Seeds 200–204、1%、24→32：

- paired CP：0.0900±0.0110；
- ordinary Geo-CP：0.1160±0.0088；
- IP-NF：0.1865±0.0202；
- wrong paired：1.6138±0.1380。

这些 seed 用于确认该 regime 值得冻结，不进入最终统计。

### frozen confirmation

- seeds：300–309；
- ratio：1%；
- train/test resolution：24→32；
- model/rank/steps 不再改变；
- 每个 seed 内先聚合 9 个 unseen tasks，再做 paired inference。

| 模型 | unseen NRMSE | high-band NRMSE |
|---|---:|---:|
| Paired geometry CP | **0.0952±0.0144** | **0.00404±0.00092** |
| Ordinary geometry CP | 0.1113±0.0213 | 0.00497±0.00113 |
| IP-NF | 0.1825±0.0173 | 0.00728±0.00122 |
| Wrong paired CP | 1.5598±0.1827 | 0.08799±0.01176 |
| Raw F-INR Tucker | 1.8167±0.2535 | 0.06079±0.00217 |
| SIREN | 1.0944±0.0366 | 0.05938±0.00318 |

统计：

- paired CP vs ordinary Geo-CP：14.5% 改善，CI 3.4–23.8%，\(p=0.0391\)；
- paired CP vs IP-NF：47.9%，CI 41.5–53.2%，\(p=0.00195\)；
- paired CP vs wrong paired：93.9%，\(p=0.00195\)；
- high-band vs Geo-CP：18.7%，\(p=0.0332\)；
- high-band vs IP-NF：44.4%，\(p=0.00195\)。

### T5 支持的精确结论

在具有近似 multilinear time × intrinsic-space structure 的物理场中：

1. 正确 geodesic geometry 是必要的；
2. 普通 functional tensorization 不够，intrinsic phase 必须进入 spatial factor；
3. phase-paired tensor structure 比普通 Geo-CP 有额外收益；
4. 显式 tensor bottleneck 在 1% 下比 monolithic geometry INR 更有样本效率；
5. 该结论不推广到 moving localized envelope。

## Baseline 因果解释汇总

| 比较 | 能得出的结论 |
|---|---|
| Paired CP vs wrong paired | shortest-path geometry 的贡献 |
| Paired CP vs ordinary Geo-CP | paired phase factorization 的贡献 |
| Paired CP vs IP-NF | 显式 tensor bottleneck 相对 joint geometry model 的贡献 |
| Paired CP vs raw F-INR Tucker | intrinsic geometry 相对普通 functional tensor 的贡献 |
| Tucker vs diagonal Tucker | dense cross-mode core 的贡献 |
| phase vs no-phase | phase bank 的贡献 |
| T5 vs T3/T4 | 数据近似 multilinear rank 是方法成功的前提，而不是任意任务难度 |

## 最终支持与不支持的结论

### 已支持

- 1% observation 下对 unseen geometry 和新 resolution 的迁移；
- correct geometry 与 explicit tensorization 的独立正贡献；
- paired carrier 对 high-frequency reconstruction 的作用；
- 近似低秩物理场中的 sample efficiency。

### 未支持

- 对任意 nonlinear/moving-envelope wave field 都优于 IP-NF；
- 已学得未知 travel metric；当前 shortest-path geometry 是已知 metadata；
- 对真实多几何 acoustic/scattering 数据已经验证；
- 参数更少：paired CP 比 IP-NF 参数更多，优势是稀疏样本下的 inductive bias；
- raw discrete CP 能直接做 zero-shot mesh/geometry transfer。

## 复现入口

- 方法：[neural_tensor.py](../../src/geoaware/neural_tensor.py)
- 数据/geometry：[neural_geometry.py](../../src/geoaware/neural_geometry.py)
- runner：[paper_b_tensor_run.py](../../experiments/paper_b_tensor_run.py)
- analyzer：[paper_b_tensor_analyze.py](../../experiments/paper_b_tensor_analyze.py)
- 最终统计：[tensor_t5_confirm_summary.json](../paper_b/results/tensor_t5_confirm_summary.json)
- 英文主稿：[DRAFT.md](../paper_b/DRAFT.md)

## 后续 B-T6 至 B-T8：phase-envelope 的严格去留实验

按照“只有效果显著才纳入”的审阅意见，先后实现 learned rank-Q separable envelope 与 fixed-RBF envelope Tucker。两者均严格分开处理 distance/time，不存在 joint MLP bypass，并修复了 baseline 初始化随模型列表变化的公平性问题。

Learned Q=2 将 moving-envelope NRMSE 从 paired CP 的 `0.897±0.033` 降到 `0.644±0.118`，但没有稳定超过 IP-NF `0.624±0.027`；RBF Tucker 为 `0.684±0.088`。因此两者均不进入 Paper B 主方法，只作为负结果保留。完整过程见[三轮持续迭代记录](三轮持续迭代记录.md)。
