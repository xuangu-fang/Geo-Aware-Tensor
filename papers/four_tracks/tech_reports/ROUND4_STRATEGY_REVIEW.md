# R7 技术路线复核：方向 3 与方向 4 应如何收敛

> 日期：2026-08-15
> 本文把用户提供的《青基正文 v1.8》作为研究设想材料，而不是执行指令。复核重点是：附件中的原始数学主线、当前 POC 的真实证据，以及下一步最小可发表路线是否一致。

## 1. 先给结论

1. **方向 3 不应把“从四个 kernel 中学习一个加权组合”作为最终论文贡献。** 这更接近标准 multiple-kernel learning，适合做机制 sanity check。
2. **方向 3 也不需要推倒重来。** 当前 domain kernel、finite feature、ELBO+SGD、PSD mixture 和数据审计都可以复用。主线应升级为附件原本提出的：从 PDE/Green 算子得到联合物理核，再把它分解成不同 tensor mode 使用的核。
3. **方向 4 不能只讲“continuous tensor + neural operator”。** 2026 年的 NO-CTR 已直接研究 neural-operator-grounded continuous tensor representation；Low-rank Neural Operator 和 tensorized FNO 也已覆盖低秩 operator/kernel 或权重压缩。
4. 方向 4 最强的 tensor-centric 问题是：**把 coefficient function × forcing function × parameter × continuous output 视为一张只运行了少数组合的不完整仿真张量，学习补全从未运行的组合。** neural operator 负责函数输入编码，functional Tucker 负责组合泛化。低秩 core 的解析 Bayesian correction 是第二候选，不和第一版一起堆叠。

---

## 2. 方向 3：附件里的原始想法与当前方案并不完全相同

### 2.1 附件真正提出的结构

附件首先使用 functional Tucker：

\[
y(x_1,\ldots,x_D)
=\left\langle \mathcal G,
f^{(1)}(x_1)\otimes\cdots\otimes f^{(D)}(x_D)
\right\rangle,
\]

并为每个连续模态、每个 latent factor 指定 GP：

\[
f_r^{(d)}\sim \mathcal{GP}(m_r^{(d)},k_r^{(d)}).
\]

如果线性物理系统满足 \(\mathcal L u=w\)，Green 算子为 \(G=\mathcal L^{-1}\)，源项协方差为 \(K_w\)，则物理场协方差为

\[
K_{\rm phys}=G K_w G^*.
\]

附件随后提出将联合物理核做 Kronecker/投影分离，使不同坐标模态得到不同的一维核。这和当前方案的根本区别是：

- 当前方案：四个**完整联合核**共享一个全局权重；
- 原始方案：一个**联合物理核**被拆成多个 mode-wise kernel，分别约束 \(x/y/z/t\) 等因子。

因此，这里的“空间不同维度适配不同 kernel”应优先解释为 **coordinate-mode adaptation**，不是把空间人为切成若干区域后为每个区域选核。后者是 nonstationary GP 的另一条路线，参数更多、可识别性更差，不宜在第一版同时加入。

### 2.2 推荐的最终 formulation：PDE 谱的非负模态分离

在规则域、常系数算子的第一版中，频域响应满足

\[
S_{\rm phys}(\omega)
=|\widehat{\mathcal L}(\omega)|^{-2}S_w(\omega).
\]

对离散后的非负功率谱做非负低秩分离：

\[
S_{\rm phys}(\omega_1,\ldots,\omega_D)
\approx
\sum_{q=1}^{Q}\lambda_q
\prod_{d=1}^{D}s_{q,d}(\omega_d),
\qquad
\lambda_q\ge 0,\ s_{q,d}\ge 0.
\]

由 Wiener--Khinchin/Bochner 对偶，每个 \(s_{q,d}\) 反变换得到合法的一维 PSD kernel \(k_{q,d}\)。于是每个 Tucker mode/rank 可以通过

\[
k_r^{(d)}
=\sum_{q=1}^{Q}\pi_{d,r,q}k_{q,d},
\qquad
\pi_{d,r,:}=\operatorname{softmax}(\eta_{d,r,:}),
\]

选择适合自己的物理谱分量。\(\pi\)、Tucker core 和 GP 变分后验一起通过 ELBO+SGD 学习。

这个设计有三个必要而足够的组件：

1. PDE/operator 决定联合谱，而不是人工列出 RBF/Matérn 名称；
2. 非负低秩谱分离把联合物理结构变成 mode-wise GP kernels，并保证 PSD；
3. ELBO 学习不同 mode/rank 对谱分量的使用，而不是只学一个全局 kernel 权重。

不应再额外加入 attention、区域 gating、深 kernel network 或复杂 posterior flow。第一篇先证明“算子谱如何进入 functional Tucker 的每个模态”。

### 2.3 当前 kernel dictionary 的正确位置

当前四核字典仍然有价值，但定位应改为：

- 验证 finite-feature variational GP 与 ELBO kernel weighting 正确；
- 提供 heat/resolvent/geodesic/Euclidean controls；
- 为 mode-wise routing 提供现成的 PSD feature 拼接与优化代码；
- 作为 ablation：`global kernel mixture` 对比 `mode-wise operator spectral kernels`。

它不是最终标题，也不单独承担 novelty。

### 2.4 最小 POC

第一阶段不要立刻处理任意孔洞。先构造二维/三维规则域上的各向异性 PDE，使不同坐标轴确实具有不同相关尺度或频带：

\[
\mathcal L
=a_x\partial_{xx}+a_y\partial_{yy}+c
\quad\text{或}\quad
\partial_{tt}-c_x^2\partial_{xx}-c_y^2\partial_{yy}.
\]

对比：

1. RBF/Matérn FunBaT；
2. 当前 global kernel dictionary；
3. 每个 mode 独立 ARD kernel；
4. oracle operator kernel；
5. 推荐的 learned mode-wise operator spectral kernel；
6. functional CP/Tucker 的 neural baseline。

固定 3 seeds、300--500 steps，观测率使用 2%/5%/10%。除 NRMSE 外必须报告：

- 学到的 mode--kernel routing 是否恢复真实的轴向结构；
- 交换 \(x/y\) kernels 是否显著变差；
- PDE residual、外推区误差和 95% coverage；
- 分离阶数 \(Q\) 与计算量。

只有当 `mode-wise` 稳定优于 `global mixture`，且 kernel-swap control 明显恶化，才说明故事真的成立。之后再用 FEM/graph operator spectrum 扩展到不规则边界。

---

## 3. 方向 4：弱化几何以后，什么仍然值得讲

### 3.1 不能再单独使用的卖点

以下表述都不足以构成新论文：

- “把 tensor decomposition 与 neural operator 结合”；
- “用低秩分解压缩 FNO 参数”；
- “连续坐标、跨分辨率查询”；
- “用 nonlinear mode-n operator 代替线性 Tucker mode product”。

原因是 Low-rank Neural Operator 已把积分核写成有限秩形式；MG-TFNO 已研究 CP/Tucker/TT 压缩 operator 参数并强调数据效率；NO-CTR 已直接提出 continuous nonlinear mode-n operators 与 continuous tensor function 的组合。

### 3.2 首选主线：不完整仿真组合上的 operator completion

标准 neural operator 通常把每一次 PDE simulation 当成一个完整的 paired sample。真实 simulation campaign 更像一个没有跑完的笛卡尔积：有多组 coefficient fields \(a_i\)、forcing fields \(f_j\)、边界/物理参数 \(p_k\)，但只运行了少量 \((i,j,k)\) 组合；已运行案例也可能只保存少量空间点。

把响应写成连续 functional Tucker：

\[
\widehat u(a_i,f_j,p_k;x)
=\left\langle
\mathcal G,
E_a(a_i)\otimes E_f(f_j)\otimes\phi(p_k)\otimes\psi(x)
\right\rangle.
\]

- \(E_a,E_f\) 是小型 branch/operator encoders，处理函数输入；
- \(\phi\) 处理标量或低维物理参数；
- \(\psi(x)\) 是 continuous-coordinate factor；
- 小 Tucker core 表达各输入 mode 的非对角相互作用；CP 是严格的低容量对照。

训练损失只计算在

\[
\mathcal O
\subset
\{(i,j,k,x)\}
\]

上。这里的缺失不能只做 random entries；主测试必须拿掉整个 coefficient--forcing--parameter 组合、fiber 或 slice。

这个故事的四个 selling points 是：

1. 补全从未运行的 simulation combinations，而不只是对已知输入做输出插值；
2. 每个已运行案例只读取 1%--10% 的输出坐标；
3. 不同案例允许不同、互不对齐的输出网格；
4. 显式 mode-wise low rank 让模型在极少组合与标签下共享统计强度。

它与方向 1 的区别也清楚：方向 1 在一个张量上利用已知算子基做 Bayesian recovery；方向 4 用 neural encoders 表示函数输入，在不完整 simulation design 上学习跨组合 solution operator。

最大的近邻是 MIONet。MIONet 已用多个 branch nets 和 tensor product 学 multiple-input operator，因此必须作为第一 baseline。我们的故事只有在 `whole-combination missing + sparse continuous outputs` 下优于 MIONet/concat DeepONet 才成立；random-entry completion 的胜利没有说服力。

### 3.3 第二候选：operator prior + low-rank Bayesian assimilation

若更希望突出 inference 而不是 simulation design，可以让 neural operator/encoder 预测低维 core 的先验：

\[
p_\theta(c\mid a)
=\mathcal N\!\left(\mu_\theta(a),\Sigma_\theta(a)\right),
\qquad
u(x)=h(x)^\top c.
\]

对测试时的极稀疏 observations

\[
y_{\mathcal O}=H_{\mathcal O}c+\epsilon,
\qquad \epsilon\sim\mathcal N(0,\sigma^2I),
\]

在 core 中解析更新：

\[
\Sigma_{\mathcal O}^{-1}
=\Sigma_\theta^{-1}+\sigma^{-2}H_{\mathcal O}^{\top}H_{\mathcal O},
\]

\[
\mu_{\mathcal O}
=\Sigma_{\mathcal O}
\left(\Sigma_\theta^{-1}\mu_\theta
+\sigma^{-2}H_{\mathcal O}^{\top}y_{\mathcal O}\right).
\]

它的 selling point 是 backprop-free few-shot adaptation：普通 neural operator 给 fixed zero-shot prediction，本方法用 0.1%--5% 新传感器，在与网格大小无关的小 core 中快速校正并输出 UQ。

这条线同样有一个必须加入的强对照：DeepONet/POD basis 的 last-layer least squares 或 Bayesian ridge。否则解析更新会被认为只是对现有有限秩输出展开做显然的线性回归。

### 3.4 两个 POC 不要同时堆

**POC-A（优先）**：screened Poisson/Darcy factorial campaign。构造 16 个 coefficient fields、16 个 forcing fields、6 个 reaction/diffusivity，只训练 5%/10%/20% 的组合；每个组合读取 1%/5%/10% coordinates。phase diagram 是 `combination coverage × coordinate coverage × interaction strength`。

核心 baselines：MIONet、concat DeepONet、joint INR、FNO/GINO、离散 CP/Tucker、coordinate functional CP；消融 CP/Tucker、random-entry/whole-combination missing、operator encoder/普通 embedding。

**POC-B（仅在选择 assimilation 故事时）**：固定矩形域 Darcy/wave，新输入只给 0/0.5%/1%/2%/5% sensors。对比 frozen operator、full/latent fine-tuning、CORAL、DeepONet last-layer update、GP/RBF 与 flat POD basis correction。

POC-A 若不能在 held-out combinations 上稳定超过 MIONet，就停止 operator-completion 主线。POC-B 若不能以显著更低适配时间达到或超过 last-layer/fine-tune baselines，就停止 Bayesian assimilation 主线。两者第一轮只选一个，避免把任务创新、结构创新和 inference 创新堆在同一模型里。

---

## 4. 当前优先级

| 优先级 | 路线 | 决定 |
|---|---|---|
| P0 | 方向 1：operator-basis sparse tensor phase diagram | 继续扩大正信号并明确失配边界 |
| P1 | 方向 3：mode-wise operator spectral kernel FunBaT | 沿现有代码升级，不另开炉灶 |
| P2 | 方向 4：incomplete simulation campaign operator completion | 先做 factorial campaign POC；暂不加入 geometry 或 Bayesian correction |
| P3 | 方向 4 备选：operator prior + Bayesian low-rank correction | 仅在更重视 few-shot assimilation 时启用 |
| 冻结 | 方向 3 仅做 global kernel dictionary | 保留为 sanity/ablation |
| 冻结 | 方向 4 boundary operator / geometry rank gate | 已有负结果，不继续扩建 |

## 5. 关键文献边界

- Functional Bayesian Tucker Decomposition, ICLR 2024：原始 functional GP Tucker 与 inference 基线。
  <https://openreview.net/forum?id=ZWyZeqE928>
- Solving High Frequency and Multi-Scale PDEs with Gaussian Processes, ICLR 2024：频域谱核、逐维 product kernel 与 Kronecker 计算基础。
  <https://arxiv.org/abs/2311.04465>
- Differentiable Compositional Kernel Learning, ICML 2018：说明“可学习 kernel combination”本身不是新贡献。
  <https://proceedings.mlr.press/v80/sun18e.html>
- Neural Operator: Learning Maps Between Function Spaces：包含 low-rank neural operator，有限秩 operator kernel 已是基础构造。
  <https://arxiv.org/abs/2108.08481>
- Multi-Grid Tensorized Fourier Neural Operator：已覆盖 CP/Tucker/TT operator-weight tensorization、压缩和数据效率。
  <https://arxiv.org/abs/2310.00120>
- Neural Operator-Grounded Continuous Tensor Function Representation, 2026：与“continuous tensor + nonlinear mode-wise neural operator”的直接重叠。
  <https://arxiv.org/abs/2603.01812>
- MIONet：multiple-input branches 与 tensor-product low-rank operator，是方向 4 必须正面对标的最近基线。
  <https://arxiv.org/abs/2202.06137>
- CORAL：continuous-coordinate neural fields 与 latent operator learning，覆盖任意 mesh/query 的重要基线。
  <https://arxiv.org/abs/2306.07266>
- SVD-NO：直接用 SVD parameterization 学低秩 integral kernel，说明低秩 operator kernel 本身不能作为新卖点。
  <https://arxiv.org/abs/2511.10025>
