# 方向 3 技术报告：Domain-kernel Bayesian Functional Tucker

> 状态：**机制 POC 已有正信号，但真正的 Bayesian GP 模型尚未实现。** 目前代码只能称为“domain-kernel section conditioned neural Tucker”。本文会严格区分“已经做成的东西”和“希望投稿时声称的东西”。

## 0. 先给结论

这个方向最简洁的研究问题是：

> 在不规则边界、孔洞和变化网格上，能否把欧氏空间中的 GP functional Tucker，替换为定义在物理域本身上的 GP functional Tucker，从而同时获得稀疏观测下的几何归纳偏置、跨分辨率预测和可信不确定性？

当前证据支持较弱但明确的第一步：在相同 Tucker 网络中，把以欧氏距离构造的 RBF sections 换成由不规则域 Laplacian 构造的 intrinsic sections，在未见形状、24→32 跨分辨率、1% 训练条目条件下，三 seed 验证误差从 `0.3320` 降到 `0.2602`；加入相同局部坐标/SDF 后，两者分别为 `0.2031` 与 `0.1905`，边界误差分别为 `0.2267` 与 `0.1905`。

但这还不能证明“Bayesian GP Tucker”：当前没有显式 Gaussian prior、variational posterior、ELBO、posterior variance 或 calibration。下一里程碑必须完成真实 GP 残差因子和可检查的 posterior predictive distribution。

## 1. 四条线中本方向的独立故事

方向 1 是已知算子基底上的有限参数 Bayesian Tucker，重点是强结构先验和极低观测率下的张量恢复。方向 3 不应只是把方向 1 改一个 kernel 名字；它应当回答另外三个问题：

1. 因子是连续函数，而不是某个固定网格上的 factor table；
2. 几何通过域上的 covariance/prior 进入，而不只是通过固定谱基；
3. 新域上的 posterior adaptation 和不确定性是论文贡献，而不是只给点预测。

因此建议标题暂定为：

**Domain-kernel Bayesian Functional Tucker for Sparse Fields on Irregular Domains**

最有价值的应用场景不是完全零样本的 PDE surrogate，而是：已有多个训练域，在一个新的不规则域上只拿到极少传感器或少量工况观测，需要恢复整个多模物理场并给出 uncertainty。完全零样本可以报告，但不应成为 GP 版本唯一任务，因为独立的零均值 domain GP 在新域没有观测时，其 posterior mean 必然回到零。

## 2. 数据对象与任务定义

令第 (c) 个物理域为 ‎(Omega_c\subset\mathbb R^d)，它可以有不规则外边界和内部孔洞。一个观测写作

\[
\mathcal D=\{(c_i,s_i,a_i,x_i,y_i)\}_{i=1}^{N_{\rm obs}},
\]

其中：

- (s_i\in\Omega_{c_i})：源位置或激励位置；
- (a_i\in\mathcal A)：扩散系数、频率、Reynolds number 等工况参数；
- (x_i\in\Omega_{c_i})：查询点；
- (y_i=u_{c_i}(s_i,a_i,x_i)+\epsilon_i)：物理场值；
- ‎(epsilon_i\sim\mathcal N(0,\sigma^2))。

当前合成数据的张量语义是 `[source, diffusivity, irregular-domain node]`，每个域有 4 个 source、14 个 diffusivity 和约 300–900 个 active nodes。

需要明确区分三种任务：

### 2.1 同域 tensor completion

训练和测试来自同一个 ‎(Omega_c)，只隐藏条目。这是最接近传统 tensor completion 的设置，也最适合与 CP/Tucker/FunBaT 公平比较。

### 2.2 新域 few-shot adaptation（主任务）

测试域在训练中从未出现，但允许观察测试域的 0.1%–5% 条目，再恢复其余条目。这里 domain kernel 的 posterior adaptation 和 UQ 才真正有用。

### 2.3 新域 zero-shot operator prediction

测试域完全没有目标观测，只给几何、source 和参数。这个任务需要共享的跨域 mean function 或 cross-domain covariance；仅有互相独立的 domain GP 不能完成 zero-shot transfer。

当前 POC 实际做的是第 2.3 类：1% 只施加在四个训练形状上，验证/测试新形状一个目标值也没有看到。因此现有实验不应描述为“在新域上用 1% 观测恢复”。

## 3. 投稿版本的严格 formulation

### 3.1 域上的 Matérn-like covariance

令 (L_{\Omega_c}) 为带指定边界条件的正半定 Laplacian，特征对满足

\[
L_{\Omega_c}\phi_{cj}=\lambda_{cj}\phi_{cj}.
\]

一个谱截断的 domain Matérn covariance 可写为

\[
k_{\Omega_c}(x,x';\kappa,\nu)
=\sigma_f^2\sum_{j=0}^{J-1}
(\kappa^2+\lambda_{cj})^{-(\nu+d/2)}
\phi_{cj}(x)\phi_{cj}(x').
\]

因为 covariance 使用 ‎(phi_j(x)phi_j(x'))，单个 eigenvector 的任意正负翻转不会改变 kernel。孔洞和凹边界通过 ‎(L_{\Omega_c}) 改变传播关系；欧氏距离很近、但隔着孔洞或墙面的点，不再必然高度相关。

这里必须谨慎：Riemannian Matérn 的经典谱公式通常在紧致无边界 manifold 上陈述；本项目使用的是带 reflecting/Neumann 边界的离散 graph Laplacian。Neumann SPDE 会引入真实的边界 covariance effect。论文中必须把边界条件当作 prior 定义的一部分，并做 Dirichlet/Neumann/mismatched BC 消融，不能直接引用无边界结论后声称完全等价。

### 3.2 真正的 geometry-aware functional Tucker

最干净的三模模型是

\[
u_c(s,a,x)
=\sum_{p=1}^{R_s}\sum_{q=1}^{R_a}\sum_{r=1}^{R_x}
G_{pqr}\,F^{(s)}_{cp}(s)\,F^{(a)}_q(a)\,F^{(x)}_{cr}(x).
\]

为 source 和 spatial 因子使用同一个域 kernel：

\[
F^{(s)}_{cp}\sim\operatorname{GP}(m^{(s)}_{\theta,p},k_{\Omega_c}),\qquad
F^{(x)}_{cr}\sim\operatorname{GP}(m^{(x)}_{\theta,r},k_{\Omega_c}),
\]

参数因子可采用一维 Matérn/RBF GP：

\[
F^{(a)}_q\sim\operatorname{GP}(m^{(a)}_{\theta,q},k_a).
\]

小 core 使用

\[
\operatorname{vec}(G)\sim\mathcal N(0,\tau_G^{-1}I).
\]

共享 mean (m_\theta(c,x)) 由纯几何输入（坐标、SDF、工况和允许的 domain descriptor）产生。它承担 zero-shot transfer；domain GP residual 承担新域 few-shot adaptation、局部几何平滑和 uncertainty。若去掉共享 mean，不同域的 GP 独立，则测试域零观测时 posterior mean 为零，这是模型性质而不是训练技巧能解决的问题。

### 3.3 为什么 source 与 query 应分开

当前 POC 把 ‎(k_{\Omega}(x,s))、(x)、(s) 一起送入一个 spatial MLP，因此严格说不是 source × parameter × space 的三模 Tucker。投稿模型应把 source factor 与 query factor分开。对于线性 elliptic PDE，其 Green function 本身就有谱展开

\[
G_\Omega(x,s)\approx\sum_j \rho_j\phi_j(x)\phi_j(s),
\]

所以用共享 domain kernel prior 分别约束 source 和 spatial factors 既更符合张量故事，也更便于解释孔洞为何起作用。

## 4. 当前代码究竟实现了什么

当前实现位于：

- `src/geoaware/domain_kernels.py`
- `src/geoaware/functional_tucker.py`
- `experiments/run_four_track_fast_poc.py`

当前 intrinsic section 为

\[
z_{\Omega,q}(x,s)=\operatorname{RMSNorm}\left[
\frac1J\sum_{j=0}^{J-1}\phi_j(x)\phi_j(s)
(1+\alpha_q\lambda_j)^{-p}\right],
\]

其中 (J=48)、‎(alpha_q\in\{0.03,0.1,0.3,1,3\})、(p=1.5)。eigenvalue 先除以该图第一个正 eigenvalue，basis 又做 empirical-‎(L^2) normalization，最后每个 section channel 按其自身 RMS 标准化。

这些操作对跨分辨率数值稳定很有帮助，但它们改变了 covariance amplitude，也没有学习 ‎(kappa,\nu,\sigma_f)。因此这些 section 应叫“由 covariance 启发的 geometry features”，不能当作已经校准的 GP covariance matrix。

点预测模型是

\[
\widehat y=\sum_{g,p,r} C_{gpr}
A_g(d_{\Omega}) B_p([\log a,a]) H_r(z),
\]

其中 (d_\Omega\in\mathbb R^7) 是手工 domain descriptor；纯 kernel 版本令 (z=z_\Omega(x,s))，当前默认版本令

\[
z=[z_\Omega(x,s),x,\operatorname{SDF}_\Omega(x),s,\|x-s\|,1].
\]

‎(A,B,H) 都是两层 GELU MLP，使用 AdamW 训练。这个模型有显式 Tucker core，但它是**确定性 neural functional Tucker**。普通 weight decay 不是显式 GP coefficient prior；当前没有 posterior distribution。

### 4.1 “composite kernel”命名需要纠正

当前代码只是把 intrinsic sections 与局部坐标/SDF **拼接后送入非线性 MLP**。这不等于

\[
k_{\rm composite}=k_\Omega+k_{\rm local}
\quad\text{或}\quad
k_\Omega k_{\rm local}.
\]

本报告以后称它为 `intrinsic_plus_local_inputs`，而不是 additive/composite GP kernel。真正的 composite kernel 必须直接构造 PSD covariance，并在 GP/KRR posterior 中使用。

### 4.2 “topology-erased”消融当前并未真正抹掉 topology

旧配置 `topology_erased_kernel_tucker` 只把 intrinsic spectral sections 换成 rectangle/bounding-box sections，但仍然给模型正确的：

- SDF；
- query/source 坐标；
- 从真实 fluid mask、边界距离统计得到的 domain descriptor。

所以它只能解释为 **bbox-kernel channel ablation with correct local geometry metadata**。它不能支持“抹掉所有 topology 后显著变差”这一强结论。真正 topology-erased control 应同时去掉 SDF、hole/component count、真实 domain descriptor，并只保留 bounding box 坐标和 rectangle kernel。

## 5. inference 路线：从 POC 到完整 Bayesian 模型

### Stage 0：当前 deterministic feature POC

目标：先验证 intrinsic covariance feature 是否比欧氏 feature 更适合孔洞/凹边界。

- point estimate：AdamW；
- uncertainty：无；
- 可声称：geometry-feature mechanism POC；
- 不可声称：GP、MAP、Bayesian posterior、calibrated UQ。

### Stage 1：固定因子 + Bayesian core

给定确定性 factor features，单个 observation 对 core 是线性的。令

\[
a_i=F^{(s)}(s_i)\otimes F^{(a)}(a_i)\otimes F^{(x)}(x_i),
\]

则

\[
y_i=a_i^\top g+\epsilon_i,
\quad g=\operatorname{vec}(G).
\]

Gaussian prior 下可精确计算

\[
\Sigma_g^{-1}=\tau_G I+\sigma^{-2}A^\top A,
\qquad
\mu_g=\sigma^{-2}\Sigma_g A^\top y.
\]

这一步可以快速提供“conditional-on-features”的 uncertainty，但不是完整 GP uncertainty。它适合做推断单元测试，也可作为方向 1 与方向 3 之间的桥梁。

### Stage 2：单 GP residual mode 的真实 variational posterior

先只随机化 spatial factor：

\[
F^{(x)}_{cr}(x)=m_{\theta,r}(c,x)
+\Phi_c(x)\operatorname{diag}(\rho_c^{1/2})w_{cr},
\quad w_{cr}\sim\mathcal N(0,I),
\]

使用 whitened variational posterior

\[
q(w_{cr})=\mathcal N(\mu_{cr},S_{cr}).
\]

优化

\[
\mathcal L=
\sum_{i\in\mathcal O}\mathbb E_q[\log p(y_i\mid F,G)]
-\sum_{c,r}\operatorname{KL}[q(w_{cr})\|p(w_{cr})]
-\operatorname{KL}[q(G)\|p(G)].
\]

先用 diagonal (S)，通过 reparameterization Monte Carlo 训练。只有 Stage 2 完成后，代码才能诚实称为 approximate GP factor posterior。

### Stage 3：source + space 双 domain-GP factor

将 source 与 query spatial factors 都随机化，保留一维 parameter GP 或确定性 factor。主要风险是乘法因子的 scale/permutation non-identifiability 和 Monte Carlo 方差。建议：

- whitened coefficients；
- factor RMS/orthogonality regularization；
- CP-diagonal core 初始化；
- 对小 core 使用条件 Gaussian 更新或 natural gradient；
- 避免一开始同时随机化所有 modes。

### Stage 4：inducing / inter-domain scalable inference

若每个域节点很多，使用 inducing variables ‎(u=f(Z)) 和

\[
q(f,u)=p(f\mid u)q(u)

\]

的 sparse variational GP。复杂度可从 dense GP 的 (O(N^3)) 降到典型的 (O(NM^2))。在不同域之间无法直接共享 inducing coordinates 时，可共享 kernel hyperparameters 和 mean network，同时为每个域选择 geodesic farthest-point inducing nodes。

## 6. 当前合成数据审计

数据由 `simulate_screened_elliptic` 独立求解

\[
(\operatorname{diag}(r(x,a))+aL_{\rm physics})u=f_{s,a}

\]

得到，边界条件是所有外边界和孔洞边界上的 reflecting zero-flux。训练 learner 看到的是 unweighted geometry operator，而 simulator 使用由 material speed 加权的 physics operator，因此不是把 solver 的精确 diagonalization basis 直接交给模型。

### 6.1 做得正确的地方

- field 由独立稀疏线性求解器生成，不调用 learner；
- linear residual 小于数据 gate 的 `1e-8`；
- train/validation/test 按 geometry name 分割，同一形状的两个分辨率不会跨 split；
- target normalization 只使用训练域被观察条目；
- SDF、graph、坐标和 descriptor 都由 geometry 生成，不使用 field target；
- 24 训练、32 验证/测试检查了基本跨分辨率能力；
- hole shape 完全未出现在四个训练 geometry 中，是有价值的 topology extrapolation stress test。

### 6.2 没有发现直接 target leakage，但有八个证据风险

1. 只有 6 个手工形状：4 train、1 validation、1 test，macro 指标实际上是单一形状指标，不能估计形状分布上的方差。
2. `wavy_with_hole` 已经被三 seed 读取、汇报并用于方法判断，今后不能继续称为 untouched final test。
3. 当前 validation 只有 `slanted_channel`，hyperparameter 很容易针对一个形状过拟合。
4. 1% mask 是 entry-wise random mask，不是固定传感器；它通常覆盖所有 source/parameter levels，可能比真实 sparse sensing 容易。
5. 四个 source anchors 和 14 个 diffusivity 在所有域相同，source snapping 和参数规律高度规则。
6. target 是特意设计得平滑且 boundary-sensitive 的 screened elliptic family，是 method-favorable synthetic gate，不是外部证据。
7. simulator 的 material speed 本身含 boundary-distance 项，进一步增强了 geometry signal；这可以作为正控，但必须另有 geometry-irrelevant 负控。
8. 旧 runner 用单个随机 minibatch loss 选 best checkpoint，会放大 seed variance。新实验已经改为定期计算全部 observed entries 的 loss；旧 JSON 不应和新协议混算。

### 6.3 指标问题

当前

\[
\operatorname{NRMSE}_{\rm boundary}=
\frac{\operatorname{RMSE}_{x\in\partial\Omega_h}}
{\operatorname{Std}(y_{x\in\partial\Omega_h})}

\]

只取一层离散 boundary nodes。需要补充：

- 以全域 target RMS/std 归一化的 boundary RMSE，避免边界局部方差很小时分母不稳定；
- 距边界 1、2、4 个 mesh spacing 的 band curve；
- 外边界与孔洞边界分开；
- near-hole shadow region；
- global relative (L^2)、per-source、per-parameter macro；
- full Bayesian 后加入 NLL、CRPS、50/90/95% coverage 和 sharpness。

## 7. 新完成的机制消融

独立实验：`experiments/track3_kernel_input_ablation.py`。

协议：四个训练形状、`r24`；一个未见验证形状 `slanted_channel_r32`；1% random entries；900 steps；每 50 steps 用全部 observed training entries 选 checkpoint；三 seeds；test geometry 未读取。

| 输入 | Validation NRMSE | Boundary NRMSE |
|---|---:|---:|
| intrinsic sections only | **0.2602±0.0055** | **0.2809±0.0086** |
| Euclidean RBF sections only | 0.3320±0.0212 | 0.3080±0.0381 |
| intrinsic + identical local inputs | **0.1905±0.0219** | **0.1905±0.0188** |
| Euclidean RBF + identical local inputs | 0.2031±0.0297 | 0.2267±0.0152 |

解释：

- 在 kernel-section-only 的参数匹配比较中，intrinsic sections 全域相对改善约 21.6%；
- 加上相同 SDF/坐标后，全域差距缩小到约 6.2%，说明局部几何特征吸收了部分作用；
- 但 boundary NRMSE 仍改善约 16.0%，与“intrinsic covariance 对边界传播更重要”的机制一致；
- 只有三 seeds 和单一 validation shape，仍是正信号，不是论文级结论；
- 这项比较验证的是 neural input representation，不是 GP posterior。

结果文件：

- `papers/four_tracks/results/track3_kernel_input_ablation_seed0.json`
- `papers/four_tracks/results/track3_kernel_input_ablation_seed1.json`
- `papers/four_tracks/results/track3_kernel_input_ablation_seed2.json`

旧 hole POC 三 seed 中，domain-kernel Tucker 为 `0.1526±0.0148`，bbox-kernel-with-correct-local-geometry 为 `0.1731±0.0140`；但因消融并未完全抹掉 topology、test 已被读取、checkpoint 协议较弱，这组数字只能用于生成 hypothesis，不能作为最终 paper table。

## 8. baseline 审计与公平实现要求

### 8.1 必须有的 sanity baselines

| Baseline | 作用 | 注意事项 |
|---|---|---|
| zero / observed global mean | absolute skill gate | 只使用允许的训练观测 |
| per-source / per-parameter mean | 检查任务是否只靠 mode marginal 就能解决 | unseen level 必须定义回退规则 |
| nearest observed / graph harmonic interpolation | 强稀疏传感器 baseline | 仅同域/few-shot，不适用于零样本 |

### 8.2 方法匹配的 tensor/INR baselines

| Baseline | 为什么必须有 | 公平约束 |
|---|---|---|
| discrete CP / Tucker | 传统 tensor completion 下界 | 只在同域 completion 使用 |
| neural functional CP | 检查 Tucker core 是否真的需要 | 输入、rank budget、训练预算匹配 |
| neural functional Tucker | 用户明确要求的直接对标 | 同样的 source/parameter/space 分模 |
| joint coordinate/SDF INR | 检查收益是否只来自输入特征 | 参数量和 validation protocol 匹配 |
| F-INR CP/Tucker | 当前 functional neural tensor 的直接前沿 baseline | 使用官方实现或逐项复现检查 |

当前 shared POC 里的 joint INR 与各 Tucker 参数量不同，训练动力学也不同；只能作为快速 sanity check，不是最终 capacity-matched comparison。

### 8.3 GP/kernel baselines

| Baseline | 核心问题 |
|---|---|
| exact Euclidean RBF GP / kernel ridge | 不规则域 kernel 是否优于普通欧氏 kernel？ |
| exact intrinsic domain GP / KRR | tensor factorization是否比单一 geometry GP 有额外价值？ |
| product-kernel GP (k_s k_a k_x) | Tucker core 是否优于标准 separable GP？ |
| additive/composite GP | intrinsic 与 SDF/local covariance 是否互补？ |
| FunBaT | GP functional Tucker 的最直接 baseline |
| GPTF / nonlinear Bayesian tensor | GP 是用作 factor prior 还是 latent-factor-to-output nonlinear map，哪种更合适？ |

特别重要：本项目账号下已有 [Functional Bayesian Tucker Decomposition (FunBaT)](https://openreview.net/pdf?id=ZWyZeqE928) 的官方实现。它在每个连续 mode 上放独立 GP prior，并用 SDE/state-space message passing 做 scalable inference。方向 3 最自然的研究定位不是绕开 FunBaT，而是：

> 将 FunBaT 的欧氏/一维 GP functional prior 推广为带边界条件的不规则域 prior，并解决跨域 transfer、graph/mesh inference 和 hole topology。

最终 baseline 应直接运行原版 FunBaT；同时实现“只替换 spatial kernel、其他 inference 不变”的最小改动版本，才能把贡献定位清楚。

### 8.4 geometry neural operator baselines

GINO、Geo-FNO、DAFNO/相关 arbitrary-domain operator 并非 Bayesian tensor baseline，但在跨形状 PDE prediction 上是必须面对的强模型。比较时应给两套预算：

- full-supervision operator learning；
- 与本文相同的 sparse-label / sparse-sensor supervision。

不能让本文只见 1% labels、而 baseline 用 full fields 后直接比较；也不能反过来把 neural operator 限制在不合理的单点 regression 接口。

## 9. 数据集路线

### 9.1 Synthetic-v2：必须先扩增

现有六形状只适合 smoke。下一版至少生成 100–300 个参数化 domain：

- outer boundary：Fourier radial、star、slanted、L/U notch、dumbbell；
- hole count：0/1/2/3；
- hole shape/position/size 独立随机；
- narrow passage width 分层；
- 32/48/64 三分辨率；
- source 位置和 diffusivity grid 随 domain 随机化；
- geometry-family-disjoint split，而不仅是 random seed split。

必须同时生成三组方程：

1. geometry-positive：边界显著影响场；
2. geometry-neutral：内部局部响应占主导，防止所有任务都偏袒 domain kernel；
3. boundary-condition mismatch：Dirichlet/Neumann/Robin，测试 kernel BC 是否选对。

### 9.2 AirfRANS：第一外部数据优先级最高

[AirfRANS 官方库](https://github.com/Extrality/airfrans_lib)提供 1000 个不同 NACA airfoil 的 RANS 解、Reynolds number/angle-of-attack 变化和官方 `full/scarce/reynolds/aoa` 任务。优点是网格不规则、边界变化真实、任务和数据文档完整。缺点是没有内部孔洞，而且场是 point cloud CFD，不天然组成共享 node mode。

建议任务：给定 airfoil geometry、Re/AoA 和 0.1%–5% scattered field sensors，恢复 pressure/velocity；按 airfoil identity 分 split。先从官方 `scarce` task 做小规模 GP feasibility，再扩到 full。

### 9.3 Geo-FNO / NeuralOperator 几何数据

[Geo-FNO 官方仓库](https://github.com/neuraloperator/Geo-FNO)提供 elasticity、plasticity、airfoil 和 pipe 数据，覆盖 point cloud/mesh/design-parameter 输入；原仓库已明确标记 deprecated，因此 baseline 代码应使用维护中的 [NeuralOperator](https://github.com/neuraloperator/neuraloperator)，数据与原实验协议用于复现。Elasticity/pipe 可作为第二外部数据，GINO/Geo-FNO 是直接强 baseline。

### 9.4 内部孔洞外部证据

目前没有找到一个同时满足“大量不同 hole topology、连续物理场、许可清晰、官方 split”的现成标准 benchmark。因此不要为了“外部”标签勉强选不匹配数据。更可靠的做法是：公开 Synthetic-v2 的生成器、mesh、PDE residual audit 和 hash manifest，并另外用 AirfRANS/Geo-FNO 证明不是只在自造数据上工作。

## 10. 测试与可复现性

本轮新增 `tests/test_domain_kernels.py`：

1. intrinsic sections 对 eigenvector sign flips 不变；
2. Euclidean RBF sections 在 source node 取最大值；
3. 非正 lengthscale 和空 lengthscale 被拒绝。

通过情况：`5 passed`。

完整 Bayesian implementation 还必须新增：

- covariance symmetry/PSD test；
- graph permutation equivariance test；
- duplicate/degenerate eigenvalue basis-rotation invariance test；
- 24/32/64 kernel diagonal与effective range convergence；
- KL 为非负且 (q=p) 时为零；
- posterior variance 随 nearby observations 增加而下降；
- exact small GP 与 variational GP 的 posterior mean/variance 对齐；
- predictive coverage synthetic calibration；
- train-only normalization 和 split leakage automated audit。

特别注意：sign invariance 不足以处理重复 eigenvalue 子空间内的任意 orthogonal rotation。完整 test 应验证整个 degenerate eigenspace 的 projector invariance。

## 11. 论文级实验矩阵

### 主表 A：同域 sparse tensor completion

- 数据：Synthetic-v2、AirfRANS subset、Geo-FNO elasticity/pipe；
- ratio：0.5%、1%、2%、5%、10%；
- masks：entry-random、fixed-sensor、missing-parameter-block；
- baselines：CP/Tucker、neural CP/Tucker、FunBaT、Euclidean product GP、intrinsic GP、本文；
- 指标：global/boundary relative (L^2)、NLL、coverage。

### 主表 B：unseen-domain few-shot adaptation

- shot ratio：0、0.1%、0.5%、1%、5%；
- shape-family-disjoint 与 topology-disjoint 两种 split；
- 报告 error-vs-shot 和 calibration-vs-shot curve；
- 0-shot 只评价共享 mean，few-shot 才评价 GP residual 增益。

### 主表 C：跨分辨率

- train 24/32，test 48/64 或原始 unstructured mesh；
- 固定物理坐标 source/sensor，不固定 node index；
- 报告 kernel truncation (J) 与 mesh size 的敏感度。

### 核心 ablations

1. intrinsic Matérn vs Euclidean RBF；
2. correct BC vs wrong BC；
3. correct domain vs bbox vs shuffled graph；
4. GP residual only vs neural mean only vs mean + GP residual；
5. CP core vs Tucker core；
6. source GP only vs spatial GP only vs both；
7. exact/spectral/inducing approximation；
8. kernel hyperparameter learned vs frozen；
9. hole count、narrow passage 和 boundary distance 分层。

### 统计协议

- POC selection：3 seeds，仅 validation；
- confirmation：至少 10 新 seeds；
- test configuration 只冻结一次；
- paired bootstrap over geometries，而不是只对 optimizer seeds 做 t-test；
- 同时报告 mean、std、median、95% CI 和 per-geometry scatter。

## 12. go/no-go 门槛

### 升级为主会完整论文

同时满足：

1. 至少 3 个数据集，其中至少 1 个外部数据；
2. strongest baseline 的 absolute NRMSE 明显低于无效区，并且本文 global error 相对改善至少 10%，或 boundary/hole metric 改善至少 15%；
3. 改善在至少 10 seeds 和多个 held-out geometries 上稳定，95% paired CI 不跨 0；
4. 比 FunBaT 与 Euclidean GP 有清楚增益；
5. GP posterior 在 NLL/coverage 上优于 point model 和简单 ensemble；
6. correct-domain/wrong-domain ablation 能定位几何机制。

### 作为 B 类/学生项目

如果点预测正信号稳定，但完整 posterior inference 或外部数据未达到主会标准，可将贡献收敛为“graph-domain kernel functional Tucker + sparse field reconstruction”，但仍不得把 deterministic feature MLP 称为 Bayesian GP。

### 停止或合并

若 intrinsic kernel 相对 Euclidean/SDF 在扩增 geometry family 后优势小于 5%，或只有合成 hole case 有效，则不再单独成 paper：把 domain kernel 作为方向 4 的一个 geometry feature / UQ extension 即可。

## 13. 接下来三轮最小迭代

### Round 1：证据修正，不扩模型

- 扩增 100+ geometry Synthetic-v2；
- 增加 true topology-erased、Euclidean RBF、intrinsic-only、local-only controls；
- 将 boundary 分成 outer/hole 多 band；
- 新建从未读取的 confirmation test family。

### Round 2：真实 GP 的最小闭环

- neural mean + 单 spatial GP residual；
- whitened diagonal variational posterior；
- Gaussian core posterior；
- exact small-domain GP 对齐 test；
- few-shot test-domain protocol 和 calibration。

### Round 3：直接对标 FunBaT

- 运行官方 FunBaT；
- 只替换 spatial RBF/SDE prior 为 domain spectral/SPDE prior；
- source 与 query factors 分模；
- AirfRANS scarce subset；
- 决定继续完整论文、降级学生项目，还是合并到方向 4。

## 14. 一手参考与代码

- [Functional Bayesian Tucker Decomposition, ICLR 2024](https://openreview.net/pdf?id=ZWyZeqE928)：最直接的 functional GP Tucker formulation、SDE prior 与 message-passing inference baseline；[官方代码](https://github.com/xuangu-fang/Functional-Bayesian-Tucker-Decomposition)。
- [Matérn Gaussian Processes on Riemannian Manifolds](https://arxiv.org/abs/2006.10160)：Laplace–Beltrami 谱构造、有限截断和 inducing inference 的理论依据；其主要公式是无边界 manifold，需要谨慎迁移到本项目边界域。
- [Lindgren, Rue & Lindström, 2011](https://doi.org/10.1111/j.1467-9868.2011.00777.x)：Matérn field 的 SPDE/GMRF 表示以及 Neumann boundary effect。
- [Variational Learning of Inducing Variables in Sparse GPs, AISTATS 2009](https://proceedings.mlr.press/v5/titsias09a.html)：标准 inducing-variable variational inference。
- [Gaussian Process Nonparametric Tensor Estimator, ICML 2016](https://proceedings.mlr.press/v48/kanagawa16.html)：GP 与 nonlinear tensor estimation 的经典对照。
- [Nonparametric Decomposition of Sparse Tensors, ICML 2021](https://proceedings.mlr.press/v139/tillinghast21a.html)：sparse tensor、GP/RFF inference 与 CP/GPTF baselines。
- [AirfRANS 官方数据与 loader](https://github.com/Extrality/airfrans_lib)。
- [Geo-FNO 官方数据/旧实现](https://github.com/neuraloperator/Geo-FNO)；[维护中的 NeuralOperator/GINO 实现](https://github.com/neuraloperator/neuraloperator)。

## 15. 最诚实的一句话进度

我们已经证明“intrinsic domain covariance sections 比参数匹配的欧氏 RBF sections 更适合当前不规则域边界任务”这一小机制在三 seed 验证上成立；还没有证明“Bayesian Functional Tucker 比 functional/neural Tucker 或真正 GP baseline 更好”。下一步不应继续给 MLP 加名词，而应先扩大形状协议，再实现一个最小但真实的 GP posterior 闭环。
