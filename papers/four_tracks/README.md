# 四条研究线的统一地图（当前单一事实源）

更新时间：2026-08-15。

快速 POC 的逐轮负信号、修正与冻结记录见
[`ITERATIONS.md`](ITERATIONS.md)。
四条线各自的完整 formulation、实现、inference、数据、baseline 与测试审计见
[`tech_reports/`](tech_reports/README.md)。

根据《青基正文 v1.8》重新复核后的方向 3/4 路线决策见
[`R7 技术路线复核`](tech_reports/ROUND4_STRATEGY_REVIEW.md)。简要结论是：方向 3 的 global kernel dictionary 降级为机制 sanity，候选主线回到 PDE 联合谱的 mode-wise kernel 分离；方向 4 的 geometry operator 已收口，候选新任务改为不完整 simulation combinations 上的 functional tensor operator completion。这两项尚未写成既成结果。

## 一句话总纲

这个项目只研究一个核心问题：**当物理场定义在不规则边界、孔洞或非均匀网格上时，怎样把这种几何放进低秩函数张量，而不是只把坐标交给一个黑盒网络。**

四条线不是互相替换，而是四种不同的几何入口：

| 方向 | 几何放在哪里 | 最短故事 | 当前优先级 | 合适定位 |
|---|---|---|---|---|
| 1. Operator-informed Bayesian Tensor | 因子的算子谱先验 | operator factor space + CP/Tucker decoder | P1，条件 GO | ratio×mismatch phase diagram 已跑通 |
| 2. Phase-factorized Wave Tensor | 波的传播相位 | 三角恒等式把行波变成显式低秩 CP | P4，STOP / DOWNGRADE | 保留机制示例或学生项目，不再占主实验预算 |
| 3. Domain-kernel Bayesian Functional Tensor | 域上的 GP covariance | ELBO 学习 geometry-kernel dictionary | P0，机制 GO | matched/near-match 正；真实 PDE hybrid 尚中性 |
| 4. Geometry-coordinate Functional CP | 坐标、SDF、source 进入连续 CP factor | 最小几何函数 CP | P2，收口 | NO、边界积分、rank gate 均未稳定增益 |

这里的“优先级”是投入顺序，不是价值排序。方向 1 和 2 的代码、结果与论文草稿均保留。

## 本轮固定预算结论

本轮统一采用早期筛选预算：默认 3 seeds、300--500 updates、validation-only，只有过 gate 才扩预算。

| 方向 | 当前最可信结果 | 决策 |
|---|---|---|
| 1 | calibrated mismatch：\(\delta\le .30\) 时 2/5/10% 均稳定胜；2% 在 `.30--.45`、5/10% 在 `.45--.60` 反转 | bias--variance phase boundary 已连续跑通 |
| 2 | clean traveling harmonic：correct phase `1.4859±0.0806`，zero `1.0000` | STOP / DOWNGRADE |
| 3 | matched mixture `0.0725±0.0046`，near-match `0.1432±0.0096`，elliptic mixture `0.3116±0.0055` | 机制 GO；global dictionary 降为 sanity，候选主线改为 mode-wise operator kernels |
| 4 | local CP ID/OOD `0.2480/0.2553`；DeepSets gate `0.2533/0.2550`，仅 1/3 seeds 胜 local | 正式收口 local geometry CP |

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

本轮审计发现并修复了一个实质问题：前向 factor 做 unit-RMS normalization，旧谱惩罚却作用在 raw coefficients，因此缩小系数可降低惩罚而不改变预测。修正后 2% seed-30 smoke 为 `0.155`，operator CP `0.410`，flat GP `0.597`，wrong Tucker `1.489`；正信号保留，但旧 10-seed 主表必须重跑。新的 neural functional Tucker 在 500→2000 steps 从 `0.523` 改善到 `0.307`，也证明所有 baseline 机械地同跑 500 steps 不公平。

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

这是一条**特殊场结构**路线，不再包装成任意 PDE 的通用 geometry model。最终 clean traveling-harmonic R2 中，zero 为 `1.0000`，correct paired phase 为 `1.4859±0.0806`，wrong Euclidean phase 为 `2.5336±0.0849`。正确传播坐标提供了相对机制信号，却没有形成绝对有效预测；方向 2 已按预注册门槛 **STOP / DOWNGRADE**，不再进入 WaveBench 或继续增加 phase 组件。

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

当前代码已从单一 kernel-section MLP 升级为 finite-feature geometry-kernel dictionary：Matérn/resolvent、heat/diffusion、graph-geodesic RBF 与 Euclidean control 形成显式 PSD features；nonnegative mixture weights、full-covariance \(q(u)\)、likelihood 与 posterior variance由 mini-batch ELBO+SGD 联合学习。它可以称 finite-feature variational GP，但还不能称完整 Bayesian functional Tucker。

### 与方向 1 的本质区别

- 方向 1 的对象是一个固定离散张量，算子主要是 mode table 的正则器。
- 方向 3 的对象是连续因子函数，kernel 定义函数空间；跨网格预测是方法语义的一部分。
- 方向 1 强调“便宜、显式、条件 core 后验”；方向 3 强调“非欧域 GP、少样本适配和不确定性”。

### 快速 POC 结果

在 3 train、2 unseen validation geometries、1% observations、3 seeds×400 steps 下：

| 数据层 | Learned mixture NRMSE | 解释 |
|---|---:|---|
| matched heat-GP sanity | `0.0725±0.0046` | heat 权重平均最高 `0.519`，ELBO 能近似识别真核 |
| perturbed near-match | `0.1432±0.0096` | 优于单 heat `0.1914`，不是只在 exact match 有效 |
| screened elliptic mismatch | `0.3116±0.0055` | 只略优于 Euclidean `0.3251` |

elliptic 上 neural CP mean 为 `0.2073±0.0105`，加入 heat/mixture GP residual 为 `0.1985±0.0267 / 0.2054±0.0275`，跨 seeds 不稳定。因此当前正贡献是 kernel dictionary 与 evidence selection 的机制闭环，不是通用 neural hybrid SOTA。

## 方向 4：Geometry-coordinate Functional CP

### 最简公式

\[
u(\Omega,p,x,s)=\sum_{r=1}^{R}w_r\,
a_r(s)\,b_r(p)\,c_r(x,s,d_{\partial\Omega}(x)).
\]

三个 factor 都是连续函数，交互经过小 CP bottleneck。当前 \(d_{\partial\Omega}\) 是活动域内到最近外边界/孔洞边界的正距离；它不是在共享 ambient grid 上定义的完整 signed distance field。

### 独特贡献

1. 可在不同节点数与不同网格上查询，不要求规则矩形张量。
2. interior boundary distance 让孔洞和外边界的局部信息进入 factor。
3. 与 joint INR 的区别是显式 rank bottleneck 与 mode sharing，而不是独占几何输入。

### 当前证据边界

- 新协议含 48 train 0/1-hole、8 ID validation、8 双孔 topology-OOD validation；8 个 test specs 已冻结但未读。
- 1% labels、3 seeds×400 steps、共享 case schedule 下，coordinate/SDF CP 为 `0.2480±0.0047`（ID）与 `0.2553±0.0049`（OOD）。
- 最好的 unmasked Geometry-NO-CP 为 `0.2840±0.0065 / 0.2958±0.0029`；hard-masked 版本更弱，同 encoder dense head 为 `0.7296±0.0300 / 0.7738±0.0218`。
- NO 的 observed loss 很低但新几何误差更高，属于稀疏过拟合。一次小门控 NO residual 修正仍未超过 CP，当前不能把 FNO 融合写成正贡献。
- Boundary-integral CP 的弱 OOD 改善没有通过 hole-token/type 消融；Boundary-DeepSets rank gate 也只在 1/3 seeds 胜 local CP。
- 最终结论是“低秩 local geometry CP 比 dense operator 稳定”；FNO、boundary integral、descriptor/DeepSets rank modulation 全部只作负消融。

## 四条线共享什么，不共享什么

共享：

- `DomainCase`：coordinates、mesh/graph operator、boundary mask、interior boundary distance/ambient SDF、source、physical parameters、target field；
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

1. **优先发展方向 3。** 把 kernel dictionary 的 matched/near-match 机制写清，并增加 Task C 新域 few-shot posterior adaptation；先做 calibration，不增加更复杂 VI。
2. **方向 1 扩展二维 phase diagram。** 连续扫描 operator mismatch strength × 2/5/10% ratio，冻结 CP/Tucker decoder 选择规则，再转一个 non-aligned PDE/GP truth。
3. **方向 4 与方向 2停止结构扩张。** 方向 4 只维护 geometry-coordinate/SDF CP 作为简洁学生项目/共享 baseline；所有 operator 融合保留为负消融。

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
