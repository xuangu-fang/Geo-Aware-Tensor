# Paper A 中文迭代记录：算子因子上的 Bayesian CP

本记录对应英文 [TENSOR_ITERATIONS.md](../paper_a/TENSOR_ITERATIONS.md)，集中说明每轮的假设、最小改动、实验、结果、失败原因与下一步决策。更早的 graph×time operator-GP 工作保留在英文 `paper_a/ITERATIONS.md`，但它现在只是 precursor baseline，不是本文的中心模型。

## A0：为什么必须重构旧故事

旧 `ExactFeatureBayes` 对所有 joint product eigenfeatures 做 Bayesian linear regression：

\[
y_\Omega=\Phi_\Omega w+\epsilon.
\]

即使把 \(w\) reshape 成 spectral coefficient tensor，它仍然具有以下问题：

- 没有显式 CP/Tucker low rank；
- 没有被学习的 mode factor；
- posterior uncertainty 在 joint feature weights 上，而不是 tensor factors/core 上；
- 正确/错误几何比较只能证明 kernel geometry，不能证明 tensorization。

因此 refocus 后，dense operator GP 被保留为强 baseline，主模型必须回到显式 Bayesian CP。

## T1：最小 conditional Bayesian spectral CP

### 研究假设

只要把传统 CP factor 放入正确的 operator eigenspace，在 0.5%–1% observation 下就应同时优于：

1. 相同 CP 但错误几何；
2. 无 operator side information 的 discrete CP；
3. 使用正确几何但没有低秩结构的 dense operator GP。

### 模型

\[
Y_{ijk}=\sum_{r=1}^{R}a_r
(\Phi_1u_{1r})_i(\Phi_2u_{2r})_j(\Phi_3u_{3r})_k+\epsilon_{ijk}.
\]

Factor mean 通过带 operator-frequency penalty 的 MAP 拟合；给定 factor 后，对 component amplitudes 做 exact conditional Gaussian posterior，并尝试 MacKay-style ARD。

### 数据与 protocol

- tensor：`20×28×36` time × bounded range × periodic angle；
- generator：rank-4 operator CP + mild local off-model residual；
- mask：0.5%/1% random 和 periodic gap；
- baselines：wrong CP、discrete CP、flat operator GP；
- 初始 rank：10；
- optimization：随机 factor initialization。

### 结果

0.5% random 三个 seed：

| 模型 | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| Geo Bayesian CP | 1.139 | 1.381 | 1.016 |
| Wrong geometry CP | 1.446 | 1.584 | 1.256 |
| Discrete CP | 1.421 | 1.382 | 1.170 |
| Flat operator GP | 0.646 | 0.758 | 0.863 |

1% seed 0 时 Geo-CP 为 0.708，flat GP 为 0.539。

### 结论与诊断

- 正确几何在 CP 内部确实有帮助；
- 但 CP 明显输给 flat GP，tensor contribution 不成立；
- ARD 与 no-ARD 的预测几乎完全相同，effective rank 停在上限。

失败的主要原因不是 CP 表示能力，而是极稀疏条件下的 multiplicative optimization。随机 factor 很难进入正确 basin；在 factor column 还没有物理意义时，conditional amplitude ARD 也无法判断哪些 component 应被删除。

### 决策

提前停止完整 sweep，不继续用更多 seed 计算一个已经被 cheap falsification 否定的配置。下一轮只修初始化，不增加模型组件。

## T2：operator posterior → CP-ALS 初始化

### 唯一改动

1. 用 observation-only dense operator model 得到完整 posterior mean；
2. reshape 成张量；
3. 运行普通 CP-ALS；
4. 把 factor 投影回各 mode operator basis；
5. 优化完全相同的 Bayesian CP objective。

这只是 optimization repair。Dense GP 不留在最终 predictor 内，也不读取 unobserved target。

### cheap pilot

0.5% random、seed 0、rank 8：

- correct BCP：0.543；
- wrong BCP：1.396；
- flat GP：0.646。

首次同时通过 geometry 与 tensor 两个因果门槛。

### 三 seed 探索

| mask | correct BCP | wrong BCP | flat GP |
|---|---:|---:|---:|
| random 0.5% | 0.660 | 1.356 | 0.756 |
| periodic gap 0.5% | 0.814 | 1.336 | 0.809 |

Random mask 下 correct BCP 3/3 优于 flat GP；periodic gap 下则基本打平，seed 2 明确失败。

### 新暴露的问题

- ARD/no-ARD point prediction 仍几乎相同；
- effective rank 保持为 8；
- amplitude-only posterior 的 95% coverage 只有约 0.06–0.14；
- 即使 no-ARD coverage 也只有约 0.22–0.47。

这说明 point estimator 已改善，但 Bayesian tensor claim 仍不成立：factor uncertainty 不能被忽略。

### 决策

保留 random-mask 的 tensor 正信号，把 periodic-gap 记录为 limitation。T3 只处理 factor uncertainty 和 ARD falsification。

## T3：factor Laplace correction；ARD 被否证

### 最小改动

保持 factor mean 与 point predictor 不变，为每个 spectral factor coefficient 加 diagonal Gauss–Newton/Laplace variance，再通过 delta method 传播到预测方差。

### 结果

0.5% random、三 seed：

- no-ARD coverage：0.633、0.616、0.330；
- no-ARD NLL：2.39、3.67、13.57；
- 50% selective RMSE reduction：0.37、0.53、0.50。

Laplace correction 明显改善了 uncertainty ranking，但仍然欠覆盖。

ARD 结果更差：

- coverage 只有 0.105、0.160、0.122；
- effective rank 仍为 8；
- rank cap 提高到 12，并做三次 factor/ARD alternating cycle 后，12 个 component 仍全部保留。

### 结论

当前 conditional Type-II ARD 在极稀疏非共轭 CP 中失败。它在 factor 未充分辨识时过早压缩 noise/component uncertainty，产生过度自信，却没有真正删 rank。

### 决策

- 最终方法使用 fixed rank；
- ARD 作为明确负结果，不再以“自动秩学习”包装；
- T4 只加入严格 observation-only split calibration，修 posterior dispersion，不改变 point model。

## T4：严格 split calibration 与最终确认

### 校准协议的两次审计

第一版 preliminary fit 虽然只用 75% observation，但 normalization 的 center/scale 来自全部 observation，存在轻微 calibration-distribution leakage。审计后改为：

1. preliminary center/scale 只由 75% fit observations 计算；
2. initializer、factor fit、posterior 都只用 75%；
3. calibration target 用同一 train-only affine transform；
4. flat GP 使用完全相同 split calibration；
5. 最终 point model 才在全部 observations 上重拟合。

严格 seed-0 pilot：

| 模型 | NRMSE | NLL | coverage | width |
|---|---:|---:|---:|---:|
| Correct BCP | 0.533 | 0.710 | 0.981 | 2.482 |
| Wrong BCP | 1.404 | 1.716 | 0.961 | 5.978 |
| Flat GP | 0.646 | 0.982 | 0.985 | 3.881 |

### observation-ratio 选择

Fresh seed 10 的 ratio pilot：

- 1%：Geo-CP 0.593，flat GP 0.565，仍近似持平；
- 2%：Geo-CP 0.149，flat GP 0.435，wrong CP 2.839。

这支持“factor 需要达到可辨识阈值”的解释。Seeds 20–24 用于探索性确认该 ratio，但不进入最终 confirmatory inference。

### 最终 frozen confirmation

- ratio：2%；
- seeds：30–39，完全未参与方法或 ratio 选择；
- rank：8；
- steps：800；
- mask：random；
- models：Geo-BCP、wrong BCP、discrete BCP、flat operator GP；
- UQ：factor Laplace + strict split calibration。

结果：

| 模型 | NRMSE | NLL | coverage95 | width95 |
|---|---:|---:|---:|---:|
| Geo Bayesian CP | **0.198±0.037** | **−0.158±0.180** | 0.983±0.011 | 1.080±0.251 |
| Flat operator GP | 0.406±0.031 | 0.387±0.085 | 0.948±0.027 | 1.569±0.230 |
| Wrong BCP | 2.459±0.381 | 2.193±0.137 | 0.977±0.015 | 11.408±2.474 |
| Discrete BCP | 1.406±0.049 | 29.895±14.615 | 0.265±0.056 | 0.648±0.119 |

所有 10 个 seed 中，Geo-BCP 都优于三个 baseline：

- 相对 flat GP：NRMSE 改善 51.2%，CI 45.0–57.1%，\(p=0.00195\)；
- 相对 wrong BCP：改善 91.9%，\(p=0.00195\)；
- 相对 discrete BCP：改善 85.9%，\(p=0.00195\)；
- NLL 相对 flat GP 的 absolute difference 为 −0.545，\(p=0.00195\)。

### 0.5% frozen 结果的诚实解释

在 seeds 10–14：

| 模型 | NRMSE | NLL | coverage | selective gain50 |
|---|---:|---:|---:|---:|
| Geo-BCP | 0.751±0.113 | 0.967±0.139 | 0.933±0.029 | 0.433±0.104 |
| Flat GP | 0.725±0.038 | 1.047±0.217 | 0.882±0.063 | 0.277±0.042 |

Geo-BCP 不在 point NRMSE 上胜出；它的正信号是 UQ calibration 和 uncertainty ranking。因此主论文不能说“越低 observation 越一定更好”，而应说低秩 factor 需要最小可辨识信息量。

## 格式与公共数据 stress tests

### Tucker-format mismatch

在 `(4,5,5)` multilinear-rank Tucker generator 上，0.5% seed 0：

- Geo-CP：0.928；
- wrong CP：1.336；
- flat GP：0.842。

正确 geometry 仍有用，但 CP compression 不如 flat model。这直接指向下一版 Bayesian spectral Tucker，而不是盲目增加 CP rank。

### Active Matter

1% seed 0：

- Geo-CP：1.384；
- wrong CP：2.300；
- discrete CP：1.251；
- flat GP：1.340。

Geo-CP 的 uncertainty calibration 有改善，但 point reconstruction 不是最佳。当前只能作为 public-data stress，不是 headline。

## 最终支持与不支持的结论

### 已支持

1. 正确 operator geometry 能显著改善相同 CP 架构；
2. 在 2% 且 CP-plausible 的 regime 中，显式低秩 CP 优于正确几何的 flat GP；
3. factor-level uncertainty 对极稀疏 Bayesian tensor 是必要的；
4. strict observation-only calibration 能得到可用 posterior dispersion。

### 未支持

1. ARD 自动恢复 rank；
2. 0.5% point reconstruction 必然优于 flat GP；
3. CP 可以处理任意 Tucker core；
4. 当前方法在 Active Matter 上已具备公开数据优势；
5. compact representation 等于更快训练——当前 initializer/calibration 使其明显慢于 flat GP。

## 复现入口

- 方法：[tensor_bayes.py](../../src/geoaware/tensor_bayes.py)
- 数据：[tensor_data.py](../../src/geoaware/tensor_data.py)
- runner：[run_tensor_bayes.py](../../experiments/run_tensor_bayes.py)
- analyzer：[analyze_tensor_bayes.py](../../experiments/analyze_tensor_bayes.py)
- 最终表：[TABLES.md](../paper_a/tensor_results/TABLES.md)
- 英文主稿：[DRAFT.md](../paper_a/DRAFT.md)

## 后续 A-T5 至 A-T7：从 CP 提升为算子几何 Tucker

用户反馈后，Paper A 的定位已调整为“算子几何张量分解”。新增显式小核心 `OperatorBayesianTucker`，每个 mode factor 仍由对应 operator basis 与 Sobolev prior 定义，条件于 factors 对 core 做精确 Gaussian posterior。

三轮结果、容量修正、十个新 seed 和 phase diagram 详见[三轮持续迭代记录](三轮持续迭代记录.md)。冻结确认中，2% Tucker truth 的 Geo-BTucker 为 `0.125±0.015`，Geo-CP `0.367±0.055`，flat GP `0.612±0.028`；1% 下经小 core/短程 refinement 后为 `0.676±0.072`，仍优于 flat GP `0.752±0.030`。
