# 下一轮论文级实验计划

> **2026-08-13 状态更新：**本文件前半部分保留了扩展实验池，但不再等同于当前
> 执行清单。第七轮已经完成 Paper A block/noise 十种子确认；Paper B 的
> FNO/U-Net/persistence/path-noise 审计表明所有模型 NRMSE≈1，现已整体改判失败。
> 当前只执行：A canonical Tucker 写作与外部数据适配决策；B 重新选择能通过绝对效果
> 门槛的外部任务；复现 manifest。GINO、structured factor
> posterior、更多 envelope/impedance 组件均不是投稿前自动必做项。详见
> [`zh/第七轮发表导向迭代报告.md`](zh/第七轮发表导向迭代报告.md)。

## 1. Paper A：Operator Geometry-Aware Tensor Decomposition

### 1.1 论文主问题

在 0.5%–5% 部分观测下，mode-specific physical operators 是否能让显式 CP/Tucker 比无结构 Bayesian tensor、平坦 operator regression 和通用 neural field 更准确、更稳健，并给出更有用的不确定性？

### 1.2 主实验矩阵

| 轴 | 设置 |
|---|---|
| 数据 | controlled CP/Tucker/mixed；独立 wave/Helmholtz；The Well acoustic；CFDBench stress |
| observation | 0.5%、1%、2%、5% |
| mask | entry random、fixed sensors、fiber missing、continuous gap、boundary-biased |
| noise | 0%、5%、10%、20%；额外 sparse outliers 1% |
| geometry | correct、estimated/noisy、wrong/permuted、flat/no geometry |
| core | CP diagonal、small Tucker、over-capacity Tucker、oracle multilinear rank |
| seed | pilot 3；frozen confirmation 10 |

Primary metric：held-out NRMSE。Key secondary：NLL、90% coverage、interval width、selective risk、训练时间和峰值显存。

### 1.3 必做 baseline

1. Mean/interpolation；
2. Discrete CP/Tucker（同 rank）；
3. Bayesian CP/BPTF 或 robust Bayesian tensor factorization；
4. Graph-Laplacian regularized CP/Tucker；
5. Flat operator GP/product-feature Bayes；
6. CP-WOPT/weighted Tucker completion；
7. 参数匹配 INR/SIREN；
8. 若公共数据适用：FNO/TFNO 或 GINO。

传统 BPTF 是必须补上的 reviewer baseline；GP side-information PMF、graph-regularized tensor completion和通用 Bayesian tensor learning构成最直接的相关工作压力。[BPTF 项目页](https://www.cs.cmu.edu/~lxiong/bptf/bptf.html)，[GP side-information PMF](https://homepages.inf.ed.ac.uk/imurray2/pub/10pmf/)，[graph CP completion](https://ojs.aaai.org/index.php/AAAI/article/view/5915)，[general Bayesian tensor learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC8777296/)。

### 1.4 核心 ablation

- correct operator → wrong operator → identity operator；
- Tucker core → diagonal CP core；
- exact core posterior → point-estimate core；
- operator-HOSVD init → random init；
- observation-matched core → over-capacity core；
- factor MAP → structured factor posterior；
- known operator coefficients → perturbed/estimated coefficients。

### 1.5 最小发表门槛

- 至少一个公共/独立 solver 数据上，同时超过 flat operator GP、graph tensor 和 BPTF；
- correct/wrong geometry 差距稳定；
- UQ 至少在一个主数据上达到可信 coverage-width tradeoff；
- 负结果界定 CP/Tucker approximation boundary；
- 资源开销不超过最强 baseline 一个数量级，或明确换取了显著统计效率/UQ。

## 2. Paper B：Geometry/Phase-Aligned Neural Tensor

### 2.1 论文主问题

当多几何传播场存在共享内禀相位结构时，显式 phase-paired tensor bottleneck 是否比普通 neural tensor、IP-NF 和 geometry-informed neural operators 更具低观测样本效率和跨分辨率泛化？

### 2.2 主实验矩阵

| 轴 | 设置 |
|---|---|
| 数据 | controlled harmonic/mixed/envelope；独立 wave/Helmholtz；The Well acoustic |
| split | seen/unseen geometry；24→32/64→128 resolution；unseen source；unseen time/frequency |
| observation | 0.5%、1%、2%、5%；以 sensor count 同时报告 |
| mask | random entries、fixed sensor tracks、upstream-only、shadow/gap、boundary stratified |
| geometry | geodesic、Euclidean、noisy geodesic、estimated travel time |

Primary metric：unseen-geometry full-field NRMSE。Key secondary：boundary/shadow/high-band NRMSE、source-distance 分层误差、参数量、训练/推理成本。

### 2.3 必做 baseline

1. Ordinary geometry CP/Tucker；
2. Paired tensor 的 wrong-geometry/no-phase 控制；
3. IP-NF、SIREN、raw coordinate MLP；
4. FNO 与 tensorized FNO；
5. GINO/FNOGNO（官方 NeuralOperator 实现）；
6. U-Net/CNextU-Net（The Well 官方 baseline）；
7. 如存在 source/travel-time oracle，增加同样使用该 oracle 的 monolithic baseline。

NeuralOperator 官方库已包含 GINO、TFNO 和 Car-CFD 数据接口，可用于正式 baseline，而不应继续只用自写近似版本。[官方 NeuralOperator 仓库](https://github.com/neuraloperator/neuraloperator)，[GINO 论文](https://arxiv.org/abs/2309.00583)。

### 2.4 核心 ablation

- paired phase identity → ordinary CP；
- geodesic distance → Euclidean radius；
- known source → estimated/perturbed source；
- speed bank full → single speed/wrong speed；
- cross-resolution → same-resolution；
- tensor contraction → joint MLP；
- harmonic residual/mismatch sweep；
- phase-envelope 只作为 negative/optional appendix，不放入主模型。

### 2.5 最小发表门槛

- 至少一个独立/公共多几何数据上，在 ≤2% 观测时超过 IP-NF 与 GINO/TFNO；
- 跨网格结果不能依赖 node ID 对齐；
- oracle source/geodesic 的信息优势必须给 baseline 同样使用；
- mismatch phase diagram 明确展示 paired tensor 的适用区域和 IP-NF 的反转区域。

## 3. 下一轮执行顺序

### Round 4.1：数据 gate，不做大训练

- 独立 solver 生成 8 geometry smoke set；
- The Well 下载/读取 8 trajectory metadata subset；
- 验证 tensor schema、operator、source、mask 和 train/test 隔离；
- 输出 dataset cards 和一张数据可视化。

### Round 4.2：baseline harness

- 接入 graph CP、BPTF、FNO/TFNO、GINO；
- 做 parameter/compute 预算表；
- 单 seed overfit/sanity test。

### Round 4.3：3-seed pilot

- A：Geo-Tucker structured posterior 与数据泛化；
- B：paired tensor 对外部数据和 geometry error；
- 只有达到各自晋级条件的配置进入冻结确认。

### Round 4.4：10-seed confirmation

- 使用新 seed；
- 一次性运行；
- 生成主表、paired statistics、资源表和 failure gallery。
