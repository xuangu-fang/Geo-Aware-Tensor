# 三方向投稿 Gate 后技术收口（2026-08-16）

## 1. 最短结论

本轮不是继续增加模型名词，而是把三个问题分别压缩成可证伪的论文命题：

1. **方向 1：条件 GO。** 已知近似算子时，把其低频基底直接放进 Tucker factors，能在 10% random 与 receiver-fiber 观测下稳定降低误差；2% structured 和 source-fiber 是明确失败区。
2. **方向 3：条件 GO。** 把算子联合谱投影为合法的一维 GP kernels，在各向异性扩散上得到 5/5 fresh-seed 优势；固定 generic support floor 能对抗先验删频，但会牺牲少量 matched 精度。输运的完整 signed spectrum 当前不能表示，降为 limitation。
3. **方向 4：旧路线 NO-GO，新任务 validation-level conditional GO。** 不再让弱 Functional Tucker 对抗 MIONet，而是保留强 Spectral MIONet，研究不规则域、孔洞和极稀疏输出标签下，真实域内 heat transport features 是否带来额外信息。答案在 screened/diffusive 场上为正，在 wave stress 上为负。

方向 1、3 已达到“可以开始写论文并进入外部数据 gate”的程度，但还不是“实验已经完整可投稿”。方向 4 是新的应用型假设，不能与旧路线的负结果混在一起。

## 2. 冻结版本与入口

| 方向 | 冻结 commit | 决策 | 论文级中文报告 |
|---|---|---|---|
| 1. Operator-prior Tucker | [`3d6eeda`](https://github.com/xuangu-fang/operator-prior-tensor/commit/3d6eedaa48b51d04f374c7adebfbc71a5a750ced) | 条件 GO | [中文报告](https://github.com/xuangu-fang/operator-prior-tensor/blob/main/docs/PAPER_TECHNICAL_REPORT_ZH.md) |
| 3. Operator-spectral FunBaT | [`558829f`](https://github.com/xuangu-fang/operator-spectral-funbat/commit/558829f) | 条件 GO | [中文报告](https://github.com/xuangu-fang/operator-spectral-funbat/blob/main/docs/PAPER_TECHNICAL_REPORT_ZH.md) |
| 4. Domain-Heat MIONet restart | [`ea2180b`](https://github.com/xuangu-fang/functional-operator-completion/commit/ea2180b16d2a14fcbe25423c01645e3997c425f9) | validation-level 条件 GO | [中文报告](https://github.com/xuangu-fang/functional-operator-completion/blob/main/docs/GEOMETRY_MIONET_RESTART.md) |

三个报告均从 motivation、formulation、inference、数据、baseline、公平性、结果、限制和下一道 gate 展开。旧的长技术账本仍保留，但不再是首要阅读入口。

## 3. 方向 1：算子基底进入 Tucker

### 3.1 一句话方法

传统 neural/functional Tucker 自由学习每一维的连续 factor；方向 1 用近似 PDE 算子的低频特征作为 factor basis，只学习较小 Tucker core 和必要的缩放。几何/物理信息进入的是显式张量分解，而不是额外的黑盒正则项。

### 3.2 冻结协议

- 真实变系数 Neumann diffusion Green tensor，而非简单旋转合成数据；
- cutoff 8、core rank `(4,5,5)`、400 steps、fresh seeds 101--105；
- random、source-fiber、receiver-fiber masks；观测率 2%、5%、10%；
- 同时比较宽 Neural Functional Tucker、参数匹配 Neural Tucker 和 wrong-operator control。

### 3.3 核心结果

| mask，10% | Operator Tucker | 宽 Neural Tucker | paired wins |
|---|---:|---:|---:|
| random | **0.1645±0.0102** | 0.2065±0.0536 | 4/5 |
| receiver-fiber | **0.2165±0.0517** | 0.2695±0.1117 | 4/5 |
| source-fiber | 0.2937±0.1890 | **0.2562±0.1177** | 3/5 |

参数匹配控制为 Operator Tucker 212 参数、Neural Tucker 210 参数；wrong operator 约为 `0.94--0.96`。因此 random/receiver 的改善不能仅归因于参数更少或 Tucker decoder 本身。

### 3.4 可以讲与不能讲

- 可以讲：算子 basis 在有限观测下带来可测量的 bias--variance 优势；优势依赖 mask 与观测率。
- 不能讲：越稀疏优势越大；2% structured masks 实际明显失败。
- 论文主图应是 observation ratio × mask × operator mismatch 的 phase diagram，而不是只挑一个最好点。

## 4. 方向 3：算子谱变成 GP kernels

### 4.1 一句话方法

对近似方程 `L u = w`，先由算子得到联合功率谱，再做非负低秩分离；每个一维非负谱都对应合法 GP kernel，并进入 functional CP 的不同 mode/rank。推断采用 ELBO+SGD。为防止近似算子彻底漏掉真实频率，固定保留 25% generic spectral support。

### 4.2 为什么不是普通字典学习

主 kernel atoms 来自算子联合谱的合法投影，而非任意选择 RBF/Matérn。generic 部分也不是自由挑选最优字典：它只作为固定非零的频率安全带，且明确测量其 matched-prior 代价。最终 collapsed 实现让 4-atom 与 8-atom bank 共享相同 GP coefficient 数量，排除了“更多 atoms 等于更多后验参数”的混淆。

### 4.3 Fresh-seed 结果

冻结设置为 2% observations、rank 2、`24^3` grid、400 steps、seeds 201--205。

| anisotropic diffusion | NRMSE | paired conclusion |
|---|---:|---|
| operator-global | 0.1212±0.0606 | — |
| operator per-mode/rank | **0.1183±0.0582** | 对 generic 5/5；对 global 仅 3/5 |
| generic per-mode/rank | 0.1567±0.0990 | — |
| oracle | 0.1183±0.0588 | operator 方法匹配其均值 |

operator per-mode/rank 的 induced-spectrum cosine/L2 为 `0.977/0.204`，generic 为 `0.926/0.427`；predictive NLL 为 `-0.679` 对 `-0.451`。强证据属于 **operator-derived mode kernels**；自由 per-rank routing 只有约 2.4% 均值增益，不能单列贡献。

严格删频控制：

| setting | wrong-support operator | fixed-floor robust | paired wins |
|---|---:|---:|---:|
| reference advection | 0.672 | **0.040** | 5/5 |
| shifted advection | 0.632 | **0.085** | 5/5 |
| anisotropic diffusion | 0.615 | **0.130** | 5/5 |

matched anisotropic diffusion 上，robust 相比纯 operator 从 `0.118` 变为 `0.131`。所以第二贡献是可量化的 robustness--specificity tradeoff，不是“免费稳健”。

### 4.4 必须公开的负结果与代码审计

- rank-4 full signed-spectrum error：advection 约 `0.18`，anisotropic diffusion 为 `0.0043`。当前实数逐轴 kernels 不能表示 tilted transport 的 cross-sign coupling；advection 只作 limitation。
- 一版 strict-support 负结果来自实现错误：公共 Fourier basis 被从第一个 atom 反除构造，首 atom 被删频时也错误清零了公共高频 basis。修复为解析 Fourier basis 后只重跑 strict controls，主确认表未重训。修复前后均保存在审计 JSON 中。

## 5. 方向 4：改变任务后的 MIONet 路线

### 5.1 为什么改任务是合理的

MIONet 的乘积 branch 本身已经是一种低秩多输入交互；简单再贴一个“low-rank”标签不构成贡献。更清楚的问题是：**在训练域与测试域的孔洞拓扑不同、每个输出场只有 1%--10% 标签时，标准 MIONet 的欧氏坐标 trunk 是否缺少域内传播信息？**

新方法保留强 Spectral MIONet，只增加 source-conditioned、sign-invariant、multiscale domain heat features。这些 features 在真实不规则域上扩散，能区分“欧氏距离很近但被孔洞隔开”的点。SDF、同宽 zero features、ambient heat、single-scale heat 都作为必要控制。

### 5.2 Fresh unseen two-hole domains

| 每场输出标签 | Spectral MIONet | SDF-MIONet | Domain-Heat MIONet |
|---:|---:|---:|---:|
| 1% | 0.219±0.004 | 0.231±0.002 | **0.146±0.005** |
| 2% | 0.210±0.005 | 0.197±0.007 | **0.118±0.004** |
| 5% | 0.178±0.002 | 0.178±0.004 | **0.125±0.012** |
| 10% | 0.170±0.002 | 0.172±0.004 | **0.112±0.003** |

1% 时孔洞边界局部误差从 Spectral MIONet 的 `0.362` 降到 `0.188`。参数量仅从 126,785 增至 127,425（约 0.5%）。但是 geodesic-wave stress 中所有方法都在 `1.02--1.09`，因此当前结论只属于 screened/diffusive fields。

### 5.3 当前定位

- 旧 Functional Tucker replacement 路线仍为 NO-GO；不能用新任务覆盖旧负结果。
- 新路线是 validation-level conditional GO，适合应用导向论文或学生项目。
- 真正的新意候选不是 generic geo-aware 或 low-rank，而是稀疏标签下 source-conditioned multiscale domain transport 与 MIONet 多输入融合。
- 2/3-hole 冻结 test 尚未消费；在外部 PDE/mesh 数据和强 geometry-aware operator baseline 前，不升级为 AI 主会主线。

## 6. 下一阶段统一主实验

### 方向 1

1. 扩到至少两个非合成 PDE families，并保留 random/source/receiver structured masks；
2. 做 operator coefficient mismatch 连续曲线和 basis cutoff × tensor rank 消融；
3. 加强 functional/neural CP/Tucker，同步报告参数量、wall time 和 seed-level paired results。

### 方向 3

1. 首要 gate 是不由同一 finite atom family 直接采样的 PDE solutions；
2. 比较 FunBaT、generic functional tensor、operator-global、operator-mode、fixed-floor robust；
3. 做 1%/2%/5% 与 structured sensors；报告 NRMSE、NLL、coverage 和 spectrum diagnostics；
4. 只有实现 complex/signed-conjugate 或 cross-mode phase 后，才重新把 advection 放入正面主张。

### 方向 4

1. 消费冻结 2/3-hole test，并增加更强的 unseen-topology family；
2. 在外部 irregular-mesh screened/diffusion 数据上比较 MIONet、Geom-DeepONet/GNOT 类 baseline；
3. 消融 heat scale、source conditioning、符号不变性和 label ratio；
4. 若优势只存在当前生成器或只对 diffusion 成立，则固定为窄应用项目，不继续泛化故事。

## 7. 工程状态

- 方向 1、3、4 分别有 `6/9/7` 项测试通过；三个工作树均与 GitHub `main` 一致。
- 方向 3 的 24 个有效 JSON 已完成 finite-value 审计；三个仓库均通过 whitespace/diff 检查。
- 方向 1 与方向 3 暂时共享顶层包名 `geoaware`，同一环境连续 editable install 会冲突；当前复现使用各仓 `PYTHONPATH=src`，后续需改唯一包名或独立环境。

## 8. 最终优先级

1. **方向 3：最高方法优先级。** 数学链条最完整，但必须过非 planted PDE 数据 gate。
2. **方向 1：稳健的机制/小论文主线。** 故事最简洁，重点是 phase boundary 和 functional Tucker 强基线。
3. **方向 4：独立的任务重启。** 当前数字强，但 scope 窄、外部数据未过 gate，先按应用线推进。

下一轮的统一验收条件已经登记在 [Hub issue #19](https://github.com/xuangu-fang/Geo-Aware-Tensor/issues/19)；方向 4 的独立 frozen-topology gate 见 [子仓 issue #2](https://github.com/xuangu-fang/functional-operator-completion/issues/2)。
