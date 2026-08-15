# 三方向独立迭代收口报告（2026-08-15）

## 一页结论

本轮完成了仓库拆分、三条线的独立 3--5 轮实验和远端同步。当前不应把三条线等价推进：

1. **方向 3 优先级最高，条件 GO。** 青基中的“PDE 联合谱 → 非负可分离谱 → 不同 mode/rank GP kernel → ELBO+SGD”已跑通；最稳健版本是 operator-derived atoms 加少量 generic escape atoms。
2. **方向 1 条件 GO。** 真实变系数 diffusion 上，Operator Tucker 在 10% 观测有相对稳定优势，但 2%--5% 尚未越过方差与估计难度；故事应是可测量的 bias--variance phase boundary，不是无条件战胜 neural functional Tucker。
3. **方向 4 NO-GO。** 公平增强坐标 trunk 后，MIONet 明显领先 functional Tucker。保留数据协议和负结果，不继续堆 operator encoder。

## 冻结版本

| 方向 | 仓库 | 冻结 commit | 决策 |
|---|---|---|---|
| 1 | [operator-prior-tensor](https://github.com/xuangu-fang/operator-prior-tensor) | [`07de48d`](https://github.com/xuangu-fang/operator-prior-tensor/commit/07de48d20fae0dfc952215c9d3715ddfaf2af862) | 条件 GO |
| 3 | [operator-spectral-funbat](https://github.com/xuangu-fang/operator-spectral-funbat) | [`08bc6db`](https://github.com/xuangu-fang/operator-spectral-funbat/commit/08bc6db9133a3cd3e698060f7392a309e95e7f91) | 条件 GO，最高优先级 |
| 4 | [functional-operator-completion](https://github.com/xuangu-fang/functional-operator-completion) | [`e265748`](https://github.com/xuangu-fang/functional-operator-completion/commit/e2657482fd035f55a87bce478f906ceade1376a1) | NO-GO |

## 方向 1：Operator-prior Tensor

### 本轮真正回答的问题

固定 operator basis 的收益能否从人工 subspace rotation 扩展到真实 PDE coefficient perturbation？答案是：**在 10% 观测下有条件成立，但极稀疏区还不稳定。**

- coefficient contrast 0→2 时，真实 field 对 reference basis 的 projection residual 从 `0.0459` 增到 `0.0965`。
- contrast=1 时，10% 观测的 Operator Tucker/Neural Functional Tucker NRMSE 为 `0.158/0.189`；2% 为 `0.273/0.262`。
- cutoff 5/8/12 的 residual 为 `0.1645/0.0699/0.0253`，但 2% NRMSE 为 `0.293/0.273/0.331`。更完整的 basis 反而可能增加估计方差。
- matched core rank 扫描表明 10% 优势不只来自一个手调 rank；小 core 近似平局，默认和大 core 小幅为正。

下一步只做一个确认实验：冻结 cutoff/rank，换 5 个 fresh seeds 和 structured source/receiver fiber masks。若不能达到至少 4/5 seeds 获胜，则保持小论文/机制论文定位。

## 方向 3：Operator-spectral FunBaT

### 最终保留的高级方法

先由已知或近似 PDE 的频率响应形成联合谱，再将它非负低秩分解成每个 tensor mode 的一维合法谱原子；functional CP/Tucker 中每个 mode/rank 使用这些原子的 soft mixture，并用 ELBO+SGD 推断。少量 generic atoms 只负责在算子错配时补足缺失频率支撑。

关键结果：

| 方法 | 1% | 2% | 5% |
|---|---:|---:|---:|
| global dictionary | 0.593 | 0.088 | 0.045 |
| per-mode/rank | **0.482** | 0.072 | **0.033** |
| oracle | 0.302 | **0.047** | 0.033 |
| swapped control | 0.811 | 0.897 | 0.604 |

- rank-4 非负联合谱分离误差：diffusion `0.0028`、advection `0.0325`、wave `0.1079`。波动 dispersion surface 明显更难。
- atom top-1 恢复只有 22%--33%；相关 kernel atoms 在预测上可替代，不能声称“发现真实 kernel 标签”。
- 严格删除正确 prior 的高频 support 后，operator-only NRMSE 为 `0.631`；加入 generic dictionary 后恢复为 `0.068`。这是混合版本最关键的因果性证据。

推荐论文故事是 **operator spectrum 的低秩合法投影 + misspecification-robust adaptation**，不是普通 dictionary learning。

## 方向 4：Functional Operator Completion

本轮采用 whole-simulation-combination holdout；训练组合内部再做 1%/5%/10% coordinate subsampling。数据为变系数 screened-Poisson factorial campaign，并实现 transductive CP/Tucker、MIONet-style 和 Joint-INR。

- 60% 组合、10% 坐标时，原始 Tucker/MIONet 为 `0.576±0.037 / 0.536±0.036`。
- 双方加入相同 Fourier coordinate lifting 后，变为 `0.317±0.081 / 0.098±0.006`。
- 高 coefficient contrast 时仍为 `0.346±0.090 / 0.141±0.013`。

这说明原始接近来自共同的弱 coordinate representation。继续给 Tucker 加完整 function encoder 会把方法逐步变成已有 MIONet，而不是产生清楚的新 tensor contribution。因此停止主线，但保留严格 split 协议和所有负结果。

## 复现与工程说明

- 三个仓库的冻结 commit 均已推送到各自 `main`，工作树与远端一致。
- Track 1/3/4 分别有 `3/4/4` 项测试通过，并完成 diff whitespace 检查；实验 JSON 已做结构审计。
- Track 1 和 Track 3 暂时共享历史包名 `geoaware`；若在同一环境连续 editable-install 会冲突。当前可靠运行方式是在各自仓库使用 `PYTHONPATH=src python -m pytest -q`，长期维护时应改为唯一包名或每仓独立虚拟环境。

更详细的公式、实现、baseline、数据协议和逐轮决策都保存在各独立仓库的 `docs/TECHNICAL_REPORT.md` 与 `docs/ITERATIONS.md`。

下一步已转成可验收 GitHub 门槛：[Track 1 fresh-seed gate](https://github.com/xuangu-fang/operator-prior-tensor/issues/1)、[Track 3 robust-bank confirmation](https://github.com/xuangu-fang/operator-spectral-funbat/issues/1)、[Track 4 stop/restart gate](https://github.com/xuangu-fang/functional-operator-completion/issues/1) 和 [Hub 方向评审](https://github.com/xuangu-fang/Geo-Aware-Tensor/issues/18)。
