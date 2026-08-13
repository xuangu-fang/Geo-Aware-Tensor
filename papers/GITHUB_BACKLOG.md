# GitHub 首批执行 Backlog

完成 GitHub 认证后，将按以下顺序建立 milestones、labels 和 issues。每个 issue 必须引用 experiment ID，并以验收条件关闭。

## M1 Dataset evidence chain

### P0 — `[DATA] Build independent multi-geometry wave/Helmholtz smoke dataset`

- Labels：`shared`, `data`, `priority:P0`, `pilot`
- Experiment：`A-DATA-R4-INDEPENDENT-WAVE-GATE`
- 验收：8 geometries、2 sources、2 resolutions；保存 operator/BC/source/material/field；dataset card、checksum 和可视化；solver 与 learner 模块隔离。

### P0 — `[DATA] Gate The Well acoustic scattering maze subset`

- Labels：`shared`, `data`, `priority:P0`, `pilot`
- Experiment：`B-DATA-R4-WELL-ACOUSTIC-GATE`
- 验收：读取 8 trajectories；确认 geometry 是否跨轨迹变化；恢复 source/initial condition；固定 64/16/32 split 方案；记录下载体积和 license。

### P1 — `[DATA] Adapt CFDBench geometry subset as an external stress test`

- Labels：`shared`, `data`, `priority:P1`, `idea`
- 验收：选择一个 geometry subset；明确 tensor modes；构造 correct/wrong/no-geometry；禁止把规则网格插值误称为原生 mesh 泛化。

## M2 Paper A submission evidence

### P0 — `[BASELINE] Add BPTF and graph-regularized CP/Tucker`

- Labels：`paper-a`, `baseline`, `priority:P0`, `pilot`
- Experiment：`SHARED-BASELINE-R4-OFFICIAL-HARNESS`
- 验收：作者/可信实现或逐式复现；统一 mask/noise；oracle rank、validation-selected rank；parameter/time/memory 表。

### P0 — `[METHOD] Gauge-fixed structured posterior for operator Tucker factors`

- Labels：`paper-a`, `method`, `priority:P0`, `pilot`
- Experiment：`A-METHOD-R4-OPTUCKER-01-RANDOM-BLOCKPOST`
- 验收：MAP/core-only/block posterior 对照；90% coverage-width；均值 NRMSE 退化不超过 5%；若失败则收窄 Bayesian claim。

### P1 — `[ABLATION] Observation-matched core and operator perturbation sweep`

- Labels：`paper-a`, `evaluation`, `priority:P1`, `pilot`
- 验收：CP/small Tucker/large Tucker；correct/noisy/wrong/identity operator；0.5/1/2/5%；3 selection seeds。

## M3 Paper B submission evidence

### P0 — `[BASELINE] Add official GINO, TFNO, FNO and The Well U-Net baselines`

- Labels：`paper-b`, `baseline`, `priority:P0`, `pilot`
- 验收：记录 official repo commit SHA；同 oracle geometry/source information；parameter-matched 和 compute-matched 两张表。

### P0 — `[EVAL] Test paired tensor on independent and The Well wave data`

- Labels：`paper-b`, `evaluation`, `priority:P0`, `pilot`
- 验收：unseen geometry/resolution/source；≤2% observations；paired vs IP-NF vs GINO/TFNO；3 selection seeds；failure gallery。

### P1 — `[ABLATION] Geometry and source uncertainty curves`

- Labels：`paper-b`, `evaluation`, `priority:P1`, `pilot`
- 验收：source shifts、boundary perturbation、material error；oracle/estimated/wrong/no geometry；性能随误差平滑曲线。

## M4 Reproducible paper freeze

### P0 — `[CONFIRM] Freeze splits, configs and fresh confirmation seeds`

- Labels：`shared`, `evaluation`, `priority:P0`, `frozen`
- 验收：两个 paper 各一个 primary metric；10 fresh seeds；paired JSON、主表、runtime/VRAM、所有负结果；test-access audit。

### P1 — `[DOC] Produce camera-ready reproducibility package`

- Labels：`shared`, `documentation`, `priority:P1`
- 验收：environment lock、data download manifests、one-command smoke、figure regeneration、license/citation、archive checksum。

## Issue 创建规则

- 不为每个 seed 单独建 issue；一个 issue 对应一个可证伪假设或数据 gate。
- pilot 与 confirm 不共用 issue，防止配置修改污染冻结记录。
- 关闭 issue 时必须附 aggregate artifact 和 REJECT/SELECT 决策。
- envelope 方法保持 `REJECTED`，除非满足 registry 中的 reopen condition。
