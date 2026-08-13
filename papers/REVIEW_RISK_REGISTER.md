# 论文审稿风险登记与应对

| ID | 潜在审稿压力/雷点 | 概率 | 影响 | 必须准备的证据 | 状态 |
|---|---|---:|---:|---|---|
| R1 | 受控数据与方法同构，结果是 tautology | 高 | 致命 | 独立 solver + 公共 The Well/CFDBench；生成器与 learner 隔离 | OPEN |
| R2 | 几何信息是 oracle，baseline 没拿到相同信息 | 高 | 致命 | oracle-matched monolithic baseline；estimated/noisy geometry 曲线 | PARTIAL：B 已有 wrong/noisy path，A 仍需 estimated operator |
| R3 | Tucker 收益只是 core 更大 | 高 | 高 | 参数/自由度匹配；wrong/identity operator；over-capacity control | PARTIAL |
| R4 | “Bayesian”只覆盖 core，factors 是 MAP | 高 | 高 | 精确表述；structured factor posterior；MAP/core-only/full posterior ablation | OPEN |
| R5 | UQ 靠 calibration split 或过宽区间 | 中 | 高 | coverage-width、NLL、cross-fit calibration、同观测预算 | OPEN |
| R6 | 初始化使用 dense GP，方法成本/信息不公平 | 高 | 中 | 初始化计时；random/HOSVD/flat init ablation；不读取 held-out target | PARTIAL |
| R7 | baseline 实现弱或未调优 | 高 | 致命 | 作者实现/官方库；版本 SHA；统一 tune budget；parameter+compute matched | MITIGATED-B / OPEN-A |
| R8 | 低 observation ratio 的定义不现实 | 中 | 高 | 同时报告 entries、sensor count、trajectory/fiber masks | OPEN |
| R9 | test leakage：geometry、normalization、rank、calibration | 中 | 致命 | split manifest；fit-time data access audit；frozen seeds | PARTIAL |
| R10 | seed 或 metric 选择导致 p-hacking | 中 | 高 | pilot/selection/confirm 分离；primary metric；全 seed paired differences | PARTIAL |
| R11 | B 只在人工 aligned regime 有效 | 高 | 高 | 外部 wave data；mismatch phase diagram；明确 scope，不宣称通用 SOTA | MITIGATED：The Well modest confirmation + long-horizon failure |
| R12 | phase-envelope 负结果削弱故事 | 低 | 低 | 将其作为边界证据；不塞入主模型 | CLOSED |
| R13 | operator construction 依赖手工 domain expertise | 中 | 高 | known/estimated/perturbed operator；构造时间和敏感性 | OPEN |
| R14 | 与 graph-regularized tensor/BPTF/GINO/TFNO 新颖性重叠 | 高 | 高 | related-work 对照；同时证明显式 operator factors、Bayesian core 和 geometry causal ablation | OPEN |
| R15 | 数据/代码太大，无法复现 | 中 | 高 | 固定小 subset manifest、下载脚本、checksum、轻量 smoke config | OPEN |
| R16 | 计算成本远高于简单 baseline | 中 | 中 | wall time、peak VRAM、预测成本、sample-efficiency 曲线 | OPEN |

## 提交前硬性检查

- R1、R2、R7、R9、R14 未降到 `MITIGATED` 前，不进入投稿冻结。
- Paper A 若 R4 未解决，标题和摘要不得使用 “fully Bayesian”。
- Paper B 若没有公共/独立数据超过 IP-NF 与至少一个 neural operator baseline，只能定位为机制/受控研究。
- 任何主表必须同时包含正确几何、错误几何、同几何 flat/joint 和显式 tensor baseline。
