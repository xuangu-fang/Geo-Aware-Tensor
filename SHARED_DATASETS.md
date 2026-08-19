# 共享物理数据与公开资源目录

更新时间：2026-08-19

本文件是三个活跃方向共同的数据入口。它回答四个问题：数据实际放在哪里、是否已经可读、它是否带几何/PDE/算子元数据、以及它适合哪个实验。存储迁移和完整性检查见 [DATA_STORAGE.md](DATA_STORAGE.md)。

## 1. 准入原则

一个数据集“能下载”不等于“适合证明 geometry/operator-aware tensor”。进入主表前必须记录：

1. 原始任务与物理方程，不能把普通时空预测改名为 tensor completion；
2. geometry 表示：规则网格、mask/SDF、mesh connectivity、boundary tags 是否可得；
3. operator 信息：离散矩阵、PDE 系数、边界条件、生成器代码分别是否可得；
4. tensor axes 的物理语义，以及 train/validation/test 在 geometry、trajectory、source 和 coordinate 上如何隔离；
5. observation mask、noise、normalization、checksum、license 和官方来源；
6. 若所有方法 held-out NRMSE 约为 1，则先判定任务未跑通，不比较微小相对提升。

大文件不进入 Git。仓库只提交下载/生成脚本、manifest、split、checksum、小型 summary 和最终图。

## 2. 本机已经存在的共享资源

公共根目录为 **/mnt/data/xuangu-fang/ai-physical-dynamics/datasets**；方向专属生成数据位于 **/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets**。

| 本机资源 | 当前状态 | 几何/算子信息 | 建议用途 |
|---|---|---|---|
| cfdbench/raw、cfdbench/extracted | 已迁移，可审计 | 多工况流场；需逐 scenario 核对 boundary/mesh metadata | Track 4 operator baseline 压力测试；Track 3 只作失配外测 |
| realpde_cylinder_subset/{raw,prepared} | 已迁移 | 圆柱边界、真实/模拟流场；精确离散算子未随数据直接提供 | Track 4 unseen-condition/geometry 测试；geometry-only prior |
| realpde_active_physics_confirmation/raw | 已迁移 | paired real/simulation，需使用冻结 metadata/split | sim-to-real 压力测试，不用于初始方法选择 |
| openfwi_curvefault_a | 已迁移 | velocity map 与多炮地震记录；source/receiver/time 语义强 | Track 1 后期非自伴/波动压力测试；Track 3 kernel stress |
| kolmogorov_mno/raw | 已迁移 | 周期流场、Re 梯度；规则网格 | Track 3 operator-spectrum 外测；不证明不规则几何 |
| active_matter_* | 已迁移 | 规则网格动态图 | 通用时空 stress；不作为几何主证据 |
| Geo-Aware-Tensor/data | 已迁移 | The Well acoustic 子集及历史不规则合成数据 | 历史复现与快速 smoke test |
| functional-operator-completion/data | 已迁移 | Domain-Heat MIONet 的不规则域/孔洞生成数据 | Track 4 当前主 POC；可为 Track 1 FEM POC 提供 geometry 规范参考 |

长实验前运行：

    cd /home/ubuntu/project/Geo-Aware-Tensor
    python3 tools/check_shared_data.py --deep

只有通过检查后才启动长实验。NFS 是工作副本，不等价于备份。

## 3. 官方公开资源

| 资源 | 官方入口 | 可用内容 | 对三个方向的判断 |
|---|---|---|---|
| PDEBench | [GitHub](https://github.com/pdebench/PDEBench)、[DaRUS 数据 DOI](https://doi.org/10.18419/darus-2986) | 多类 PDE、数据生成代码、forward/inverse baselines | Track 1/3 最优先的外部机制数据；可从生成器补存 operator metadata。多数数据为规则网格，不单独证明孔洞泛化 |
| AirfRANS | [GitHub](https://github.com/Extrality/airfrans_lib)、[dataset API](https://airfrans.readthedocs.io/en/latest/modules/dataset.html) | 1000 个 airfoil RANS simulations、非结构点云/mesh、full/scarce/Re/AoA tasks | Track 4 优先 irregular-geometry 外测；Track 1 只能称 geometry-Laplacian prior，不能声称知道完整 RANS operator |
| The Well | [GitHub](https://github.com/PolymathicAI/the_well)、[数据总览](https://polymathic-ai.org/the_well/datasets_overview/) | 统一 HDF5 schema 的大规模多物理数据；含 acoustic scattering、active matter、reaction diffusion 等 | acoustic 对孔洞/介质和波场有价值；体量很大，应先取固定小子集。geometry metadata 需逐 dataset 审计 |
| OpenFWI | [官网](https://openfwi-lanl.github.io/)、[GitHub](https://github.com/lanl/OpenFWI) | 成对地下速度模型与多炮地震观测 | source–receiver–time tensor 最自然；但高频波动和吸收边界使它适合后期压力测试 |
| RealPDEBench | [GitHub](https://github.com/AI4Science-WestlakeU/RealPDEBench)、[官网](https://realpdebench.github.io/) | cylinder、FSI、controlled cylinder、foil、combustion 的 paired real/simulation 数据与多种 NO baselines | Track 4 的 sim-to-real 与真实测量测试；不能从 paired simulation 自动推断“真实 operator 已知” |
| CFDBench | [GitHub](https://github.com/luo-yining/CFDBench) | 多种边界/几何/物理参数的流体 benchmark 与 baseline code | 已有本地副本；正式使用前需冻结 scenario、版本与 split |
| FlowBench | [论文](https://arxiv.org/abs/2409.18032) | 复杂几何上的大规模 flow simulations | 潜在 Track 4 外部 geometry gate；确认官方数据、license 和 mesh metadata 前只列候选 |

## 4. 各方向的数据优先级

| 优先级 | Track 1：Operator-prior tensor | Track 3：Operator-spectral FunBaT | Track 4：Functional operator completion |
|---|---|---|---|
| P0 controlled | 当前 1D diffusion Green tensor；新建不规则 FEM Green tensor | 当前 planted anisotropic diffusion 与 support-floor audit | 当前 Domain-Heat 0/1-hole train、2/3-hole sealed test |
| P1 external | PDEBench diffusion/reaction-diffusion/Darcy | PDEBench diffusion/advection；本地 Kolmogorov | AirfRANS；RealPDEBench cylinder |
| P2 stress | OpenFWI、The Well acoustic | OpenFWI/The Well wave、CFDBench | CFDBench、The Well acoustic、FlowBench |

P0 用于确认机制；P1 用于外部有效性；P2 用于暴露边界。不得用一个数据集同时完成调参、模型选择和最终 confirmation。

## 5. 统一 manifest 最低字段

每个 manifest 至少包含 dataset_id、official_url、license、local_root、version_or_commit、raw_checksums、physics、geometry_representation、operator_metadata、tensor_axes、train_val_test_unit、observation_protocol、normalization_scope 和 generator_or_preprocess_command。

各子仓库可以增加方向专属字段，但不能删除这些公共字段。Track 1、3、4 的具体选择见各自的 docs/DATASETS_AND_RESOURCES.md。
