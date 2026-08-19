# 共享数据与实验存储（2026-08-19）

## 1. 结论

大规模数据和后续大型实验产物统一放在 NFS，并分成跨项目公共层与本研究方向专属层：

```text
/mnt/data/xuangu-fang/
├── ai-physical-dynamics/
│   ├── datasets/           # 多个物理学习项目共享的 benchmark
│   └── benchmark-results/  # 共享 benchmark 的大型结果
└── physics-informed-tensor-learning/
    ├── datasets/           # Geo-Aware Tensor 各方向专属生成数据
    ├── runs/               # 本方向大型训练输出与 checkpoint
    ├── cache/              # 本方向可再生成的下载/特征缓存
    └── metadata/           # 本方向机器级审计
```

代码、Git 元数据、论文文档、小型 JSON/CSV 汇总和最终图片继续放在 `/home/ubuntu/project`。不要把虚拟环境或 Git 仓库整体迁到 NFS；大量小文件会显著降低环境启动和 Git 操作速度。

## 2. 本次已迁移的数据

本轮迁移 13 个 payload，共 `43,440,550,310` bytes（约 41 GiB）。切换前，每个映射均满足：

- `rsync -ani --delete` 为 0 changes；
- 源与目标 byte count 完全一致；
- 源与目标 regular-file count 完全一致；
- NumPy mmap、HDF5、OpenFWI 和 Arrow 文件头读取通过；
- Hub 60 项测试和方向 4 的 7 项测试通过。

| 原项目路径 | NFS 相对路径 | bytes | files |
|---|---|---:|---:|
| `yanjiu/data/active_matter_multi/train` | `ai-physical-dynamics/datasets/active_matter_multi/train` | 4,395,630,592 | 5 |
| `yanjiu/data/active_matter_multi/test` | `ai-physical-dynamics/datasets/active_matter_multi/test` | 1,384,120,320 | 5 |
| `yanjiu/data/active_matter_ood_locked/raw` | `ai-physical-dynamics/datasets/active_matter_ood_locked/raw` | 4,345,298,944 | 4 |
| `yanjiu/data/active_matter_transport_fresh_shift` | `ai-physical-dynamics/datasets/active_matter_transport_fresh_shift` | 1,342,177,280 | 4 |
| `yanjiu/data/cfdbench/raw` | `ai-physical-dynamics/datasets/cfdbench/raw` | 786,403,674 | 1 |
| `yanjiu/data/cfdbench/extracted` | `ai-physical-dynamics/datasets/cfdbench/extracted` | 836,587,386 | 742 |
| `yanjiu/data/kolmogorov_mno/raw` | `ai-physical-dynamics/datasets/kolmogorov_mno/raw` | 6,566,707,456 | 2 |
| `yanjiu/data/realpde_active_physics_confirmation/raw` | `ai-physical-dynamics/datasets/realpde_active_physics_confirmation/raw` | 4,316,377,840 | 8 |
| `yanjiu/data/realpde_cylinder_subset/raw` | `ai-physical-dynamics/datasets/realpde_cylinder_subset/raw` | 16,480,728,418 | 38 |
| `yanjiu/data/realpde_cylinder_subset/prepared` | `ai-physical-dynamics/datasets/realpde_cylinder_subset/prepared` | 456,453,977 | 2 |
| `measure-what-persists/data/openfwi_curvefault_a` | `ai-physical-dynamics/datasets/openfwi_curvefault_a` | 2,129,400,768 | 6 |
| `Geo-Aware-Tensor/data` | `physics-informed-tensor-learning/datasets/Geo-Aware-Tensor/data` | 387,470,951 | 248 |
| `functional-operator-completion/data` | `physics-informed-tensor-learning/datasets/functional-operator-completion/data` | 13,192,704 | 64 |

原项目路径已经替换为绝对符号链接，因此现有实验命令不需要改参数。仓库中已经跟踪的 benchmark manifest、split 和 sealed plan 没有迁走；它们仍由 Git 管理。

## 3. 每次长实验前的检查

```bash
cd /home/ubuntu/project/Geo-Aware-Tensor
python3 tools/check_shared_data.py
python3 tools/check_shared_data.py --deep
```

默认检查 NFS mount、13 个链接、目标存在性和可读性。`--deep` 还会重新计算 byte/file counts，适合冻结实验前执行。

若 `/mnt/data` 没有挂载，符号链接会显示为 broken。此时不要重新下载或生成同名数据，也不要删除符号链接；先恢复 NFS mount。

## 4. 后续数据和 runs 约定

1. 多方向共享的外部 benchmark 放到 `ai-physical-dynamics/datasets/<dataset-name>/<version>/`；只服务本方向的合成/派生数据放到 `physics-informed-tensor-learning/datasets/<track>/<version>/`。仓库只提交 manifest、checksum、split 和下载/预处理脚本。
2. 大型 checkpoint、逐步预测和 tensor dump 放到 `physics-informed-tensor-learning/runs/<repo>/<experiment-id>/`；论文表格和最终图复制回对应仓库。
3. 可重建 cache 放到 `physics-informed-tensor-learning/cache/<tool-or-dataset>/`，不可把 cache 当成唯一实验记录。
4. 数据目录应只由一个写入任务负责；训练任务默认只读，避免多个进程同时改写 NFS 数据。
5. CFDBench extracted 这类大量小文件的数据在 NFS 上随机读取较慢。高频训练前可构造版本化的 shard/container，但必须保留原始数据和转换 manifest。

## 5. 可靠性边界

NFS 是当前唯一工作副本，不等价于备份。Git 只保护代码和小型元数据，不保护这些 payload。重要外部数据应保留官方 URL/checksum；不可再生成的数据需要另行配置快照或第二副本。
