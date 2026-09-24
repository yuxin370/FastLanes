# 图像压缩数据保留与清理状态

2026-09-22 已完成目录归并和过时副本清理。完整名称、来源和新旧路径对照见[统一数据清单](../../../docs/DATASETS.md)。旧的搬移和隔离命令已撤下，无需再次执行清理脚本。

## 当前目录

压缩副本统一位于 `galp/data/compressed/`：

| 用途 | 目录（相对于统一入口） |
|---|---|
| ImageNet-512 在线推理源 | `imagenet512_val_block_major/` |
| B6 premix 训练源 | `imagenet512_train_block_major_premixed/dct/` |
| Compact-v3 训练 | `imagenet512_train_compact_v3/dct/` |
| Compact-v3 验证 | `imagenet512_val_compact_v3/` |
| 测试用 1K/10K 数据 | `fixtures/imagenet512_train_{1000,10000}_compact_v3/` |
| 离线模型目标备份 | `backup/offline_model_input/` |

离线目标备份已经完成模型 crop/resize 等预处理，仅用于历史复现，**不能作为正式系统性能实验输入**。正式源保留源 JPEG 的完整空间和频率。

JPEG 仍位于 `galp/data/system_rgbnomore/e2e_v3/imagenet_512/{train,val}`，工作用样本清单位于同级 `training_manifests_official_v3/`；原始分辨率验证 JPEG 保留在 `e2e_v2/imagenet/val`。checkpoint 和实验结果未搬移。

## 已完成清理

用户执行清理脚本并返回 `CLEANUP_COMPLETE`。随后确认全部 6 个目标目录已不存在，包括本地 19 个过时目录所在的暂存区、重复的旧 block-major 验证副本、空的失败输出及已归并的三个外部来源目录。按删除前文件逻辑大小计，共约 223.719 GB。

保留的 22 份 manifest、4,434 个 shard 的数据和 metadata 文件均存在，大小符合 manifest；4 份完整离线模型目标备份保留，当前 train/val manifest 和 premix mapping 引用有效。

原始实验 JSON、冻结配置及 checkpoint 保留当时的来源记录。旧运行合同中的绝对路径和摘要不会因修改脚本默认值而自动更新；使用当前数据重新运行时，应通过现有入口新建运行合同，不能直接把旧合同当作已迁移配置。

## 当前 Compact-v3 检查命令

从仓库根目录运行：

```bash
PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
DATA="$PWD/galp/data/compressed"

"$PY" -B -m galp.benchmarks.system_rgbnomore.training.manifest_preflight \
  "$DATA/imagenet512_train_compact_v3/dct/manifest.bin" \
  --expected-manifest-version 3 \
  --expected-physical-layout image-major-vector-rowgroups \
  --expected-spatial-order tiled-z32 \
  --expected-image-count 1281167

"$PY" -B -m galp.benchmarks.system_rgbnomore.training.manifest_preflight \
  "$DATA/imagenet512_val_compact_v3/manifest.bin" \
  --expected-manifest-version 3 \
  --expected-physical-layout image-major-vector-rowgroups \
  --expected-spatial-order tiled-z32 \
  --expected-image-count 50000
```
