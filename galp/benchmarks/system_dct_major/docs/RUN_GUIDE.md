# DCT-major benchmark run guide

## 运行前提

正式 GALP 路径需要：

1. DCT-major `manifest.bin` 及其完整 shard 集合；
2. 与 manifest 和 crop descriptor 匹配的 `BLOCK_MAJOR_ACCESS_V1` sidecar；
3. 当前构建的 `_galp_direct_dct` binding；
4. RGB-no-more DCT/RGB checkpoints；
5. 无 shuffle 的 canonical ImageNet validation 样本顺序。

sidecar 必须在 contract 快照前完成物化。runner 不允许 native execution 在 cold
计时中首次创建 sidecar，因为这会让 storage gate、cache eviction 集合和输入快照
失效。

构建并检查接口：

```bash
cmake --build build --target _galp_direct_dct -j2
PYTHONPATH=.:build/galp/torch \
  /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m galp.benchmarks.system_dct_major.run --help
```

正式 store 使用现有 `imagenet512_val_block_major`，源 JPEG 为
`galp/data/system_rgbnomore/e2e_v3/imagenet_512/val`。这两者来自同一份
512×512、4:2:0 数据视图；不能混用原始分辨率 JPEG 或离线生成模型目标的备份。
数据已完成构建及访问索引生成，无需重新压缩；创建其他副本时必须选择新的输出目录。
完整目录说明见[数据清单](../../../docs/DATASETS.md)。

可在计时外验证已有数据：

```bash
build/galp/tools/jpeg_dct/galp_jpeg_dct_tool \
  --verify-manifest galp/data/compressed/imagenet512_val_block_major/manifest.bin \
  --verify-workers 16 \
  galp/data/system_rgbnomore/e2e_v3/imagenet_512/val
```

全量 verifier 必须报告 `coefficient_mismatches: 0` 和 `exact: true`。

当前正式目录已包含 `access/`。仅在新建数据副本时，于计时外构建 block-major descriptors。K64 和 configurable selection
必须命中同一份 ownership schedule；物理 byte/range 计划仍按各自 selection
独立生成。

```bash
cmake --build build --target galp_block_major_access_tool _galp_direct_dct -j2
build/galp/tools/jpeg_dct/galp_block_major_access_tool \
  galp/data/compressed/imagenet512_val_block_major/manifest.bin \
  --output-dir galp/data/compressed/imagenet512_val_block_major/access \
  --output-json galp/data/compressed/imagenet512_val_block_major/access/build.json
```

当前 scheduled-range profile 还要求首分片的
`access/shard_000000.active_output_schedule.bin`。若尚未生成，在 GPU 可用时先于
contract 快照执行一次完整首分片读取；这一步不属于计时：

```bash
PYTHONPATH=.:build/galp/torch \
GALP_BLOCK_MAJOR_ACCESS_DIR="$PWD/galp/data/compressed/imagenet512_val_block_major/access" \
GALP_PHASE6_NATIVE_PHYSICAL=1 \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python - <<'PY'
from galp.profiles.rgbnomore import VALIDATION_CENTER_CROP_512
from galp.torch import DirectDctReader

reader = DirectDctReader(
    "galp/data/compressed/imagenet512_val_block_major/manifest.bin",
    module_path="build/galp/torch",
)
pipeline = reader.pipeline(VALIDATION_CENTER_CROP_512, coefficients=range(32))
pipeline.start([list(range(1024))])
next(pipeline)
pipeline.close()
PY
test -f galp/data/compressed/imagenet512_val_block_major/access/shard_000000.active_output_schedule.bin
```

## Contract preflight

```bash
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m galp.benchmarks.system_dct_major.run \
  --preset e2e \
  --workload feature-extraction \
  --sample-count 1024 --repeats 1 \
  --dct-major-manifest galp/data/compressed/imagenet512_val_block_major/manifest.bin \
  --dct-major-label-map galp/data/compressed/imagenet512_val_compact_v3/labels.json \
  --block-major-access-dir galp/data/compressed/imagenet512_val_block_major/access \
  --dct-coeffs first:32 \
  --raw-mask-oracle-dir galp/benchmarks/coefficient_mask_evaluator/runs/imagenet_val_k1_64_20260816_h100 \
  --output-dir /tmp/galp-dct-major-contract \
  --dry-run
```

检查 `contract.json` 中：

- pipeline 为 `dct_major_pushdown/dct_major_coefficient_pushdown/rgbnomore/dali/ffcv/pytorch`；
- GALP baseline 为 `all`/K64，GALP-DCT-pushdown 为显式 `first:32`/K32；
- GALP runtime profile 为 `block-major-p4-scheduled-bounded-110-v1`；
- block-major sidecar 文件全部进入 immutable input snapshot；
- sample ordinal、label 和 `galp_image_id` 对齐；
- 没有 planner、workset、cache、kernel 或 bounded-read 的 Python 配置字段。

## Smoke

```bash
PYTHONPATH=.:build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m galp.benchmarks.system_dct_major.run \
  --preset e2e \
  --workload feature-extraction \
  --sample-count 1024 --repeats 1 \
  --pipelines dct_major_pushdown dct_major_coefficient_pushdown rgbnomore dali ffcv pytorch \
  --dct-coeffs first:32 \
  --dct-major-manifest galp/data/compressed/imagenet512_val_block_major/manifest.bin \
  --dct-major-label-map galp/data/compressed/imagenet512_val_compact_v3/labels.json \
  --block-major-access-dir galp/data/compressed/imagenet512_val_block_major/access \
  --output-dir /tmp/galp-dct-major-feature-smoke
```

再以 `--workload evaluation` 和新输出目录执行一次。生产 profile 要求完整 manifest
shard，因此 suite smoke 使用第一个完整 shard（当前 manifest 为 1,024 张）、零 warmup、
1 个 repeat。

## Formal

`first:32` evaluation 需要完整的 raw-mask oracle。设置为已完成评估的目录，例如本地
归并后的运行：

```bash
RAW_MASK_ORACLE_DIR="$PWD/galp/benchmarks/coefficient_mask_evaluator/runs/imagenet_val_k1_64_20260816_h100"
```

```bash
PYTHONPATH=.:build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m galp.benchmarks.system_dct_major.run \
  --preset e2e \
  --workload evaluation \
  --raw-mask-oracle-dir "$RAW_MASK_ORACLE_DIR" \
  --pipelines dct_major_pushdown dct_major_coefficient_pushdown rgbnomore dali ffcv pytorch \
  --dct-coeffs first:32 \
  --dct-major-manifest galp/data/compressed/imagenet512_val_block_major/manifest.bin \
  --dct-major-label-map galp/data/compressed/imagenet512_val_compact_v3/labels.json \
  --block-major-access-dir galp/data/compressed/imagenet512_val_block_major/access \
  --sample-count 50000 \
  --output-dir /tmp/galp-dct-major-evaluation-50k
```

E2E 默认 batch 50、零 warmup、5 个 repeat。`--sample-count` 会推导 measured
batch 数并保留 partial tail。DCT-major 生产 profile 要求 warmup 为 0。

冷启动策略可在两种实验边界间选择：

- `application-overlapped`：允许首批数据 prefetch 与应用启动重叠；
- `controlled-io`：模型 prime 完成后才开始 cold I/O。

`--evict-pipeline-file-cache` 只驱逐当前 pipeline 的不可变输入页。两者是实验
协议，不是 GALP reader 的底层实现开关。

## 完整 suite

```bash
PYTHONPATH=.:build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m galp.benchmarks.system_dct_major.run_suite \
  --raw-mask-oracle-dir "$RAW_MASK_ORACLE_DIR" \
  --dct-major-manifest galp/data/compressed/imagenet512_val_block_major/manifest.bin \
  --dct-major-label-map galp/data/compressed/imagenet512_val_compact_v3/labels.json \
  --block-major-access-dir galp/data/compressed/imagenet512_val_block_major/access \
  --dct-coeffs first:32 \
  --output-dir /tmp/galp-dct-pushdown-k32-suite \
  --dry-run
```

计划固定为：

1. 只生成并冻结 semantic contract，不运行 pipeline；
2. K64 regression 与 K32/K16/list raw-mask semantic gate；
3. feature-extraction smoke；
4. evaluation smoke；
5. formal feature-extraction；
6. formal evaluation；
7. DCT/RGB × feature/evaluation 四个 model-only ceilings。

`--dry-run` 只生成 `suite_plan.json`。实际执行时使用 fresh 目录；中断后以完全
相同的参数和 `--resume` 继续。已完成 phase 按 marker 跳过，存在未完成输出的
phase 不会被覆盖。

## 参数边界

Python 可以设置数据/模型路径、pipeline 集合、workload、feature stage、是否物化
features、batch、样本数、测量长度、repeat、外层 workers、semantic sample 数、
cold protocol、文件页驱逐、通用 prefetch factor、hash 开关、device 和 seed。

Python 不再设置：segment size、full/legacy/image-major 模式、planless/fixed-items、
workset、decode rowgroups、rowgroup prefetch、cache、kernel launch、stream priority、
double buffer、crop execution、read amplification 或 bounded-read 参数。这些由原生
profile 固定。

## 结果验收

检查 `validation.json`、`results.json` 和 `report.md`：

- 同域语义阈值通过；
- 五条 pipeline 的 sample trace 完全一致；
- K64 omitted/all tensors、logits、Top-1/Top-5 完全一致；
- native K32 的 full predictions 与冻结的 raw-mask oracle 一致，且 accuracy delta 不超过 0.05 pp；
- K32 selected payload 和实际 bounded physical bytes 都低于 K64；
- DCT-major physical byte/vector/block/rowgroup/pread 计数完整；
- crop pushdown 的物理读取和 decode 工作量低于 full-input reference 估算；
- runtime profile、binding fingerprint 和 sidecar snapshot 与 contract 一致；
- transient memory 统计覆盖异步 completion 前仍存活的 arena。

历史 segment sweep、legacy/planless ABBA、crop ABBA 和 plan-audit 文档只作为实验
记录，不再是当前运行协议。
