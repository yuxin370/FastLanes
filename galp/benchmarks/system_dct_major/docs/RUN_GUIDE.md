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
PYTHONPATH=build/galp/torch \
  /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run.py --help
```

## Contract preflight

```bash
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run.py \
  --preset smoke \
  --workload feature-extraction \
  --dct-major-manifest galp/data/imagedataset_dct/ImageNet-val/manifest.bin \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-contract \
  --dry-run
```

检查 `contract.json` 中：

- pipeline 为 `dct_major_pushdown/rgbnomore/dali/pytorch`；
- GALP runtime profile 为 `block-major-p4-scheduled-bounded-110-v1`；
- block-major sidecar 文件全部进入 immutable input snapshot；
- sample ordinal、label 和 `galp_image_id` 对齐；
- 没有 planner、workset、cache、kernel 或 bounded-read 的 Python 配置字段。

## Smoke

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run.py \
  --preset smoke \
  --workload feature-extraction \
  --pipelines dct_major_pushdown rgbnomore dali pytorch \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-feature-smoke
```

再以 `--workload evaluation` 和新输出目录执行一次。Smoke 默认 batch 2、零
warmup、2 个 measured batch、1 个 repeat。

## Formal

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run.py \
  --preset e2e \
  --workload evaluation \
  --pipelines dct_major_pushdown rgbnomore dali pytorch \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
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
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run_suite.py \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-suite \
  --dry-run
```

计划固定为：

1. feature-extraction smoke；
2. evaluation smoke；
3. formal feature-extraction；
4. formal evaluation；
5. DCT/RGB × feature/evaluation 四个 model-only ceilings。

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
- 四条 pipeline 的 sample trace 完全一致；
- DCT-major physical byte/vector/block/rowgroup/pread 计数完整；
- crop pushdown 的物理读取和 decode 工作量低于 full-input reference 估算；
- runtime profile、binding fingerprint 和 sidecar snapshot 与 contract 一致；
- transient memory 统计覆盖异步 completion 前仍存活的 arena。

历史 segment sweep、legacy/planless ABBA、crop ABBA 和 plan-audit 文档只作为实验
记录，不再是当前运行协议。
