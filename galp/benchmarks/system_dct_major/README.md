# DCT-major locality benchmark

本目录测试 GALP 的物理顺序：

```text
component -> spatial DCT block -> image
```

storage API 名称为 `kSpatialMajorImageMinor`，benchmark 中简称 DCT-major。

## 正式矩阵

| Pipeline | 输入域 | 用途 |
| --- | --- | --- |
| `dct_major_pushdown` | DCT | GALP block-major 原生 crop pushdown |
| `rgbnomore` | DCT | RGB-no-more 严格语义参考 |
| `dali` | RGB | nvJPEG/GPU transform 部署参考 |
| `pytorch` | RGB | PIL/torchvision 部署参考 |

历史 `full`、`legacy_pushdown`、三代 image-major 和 planless/fixed planner A/B
不再属于正式 pipeline，也不能通过 Python contract 恢复。DCT 同域路径共享 DCT
checkpoint；RGB 路径共享 RGB checkpoint。

支持两个 workload：

- `feature-extraction`：输出 `[N,192]` penultimate features；
- `evaluation`：输出 `[N,1000]` logits 并统计 Top-1/Top-5。

## 不变量

- 所有 sampler/reader 都是 `shuffle=false`；
- `drop_last=false`，支持 partial tail；
- sample ordinal 等于物理 `galp_image_id`；
- DCT 请求全部 64 个系数并使用 FP32 模型；
- 生产运行要求完整 manifest shard 和预先物化的 `BLOCK_MAJOR_ACCESS_V1` sidecar；
- transformed DCT 允许最多一个归一化整数级误差（`1/1020`）；
- physical bytes、vectors、source blocks、rowgroups 和 preads 必须写入结果；
- crop-pushdown 只有在 bytes、decoded vectors 和 source blocks 均小于完整输入时才成立。

## 语义 profile 与原生运行策略

`dct_major_pushdown` 使用语义 profile
`rgbnomore-validation-center-crop-512-v1`，由它定义 64×64 block crop reference、
28×28/14×14 输出网格和 FP32 归一化。其原生 runtime policy 固定为
`block-major-p4-scheduled-bounded-110-v1`：

- planless execution，decoded-rowgroup cache/plan cache 为 0；
- decode rowgroup batch 64，workset capacity 512 MiB；
- rowgroup prefetch `depth/workers/min-batches = 16/8/1`；
- limited-overlap、低优先级 stream、异步 completion；
- transform launch 512 blocks / 512 CTAs；
- scheduled bounded io_uring；
- global read amplification 1.10，local amplification 继承，run 上限为 rowgroup；
- double buffer 由 runtime policy 自动选择。

Python contract 只记录 runtime policy 身份，不记录上述 planner、allocator、I/O 或 kernel
细节。

## Quick start

只生成 contract：

```bash
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run.py \
  --preset smoke \
  --workload feature-extraction \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-dry-run \
  --dry-run
```

四管线 feature smoke：

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run.py \
  --preset smoke \
  --workload feature-extraction \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-feature-smoke
```

正式 50K evaluation：

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run.py \
  --preset e2e \
  --workload evaluation \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-eval-50k
```

生产 DCT-major 要求 `warmup-batches=0`，避免 sidecar/cold-I/O contract 在运行后
发生变化。详细说明见 [run guide](docs/RUN_GUIDE.md)。

## 完整 suite

`run_suite.py` 固定执行：feature smoke、evaluation smoke、两次四管线 formal 和
四个 DCT/RGB model-only ceiling。它不再做 segment sweep、自动选优、legacy
ABBA、crop A/B 或 plan-audit compare。

```bash
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run_suite.py \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-complete \
  --dry-run
```

移除 `--dry-run` 后执行；中断后可使用相同参数加 `--resume`。runner 不覆盖不完整
phase。

## 测试

```bash
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m unittest discover -s galp/benchmarks/system_dct_major/tests -v
```

`diagnostics/` 用于分析，不定义生产 API。历史 crop ABBA runner 已删除。
