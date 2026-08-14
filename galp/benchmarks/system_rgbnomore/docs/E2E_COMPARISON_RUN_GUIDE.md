# RGB-no-more 端到端对比运行指南

## 当前接口

统一入口是：

```bash
galp/benchmarks/system_rgbnomore/inference/run.py
```

正式 pipeline 只有 `galp`、`rgbnomore`、`dali`、`pytorch`。旧的
`galp_planless`、`galp_fixed_items` 名称及其 A/B 参数不再接受；`galp` 始终
使用 native production profile。

先确认 binding 和帮助信息：

```bash
cmake --build build --target _galp_direct_dct -j2
PYTHONPATH=build/galp/torch \
  /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_rgbnomore/inference/run.py --help
```

## 数据与模型

正式 DCT checkpoint 的输入是预处理后的 `512×512` JPEG。推荐路径：

```text
RGB JPEG:   galp/data/system_rgbnomore/e2e_v3/imagenet_512
Index CSV:  galp/data/system_rgbnomore/e2e_v2/indexbase_val.csv
GALP:       galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32_rgbnomore512/manifest.bin
Labels:     galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32_rgbnomore512/labels.json
RGB model:  galp/data/system_rgbnomore/e2e_v2/checkpoints/imgnetRGBViTTi_ep300_74.1.pth
DCT model:  galp/data/system_rgbnomore/e2e_v2/checkpoints/imgnetDCTViTTi_ep300_75.1.pth
```

默认 `--dct-source-image-size 512` 会扫描 JPEG SOF 并拒绝数据语义不一致。
只有自定义 checkpoint/recipe 才应传 `0`。

首次创建或数据发生受控变化后，可显式生成 payload fingerprint：

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset smoke \
  --output-dir /tmp/galp-fingerprint-preflight \
  --refresh-galp-payload-fingerprints \
  --dry-run
```

普通运行只核对缓存的 fingerprint 和文件快照，不在计时前隐式扫描全部 shard。

## Smoke

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset smoke \
  --pipelines galp rgbnomore dali pytorch \
  --output-dir /tmp/galp-rgbnomore-smoke
```

Smoke 使用 batch 2、1 个 warmup batch、2 个 measured batch、1 个 repeat。

## 正式 E2E

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset e2e \
  --pipelines galp rgbnomore dali pytorch \
  --output-dir /tmp/galp-rgbnomore-e2e \
  --data-root galp/data/system_rgbnomore/e2e_v3/imagenet_512 \
  --index-csv galp/data/system_rgbnomore/e2e_v2/indexbase_val.csv \
  --galp-manifest galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32_rgbnomore512/manifest.bin \
  --galp-label-map-json galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32_rgbnomore512/labels.json \
  --rgb-checkpoint galp/data/system_rgbnomore/e2e_v2/checkpoints/imgnetRGBViTTi_ep300_74.1.pth \
  --dct-checkpoint galp/data/system_rgbnomore/e2e_v2/checkpoints/imgnetDCTViTTi_ep300_75.1.pth \
  --torch-binding-dir build/galp/torch
```

E2E 默认 batch 50、零 warmup、1000 个 measured batch、5 个 repeat。GPU/NUMA
绑定应由调用环境设置，例如 `CUDA_VISIBLE_DEVICES` 和 `numactl`；它们不写成 GALP
实现配置。

## 可覆盖参数的边界

可覆盖的是实验输入和工作负载：pipeline 集合、路径、checkpoint、batch、测量
长度、repeat、外层 workers、semantic sample 数、seed、device、precision 和
PyTorch/DALI prefetch factor。

下列内容不再是 Python 参数：planless/fixed-items、cache、decode batching、
workset、rowgroup prefetch、kernel blocks/CTAs、stream priority、async completion、
double buffer、crop execution、bounded read 和完整 RGB-no-more grid 参数。它们属于
binding 校验过的 `compact-v3-planless-limited-o512-c512-v1` profile。

## 结果检查

每个输出目录至少应包含：

- `contract.json` 与 canonical sample manifest；
- `pipeline_galp.json`、其他 pipeline JSON 和语义 artifact；
- `results.json`、`results.csv`、`report.md`；
- `validation.json`、`commands.json` 和日志。

发布前检查 `validation.json` 通过、样本 trace 一致、DCT/RGB 同域语义 gate
通过，并确认正式结果来自 fresh 输出目录。诊断脚本的单项结果不能替代统一 E2E
contract。
