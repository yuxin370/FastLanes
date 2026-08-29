# GALP RGB-no-more 端到端 Benchmark

## 正式入口

推理只保留四条有明确语义的管线：

| Pipeline | 输入域 | 实现 |
| --- | --- | --- |
| `galp` | JPEG DCT | GALP 原生 Direct-DCT 生产 profile |
| `rgbnomore` | JPEG DCT | RGB-no-more 参考 reader |
| `dali` | RGB | DALI JPEG decode 与 GPU 预处理 |
| `pytorch` | RGB | PyTorch/torchvision 参考路径 |

`galp_planless`、`galp_fixed_items` 等实现版本名已经删除。GALP 对外只暴露
`galp`，其 planner、缓存、kernel launch、stream 和 I/O 策略由 C++ 生产
profile 固定，不再进入 Python contract 或命令行。

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset e2e \
  --pipelines galp rgbnomore dali pytorch \
  --output-dir /tmp/galp-e2e \
  --data-root galp/data/system_rgbnomore/e2e_v3/imagenet_512 \
  --index-csv galp/data/system_rgbnomore/e2e_v2/indexbase_val.csv \
  --rgb-checkpoint galp/data/system_rgbnomore/e2e_v2/checkpoints/imgnetRGBViTTi_ep300_74.1.pth \
  --dct-checkpoint galp/data/system_rgbnomore/e2e_v2/checkpoints/imgnetDCTViTTi_ep300_75.1.pth \
  --galp-manifest galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32_rgbnomore512/manifest.bin \
  --galp-label-map-json galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32_rgbnomore512/labels.json \
  --torch-binding-dir build/galp/torch
```

发布的 DCT checkpoint 对应预先 resize 并重新编码为 `512×512` JPEG 的
ImageNet。正式 runner 默认检查 JPEG SOF 尺寸；`--dct-source-image-size 0`
只适用于具有不同数据配方的自定义 checkpoint。

## 计时与比较边界

主指标覆盖完整 measured window：数据读取、JPEG/DCT decode、变换、H2D、
模型 forward、指标统计和最终 CUDA 同步。模型创建、checkpoint 加载、contract
生成和 warmup 不计时。

- `galp` 与 `rgbnomore` 使用同一个 DCT checkpoint；
- `dali` 与 `pytorch` 使用同一个 RGB checkpoint；
- DCT 与 RGB 的比值用于系统部署参考，不作逐元素跨域等价声明；
- 每条管线使用相同的 canonical 样本顺序，禁止隐式 shuffle。

Preset：

| Preset | Batch | Warmup | Measured batches | Repeats |
| --- | ---: | ---: | ---: | ---: |
| `smoke` | 2 | 1 | 2 | 1 |
| `e2e` | 50 | 0 | 1000 | 5 |

批量、测量长度和 worker 等工作负载参数仍可覆盖，并写入 contract。GALP
内部运行策略不属于工作负载参数，不能从 Python 覆盖。

## GALP 原生生产 profile

推理使用语义 profile `rgbnomore-validation-v1`，由它定义标准 crop、输出网格和
FP32 RGB-no-more 数值合同。其 native runtime policy
`compact-v3-planless-limited-o512-c512-v1` 固定：

- planless transformed-grid 路径；
- decode rowgroup batch 64，decode workset 512 MiB；
- rowgroup prefetch depth/workers/min-batches 为 `16/8/1`；
- decoded-rowgroup cache 与 plan cache 关闭；
- 单 batch limited-overlap、跨 batch bounded two-slot overlap、低优先级 stream、异步 completion；
- transform launch 上限为 512 blocks / 512 CTAs；
- double buffer 由 runtime policy 自动管理。

FP32 grid 的反量化 `(coefficient + 4) / 1020` 属于语义 profile，不属于运行策略。

Python 只提交 `image_ids` 和高层 transform 描述并读取结果；output-slot admission、
future 生命周期、buffer keepalive 和 reclaim 由 native 层封装。

## 常用命令

四管线 smoke：

```bash
python3 galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset smoke \
  --output-dir /tmp/galp-system-smoke
```

只比较 DCT：

```bash
python3 galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset e2e \
  --pipelines galp rgbnomore \
  --output-dir /tmp/galp-dct-e2e
```

只生成 contract 和命令：

```bash
python3 galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset smoke \
  --output-dir /tmp/galp-system-dry-run \
  --dry-run
```

完整参数以 `run.py --help` 为准。详细执行说明见
[E2E comparison guide](docs/E2E_COMPARISON_RUN_GUIDE.md)。

## 训练入口

训练使用 `training/run.py`，管线仍为 `galp`、`rgbnomore`、`dali`、`pytorch`。
模型固定为 RGB-no-more ViT-Ti，augmentation recipe 固定为 published v1，
precision 固定为 FP32，因此原来的三个单值选项已经删除。

`--workers` 是外层数据 worker/workload 参数，不再转发为 native rowgroup
prefetch worker 数。所有 pipeline 使用已冻结的 production lookahead；训练入口不再
提供 `--prefetch-depth`，GALP 的有界预取完全由 native runtime policy 拥有。
详见 [training guide](docs/TRAINING_BENCHMARK_RUN_GUIDE.md)。

## 输出与诊断边界

正式输出包括 `contract.json`、sample manifest、`pipeline_<name>.json`、语义
artifact、`results.json/csv`、`report.md` 和 `validation.json`。发布结果只从统一
runner 生成。

`diagnostics/` 下工具用于定位语义或性能回归，不构成可配置的生产 API；历史
scheduler matrix、fixed-items/planless A/B 和 crop-I/O A/B runner 已删除。

CPU 审计测试：

```bash
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m unittest -v galp.tests.test_system_benchmark galp.tests.test_training_benchmark
```
