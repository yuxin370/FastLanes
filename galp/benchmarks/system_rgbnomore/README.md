# GALP 统一端到端 Benchmark

## 1. 唯一正式入口

对外性能数据只由以下入口产生：

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset e2e \
  --pipelines galp galp_legacy rgbnomore dali \
  --output-dir /tmp/galp-e2e \
  --data-root galp/data/system_rgbnomore/e2e_v2/imagenet \
  --index-csv galp/data/system_rgbnomore/e2e_v2/indexbase_val.csv \
  --rgb-checkpoint galp/data/system_rgbnomore/e2e_v2/checkpoints/imgnetRGBViTTi_ep300_74.1.pth \
  --dct-checkpoint galp/data/system_rgbnomore/e2e_v2/checkpoints/imgnetDCTViTTi_ep300_75.1.pth \
  --galp-manifest galp/data/system_rgbnomore/e2e_v2/dct/manifest.bin \
  --galp-label-map-json galp/data/system_rgbnomore/e2e_v2/dct/labels.json \
  --torch-binding-dir build/galp/torch \
  --refresh-galp-payload-fingerprints
```

该命令固定使用 `galp/data/system_rgbnomore/e2e_v2` 中配套的 JPEG、manifest v2、
label sidecar、index 和 checkpoints，避免混用不同数据生成批次。benchmark 实现、
contract、pipeline 和校验仍全部位于本 benchmark 目录中。

这套 benchmark 只测完整推理路径，不再把 loader、forward-only、kernel
或训练 step 和端到端数据放在同一张对比表中。旧的独立 RGB/DCT/DALI
baseline、comparison runner、summarizer 和 validator 已移除，其能力全部由
`system_rgbnomore` 的 contract、pipeline 和 validate 三层覆盖。

JPEG-DCT crop 报告严格区分 requested blocks、selected/decoded vectors 与
physical compressed bytes。生产稀疏范围读、manifest-v3 上界原型、三路强制
A/B 模式和验收命令见
[JPEG-DCT crop I/O architecture](../../docs/architecture/jpeg_dct_crop_io.md)。

## 2. 目录布局

| 目录 | 职责 |
| --- | --- |
| `inference/` | 正式端到端推理入口、pipeline、模型构建和结果校验 |
| `training/` | 训练 runner、数据 adapter、优化器、指标、schema 与产物校验 |
| `dataset/` | canonical sample manifest 与 GALP 数据集准备工具 |
| `diagnostics/` | Direct-DCT、pushdown、manifest 和调度策略诊断 |
| `shared/` | 推理、训练和数据准备共同使用的基础工具 |
| `docs/` | 运行指南、设计 RFC、里程碑和实验记录 |

根目录仅保留本总览；各目录中的 `run.py`、`validate.py` 等名称由所在功能目录限定。

## 3. 覆盖矩阵

| Pipeline | 输入域 | 数据路径 | 模型 |
| --- | --- | --- | --- |
| `galp` | JPEG DCT | GALP sharded Direct-DCT、融合 transformed-grid pushdown、有界有序 batch prefetch | RGB-no-more JPEG-Ti |
| `rgbnomore` | JPEG DCT | RGB-no-more 原生 DCT reader 与验证变换 | 同一 DCT checkpoint |
| `dali` | RGB | DALI file reader、mixed JPEG decode、GPU resize/crop/normalize | RGB-no-more RGB ViT-Ti |
| `pytorch` | RGB | canonical manifest、PyTorch DataLoader、RGB-no-more 验证变换 | 同一 RGB checkpoint |

这四条路径覆盖两组有意义的对比：

- `galp` 与 `rgbnomore`：相同 DCT 模型、checkpoint、样本和语义；
- `dali` 与 `pytorch`：相同 RGB 模型、checkpoint、样本和高层预处理契约。

DCT 与 RGB 使用输入域专用 checkpoint，只做系统级参考，不做逐元素跨域比较。

## 4. 端到端计时边界

主指标为完整 measured window 的 images/s 和逐 batch 延迟分布。单 batch 从请求
数据开始，到以下工作全部完成为止：

1. 文件读取；
2. JPEG decode 或 DCT 解压；
3. resize/crop/normalize 或 DCT transform；
4. host-to-device；
5. 模型 forward；
6. top-1/top-5 统计；
7. CUDA 同步。

模型创建、checkpoint 加载、pipeline build、manifest 生成和 warmup 不计时。
各 pipeline 使用其生产级预取机制；GALP 默认维护深度为 2 的有序 Direct-DCT prefetch 队列，使后续两个 batch 的 CPU 读取/规划可以与当前 batch forward 重叠；
单个 image-major workset 内也启用并行 rowgroup 读取。
每 batch 的同步边界保持一致，因此吞吐和延迟可复核。

## 5. Preset

| Preset | Batch | Warmup batches | Measured batches | Repeats | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| `smoke` | 2 | 1 | 2 | 1 | 接口、依赖、语义和输出契约检查 |
| `e2e` | 64 | 5 | 20 | 5 | 正式端到端性能结果 |

`e2e` 汇总时保留全部原始 repeat，但排除 repeat 0 后计算正式分布，降低首次 page-cache
状态影响。所有维度仍可用命令行显式覆盖；覆盖后的值会写入不可变 contract。

## 6. 公平性和完整性

运行前生成 `contract.json` 和 canonical sample manifest，固定：

- 样本 ID/path、顺序、内容 SHA-256 和 RGB-no-more label；
- batch、workers、warmup、measurement、repeats、precision 和 device；
- RGB/DCT checkpoint 及 SHA-256；
- 两个输入域的预处理契约；
- GALP cache、decode batching 和 prefetch 参数；
- `manifest.bin` 引用的每个 FLS/metadata payload 摘要；
- 语义阈值和计时边界。

GALP 正式路径默认使用 `rgbnomore-val-pushdown`。运行时强制检查融合
transformed-grid 路径被使用，并拒绝 generic projection fallback。

大文件摘要在数据准备或显式 `--refresh-galp-payload-fingerprints` 时只生成一次，
并缓存于 `manifest.bin.payload_fingerprints.json`。正式 benchmark 默认只做
device/inode/size/mtime/ctime 快照检查；快照变化才复算对应文件，且所有检查均在
计时区外。缺少 cache 时正式 runner 会直接报错，不会隐式读取整个 shard 数据集。

每次运行同时输出：

- throughput repeat 分布；
- batch latency mean/p50/p95/p99；
- top-1/top-5；
- CPU process time；
- Torch 可见的 peak allocated/reserved GPU memory；
- loader/H2D/forward 与 GALP native stage 诊断；
- canonical measured sample trace；
- 语义 tensor/logit artifact。

## 7. 常用命令

四管线 smoke：

```bash
python3 galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset smoke \
  --output-dir /tmp/galp-system-smoke
```

正式端到端：

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset e2e \
  --pipelines galp galp_legacy rgbnomore dali \
  --output-dir /tmp/galp-system-e2e \
  --data-root galp/data/system_rgbnomore/e2e_v2/imagenet \
  --index-csv galp/data/system_rgbnomore/e2e_v2/indexbase_val.csv \
  --rgb-checkpoint galp/data/system_rgbnomore/e2e_v2/checkpoints/imgnetRGBViTTi_ep300_74.1.pth \
  --dct-checkpoint galp/data/system_rgbnomore/e2e_v2/checkpoints/imgnetDCTViTTi_ep300_75.1.pth \
  --galp-manifest galp/data/system_rgbnomore/e2e_v2/dct/manifest.bin \
  --galp-label-map-json galp/data/system_rgbnomore/e2e_v2/dct/labels.json \
  --torch-binding-dir build/galp/torch \
  --refresh-galp-payload-fingerprints
```

只比较 DCT 两条路径：

```bash
python3 galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset e2e \
  --pipelines galp rgbnomore \
  --output-dir /tmp/galp-dct-e2e
```

生成 contract 和命令但不启动 GPU：

```bash
python3 galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset smoke \
  --output-dir /tmp/galp-system-dry-run \
  --dry-run
```

## 8. 输出

输出目录包含：

- `contract.json`、`sample_manifest.json`、`canonical_rgbnomore_index.csv`；
- `pipeline_<name>.json` 和 `semantic_<name>.npz`；
- `results.json`：完整 contract、raw repeats、aggregate 和语义结果；
- `results.csv`：每 pipeline/repeat 一行；
- `report.md`：端到端主表、可比性和 caveat；
- `validation.json`、`commands.json`、日志与运行元数据。

正式发布以 `report.md` 和 `results.json` 为准，禁止从诊断脚本拼接另一套结果。

## 9. 非正式性能工具

以下工具保留，但不属于正式对比矩阵：

- `diagnostics/direct_dct.py`：GALP loader/forward/train 与内部阶段诊断；
- `diagnostics/validate_pushdown.py`：pushdown 逐元素语义验证；
- `diagnostics/scan_manifests.py`：数据集 manifest 与 label-map 元数据检查；
- `diagnostics/scheduler_matrix.py`：Direct-DCT/model CUDA 调度策略矩阵；
- `dataset/prepare_dataset.py`：生成并验证正式 benchmark 使用的 GALP 数据集；
- `galp/benchmarks/micro_bench.cu`：算子 microbenchmark；
- `galp/benchmarks/compressor_bench.cu`：codec/compressor benchmark。

它们用于定位瓶颈和回归，不能替代统一端到端 benchmark。

## 10. 训练 Benchmark

训练使用独立入口 `training/run.py`，不会改变上面的正式推理入口或推理 schema。训练矩阵固定为两组同域比较：

| 比较组 | Pipeline | 输入 | 正式模型 |
| --- | --- | --- | --- |
| DCT | `galp` / `rgbnomore` | Y `[B,1,28,28,8,8]`、CbCr `[B,2,14,14,8,8]` | RGB-no-more ViT-Ti DCT，5,642,728 个可训练参数 |
| RGB | `dali` / `pytorch` | RGB `[B,3,224,224]` | RGB-no-more ViT-Ti RGB，5,716,456 个可训练参数 |

完整运行方法、manifest 格式、checkpoint 严格字段、状态解释和产物列表见
[`docs/TRAINING_BENCHMARK_RUN_GUIDE.md`](docs/TRAINING_BENCHMARK_RUN_GUIDE.md)。CPU 可执行的审计测试为：

训练 runner 默认使用 `--execution-mode audit`；生产吞吐必须显式使用
`--execution-mode runtime`。mode 会写入 contract、commands、pipeline 和 results，resume
不能跨 mode。runtime loader 诊断和当前环境限制见
[`docs/TRAINING_RUNTIME_BOTTLENECK_REPORT_2026-07-21.md`](docs/TRAINING_RUNTIME_BOTTLENECK_REPORT_2026-07-21.md)。

```bash
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m unittest -v galp.tests.test_training_benchmark
```

没有 CUDA 时，`galp`、`dali` 和真实 GPU train smoke 明确记为环境 skip；不会伪造通过结果。
