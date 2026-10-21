# FastLens final-paper reproduction

This directory follows **Sections 6.2–6.5 of [fastlens-final.pdf](../../docs/paper/outputs/fastlens-final.pdf)** (PDF date
2026-10-18). It is the reviewer entry point. Development checks are in
[`../development/`](../development/README.md); reader/model implementations remain
shared in `../baselines/` and the model/training packages. Section-specific workers
live beside their `run.py`; workers reused across sections live in `shared/`.
Process execution and logging are in `../execution/`. Historical queue, retry and
continuation scripts are in `../development/historical/`.

The final PDF contains six RGB baselines. The current paper LaTeX
has since changed and does not compile to this exact final PDF.

## Start here

Run from the repository root with a Python environment containing PyTorch,
torchvision, NumPy, Pillow, einops, psutil and matplotlib. JPEG-B8 experiments also
need the RGB-no-more `dct_manip` extension. DALI, FFCV, CoorDL and L3 are needed
only for their respective baseline measurements; Nsight Systems is needed for
CUPTI diagnostics. ImageNet and pretrained checkpoints are external inputs.

```bash
export CC=clang CXX=clang++
export RGBNOMORE_ROOT="$HOME/RGB-no-more"
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"
python -m fastlens.benchmarks.paper.reproduce build
python -m fastlens.benchmarks.paper.reproduce figures
```

Measurements use a dedicated GPU without MPS sharing. If another MPS daemon is
running, select a separate pipe directory before launching GPU experiments:
`export CUDA_MPS_PIPE_DIRECTORY="$TMPDIR/fastlens-paper-no-mps"` and
`mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"`. This leaves the existing daemon and jobs alone.

`build` configures Release mode and builds the storage bridge, Torch extension
and input-boundary control library. Use `--build-dir` for another build directory;
pass that directory's Torch module/control library to the corresponding experiment.
`figures` redraws the ten published plots from the [published result directory](results/README.md);
it needs no GPU, dataset, native build or checkpoint. It writes PDF/PNG under
`results/reproduced/paper/figures/` and does not alter measured records.
`--output DIR` changes the destination; `--artifact-root DIR` selects another
repository-shaped published artifact tree containing `fastlens/benchmarks/paper/results/`.
Replotting does not run measurements.

## Experiments, organized by final-PDF section

Every experiment directory has `run.py`. Run it as a module from the repository
root, replacing `/` by `.` and appending `.run`, as in the examples below.
`run` is the default action. `--help` shows options; `run --dry-run` prints the
commands without accessing GPUs or launching measurements. Timed jobs execute
serially; independent source preparation runs in parallel. New measurements go
to `results/reproduced/<section>/<experiment>/`; use a fresh `--output` when rerunning.

| Directory | Final-PDF result | Default measured protocol |
| --- | --- | --- |
| [section_6_2_1/matched_dct](section_6_2_1/matched_dct/README.md) | Fig. 9 | Seven DCT models; 4090 + H100; 3 trials; training and inference |
| [section_6_2_2/shape_preserving_selection](section_6_2_2/shape_preserving_selection/README.md) | Table 2 | H100; MobileNet24, ResNet24, ViT-Ti; D4/D6/D8; 5 paths; 1 trial |
| [section_6_2_2/compact_input](section_6_2_2/compact_input/README.md) | Fig. 10 | ResNet-50, MobileNetV2, ViT-Ti; D3/D15; both GPUs; 5 paths |
| [section_6_2_3/deployment](section_6_2_3/deployment/README.md) | Fig. 11a | Five DCT profiles vs six RGB baselines; both GPUs; 3 trials |
| [section_6_2_4/resident_training](section_6_2_4/resident_training/README.md) | Fig. 11b | Same training protocol; resident reference cycles 8 real batches/512 images |
| [section_6_2_5/training_convergence](section_6_2_5/training_convergence/README.md) | Fig. 12 | SwinV2/MobileNet24; shared and independent crops; 300 epochs |
| [section_6_3/storage](section_6_3/storage/README.md) | Table 3 storage evidence | Actual file sizes, including referenced metadata/indexes/padding |
| [section_6_3/h2d_traffic](section_6_3/h2d_traffic/README.md) | Fig. 14 | 4090; ViT-Ti and MobileNet24; separate training/inference CUPTI captures |
| [section_6_4/projection_and_staging](section_6_4/projection_and_staging/README.md) | Fig. 13 + projection controls | Crops 448/160; 3 staging trials; input-only projection is separate |
| [section_6_4/compressed_transfer_and_tensor](section_6_4/compressed_transfer_and_tensor/README.md) | Fig. 15 | MobileNet24; 16,384 images; raw/compressed and grid/direct controls |
| [section_6_5_1/block_major_locality](section_6_5_1/block_major_locality/README.md) | Fig. 16 | 4,096 images; shared/correlated/independent crops; batches 64/1,024 |
| [section_6_5_2/read_bandwidth](section_6_5_2/read_bandwidth/README.md) | §6.5.2 | 125/500/2,000 MB/s; crops 448/160; 3 trials on each GPU |

Examples (no benchmark parameters need to be copied from the manuscript):

```bash
python -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run plot
python -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_2_2.compact_input.run run --gpus H100
python -m fastlens.benchmarks.paper.section_6_5_2.read_bandwidth.run run --gpus 4090
```

The three end-to-end sections reuse the same timing implementations and published
records. Their training defaults are 65,536 distinct images, batch 64,
accumulation 16, 32 warmup updates, restored model/optimizer/scheduler/RNG state,
basic crop/flip/normalization, no RandAugment or Mixup, seed 11997733, 8 host threads.
ViT-Ti uses FP32; other models use BF16 autocast. Inference covers all 50,000 images,
uses FP32 and batch 64 (Swin: 50), excludes 20 model warmup batches, and includes
reader startup and metrics in the timed interval. TF32 and CUDA graphs are off.
The default NUMA node is 0, except the ResNet/4090 extension uses node 1.

## Prepare compact and selected-frequency inputs

The default data tree is `fastlens/data/`. First prepare ImageNet-512 and the native
FLS layouts using the existing [dataset instructions](../baselines/system_rgbnomore/README.md)
and [DCT-major instructions](../system_dct_major/README.md). JPEG-B8 measurements
use these existing 512x512 JPEGs (bilinear resize, JPEG quality 75, 4:2:0), not a
second JPEG encoding. Then:

```bash
python -m fastlens.benchmarks.paper.prepare_sources
python -m fastlens.benchmarks.paper.section_6_2_2.compact_input.run prepare
python -m fastlens.benchmarks.paper.section_6_2_2.shape_preserving_selection.run prepare
```

`prepare_sources` freezes the first 8,194 physical training entries from the
premixed `ordered_mapping.csv` and the official 50,000-image validation selection,
then builds full-frequency FS stores. Outputs default to `fastlens/data/paper/train8194`
and `fastlens/data/paper/val50000`. Override `--jpeg-root`, `--mapping`, `--val-selection`
and `--output` for another installation. The experiment entry points expose matching
`--source-manifest`, `--fs-store`, `--prepared` and `--native-manifest` options.
Changing population counts is a development smoke test, not paper reproduction.

Compact preparation fits D3 and D15 normalization and builds both prepared IC
layouts. Its timed defaults are batch 64, pool 256, 8 workers, 2 model CPU threads,
FP32, SGD (lr .001, momentum .9, weight decay 4e-5), seed 20261014, no augmentation,
60 s warmup and at least 180 s per repeat. The default repeats match each published
point: all five ResNet-50/D3/4090 points and the two ViT-Ti/D3/4090 IC points use
three repeats; the other 53 use one. `--repeats` explicitly overrides this.
Both frequency experiments bind timed readers/models to eight CPU cores
(`--cpus 0-7`); change CPU IDs to match the host while retaining the eight-core budget.

Shape-preserving preparation creates D4/D6/D8 contracts and IC fixed-frequency
stores, shared/group-size-256. FS/FastLens retain full-frequency stores. Defaults:
batch 64, pool 256, 8 DataLoader workers, prefetch 2, 2 model CPU threads,
source order, seed 11997733, atol/rtol 1e-5, decode budget 1 GiB, reader budget
2 GiB, host-input budget 32 GiB, decoded caches CPU/FastLens 0, IC 1 GiB, FS 512 MiB.
Each path runs a fresh input/logit verification before 60 s warmup and at least
180 s measurement. Both suites keep the tail and preserve the measured population.

## Published evidence versus a new measurement

`plot`/`table` reads the retained published CSV/JSON inputs in `paper/results/`. `run` writes fresh
raw measurements; it does not substitute new numbers into the paper or overwrite
historical evidence. Plotting now reads the consolidated result directory,
including the full tables and 522 end-to-end trial records. The per-experiment READMEs identify the source records
and the external inputs needed by each workload.

The end-to-end inference and H2D replay workloads still require the original
full-population prediction/contract artifacts under `results/` in addition to
ImageNet/checkpoints. Those large artifacts are **not included** with the small
plotting CSVs. H2D replay also reads `results/paper/selected_max_20261008/` and its
referenced command records. Without that artifact bundle, these entries cannot
perform a fresh full experiment; `--help`, `--dry-run` and all published plots
remain usable. The placeholders printed for accepted inference references in
`--dry-run` are resolved from those artifacts when executing. Do not interpret
successful command generation as a completed performance run.

On the current machine, preparing the historical H2D inference replay also
detects changed file identity for
`fastlens/data/compressed/imagenet512_val_block_major/access/shard_000000.active_output_schedule.bin`.
The historical contract's size/mtime check fails before measurement. Source-code
paths in newly generated replay contracts now follow the `baselines/` layout,
but the dataset check remains enforced. Use the matching historical dataset
bundle, or perform a fresh input validation before collecting a new H2D run;
this replay is not currently verified end to end.

### 结果口径与已知限制

- 四组源数据共522条逐次记录（含12条未用于最终图的 ResNet resident 对照），原始大型运行树不包含在仓库中。
  按逐轮比值中位数再跨七配置取算术平均，RTX4090/H100 训练为3.80×/4.20×，
  推理为6.17×/7.95×，与最终 PDF 一致。
- 六条 RGB baseline 下，训练领先9/10，推理为最快 RGB 的97.7%，
  DCT resident 达成率99.1%。
- FastLens 训练存储107.317 GB、JPEG 的2.12倍；compressed-transfer 对照的
  H2D 减少87.7%，均与原始计数一致。
- Table 2 的45条导出记录与最终 PDF 逐项一致，每配置一次正式测量。Compact 的60个点中53个单次、7个三次中位数，
  不能描述成统一三次重复统计。
- 四条收敛曲线各300个 epoch。Swin shared/independent 最终 Top-1 为
  78.888/79.052%，MobileNet 为71.976/71.764%。
- 完整50K推理、65,536-image窗口和300-epoch训练需要准备对应外部输入。
  FS 预取路径曾触发输入一致性门禁，差异尚未稳定定位；验证阈值保持1e-5，
  不能将该路径标为完整验证通过。
- 仍未通过的既有检查：`FastLensDirectDctTorchRuntimeSmoke` 在 RTX4090、
  `imagenet512_val_compact_v3/manifest.bin` 上报
  `consumer-stream delay completed before lifetime verification`。
  未设置 `FASTLENS_DIRECT_DCT_TEST_MANIFEST` 时该检查会跳过，不能把跳过当通过；
  换用 block-major 验证 manifest 则在异步 planless 输出契约处被拒绝。
  PLS GPU smoke 可独立验证；它不能替代该生命周期检查。复查命令如下，GPU 序号按本机配置选择：

```bash
CUDA_VISIBLE_DEVICES=2 \
FASTLENS_DIRECT_DCT_TEST_MANIFEST="$PWD/fastlens/data/compressed/imagenet512_val_compact_v3/manifest.bin" \
FASTLENS_REQUIRE_CUDA_SMOKE=1 ctest --test-dir build \
  -R '^FastLensDirectDctTorchRuntimeSmoke$' --output-on-failure
```

nvCOMP compression 属于辅助源码，需要相应的外部依赖，
不属于最终 PDF 的默认实验配置。CPU 测试或 CLI 加载检查不能替代完整 GPU 测量。
