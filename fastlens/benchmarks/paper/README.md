# FastLens final-paper reproduction

This directory follows **Sections 6.2–6.5 of [fastlens.pdf](../../docs/paper/outputs/fastlens.pdf)** (PDF date
2026-10-18). It is the reviewer entry point. Development checks are in
[`../development/`](../development/README.md); reader/model implementations remain
shared in `../baselines/` and the model/training packages. Section-specific workers
live beside their `run.py`; workers reused across sections live in `shared/`.
Process execution and logging are in `../execution/`.

The final PDF contains six RGB baselines.

## Start here

Follow the root [environment setup](../../../README.md#1-install-dependencies), then
run from the repository root with the `fastlens` environment active.
JPEG-B8 experiments also need the RGB-no-more `dct_manip` extension.
DALI, FFCV, CoorDL and L3 are needed
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

Full-population predictions and execution contracts are generated inputs rather
than plotting tables. Follow [Rebuilding reference inputs](REBUILDING_REFERENCES.md)
to create them from public model code, checkpoints and locally prepared ImageNet
data, and to capture H2D traffic from a fresh run. Pass the generated configuration with `--references results/rebuilt/references.json`.
`prepare_references` builds the full-population reference set before measured
chapter runs. H2D capture accepts `--run-dir results/reproduced` and profiles
newly completed chapter trials.

### Published measurement conventions

- The four source groups contain 522 trial records, including 12 ResNet resident
  controls unused in the final figure. Arithmetic means across the seven
  per-configuration median ratios give training speedups of 3.80x/4.20x and
  inference speedups of 6.17x/7.95x on RTX 4090/H100.
- Across six RGB baselines, FastLens leads training in 9/10 configurations;
  inference reaches 97.7% of the fastest RGB path. DCT resident attainment is 99.1%.
- The training store occupies 107.317 GB, or 2.12 times the JPEG input size.
  The compressed-transfer control reduces H2D traffic by 87.7%.
- Table 2 contains 45 measurements with one trial per configuration. Of the 60
  compact-training points, 53 use one trial and seven use three-trial medians.
- Each convergence curve contains 300 epochs. Final shared/independent Top-1
  values are 78.888/79.052% for Swin and 71.976/71.764% for MobileNet.

nvCOMP compression is an auxiliary comparison with its own dependencies and is
outside the default final-paper experiment matrix.
