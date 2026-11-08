# §6.4 — Compressed transfer and direct tensors (Figure 15)

Defaults: MobileNet24, 16,384 images, batch 64, crop 448, three paired trials.
Correctness, warmup, Nsight and memory passes are separate from timing.
`--timing-only` omits diagnostic passes. Run from the repository root.

## Prerequisites

Follow the [paper environment](../../README.md#start-here) and
[native source-DCT preparation](../../REBUILDING_REFERENCES.md#prepare-native-source-dct-inference).
The source must be a native Block-major dataset with `manifest.bin`, its referenced
FLS/metadata files, **`samples.json` as a complete physical-order array**, and
**`profile.json`**. A Published Store alone does not supply these experiment files.
Retain the sample/profile files emitted by the native dataset preparation workflow;
do not substitute the annotation output JSONL for the sample array.

Build the CPU preparation tools (CUDA toolkit required for configuration, no GPU
required for preparation):

```bash
cmake --build build --target fastlens_raw_dct_control fastlens_dct_storage \
  fastlens_block_major_access_tool --parallel
```

Measurements additionally need the Torch extension, model dependencies/checkpoint
and an idle supported GPU; resource diagnostics need Nsight Systems. The preparation
uses hardlinks for compressed FLS files, so source and output must be on the same
filesystem. Do not modify linked source FLS files after preparation.

## Prepare

```bash
python -m fastlens.benchmarks.paper.section_6_4.compressed_transfer_and_tensor.prepare_compression_control \
  --source-root fastlens/data/compressed/imagenet512_val_block_major \
  --output-root fastlens/data/compressed/figure15-controls \
  --image-count 16384 --start-image 0 --workers 4
```

Choose `--workers` according to RAM: each worker materializes a complete partition
of 64 int16 coefficient columns, plus encoder/verification working memory. A
1,024-image partition of 512×512 4:2:0 JPEGs has about 768 MiB of raw coefficients.
The default is one worker; independent partitions can run concurrently.

Selection takes a contiguous range of **whole** source partitions with exactly
16,384 images. With 1,024-image partitions, this is the first sixteen partitions,
not the old four-partition default. `--shard-ids` can select an explicit consecutive
range instead of `--start-image`; its total must still match `--image-count`.
For a functional preparation check, use a smaller count that ends on a partition
boundary. Small inputs are not the paper measurement.

```text
figure15-controls/
  preparation.json
  compressed/
    manifest.bin
    shard_000000.fls
    shard_000000.meta.bin
    samples.json
    profile.json
    geometry/
  raw_int16/
    manifest.bin
    shard_000000.fls
    shard_000000.meta.bin
    samples.json
    profile.json
    geometry/
```

Further partition files and per-tool logs accompany these entries.

## Validate

The raw writer forces uncompressed int16 encoding and verifies every real DCT
coefficient and rowgroup boundary against its compressed input. Metadata files
are byte-identical copies. Both controls use the same selected sample array and
profile, and the same manifest/index builders. Preparation checks sample counts,
order and partition geometry. It does not reuse historical run logs or cached raw
shards. A command failure prevents a usable completed preparation.

The runner validates these paired inputs before GPU selection. All four cases
use the prepared subset: direct, full-grid and compressed-transfer cases read
`compressed/`; the raw-transfer case reads `raw_int16/`. This keeps physical image
IDs, logical sample IDs, order and count aligned across both comparisons.

## Run

```bash
python -m fastlens.benchmarks.paper.section_6_4.compressed_transfer_and_tensor.run --help
python -m fastlens.benchmarks.paper.section_6_4.compressed_transfer_and_tensor.run run \
  --preparation fastlens/data/compressed/figure15-controls/preparation.json --dry-run
python -m fastlens.benchmarks.paper.section_6_4.compressed_transfer_and_tensor.run run \
  --preparation fastlens/data/compressed/figure15-controls/preparation.json --gpus 4090
```

The JSON path is explicit. `--dry-run` only prints the measurement command;
it does not generate datasets or establish numerical correctness. Actual paper
runs require exactly 16,384 prepared images and compare sample IDs, predictions,
checkpoint, precision and batch size across the cases.

## Analyze and outputs

New results go under `results/reproduced/section_6_4/compressed_transfer_and_tensor/`,
with per-GPU `core/` directories, individual `N_16384.json` records,
`timings.csv`, diagnostic summaries and the final case summary. Interpret timings
separately from profiler runs; normalize each metric to its matched control.

```bash
python -m fastlens.benchmarks.paper.section_6_4.compressed_transfer_and_tensor.run plot
```

Plotting uses retained [published results](../../results/README.md), not newly
measured outputs. Historical records refer to `results/paper_rerun_20261007/summary.csv`
and separate diagnostics under `results/paper_rerun_20261006/{4090,H100}/core/`.
The historical raw data directories are not required to regenerate the published
figure. See the [paper guide](../../README.md) for evidence boundaries.
