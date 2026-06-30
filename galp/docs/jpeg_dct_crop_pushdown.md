# JPEG DCT Vector-Level Crop Pushdown

This note summarizes the current JPEG DCT / FastLanes vector-level crop
pushdown implementation. It is intended as the implementation/status reference
for benchmark reports and follow-up optimization work.

## Goal

The pushdown path should decode only the FastLanes vectors needed by a pixel
crop when that saves more decode work than the fixed overhead it introduces.
The runtime must stay general: decisions are based on crop/block/vector/rowgroup
counts, estimated worksets, and gather items, not dataset names.

## Data Flow

```text
pixel crop
  -> component-space DCT block range
  -> block row references: shard, rowgroup, row_in_rowgroup
  -> selected FastLanes vector chunks
  -> per-rowgroup runtime policy: selected vectors or full rowgroup
  -> batch-level FastLanes workset
  -> selected-vector decode kernel(s)
  -> batch gather to block-major 64-coefficient DCT blocks
```

The CPU planner lives in `JpegDctShardDatasetReader::PlanDeviceDctBatch()` and
`PrepareDeviceDctBatch()`. It produces image layouts, output block metadata,
touched rowgroups, selected-vector chunks, selected/full vector counts, and the
runtime-policy decision for each rowgroup. `ReadPreparedDeviceDctBatch()` then
executes the prepared plan without repeating selected-vector sort/unique or
gather remapping.

The selected-vector decision is not only bookkeeping. During execution,
`prepare_decoded_rowgroup_work()` attaches the rowgroup's compact
`selected_vectors` to `DecodedRowgroupWork` whenever the runtime policy chooses
selected-vector decode. `execute_decoded_rowgroup_batch()` then calls
`append_jpeg_rowgroup_columns()`, which passes those selected vectors into
`runtime::append_column_to_workset()`. At the FastLanes workset layer this sets
`Batch<T>::work_items_explicit`, emits one `WorkItemAny` per selected vector
chunk, and shrinks `DeviceExpression<T>::output_n_values` to the selected chunk
width. The mixed/typed decode launch consumes those explicit work items, so the
decompress kernels are scheduled for selected chunks instead of the touched
rowgroup's full vector range.

## Runtime Policy

The rowgroup-level policy chooses selected-vector decode only when:

- selected chunks fit the configured multi-vector unpack width;
- selected/full vector ratio is below the configured threshold;
- the crop saves at least the configured minimum vector count.

Otherwise the rowgroup falls back to full-rowgroup decode. This protects tail
chunks and avoids paying selected-vector overhead for small savings.

`pipeline_benchmark --mode auto` uses prepared crop/full plans to choose
pushdown or full-then-crop per window. The decision uses selected/full block
ratio, selected/full vector ratio, touched/full rowgroup ratio, average full
blocks per rowgroup, estimated worksets, and estimated gather items. Workset
estimates are grouped by shard and respect `--decode-batch-rowgroups`, matching
the execution path.

## Launch and Sync Shape

Newly decoded rowgroups are accumulated and submitted in decode batches rather
than one workset per rowgroup. The batch limit is controlled by
`--decode-batch-rowgroups` and defaults to 64 rowgroups.

For each decoded batch:

- one FastLanes workset is built and uploaded;
- selected-vector or full-rowgroup decode kernels are launched by the workset;
- one decoded gather kernel handles all gather items in the batch;
- one dense-cache materialization kernel is launched only when dense cache
  insertion is needed.

Dense-cache hits are gathered through a cache-hit stream and can hand off to the
next decoded batch via a CUDA event. The JPEG DCT hot path does not use
`cudaDeviceSynchronize()`. Remaining internal waits are event-level waits used
for resource reuse:

- cached-gather drain when its scratch buffer must grow or legacy default-stream
  ordering requires host ordering;
- decoded-batch completion before reusing rowgroup metadata, workset arenas, and
  scratch buffers.

Both are counted through `internal_sync_count` and split into
`cached_gather_sync_count` and `decoded_batch_sync_count`.

## Scratch Reuse

`JpegDctDeviceScratch` owns reusable host staging vectors, device scratch
buffers, CUDA events/streams, and the decode workset. The main reusable device
scratch buffers are:

- coefficient pointer/source bindings for decoded gather;
- decoded gather batch items;
- dense materialization batch items;
- cached-gather batch items.

Device scratch capacity grows geometrically from a small capacity floor.
`scratch_allocation_count` counts capacity growth events, not every upload.
`scratch_upload_count` counts metadata uploads to those reusable buffers.

## Instrumentation

The pipeline benchmark prints the counters needed to explain pushdown overhead:

- selected/full vector counts and ratios;
- planned vs actual saved vectors;
- touched rowgroups and workset count;
- decode/gather/cache-materialization launch counts;
- workset uploads and scratch uploads/allocations;
- internal sync counts and cached-gather event handoffs;
- dense cache hit/miss and sparse-vector-cache placeholder counters;
- planning, workset build/upload, decode, gather, read/decode, transform, and
  `total_ms` timings;
- compare-mode speedup and saved-time fields derived from stage `total_ms`;
- runtime-policy decision and reason.

## Correctness Coverage

CPU-side planning tests cover:

- aligned crops;
- unaligned crops that expand to multiple DCT blocks;
- clipped boundary crops;
- full-image sentinel crops (`width == 0 || height == 0`);
- out-of-bounds crop starts;
- request-major output layout;
- cross-component planning;
- cross-rowgroup planning;
- selected-vector chunk compaction and remapping;
- multi-vector tail overrun fallback.
- FastLanes workset construction for selected vectors: explicit compact
  `WorkItemAny` chunks, reduced `output_n_values`, and workset-level tail
  overrun rejection.

GPU correctness still requires running `pipeline_benchmark --mode compare` on a
CUDA-capable machine. That command verifies that crop pushdown output matches
full-decode-then-crop output.

Useful no-GPU validation while developing the runtime:

```bash
ctest --test-dir build -R GalpJpegDctNoDeviceSynchronize --output-on-failure
ctest --test-dir build -R GalpPipelineBenchmarkSummaryScript --output-on-failure
ctest --test-dir build -L galp-workset-cpu --output-on-failure
./build/galp/tests/galp_tests --gtest_filter='WorksetSelectedVectors.*'
./build/galp/tests/galp_tests --gtest_filter='JpegDct.DeviceBatchPlanPreview*:JpegDct.SelectedVectorPlanning*:JpegDct.RuntimePolicy*:JpegDct.AutoPipeline*'
```

`GalpJpegDctNoDeviceSynchronize` is a static guard for the JPEG DCT hot path.
It rejects both `cudaDeviceSynchronize()` and `cudaStreamSynchronize()` in the
planner/executor/benchmark-support files; the remaining allowed waits are the
explicit CUDA event waits counted by `internal_sync_count`.
`GalpPipelineBenchmarkSummaryScript` checks that saved benchmark output can be
converted into the reporting table, including `image_size`, derived
`crop_size`, and speedup/saved-time fields.

## Benchmark Protocol

Use both small-image and medium/large-image scenarios:

```bash
# Correctness and benefit check.
./build/galp/tools/galp_cli pipeline_benchmark <large-or-medium-manifest.bin> \
  --crop 30 40 224 224 \
  --window-images 128 \
  --cache-capacity-mib 1024 \
  --mode compare

# Fixed-overhead pressure check.
./build/galp/tools/galp_cli pipeline_benchmark <small-image-manifest.bin> \
  --crop 0 0 16 16 \
  --window-images 128 \
  --cache-capacity-mib 1024 \
  --mode compare

# Runtime policy check.
./build/galp/tools/galp_cli pipeline_benchmark <manifest.bin> \
  --crop 0 0 16 16 \
  --window-images 128 \
  --cache-capacity-mib 1024 \
  --mode auto
```

For timing-only sweeps, use `--no-verify` only after the same dataset/crop/window
configuration has passed `compare` with verification enabled.

To preserve raw evidence and generate a table in one pass:

```bash
./build/galp/tools/galp_cli pipeline_benchmark <manifest.bin> \
  --crop 30 40 224 224 \
  --window-images 128 \
  --cache-capacity-mib 1024 \
  --mode compare | tee pipeline_benchmark.log

python3 scripts/my_tool/summarize_pipeline_benchmark.py \
  --dataset <dataset> --image-size <width>x<height-or-varies> \
  --require-match --require-default-fields \
  pipeline_benchmark.log
```

## Reporting Template

Report at least these fields for each dataset/crop pair:

| dataset | image size | crop size | mode | outputs_match | pushdown_selected_vector_ratio | full_then_crop_selected_vector_ratio | pushdown_total_ms | full_then_crop_total_ms | pushdown_speedup_vs_full_then_crop | pushdown_saved_ms_vs_full_then_crop | pushdown_plan_ms | pushdown_read_decode_ms | pushdown_decode_ms | pushdown_gather_ms | pushdown_workset_count | pushdown_decode_kernel_launch_count | pushdown_gather_kernel_launch_count | pushdown_scratch_allocation_count | pushdown_internal_sync_count | pushdown_runtime_policy_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

For `auto` runs, also report:

- `auto_pushdown_windows`;
- `auto_full_then_crop_windows`;
- `auto_policy_selected_vector_ratio`;
- `auto_policy_estimated_pushdown_worksets`;
- `auto_policy_estimated_full_worksets`;
- `auto_policy_reason`.

The main comparison is `pushdown_speedup_vs_full_then_crop` for `compare`
mode. This is `full_then_crop_total_ms / pushdown_total_ms`; the companion
`pushdown_saved_ms_vs_full_then_crop` reports the absolute time saved. For
`auto` mode, use the selected-path `auto_total_ms` plus policy-window counts.

Saved benchmark logs can be converted to a report table with:

```bash
python3 scripts/my_tool/summarize_pipeline_benchmark.py --format markdown \
  --dataset ImageNet --image-size varies imagenet_pipeline_benchmark.log

python3 scripts/my_tool/summarize_pipeline_benchmark.py --format csv \
  --image-size 32x32 cifar10_pipeline_benchmark.log

python3 scripts/my_tool/summarize_pipeline_benchmark.py --format markdown \
  --datasets tiny-imagenet,svhn,cifar10 \
  --image-sizes 64x64,32x32,32x32 \
  small_image_pipeline_benchmark.log
```

The summarizer accepts stdin as `-`, extracts every `Pipeline benchmark
results:` block, derives `crop_size` from `crop: x,y,w,h`, and derives
speedup/saved-time fields from `total_ms` if an older log does not print them
directly. Use `--image-size` for one dataset, or `--datasets` and
`--image-sizes` for logs that contain multiple result blocks, because
`pipeline_benchmark` output does not carry source image dimensions or manifest
labels. Use `--require-match` for correctness logs so the command fails if any
result block reports `outputs_match` other than `1`. Use
`--require-default-fields` for benchmark evidence so the command also fails when
the log is missing any field from the reporting table.

## Verification Status

Current CPU-side checks prove crop planning, selected-vector mapping, runtime
policy helpers, and benchmark timing aggregation. They do not prove GPU kernel
correctness or performance.

Required GPU evidence before marking the implementation complete:

- `pipeline_benchmark --mode compare` with `outputs_match: 1` for boundary,
  unaligned, cross-rowgroup, and cross-component crop cases;
- small-image overhead stress results;
- medium/large-image benefit results;
- counter comparison showing reduced selected vectors, worksets/launches where
  applicable, bounded scratch allocation growth, and no device-wide sync in the
  JPEG DCT hot path.

## Delivery Checklist

This section maps the current implementation to the requested deliverables.

1. Code changes:
   vector-level crop selection is planned in `PlanDeviceDctBatch()` and executed
   through FastLanes explicit work items. `pipeline_benchmark` supports
   pushdown, full-then-crop baseline, compare, and auto modes.
2. Synchronization removed:
   the JPEG DCT planner/executor/benchmark-support hot path has no
   `cudaDeviceSynchronize()` or `cudaStreamSynchronize()`. A CTest static guard
   enforces this. Remaining waits are CUDA event waits counted by
   `internal_sync_count`.
3. Launches merged:
   decoded rowgroups are accumulated into batch-level worksets, and gather items
   are submitted through batch gather kernels. `workset_count`,
   `decode_kernel_launch_count`, and `gather_kernel_launch_count` expose the
   actual launch shape.
4. Scratch reused:
   `JpegDctDeviceScratch` is owned by the reader and reused across windows. It
   owns reusable device metadata buffers, host staging vectors, events/streams,
   and the decode workset. `scratch_allocation_count` reports capacity growth.
5. Data flow:
   crop-to-block, block-to-rowgroup, row-in-rowgroup-to-vector, selected-vector
   workset build, decode, and gather are described in the Data Flow section.
6. Correctness tests:
   CPU tests cover crop planning, selected-vector compaction/remapping, runtime
   policy, auto-policy estimates, and workset explicit selected-vector chunks.
   GPU correctness still requires `pipeline_benchmark --mode compare`.
7. Benchmark comparison:
   `pipeline_benchmark` prints selected/full vector ratios, plan/decode/gather
   timings, launch counts, scratch allocations, sync counts, policy decisions,
   and compare-mode speedup fields. GPU timing tables must be filled from a
   CUDA-capable run.
8. Deferred optimizations:
   sparse vector cache, batch ownership that removes decoded-batch event waits,
   and more detailed measured-cost policy terms are intentionally deferred until
   GPU evidence shows they are worth the added complexity.
9. Final judgment:
   fixed overhead is structurally lower than the previous per-rowgroup path, but
   the implementation cannot be called complete until GPU compare correctness
   and small/medium/large benchmark counters prove the overhead is low enough in
   practice.

## Decision Guide

Use vector-level crop pushdown when the prepared plan shows clear decode-work
savings:

- `selected_vector_ratio` is well below the full decode path;
- crop output is small enough that gather/layout work does not dominate;
- touched rowgroups can be submitted in a small number of decode batches;
- the rowgroup runtime policy reports mostly `selected-vector` instead of
  `full-rowgroup`.

Use full decode plus crop when the fixed costs are likely to dominate:

- crop covers almost all blocks or vectors;
- the window has very few full-image blocks, making launch/workset overhead hard
  to amortize;
- the crop touches most rowgroups;
- estimated pushdown worksets do not shrink versus full decode and selected
  vector savings are modest;
- rowgroups are tiny, so vector-level selectivity does not reduce enough kernel
  work to pay for planning, workset upload, gather, and synchronization.

`pipeline_benchmark --mode auto` follows this guide with explicit thresholds and
prints the chosen reason through `auto_policy_reason`. The policy is deliberately
simple: it is a guardrail against bad pushdown choices, not a learned performance
model.

## Current Limitations

- Sparse decoded-vector cache is not implemented. Sparse cache counters are
  printed but remain zero.
- The decoded-batch path still waits at batch end before reusing rowgroup
  metadata, workset arenas, and scratch. Removing this requires longer-lived
  batch ownership for those resources.
- Dense rowgroup cache still caches only full decoded rowgroups. This keeps the
  main vector-pushdown path simple but means repeated sparse vector accesses are
  not reused unless the policy falls back to full-rowgroup decode.
- The current rowgroup-level runtime policy uses selected/full vector ratio and
  saved-vector count. More detailed cost terms, such as measured cache hit
  probability or per-codec decode cost, are left for later tuning after GPU
  benchmark evidence is available.
- GPU correctness and benchmark performance must be verified on a CUDA-capable
  machine before calling the optimization complete.
