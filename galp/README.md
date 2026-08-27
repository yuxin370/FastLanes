# G-ALP

G-ALP is the GPU decompression component for FastLanes. This subtree is being
shaped into a repository-local library component with a small public facade.

The stable external surface is intentionally narrow. Most implementation
headers under `src` are private and may change without compatibility
guarantees.

## Status

- Main supported path: table-level GPU decompression for i8/i16 FastLanes
  rowgroups.
- Public API: `galp/include/galp` exposes `galp::Reader`, `galp::Table`,
  `galp::RowgroupView`, `galp::ColumnView`, and `galp::DecompressOptions`.
- Private implementation: `core/*`, `format/*`, `engine/*`, `cuda/*`,
  `codecs/*`, generated bindings, benchmark headers, and nvCOMP support.
- CUDA is required for runtime behavior and GPU correctness tests. Tests that
  need a CUDA device or local sample data should skip when those inputs are not
  available.

## Architecture

The main table path is:

```text
external consumer
  -> galp::Reader / galp::decompress_table
  -> galp::execution::decompress_table
  -> galp::runtime::execute_table_pipeline
  -> galp::format::FlsReader
  -> galp::expression::assemble
  -> galp::memory::DeviceArena / pinned pools / H2D
  -> CUDA decompression kernels
  -> optional materialization
```

The data flow is:

```text
FLS rowgroup
  -> zero-copy rowgroup view
  -> expression assembly
  -> execution workset
  -> DeviceArena/H2D upload
  -> CUDA kernel decode
  -> optional write-out and D2H materialization
```

## Internal Layout

| Area | Path | Responsibility |
|---|---|---|
| Public facade | `include/galp`, `src/api` | Stable external API |
| Format/reader | `src/format` | FLS descriptors, schema plans, rowgroup IO |
| Expression/core | `src/core` | Data model, enums, type helpers, expression assembly |
| Engine operators | `src/engine/operators` | Decode configuration, batch, column, and rowgroup operators |
| Engine table | `src/engine/table` | Table execution, runners, resources, request/pipeline state |
| Engine pipeline | `src/engine/pipeline` | Streaming pipeline, prefetch queues, pinned rowgroup pools |
| Engine materialization | `src/engine/materialization` | Metadata, zero-copy materialization, pinned D2H |
| Engine worksets | `src/engine/workset` | Workset model, streams, append, upload |
| Compression formats | `src/codecs/encodings`, `src/codecs/*.cuh` | Compressed column descriptors, shared constants, format utilities |
| Decompression primitives | `src/codecs/decode`, `src/codecs/device_ops` | Vector-layout unpackers, patchers, expanders, decompressors, ALP helpers |
| Memory | `src/cuda/memory` | CUDA RAII, DeviceArena, DevicePool, pinned host pools |
| CUDA support | `src/cuda`, `src/cuda/launch`, `src/cuda/memory` | Device helpers, launch infrastructure, memory support, kernel implementations |
| CLI | `tools/galp_cli/galp_cli.cu` | `read_table`, `benchmark`, launch measurement |
| ALP extension | `extensions/alp` | CPU ALP encode/decode support for tests and benchmarks |
| Tool support | `tools/benchmark_support`, `tools/data` | CLI benchmark support and data helpers |
| Benchmarks | `benchmarks`, `benchmarks/nvcomp` | Generated bindings, microbenchmarks, nvCOMP comparisons |
| Code generation | `scripts/codegen` | Benchmark binding generation |
| Tests | `tests` | Public API smoke tests, reader tests, CUDA/internal/integration tests; CTest labels carry test categories |

## Public API

External consumers should include only the facade:

```cpp
#include <galp/galp.hpp>

galp::Reader reader("/path/to/data.fls");

galp::DecompressOptions options;
options.write_output = true;
options.scope = galp::TableDecompressionScope::WholeTable;

galp::Table table = reader.decompress(options);
auto rowgroups = table.rowgroup_count();
auto total_columns = table.total_columns();
auto first_rowgroup_columns = rowgroups == 0 ? 0 : table.rowgroup_column_count(0);
if (rowgroups != 0 && first_rowgroup_columns != 0) {
	auto rowgroup = table.rowgroup(0);
	auto column = rowgroup.column(0);
	if (column.type() == galp::DataType::I8) {
		std::span<const int8_t> values = column.values<int8_t>();
	}
}
```

`galp::Table` exposes table metadata and read-only data views:

- `rowgroup_count()`
- `total_columns()`
- `rowgroup_column_count()`
- `rowgroup_column_counts()`
- `rowgroup(size_t) -> galp::RowgroupView`
- `empty()`

`galp::RowgroupView` exposes:

- `column_count()`
- `column(size_t) -> galp::ColumnView`

`galp::ColumnView` exposes:

- `name()`
- `type()`
- `size()`
- `values<T>() -> std::span<const T>`

Output storage is owned by `galp::Table`. `RowgroupView` and `ColumnView` are
lightweight read-only views and must not outlive their source table.

`DecompressOptions::write_output=false` enables the write-back-free path. That
path skips decoded global output writes but still returns table metadata.
`RowgroupView::column()` and `ColumnView::values<T>()` require
`write_output=true`.

The stable table path currently supports i8/i16 single-token rowgroups,
including uncompressed i8, constant i8, frequency i8/i16, FFOR i8/i16,
FFOR+SLPATCH i8/i16, dictionary i8/i16, cross-RLE i8, RLE i8/i16, and
`EXP_EQUAL` aliases. Unsupported tokens or multi-op expressions throw
`galp::UnsupportedFormatError` with token, rowgroup, column, and column name
context.

## Direct-DCT Runtime

When `GALP_WITH_JPEG_DCT=ON`, `galp/direct_dct.hpp` exposes a small
stay-on-GPU runtime facade for direct-DCT ML workloads:

The advanced premixed physical-PLS training path, including its native
crop/pool-shuffle/CUDA augmentation boundary and deployment prerequisites, is
documented in the [experimental PLS training guide](benchmarks/system_dct_major/training_pls/README.md).

```cpp
#include <galp/direct_dct.hpp>

galp::jpeg::DirectDctRuntime runtime("/path/to/manifest.bin");

galp::jpeg::JpegDctDeviceBatchOptions options;
galp::jpeg::parse_jpeg_dct_coefficient_selection("first:8", options.coefficient_selection);

auto batch = runtime.ReadBatch(
    std::vector<uint32_t> {0, 1, 2, 3},
    galp::jpeg::JpegDctCropBox {30, 40, 224, 224},
    options);

auto tensor = batch.tensor();
// tensor.data is a CUDA int16 pointer with logical shape [block_count, coefficients_per_block].
```

`DirectDctBatch` owns the underlying `JpegDctDeviceBatch`, so the CUDA pointer
returned by `DirectDctBatch::tensor()` remains valid as long as the batch or a
wrapper that owns it remains alive. The initial tensor contract is a compact
flat layout:

```text
coefficients: int16 CUDA buffer, logical shape [total_blocks, K]
K: selected DCT coefficient count, 64 for all coefficients
image_layouts: per-request global image id, block offset, block count
block_metadata: request/image/component/block coordinates for each tensor row
selected_coefficients: logical DCT coefficient id for each tensor column
```

The PyTorch extension also exposes GPU-resident metadata tensors for hot
training paths:

```text
image_offsets_tensor: int64 CUDA tensor [image_count]
image_counts_tensor: int64 CUDA tensor [image_count]
block_to_image_tensor: int64 CUDA tensor [total_blocks]
```

Use these tensors for per-image pooling or token packing on CUDA. The private
native batch's `image_layouts`, `block_metadata`, `rowgroups`, and
`execution_stats` properties materialize Python list/dict objects and are
intended for debugging and compatibility. The stable `galp.torch.DirectDctBatch`
does not expose them; benchmark tools opt into `galp.diagnostics.direct_dct`.

Applications should use the stable Python facade rather than importing the
private `_galp_direct_dct` extension.  The caller chooses a semantic output
profile; cache, planning, prefetch, stream, I/O, and launch policies remain
native-owned:

```python
from galp.profiles.rgbnomore import VALIDATION
from galp.torch import DirectDctReader

reader = DirectDctReader(
    "/path/to/manifest.bin",
    module_path="build/galp/torch",
)
pipeline = reader.pipeline(VALIDATION).start(
    [[0, 1, 2, 3], [4, 5, 6, 7]],
)
for batch in pipeline:
    output = model(batch.y, batch.cbcr)

# Optional raw JPEG zigzag-column selection.  The registered profile still
# owns crop/layout/runtime policy, and the model-ready tensor shape is unchanged.
selected = reader.pipeline(VALIDATION, dct_coeffs="first:32").start(
    [[0, 1, 2, 3]],
)
```

The public API has no future/submission gate, planner preview, manual reclaim,
rowgroup metadata, or buffer-keepalive method. The native pipeline owns bounded
prefetch, ordered CUDA submission, completion, and deferred reclamation. Tensor
ownership automatically keeps native storage alive on the consuming PyTorch
stream. Dataset and implementation diagnostics are available only from the
explicitly unstable `galp.diagnostics.direct_dct` module.

`galp.profiles.rgbnomore` owns RGB-no-more geometry, normalization, and crop
reference semantics.  Generic native runtime policies are defined separately
and are observable through `reader.profile_info(profile)` but are not Python
tuning options.

The runtime keeps the existing JPEG DCT crop and coefficient-selection
pushdown, cache, prefetch, and decode-batch behavior. Registered profiles may
also fuse DCT-grid transforms and model-ready affine conversion. It does not
perform IDCT, RGB reconstruction, or detection/segmentation collation.

The first runtime API does not accept a caller-owned CUDA stream. It delegates
planning, compressed rowgroup upload, GPU FastLanes decode, gather/projection,
cache reuse, and event handoff to the existing `JpegDctShardDatasetReader`
device-batch path. The direct-DCT facade and PyTorch export path do not issue
device-to-host copies, `cudaDeviceSynchronize()`, or `cudaStreamSynchronize()`.
The optional PyTorch wrapper records a CUDA event on the current PyTorch stream
when external tensor storage is released and delays `DirectDctBatch` destruction
until that event completes, preventing GALP's device pool from reusing a buffer
while already queued PyTorch kernels on that stream are still consuming it.
Custom-stream callers should use normal CUDA/PyTorch stream synchronization
around the returned tensor before launching dependent work on another stream.

An optional PyTorch extension is available behind `GALP_BUILD_TORCH=ON`. This
is deliberately not part of `Galp::core`'s default dependency set:

```bash
cmake -S . -B build-galp-torch -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
  -DGALP_BUILD_TORCH=ON \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-galp-torch --target _galp_direct_dct -j
PYTHONPATH=build-galp-torch/galp/torch \
  python3 galp/examples/direct_dct_pipeline_demo.py /path/to/manifest.bin \
    --batch-size 32 \
    --steps 3 \
    --train-smoke
```

The example keeps the model-ready Y/CbCr tensors on CUDA and runs a small
classifier through the public native-owned pipeline. It reports only the
versioned aggregate metrics. Prefetch depth, cache capacity, crop execution,
I/O scheduling, CUDA stream selection, and kernel launch geometry are not
command-line options.

The fixed 512 center-crop semantic profile can be selected for a compatible
DCT-major manifest without exposing its block-major runtime policy:

```bash
PYTHONPATH=build-galp-torch/galp/torch \
  python3 galp/examples/direct_dct_pipeline_demo.py /path/to/manifest.bin \
    --batch-size 32 \
    --profile validation-center-crop-512
```

The older `direct_dct_torch_end_to_end_demo.py` is a private-extension
diagnostic/compatibility harness used by pushdown validation. It is not a
model-facing API example. The CMake module also tries to discover the Torch
prefix automatically from the selected Python interpreter when `Torch_DIR` is
not already set.

Public headers must not include private implementation prefixes such as
`core/`, `format/`, `engine/`, `cuda/`, `codecs/`, benchmark,
extension, or tool-support implementation paths. Check the boundary with:

```bash
cmake --build build-galp-ninja --target galp_api_boundary_checks
```

## Build

Configure from the repository root. The examples below use `build-galp-ninja`
for the normal development build:

```bash
cmake -S . -B build-galp-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
  -DGALP_BUILD_TESTS=ON \
  -DGALP_BUILD_TOOLS=ON \
  -DGALP_BUILD_BENCHMARKS=OFF \
  -DGALP_WITH_NVCOMP=OFF
```

Build the core library, public API smoke target, tests, and CLI:

```bash
cmake --build build-galp-ninja --target galp_core galp_public_api_smoke galp_tests galp_cli -j
```

Run the usual validation set:

```bash
ctest --test-dir build-galp-ninja -L public-api --output-on-failure
ctest --test-dir build-galp-ninja -L gpu --output-on-failure
ctest --test-dir build-galp-ninja -R GalpPackageConsumerSmoke --output-on-failure
cmake --build build-galp-ninja --target galp_api_boundary_checks
```

Common CMake options:

```text
GALP_BUILD_TESTS        Build GALP tests
GALP_BUILD_TOOLS        Build galp_cli
GALP_BUILD_EXAMPLES     Build GALP examples
GALP_BUILD_BENCHMARKS   Build generated bindings and microbenchmarks
GALP_WITH_NVCOMP        Build nvCOMP compressor comparison targets
GALP_ENABLE_INSTALL     Generate install/export/package targets
```

With `GALP_BUILD_BENCHMARKS=OFF`, CMake does not run benchmark code generation
and does not create benchmark generated outputs. Use a separate benchmark build
tree when working on generated bindings or `micro_bench`:

```bash
cmake -S . -B build-galp-bench -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
  -DGALP_BUILD_TESTS=ON \
  -DGALP_BUILD_TOOLS=ON \
  -DGALP_BUILD_BENCHMARKS=ON \
  -DGALP_WITH_NVCOMP=OFF

cmake --build build-galp-bench --target generated-bindings micro_bench -j
```

Enable `GALP_WITH_NVCOMP=ON` only when building nvCOMP compressor comparison
targets:

```bash
cmake -S . -B build-galp-bench-nvcomp -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
  -DGALP_BUILD_BENCHMARKS=ON \
  -DGALP_WITH_NVCOMP=ON

cmake --build build-galp-bench-nvcomp --target compressor_bench -j
```

## JPEG DCT Pipeline Benchmark

When `GALP_WITH_JPEG_DCT=ON`, `galp_cli pipeline_benchmark` measures JPEG DCT
crop reads from a sharded DCT/FLS manifest:

```bash
./build-galp-ninja/galp/tools/galp_cli pipeline_benchmark /path/to/manifest.bin \
  --crop 64 64 512 512 --window-images 256 --mode compare

./build-galp-ninja/galp/tools/galp_cli pipeline_benchmark /path/to/manifest.bin \
  --crop 64 64 512 512 --window-images 256 --dct-coeffs list:0,2,5 --mode compare

./build-galp-ninja/galp/tools/galp_cli pipeline_benchmark /path/to/manifest.bin \
  --window-images 256 --dct-coeffs list:0,2,5 --mode compare

./build-galp-ninja/galp/tools/galp_cli pipeline_benchmark /path/to/manifest.bin \
  --crop 64 64 512 512 --window-images 256 --dct-coeffs first:8 --mode dct-compare
```

The current implementation, benchmark, and historical-document status is indexed in
[`galp/docs/README.md`](docs/README.md). The executable inference contract is documented in
[`galp/benchmarks/system_rgbnomore/docs/E2E_COMPARISON_RUN_GUIDE.md`](benchmarks/system_rgbnomore/docs/E2E_COMPARISON_RUN_GUIDE.md).
For GPU evidence collection, save raw `pipeline_benchmark` output and summarize
it with:

```bash
./build-galp-ninja/galp/tools/galp_cli pipeline_benchmark /path/to/manifest.bin \
  --crop 64 64 512 512 --window-images 256 --mode compare | tee pipeline_benchmark.log

python3 scripts/my_tool/bench_pipeline_summary.py \
  --dataset <dataset> --image-size <width>x<height-or-varies> \
  --require-match --require-default-fields \
  pipeline_benchmark.log
```

The summarizer derives crop size and speedup fields and fails when correctness
or required counters are missing. Pass `--image-size` for one result block, or
`--image-sizes` with `--datasets` for logs containing multiple result blocks,
because `pipeline_benchmark` output does not include image dimensions. The
default report keeps `dct_coeffs` visible and selects mode-specific fields for
`auto` and `dct-compare` result rows.

The pipeline modes are:

- `pushdown`: plan only the requested pixel crop, push the selected DCT blocks
  into the JPEG DCT reader and the FastLanes workset schedule, and decode only
  the selected FastLanes vector chunks.
- `baseline` or `full-then-crop`: decode full images first, then select the DCT
  blocks that intersect the same pixel crop from the decoded full output.
- `compare`: run both paths and verify that the cropped DCT coefficients match.
- `dct-compare`: keep crop pushdown fixed for both paths and compare DCT
  coefficient selection pushdown against a post-decode baseline. The pushdown
  path decodes/gathers only the selected coefficient columns; the
  `dct_post_decode` path uses the same crop requests but decodes all 64
  coefficients and logically projects the selected subset afterwards.
- `auto`: use CPU prepared plans for each window to choose pushdown or
  full-then-crop from selected/full FastLanes vector ratio, crop/full block
  ratio, touched-rowgroup ratio, average full blocks per rowgroup, and estimated
  workset/gather item counts. The selected path executes the prepared plan
  rather than planning a second time. When no crop is requested, auto prepares
  only the full-image candidate. It keeps small windows on full-then-crop to
  avoid fixed pushdown overhead, and also rejects pushdown
  when cropped gather output is already close to full output, without
  dataset-name special cases.

`--crop x y width height` is a pixel-space crop in source-image coordinates.
Omitting `--crop` benchmarks full-image output. The pushdown path currently
pushes selection to FastLanes vector/chunk granularity; rows inside a selected
FastLanes vector are still decoded as part of that vector.

The benchmark prints pushdown and full-then-crop stage counters for selected
vs. full vectors, workset uploads, decode/gather/materialize kernel launches,
scratch metadata uploads, scratch allocation growth, internal stream-local
syncs, dense cache hits/misses, cached-gather launch/event handoffs, and
runtime-policy decisions. `*_plan_ms` includes prepared-plan construction with
device planning counted once through `*_device_planning_ms`; `*_read_decode_ms`
measures prepared-plan execution without rebuilding the plan. Newly decoded
rowgroups are submitted in batches of
up to `--decode-batch-rowgroups` per FastLanes workset (default 64), so
`*_workset_count` tracks batch submissions rather than one workset per touched
rowgroup. Sparse vector cache counters are
reported as zero until a sparse decoded-vector cache is implemented.
`*_planned_selected_vector_count` reports the crop-selected vectors before the
runtime policy; `*_selected_vector_count` reports vectors actually submitted to
FastLanes decode and therefore excludes dense-cache hits. As a result,
`*_actual_saved_vector_count` includes both vector pushdown and dense-cache
decode avoidance.
In `auto` mode the output also reports chosen pushdown/full windows, auto policy
prepared-planning/policy time, aggregate auto total time, the last policy
reason, aggregate crop/full prepared-plan block and FastLanes vector counts and
ratios, touched/full rowgroup counts and ratios, average full blocks per
rowgroup, prepared-plan repeated rowgroup reuse candidates, and per-reason
window counts. Reuse candidates are a cheap policy proxy; measured dense-cache
hits and misses remain in the per-stage cache counters.
JPEG DCT readers reuse host staging vectors and device scratch buffers across
`ReadDeviceDctBatch()` windows. Scratch upload counters report device scratch
metadata copies; decoded-gather coefficient pointers and source tags are packed
into one binding array before upload. Scratch allocation counters report
reusable device-buffer capacity growth rather than per-window allocation churn;
host staging vectors, including pending rowgroup work, are reused but are not
counted as device allocations. Metadata scratch starts from a small capacity
floor and grows geometrically to avoid repeated tiny cudaMalloc/free. The JPEG
scratch also owns the reusable decode
workset, so its streams, timing events, output arena, and chunk arena capacity
are preserved across decode batches. Increasing `--decode-batch-rowgroups`
can reduce tiny worksets and batch-end syncs when the selected rowgroups are
small enough for one larger workset; `*_workset_count` is therefore a submitted
batch count, not a resource construction count. Dense-cache-hit gathers use CUDA
events to hand off to the next decoded batch stream, so cache-hit gather
synchronization is folded into an existing batch sync when a decode follows.
Cached-hit gather items are queued at device-batch scope and flushed at
decode/cache-eviction boundaries or at batch completion, instead of being forced
out at every shard boundary. Consecutive cached gathers on the cache-hit stream
reuse the cached-gather item scratch without a host wait unless the buffer needs
to grow.

`JpegDctShardDatasetReader::PlanDeviceDctBatch()` exposes the CPU-only crop
planner metadata used by the device path. It returns image layouts, selected
DCT block metadata, touched rowgroups, raw crop-selected vector counts,
runtime-policy estimated scheduled vector counts, full vector counts, and
planning time without launching GPU decode kernels or mutating the
decoded-rowgroup cache. The prepared plan also carries the compact
per-rowgroup selected-vector list, fit result, selected/full vector counts, and
runtime-policy decision. For rowgroups that will run selected-vector decode it
also carries the remapped gather items, so `ReadPreparedDeviceDctBatch()` can
submit the workset without repeating selected-vector sort/unique or gather
remapping.

## Package Consumers

G-ALP exports `Galp::core`:

```cmake
find_package(Galp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE Galp::core)
```

Verify the build-tree package consumer with:

```bash
ctest --test-dir build-galp-ninja -R GalpPackageConsumerSmoke --output-on-failure
```

`galp_core` is a CUDA static library. Device symbols are resolved during the
build so C++ consumers can link `Galp::core` without missing
`__cudaRegisterLinkedBinary_*` symbols.

Minimal examples:

- `galp/examples/public_api_reader/public_api_reader.cpp`: includes only `<galp/galp.hpp>`, prints
  rowgroup/column metadata, and reads the first column span.
- `galp/examples/cmake_consumer/CMakeLists.txt`: demonstrates
  `find_package(Galp CONFIG REQUIRED)` and `target_link_libraries(... Galp::core)`.

Build repository examples with:

```bash
cmake -S . -B build-galp-examples -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
  -DGALP_BUILD_EXAMPLES=ON \
  -DGALP_BUILD_TOOLS=ON \
  -DGALP_BUILD_TESTS=OFF \
  -DGALP_BUILD_BENCHMARKS=OFF

cmake --build build-galp-examples --target galp_public_api_reader -j
./build-galp-examples/galp/examples/public_api_reader/galp_public_api_reader data/fls/galp-test/data.fls
```

## Tests

Run all tests:

```bash
ctest --test-dir build-galp-ninja --output-on-failure
```

Common focused subsets:

```bash
ctest --test-dir build-galp-ninja -L public-api --output-on-failure
ctest --test-dir build-galp-ninja -L gpu --output-on-failure
```

Reader tests use `FLS_READER_TEST_FILE` when set. Otherwise they use:

```text
data/fls/galp-test/data.fls
```

Tests that require CUDA or sample files should skip when those inputs are not
available locally.

## CLI

`galp_cli` provides two main commands:

```bash
./build-galp-ninja/galp/tools/galp_cli read_table data/fls/galp-test/data.fls /tmp/galp_out.csv
./build-galp-ninja/galp/tools/galp_cli benchmark data/fls/galp-test/data.fls --samples 5
```

Read one rowgroup:

```bash
./build-galp-ninja/galp/tools/galp_cli read_table data/fls/galp-test/data.fls /tmp/galp_out.csv --rowgroup 0
```

The default benchmark mode is write-back-free and consume-only: decoded
registers are consumed without writing global output. Add `--include-materialize`
to measure the output-producing path:

```bash
./build-galp-ninja/galp/tools/galp_cli benchmark data/fls/galp-test/data.fls --samples 5 --include-materialize
```

Common benchmark options:

```text
--samples N
--kernel-samples N
--rowgroup N
--per-rowgroup-workset
--no-mixed-dispatch
--no-rowgroup-prefetch
--prefetch-depth N
--prefetch-workers N
--max-prefetch-storage-bytes N
--stream-target-work-items N
--stream-max-rowgroups N
--include-materialize
--reuse-table-resources
--freq-patcher stateful|branchless|hybrid[:threshold]
```

## Microbenchmarks

`micro_bench` is generated-binding driven and is built only when
`GALP_BUILD_BENCHMARKS=ON`:

```bash
cmake --build build-galp-bench --target generated-bindings micro_bench -j
./build-galp-bench/galp/benchmarks/micro_bench --help
```

The CLI uses normalized enum strings, not C++ type or enum names. For example,
use `u32`, `bit-packing`, `dummy`, and `none`, not `uint32_t`, `BP`, `Dummy`,
or `None`:

```bash
./build-galp-bench/galp/benchmarks/micro_bench \
  u32 bit-packing decompress 1 1 dummy none none 1 8 0 0 1024 1 0
```

## Benchmark Semantics

The main path is fused rowgroup prefetch:

```bash
PREFETCH_DEPTH=4 PREFETCH_WORKERS=0 STREAM_MAX_ROWGROUPS=1 \
  bash scripts/my_tool/bench_fls_end2end.sh
```

`PREFETCH_WORKERS=0` lets `galp_cli` choose the worker count from the rowgroup
count. `--samples` controls independent benchmark samples and defaults to 5.
`benchmark_wall_ms`, `query_wall_ms`, and `kernel_event_ms` report the median
sample and also expose `_min`, `_median`, and `_mean` fields. `--kernel-samples`
controls kernel replay inside each sample and defaults to 1.

Local helper scripts:

- `scripts/my_tool/bench_fls_end2end.sh`: default end-to-end benchmark, writes CSV files
  under `scripts/my_tool/end2end_res/`.
- `scripts/my_tool/bench_fls_cache.sh`: cold/warm cache comparisons using
  `CACHE_DROP_MODE`.
- `scripts/my_tool/bench_fls_io_sweep.sh`: IO and prefetch parameter matrix.
- `scripts/my_tool/profile_ncu.sh`: NCU profiler entry point.
- `scripts/my_tool/profile_nsys.sh`: NSYS profiler entry point.
- `scripts/my_tool/check_decompressed_diff.sh`: compares `galp_cli` output with FastLanes
  decompression output.

Important metrics:

- `benchmark_wall_ms`: wall clock for the whole benchmark run.
- `benchmark_wall_ms_min/median/mean`, `query_wall_ms_min/median/mean`,
  `kernel_event_ms_min/median/mean`: summaries across independent samples.
- `resource_prepare_ms`: reader and pinned-pool setup when
  `--reuse-table-resources` is enabled.
- `query_wall_ms`: query body wall time in reuse mode.
- `pipeline_active_ms`: query wall time minus pipeline setup.
- `kernel_event_ms`: kernel time measured by CUDA events.
- `read_rowgroup_ms`, `pread_ms`, `zero_copy_view_setup_ms`: rowgroup read and
  zero-copy setup costs.
- `upload_workset_ms`, `upload_dma_gpu_ms`, `h2d_bytes`, `h2d_copies`: H2D upload
  costs.
- `prefetch_wait_ms`, `prefetch_depth_block_ms`, `prefetch_byte_block_ms`:
  prefetch waiting and back-pressure.
- `write_back`, `include_materialize`, `consume_only`, `write_back_free`:
  execution-mode fields.

Host rowgroup reads, H2D upload, and kernels can overlap in the streaming
pipeline, so stage sums may exceed `benchmark_wall_ms`.

## Debug Environment

```text
GALP_ROWGROUP_TIMELINE_CSV=/tmp/galp_timeline.csv
    Writes a per-rowgroup read/build/ready/upload timeline.

GALP_MEASURE_H2D=1
    Measures upload_dma_gpu_ms with CUDA events. This diagnostic mode can reduce
    H2D/compute overlap.

GALP_DISABLE_ASYNC_H2D=1
    Disables the dedicated H2D stream and falls back to the default stream.

GALP_PINNED_ROWGROUP_PREWARM_BYTES=N
    Caps pinned rowgroup buffer prewarm bytes.

GALP_PINNED_ROWGROUP_PREWARM_MODE=full|off|none|lazy
    Controls pinned rowgroup buffer prewarm mode.

GALP_PINNED_ROWGROUP_PREWARM_SLOTS=N
    Overrides adaptive prewarm slot count.

GALP_VALIDATE_SHARED_POSITION_OFFSETS=1
    Enables shared position offset validation while debugging reader/expression
    offsets.
```

## Generated Bindings

Generated benchmark outputs are not committed. When `GALP_BUILD_BENCHMARKS=ON`,
CMake runs the generators under `galp/scripts/codegen` and writes outputs under
the build tree:

```text
${CMAKE_BINARY_DIR}/generated/galp/benchmarks/bindings/
${CMAKE_BINARY_DIR}/generated/galp/benchmarks/include/galp_bench/generated/
```

When changing generated kernel bindings, edit the generators and validate the
expected generated file manifest with:

```bash
python3 galp/scripts/codegen/check_generated_reproducible.py \
  --tmp-dir /tmp/galp_codegen_checker
```

## Development Rules

- Public consumers should depend only on `<galp/galp.hpp>` and `Galp::core`.
- `src` is a private include surface.
- Public headers must not leak `core`, `format`, `engine`, `cuda`, `codecs`,
  benchmark, extension, or tool-support implementation paths.
- For engine/format/codecs/cuda changes, build `galp_core` and `galp_cli`, then
  run the public API tests.
- For public API or package changes, run `GalpPublicApiSmoke`,
  `GalpPublicApiBoundaries`, and `GalpPackageConsumerSmoke`.
- For generated binding changes, keep generator edits and expected-manifest
  checks in the same change; generated outputs are build artifacts.

## Direction

The current work is about tightening boundaries rather than changing the core
decompression algorithms:

- A small stable facade now fronts the internal CUDA decompression implementation.
- Direct exposure of reader, dispatch, compression, and decompression internals
  is being reduced in favor of `galp::*` public types.
- nvCOMP and benchmark support are optional build slices.
- CUDA resources are moving toward RAII stream/event/pool/workset ownership.
- The default benchmark path is write-back-free for pipeline tuning.

Further modularization should keep shrinking `format/reader.cuh` and
`engine/pipeline/pipeline.cuh`, while hardening the boundaries between reader,
zero-copy planning, compression column construction, resource preparation,
prefetch integration, and chunk execution.
