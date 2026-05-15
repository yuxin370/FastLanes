# G-ALP

G-ALP is the GPU decompression component for FastLanes. This subtree is being
shaped into a repository-local library component with a small public facade.

The stable external surface is intentionally narrow. Most implementation
headers under `src/include` are private and may change without compatibility
guarantees.

## Status

- Main supported path: table-level GPU decompression for i8/i16 FastLanes
  rowgroups.
- Public API: `galp/include/galp` exposes `galp::Reader`, `galp::Table`,
  `galp::RowgroupView`, `galp::ColumnView`, and `galp::DecompressOptions`.
- Private implementation: `engine/*`, `compression/*`, `decompression/*`,
  `memory/*`, generated bindings, benchmark headers, and nvCOMP support.
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
| Format/reader | `src/include/engine/format`, `src/include/engine/reader.cuh` | FLS descriptors, schema plans, rowgroup IO, zero-copy rowgroups |
| Expression | `src/include/engine/expression.cuh` | Convert rowgroup columns into execution expressions |
| Execution | `src/include/engine/execution/*` | Decode configuration, rowgroup/table data models, public engine entry points |
| Runtime | `src/include/engine/runtime/*` | Worksets, H2D upload, streaming table pipeline, materialization |
| Execution internals | `src/include/engine/execution/internal/*` | Prefetch queues, launch glue, batch/unpack dispatch |
| Compression formats | `src/include/compression/columns`, `src/include/compression/*.cuh` | Compressed column descriptors, shared constants, format utilities |
| Decompression primitives | `src/include/decompression/primitives.cuh`, `src/include/decompression/primitives` | Vector-layout unpackers, patchers, expanders, decompressors, ALP helpers |
| Memory | `src/include/memory`, `src/memory` | CUDA RAII, DeviceArena, DevicePool, pinned host pools |
| Kernels | `src/include/engine/kernels/*` | CUDA kernel wrappers and host launch helpers |
| CLI | `tools/galp_cli.cu` | `read_table`, `benchmark`, launch measurement |
| Benchmarks | `benchmark` | Generated bindings, microbenchmarks, nvCOMP comparisons |
| Code generation | `code-generators` | Benchmark binding generation |
| Tests | `test` | Public API smoke tests, reader tests, GPU/internal tests |

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

Public headers must not include private implementation prefixes such as
`engine/`, `compression/`, `decompression/`, `memory/`, `benchmark/`, `alp/`,
`nvcomp/`, or `generator/`. Check the boundary with:

```bash
cmake --build build --target galp_api_boundary_checks
```

## Build

Configure from the repository root:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
  -DGALP_BUILD_TESTS=ON \
  -DGALP_BUILD_TOOLS=ON \
  -DGALP_BUILD_BENCHMARKS=OFF \
  -DGALP_WITH_NVCOMP=OFF
```

Common development targets:

```bash
cmake --build build --target galp_core galp_public_api_smoke galp_tests galp_cli -j
```

Common CMake options:

```text
GALP_BUILD_TESTS        Build GALP tests
GALP_BUILD_TOOLS        Build galp_cli
GALP_BUILD_BENCHMARKS   Build generated bindings and micro-benchmarks
GALP_WITH_NVCOMP        Build nvCOMP compressor comparison targets
GALP_ENABLE_MULTI_COLUMN Reserved; keep OFF, ON fails configure by design
GALP_ENABLE_INSTALL     Generate install/export/package targets
```

Keep benchmarks and nvCOMP disabled by default. Enable them only when generated
microbenchmarks or nvCOMP comparisons are part of the task:

```bash
cmake -S . -B build \
  -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
  -DGALP_BUILD_BENCHMARKS=ON \
  -DGALP_WITH_NVCOMP=ON
```

## Package Consumers

G-ALP exports `Galp::core`:

```cmake
find_package(Galp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE Galp::core)
```

Verify the build-tree package consumer with:

```bash
ctest --test-dir build -R GalpPackageConsumerSmoke --output-on-failure
```

`galp_core` is a CUDA static library. Device symbols are resolved during the
build so C++ consumers can link `Galp::core` without missing
`__cudaRegisterLinkedBinary_*` symbols.

Minimal examples:

- `galp/examples/public_api_reader.cpp`: includes only `<galp/galp.hpp>`, prints
  rowgroup/column metadata, and reads the first column span.
- `galp/examples/cmake-consumer/CMakeLists.txt`: demonstrates
  `find_package(Galp CONFIG REQUIRED)` and `target_link_libraries(... Galp::core)`.

## Tests

Run all tests:

```bash
ctest --test-dir build --output-on-failure
```

Common focused subsets:

```bash
ctest --test-dir build -L public-api --output-on-failure
ctest --test-dir build -L gpu --output-on-failure
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
./build/galp/tools/galp_cli read_table data/fls/galp-test/data.fls /tmp/galp_out.csv
./build/galp/tools/galp_cli benchmark data/fls/galp-test/data.fls --samples 5
```

Read one rowgroup:

```bash
./build/galp/tools/galp_cli read_table data/fls/galp-test/data.fls /tmp/galp_out.csv --rowgroup 0
```

The default benchmark mode is write-back-free and consume-only: decoded
registers are consumed without writing global output. Add `--include-materialize`
to measure the output-producing path:

```bash
./build/galp/tools/galp_cli benchmark data/fls/galp-test/data.fls --samples 5 --include-materialize
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

## Benchmark Semantics

The main path is fused rowgroup prefetch:

```bash
PREFETCH_DEPTH=4 PREFETCH_WORKERS=0 STREAM_MAX_ROWGROUPS=1 \
  bash scripts/my_tool/run_bench.sh
```

`PREFETCH_WORKERS=0` lets `galp_cli` choose the worker count from the rowgroup
count. `--samples` controls independent benchmark samples and defaults to 5.
`benchmark_wall_ms`, `query_wall_ms`, and `kernel_event_ms` report the median
sample and also expose `_min`, `_median`, and `_mean` fields. `--kernel-samples`
controls kernel replay inside each sample and defaults to 1.

Local helper scripts:

- `scripts/my_tool/run_bench.sh`: default end-to-end benchmark, writes CSV files
  under `scripts/my_tool/end2end_res/`.
- `scripts/my_tool/run_bench_code_cache.sh`: cold/warm cache comparisons using
  `CACHE_DROP_MODE`.
- `scripts/my_tool/run_galp_io_sweep.sh`: IO and prefetch parameter matrix.
- `scripts/my_tool/run_ncu.sh`: NCU profiler entry point.
- `scripts/my_tool/run_nsys.sh`: NSYS profiler entry point.
- `scripts/my_tool/diff.sh`: compares `galp_cli` output with FastLanes
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

`galp/benchmark/generated-bindings` contains generated outputs. When changing
generated kernel bindings, edit the generators under `galp/code-generators` and
commit the regenerated outputs in the same change.

`GALP_ENABLE_MULTI_COLUMN=ON` intentionally fails during configure.
Multi-column generated `query_multi_column` translation units compile too slowly
for the supported GALP benchmark build. Those generated files remain in the tree
temporarily for later redesign or removal.

## Development Rules

- Public consumers should depend only on `<galp/galp.hpp>` and `Galp::core`.
- `src/include` is a private include surface.
- Public headers must not leak `engine`, `compression`, `decompression`,
  `memory`, `benchmark`, or `nvcomp` implementation paths.
- For execution/runtime/memory changes, build `galp_core` and `galp_cli`, then
  run the public API tests.
- For public API or package changes, run `GalpPublicApiSmoke`,
  `GalpPublicApiBoundaries`, and `GalpPackageConsumerSmoke`.
- For generated binding changes, keep generator edits and regenerated outputs in
  the same change.

## Direction

The current work is about tightening boundaries rather than changing the core
decompression algorithms:

- A small stable facade now fronts the internal CUDA decompression implementation.
- Direct exposure of reader, dispatch, compression, and decompression internals
  is being reduced in favor of `galp::*` public types.
- nvCOMP and benchmark support are optional build slices.
- CUDA resources are moving toward RAII stream/event/pool/workset ownership.
- The default benchmark path is write-back-free for pipeline tuning.

Further modularization should keep shrinking `engine/reader.cuh` and
`engine/runtime/pipeline.cuh`, while hardening the boundaries between reader,
zero-copy planning, compression column construction, resource preparation,
prefetch integration, and chunk execution.
