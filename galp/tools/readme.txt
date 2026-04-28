GALP CLI Quick Usage

Replace the placeholders below:
- <GALP_CLI> = /path/to/FastLanes/build/galp/tools/galp_cli
- <FLS_FILE> = /path/to/FastLanes/data/fls/galp-test/data.fls

1) Decompress full table to CSV (all optimizations enabled by default)
<GALP_CLI> read_table <FLS_FILE> /tmp/out.csv

2) Decompress one rowgroup only
<GALP_CLI> read_table <FLS_FILE> /tmp/out.csv --rowgroup 0

3) Benchmark (default: streaming + mixed-dispatch + zero-copy rowgroup materialization)
<GALP_CLI> benchmark <FLS_FILE> --samples 100

4) Benchmark per-rowgroup worksets
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --per-rowgroup-workset

5) Benchmark with typed-batch launches instead of mixed-dispatch
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --no-mixed-dispatch

6) Benchmark with global write-back enabled
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --write-back

7) Benchmark with custom streaming chunk thresholds
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --stream-target-work-items 131072 --stream-max-rowgroups 4

8) Frequency: branchless patcher (extended + PrefetchAllBranchless)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --freq-patcher branchless

9) Frequency: hybrid selection (stateful/branchless by exception density)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --freq-patcher hybrid:6

Benchmark output metrics
  benchmark_wall_ms                End-to-end wall clock of the whole benchmark run.
  kernel_event_ms                  Accumulated GPU event time spent inside benchmark kernel launches.
  read_rowgroup_ms                 Aggregated host-side rowgroup load time.
  file_read_ms                    File IO time only.
  rowgroup_build_ms               Host-side rowgroup/column build time after IO.
  append_expr_ms / upload_workset_ms / ...
                                   Remaining host-side stage timings accumulated by stage.
  output_arena_bytes              Total device output bytes reserved across chunks/worksets.
  prefetch_wait_ms                 Time the consumer waited for background rowgroup prefetch.
  Notes:
  - In streaming mode, host stages and GPU execution can overlap, so the stage sums may exceed
    benchmark_wall_ms.
  - samples scales kernel_event_ms, but does not multiply the host build/upload/release stages,
    because the workset is built once and the kernel is replayed samples times.
  - Aliases end_to_end_ms and kernel_ms are also printed.

Options
  --rowgroup N                      Only process the given rowgroup.
  --samples N                       Number of benchmark repetitions (default: 1).
  --no-header                       Skip CSV header (read_table mode).
  --out PATH                        Output CSV path (read_table mode).
  --iters N                         Launch measurement iterations (default: 100000).
  --grid N                          Launch grid size for measurement (default: 1).
  --block N                         Launch block size for measurement (default: 1).
  --per-rowgroup-workset            Use one independent workset per rowgroup.
  --no-mixed-dispatch               Use typed-batch launches (one kernel per active type per sample)
                                    instead of mixed-dispatch (default: mixed-dispatch, one launch).
  --no-rowgroup-prefetch           Disable background rowgroup prefetch in whole-table benchmark.
  --prefetch-depth N               Number of rowgroups to prefetch ahead (default: 2).
  --prefetch-workers N             Number of background rowgroup prefetch workers (default: 2).
  --write-back                      Force benchmark kernels to write decompressed outputs to global memory.
  --stream-target-work-items N      Chunk flush threshold by work_items in whole-table streaming
                                    benchmark (default: 262144).
  --stream-max-rowgroups N          Chunk flush threshold by rowgroup count in whole-table streaming
                                    benchmark (default: 8, 0 disables rowgroup-cap flushing).
  --freq-patcher MODE               FREQ patcher mode: stateful, branchless, or hybrid[:threshold].
Environment Variables
  GALP_DISABLE_ASYNC_H2D=1          Disable dedicated h2d_stream entirely; GPU allocation and
                                    H2D transfers fall back to the default stream.

Defaults
  All optimizations are enabled by default:
  - Mixed-dispatch (single kernel launch per sample per chunk)
  - Streaming double-buffer pipeline (async overlap of H2D and kernel)
  - Background rowgroup prefetch in whole-table benchmark mode
  - Zero-copy rowgroup parsing (host columns point into backing buffer)
  - Shared DeviceArena per workset / chunk (single staged allocation and upload for appended expressions)
  - Async h2d_stream for overlapping uploads with compute

  To run a launch-heavy baseline:
    GALP_DISABLE_ASYNC_H2D=1 \
    <GALP_CLI> benchmark <FLS_FILE> --samples 100 \
      --per-rowgroup-workset --no-mixed-dispatch

Notes
- `--per-rowgroup-workset` processes each rowgroup as an independent workset (no cross-rowgroup batching).
- Frequency patcher defaults to the stateful path. Use `--freq-patcher branchless` for the branchless
  path, or `--freq-patcher hybrid[:threshold]` to select per-column by exception density.
- Set GALP_MEASURE_H2D=1 to synchronously measure upload_dma_gpu_ms with CUDA events. This is a
  diagnostic mode and disables some H2D/compute overlap while measuring.
- GALP_PINNED_ROWGROUP_PREWARM_BYTES caps rowgroup pinned-buffer prewarming (default: 536870912).
  Set it to 0 to disable startup prewarming. The pipeline prewarms only active prefetch slots.

H2D Transfer Architecture
  A workset uses a shared DeviceArena to aggregate all appended expressions into one
  staged upload. Each column packs its sub-arrays via copy_to_device(arena, out), and
  arena.upload() performs one aggregated allocation/upload for the workset. Execution
  metadata (DeviceExpression arrays, work items, mixed slots) is packed into the same
  arena so the entire workset is transferred in a single H2D operation. Allocation sizes are rounded
  to power-of-2 buckets (min 64KB) for DevicePool cache efficiency.
