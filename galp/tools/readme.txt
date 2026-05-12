GALP CLI Quick Usage

Replace the placeholders below:
- <GALP_CLI> = /path/to/FastLanes/build/galp/tools/galp_cli
- <FLS_FILE> = /path/to/FastLanes/data/fls/galp-test/data.fls

1) Decompress full table to CSV (all optimizations enabled by default)
<GALP_CLI> read_table <FLS_FILE> /tmp/out.csv

2) Decompress one rowgroup only
<GALP_CLI> read_table <FLS_FILE> /tmp/out.csv --rowgroup 0

3) Benchmark (default: write-back-free streaming + mixed-dispatch + zero-copy rowgroup parsing)
<GALP_CLI> benchmark <FLS_FILE> --samples 5

4) Benchmark per-rowgroup worksets
<GALP_CLI> benchmark <FLS_FILE> --samples 5 --per-rowgroup-workset

5) Benchmark with typed-batch launches instead of mixed-dispatch
<GALP_CLI> benchmark <FLS_FILE> --samples 5 --no-mixed-dispatch

6) Benchmark including host materialization
<GALP_CLI> benchmark <FLS_FILE> --samples 5 --include-materialize

7) Benchmark with custom streaming chunk thresholds
<GALP_CLI> benchmark <FLS_FILE> --samples 5 --stream-target-work-items 131072 --stream-max-rowgroups 4

8) Frequency: branchless patcher (extended + PrefetchAllBranchless)
<GALP_CLI> benchmark <FLS_FILE> --samples 5 --freq-patcher branchless

9) Frequency: hybrid selection (stateful/branchless by exception density)
<GALP_CLI> benchmark <FLS_FILE> --samples 5 --freq-patcher hybrid:6

10) Write per-rowgroup pipeline timeline while benchmarking
  GALP_ROWGROUP_TIMELINE_CSV=/tmp/galp_timeline.csv \
    <GALP_CLI> benchmark <FLS_FILE> --samples 1 --prefetch-workers 0 --prefetch-depth 4 --stream-max-rowgroups 1

Benchmark output metrics
  benchmark_wall_ms                End-to-end wall clock of the whole benchmark run.
  benchmark_wall_ms_min/median/mean
                                   Summary over independent benchmark samples.
  resource_prepare_ms              Reader/pinned-pool setup time before a steady-state query
                                   when --reuse-table-resources is enabled.
  query_wall_ms                    Query wall clock; excludes resource_prepare_ms in reuse mode.
  query_wall_ms_min/median/mean    Summary over independent benchmark samples.
  pipeline_active_ms               query_wall_ms minus pipeline setup inside the query.
  kernel_event_ms                  Accumulated GPU event time spent inside benchmark kernel launches.
  kernel_event_ms_min/median/mean  Summary over independent benchmark samples.
  read_rowgroup_ms                 Aggregated host-side rowgroup load time.
  file_read_ms                    Aggregated legacy read/setup stage before rowgroup build.
  rowgroup_build_ms               Aggregated host-side rowgroup/column build time after IO.
  pinned_acquire_ms               Pinned rowgroup buffer lease time before file IO.
  pread_ms                        Time spent in File::ReadRangeUnchecked / pread.
  pread_wall_ms                   Wall span from first pread start to last pread end.
  zero_copy_view_setup_ms         Zero-copy RowgroupView/descriptor setup after pread.
  read_wall_ms                    Wall span from first rowgroup load start to last rowgroup build end.
  file_read_wall_ms               Wall span of the legacy file_read_ms stage.
  storage_bytes                   Sum of rowgroup descriptor m_size bytes read from the FLS file.
  storage_read_gbps               storage_bytes / pread_wall_ms, using decimal GB/s.
  append_expr_ms / upload_workset_ms / ...
                                   Remaining host-side stage timings accumulated by stage.
  output_arena_bytes              Total device output bytes reserved across chunks/worksets.
  write_back / include_materialize
                                  1 when --include-materialize is enabled.
  consume_only / write_back_free  1 only for the default no-materialize benchmark path.
  prefetch_wait_ms                 Time the consumer waited for background rowgroup prefetch.
  prefetch_depth_block_ms          Time prefetch workers spent blocked by prefetch_depth back-pressure.
  Notes:
  - In streaming mode, host stages and GPU execution can overlap, so the stage sums may exceed
    benchmark_wall_ms.
  - storage_read_gbps uses pread_wall_ms so it excludes pageable allocation and zero-copy setup.
  - samples is the number of independent benchmark samples used for min/median/mean.
  - kernel_samples scales kernel_event_ms within each sample, but does not multiply the host
    build/upload/release stages, because the workset is built once and the kernel is replayed
    kernel_samples times.
  - Aliases end_to_end_ms and kernel_ms are also printed.

Options
  --rowgroup N                      Only process the given rowgroup.
  --samples N                       Number of independent benchmark samples for min/median/mean
                                    (default: 5).
  --kernel-samples N                Kernel replays inside each benchmark sample (default: 1).
  --no-header                       Skip CSV header (read_table mode).
  --out PATH                        Output CSV path (read_table mode).
  --iters N                         Launch measurement iterations (default: 100000).
  --grid N                          Launch grid size for measurement (default: 1).
  --block N                         Launch block size for measurement (default: 1).
  --per-rowgroup-workset            Use one independent workset per rowgroup.
  --no-mixed-dispatch               Use typed-batch launches (one kernel per active type per sample)
                                    instead of mixed-dispatch (default: mixed-dispatch, one launch).
  --no-rowgroup-prefetch           Disable background rowgroup prefetch in whole-table benchmark.
  --prefetch-depth N               Number of rowgroups to prefetch ahead (default: 4).
  --prefetch-workers N             Number of background rowgroup prefetch workers
                                    (default: 0=auto by rowgroup count).
  --max-prefetch-storage-bytes N    Fused prefetch compressed-byte budget
                                    (default: prefetch_depth * max rowgroup bytes).
  --include-materialize             Benchmark the output-producing path by enabling device
                                    write-out and pinned D2H result materialization.
  --reuse-table-resources           Prepare reader and pinned rowgroup pool before the query timer;
                                    useful for steady-state measurements.
  --stream-target-work-items N      Chunk flush threshold by work_items in whole-table streaming
                                    benchmark (default: 262144).
  --stream-max-rowgroups N          Chunk flush threshold by rowgroup count in whole-table streaming
                                    benchmark (default: 1, 0 disables rowgroup-cap flushing).
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
  - Write-back-free benchmark kernels (decoded registers are consumed, not written to global output)
  - Shared DeviceArena per workset / chunk (single staged allocation and upload for appended expressions)
  - Async h2d_stream for overlapping uploads with compute

  To run a launch-heavy baseline:
    GALP_DISABLE_ASYNC_H2D=1 \
    <GALP_CLI> benchmark <FLS_FILE> --samples 5 --kernel-samples 100 \
      --per-rowgroup-workset --no-mixed-dispatch

Notes
- `--per-rowgroup-workset` processes each rowgroup as an independent workset (no cross-rowgroup batching).
- Frequency patcher defaults to the stateful path. Use `--freq-patcher branchless` for the branchless
  path, or `--freq-patcher hybrid[:threshold]` to select per-column by exception density.
- Set GALP_MEASURE_H2D=1 to synchronously measure upload_dma_gpu_ms with CUDA events. This is a
  diagnostic mode and disables some H2D/compute overlap while measuring.
- GALP_PINNED_ROWGROUP_PREWARM_BYTES caps rowgroup pinned-buffer prewarming (default: 536870912).
- GALP_PINNED_ROWGROUP_PREWARM_MODE=full restores eager prewarming of the whole prefetch working set;
  off/none/lazy disables rowgroup pinned-buffer prewarming.
- GALP_PINNED_ROWGROUP_PREWARM_SLOTS=N overrides the adaptive prewarm slot count.
- GALP_ROWGROUP_TIMELINE_CSV writes a per-rowgroup CSV with read/build/ready/upload timestamps and queue gaps.
  By default the pipeline prewarms the stable working set: consumer-held rowgroups, active IO/build owners,
  and configured prefetch-depth slots.

H2D Transfer Architecture
  A workset uses a shared DeviceArena to aggregate all appended expressions into one
  staged upload. Each column packs its sub-arrays via copy_to_device(arena, out), and
  arena.upload() performs one aggregated allocation/upload for the workset. Execution
  metadata (DeviceExpression arrays, work items, mixed slots) is packed into the same
  arena so the entire workset is transferred in a single H2D operation. Allocation sizes are rounded
  to power-of-2 buckets (min 64KB) for DevicePool cache efficiency.
