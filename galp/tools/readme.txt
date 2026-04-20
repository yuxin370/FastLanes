GALP CLI Quick Usage

Replace the placeholders below:
- <GALP_CLI> = /path/to/FastLanes/build/galp/tools/galp_cli
- <FLS_FILE> = /path/to/FastLanes/data/fls/galp-test/data.fls

1) Decompress full table to CSV (all optimizations enabled by default)
<GALP_CLI> read_table <FLS_FILE> /tmp/out.csv

2) Decompress one rowgroup only
<GALP_CLI> read_table <FLS_FILE> /tmp/out.csv --rowgroup 0

3) Benchmark (default: streaming + mixed-dispatch + zero-copy)
<GALP_CLI> benchmark <FLS_FILE> --samples 100

4) Benchmark per-rowgroup (non-mega path)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --no-mega-kernel

5) Benchmark without streaming (synchronous chunked execution)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --no-streaming

6) Benchmark with typed-batch launches instead of mixed-dispatch
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --no-mixed-dispatch

7) Benchmark without zero-copy parsing
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --no-zero-copy

8) Benchmark with global write-back enabled
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --write-back

9) Benchmark with launch-overhead estimate
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --estimate-launch --launch-iters 10000

10) Benchmark one rowgroup with launch-overhead estimate
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --estimate-launch --launch-iters 10000 --rowgroup 0

11) Benchmark with custom streaming chunk thresholds
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --stream-target-work-items 131072 --stream-max-rowgroups 4

12) Baseline: disable overlap-heavy execution features
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --no-streaming --no-zero-copy
(with env: GALP_DISABLE_ASYNC_H2D=1)

13) Frequency: branchless patcher (extended + PrefetchAllBranchless)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --freq-prefetch-all-branchless

14) Frequency: hybrid selection (stateful/branchless by exception density)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --freq-prefetch-all-branchless --freq-hybrid-patcher --freq-branchless-threshold 6

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
  --estimate-launch                 Estimate launch overhead during benchmark.
  --launch-iters N                  Iterations for launch estimate (default: 10000).
  --no-mega-kernel                  Use per-rowgroup worksets instead of whole-table aggregation.
  --no-streaming                    Keep whole-table aggregation but use synchronous chunked execution
                                    instead of the default double-buffered streaming pipeline.
  --no-mixed-dispatch               Use typed-batch launches (one kernel per active type per sample)
                                    instead of mixed-dispatch (default: mixed-dispatch, one launch).
  --no-zero-copy                    Disable zero-copy rowgroup parsing; use the traditional
                                    parse-and-copy path (default: zero-copy enabled).
  --no-rowgroup-prefetch           Disable background rowgroup prefetch in whole-table benchmark.
  --prefetch-depth N               Number of rowgroups to prefetch ahead (default: 2).
  --write-back                      Force benchmark kernels to write decompressed outputs to global memory.
  --stream-target-work-items N      Chunk flush threshold by work_items in whole-table streaming
                                    benchmark (default: 262144).
  --stream-max-rowgroups N          Chunk flush threshold by rowgroup count in whole-table streaming
                                    benchmark (default: 8, 0 disables rowgroup-cap flushing).
  --freq-prefetch-all-branchless    Use FREQ extended format with PrefetchAllBranchless patcher.
  --freq-hybrid-patcher             Enable per-column selection between stateful and branchless FREQ
                                    patcher. Requires --freq-prefetch-all-branchless.
  --freq-branchless-threshold N     Hybrid cutoff: average exceptions per vector (default: 6).
                                    Requires both --freq-prefetch-all-branchless and --freq-hybrid-patcher.
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

  To run a pure baseline, disable everything:
    GALP_DISABLE_ASYNC_H2D=1 \
    <GALP_CLI> benchmark <FLS_FILE> --samples 100 \
      --no-streaming --no-zero-copy --no-mixed-dispatch

Notes
- `--no-zero-copy` only affects host-side parsing; the H2D transfer to GPU is unchanged.
- `--no-streaming` disables the streaming overlap but still groups rowgroups into chunks.
- `--no-mega-kernel` processes each rowgroup as an independent workset (no cross-rowgroup batching).
- Additional accepted aliases: --gpu-dispatch-kernel, --zero-copy-parse, --mega-kernel-no-stream.
- Frequency patcher defaults to the stateful path. Pass `--freq-prefetch-all-branchless` to switch to
  branchless, and additionally `--freq-hybrid-patcher` to let the engine pick per-column based on
  exception density vs `--freq-branchless-threshold`.

H2D Transfer Architecture
  A workset uses a shared DeviceArena to aggregate all appended expressions into one
  staged upload. Each column packs its sub-arrays via copy_to_device(arena, out), and
  arena.upload() performs one aggregated allocation/upload for the workset. Execution
  metadata (DeviceExpression arrays, work items, mixed slots) is packed into the same
  arena so the entire workset is transferred in a single H2D operation. Allocation sizes are rounded
  to power-of-2 buckets (min 64KB) for DevicePool cache efficiency.
