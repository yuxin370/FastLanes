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

12) Baseline: all optimizations disabled
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --no-streaming --no-zero-copy
(with env: GALP_DISABLE_ASYNC_H2D=1 GALP_DISABLE_BATCH_UPLOADER=1)

13) Frequency: branchless patcher (extended + PrefetchAllBranchless)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --freq-prefetch-all-branchless

14) Frequency: hybrid selection (stateful/branchless by exception density)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --freq-prefetch-all-branchless --freq-hybrid-patcher --freq-branchless-threshold 6

Benchmark output metrics
  benchmark_wall_ms                End-to-end wall clock of the whole benchmark run.
  kernel_event_ms                  Accumulated GPU event time spent inside benchmark kernel launches.
  read_rowgroup_ms / append_expr_ms / upload_workset_ms / ...
                                   Host-side stage timings accumulated by stage.
  Notes:
  - In streaming mode, host stages and GPU execution can overlap, so the stage sums may exceed
    benchmark_wall_ms.
  - samples scales kernel_event_ms, but does not multiply the host build/upload/release stages,
    because the workset is built once and the kernel is replayed samples times.
  - Backward-compatible aliases end_to_end_ms and kernel_ms are still printed for older scripts.

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

Environment Variables (A/B testing)
  GALP_DISABLE_CHUNK_ARENA=1        Disable chunk-level super-arena; fall back to per-column
                                    DeviceArena + BatchUploader.
  GALP_DISABLE_BATCH_UPLOADER=1     Disable batched H2D copy coalescing; each column is copied
                                    individually via cudaMemcpyAsync on the h2d_stream.
  GALP_DISABLE_ASYNC_H2D=1          Disable dedicated h2d_stream entirely; all GPU allocation and
                                    H2D transfers fall back to synchronous calls on the default
                                    stream.  Also implicitly disables BatchUploader.

  A/B test matrix:
    (default)                       → chunk-level super-arena (1 malloc + 1 memcpy per chunk)
    GALP_DISABLE_CHUNK_ARENA=1      → per-column arena + BatchUploader
    GALP_DISABLE_BATCH_UPLOADER=1   → per-column arena, no batching
    GALP_DISABLE_ASYNC_H2D=1        → sync default stream, no batching
    both ASYNC_H2D + BATCH set      → same as GALP_DISABLE_ASYNC_H2D=1

Defaults
  All optimizations are enabled by default:
  - Mixed-dispatch (single kernel launch per sample per chunk)
  - Streaming double-buffer pipeline (async overlap of H2D and kernel)
  - Zero-copy rowgroup parsing (host columns point into backing buffer)
  - Chunk-level super-arena (single cudaMalloc + single cudaMemcpy per streaming chunk), per-column DeviceArena as fallback (single cudaMalloc + single cudaMemcpy per column)
  - Async h2d_stream with BatchUploader (coalesced pinned H2D copies, used with per-column arena)

  To run a pure baseline, disable everything:
    GALP_DISABLE_ASYNC_H2D=1 GALP_DISABLE_BATCH_UPLOADER=1 \
    <GALP_CLI> benchmark <FLS_FILE> --samples 100 \
      --no-streaming --no-zero-copy --no-mixed-dispatch

Notes
- `--no-zero-copy` only affects host-side parsing; the H2D transfer to GPU is unchanged.
- `--no-streaming` disables the streaming overlap but still groups rowgroups into chunks.
- `--no-mega-kernel` processes each rowgroup as an independent workset (no cross-rowgroup batching).
- Backward-compatible aliases: --gpu-dispatch-kernel, --zero-copy-parse, --mega-kernel-no-stream.
- Frequency patcher defaults to the stateful path. Pass `--freq-prefetch-all-branchless` to switch to
  branchless, and additionally `--freq-hybrid-patcher` to let the engine pick per-column based on
  exception density vs `--freq-branchless-threshold`.

H2D Transfer Architecture
  Three tiers of H2D coalescing (highest to lowest):

  1. Chunk-level super-arena (default, GALP_DISABLE_CHUNK_ARENA unset):
     All columns in a streaming chunk share ONE DeviceArena. append_expressions()
     creates a single arena, each column packs its sub-arrays via copy_to_device(arena, out),
     and arena.upload() performs ONE cudaMallocAsync + ONE cudaMemcpyAsync for the
     entire chunk. Resolver callbacks populate device column pointers after upload.
     Allocation sizes are rounded to power-of-2 buckets (min 64KB) for DevicePool
     cache efficiency. batch.device_exprs is pre-reserved to guarantee stable
     addresses for resolver callbacks.

  2. Per-column DeviceArena (fallback, GALP_DISABLE_CHUNK_ARENA=1):
     Each column's copy_to_device(stream) creates a local DeviceArena to consolidate
     its sub-arrays (packed data, bit-widths, offsets, exceptions, etc.) into a single
     cudaMallocAsync + single cudaMemcpyAsync per column. Host data is packed into a
     contiguous pinned buffer before transfer. Sub-pointers within the arena are tagged
     as sub-allocations (no-op on cudaFree). Composed column types (FFOR, SLPATCH, RLE,
     DICT*) inline their child columns into the same arena.
     The BatchUploader coalesces per-column H2D calls and shares a single CUDA event.

  3. Raw GPUArray (GALP_DISABLE_BATCH_UPLOADER=1):
     Each sub-array is individually allocated and transferred.

  celebA 1-sample nsys comparison (chunk vs per-column):
    cudaMemcpyAsync:  479 calls / 3.3ms  vs  19,001 calls / 46ms  (-97.5% / -93%)
    cudaMallocAsync:  18,778 / 27ms      vs  37,817 / 45ms        (-50% / -40%)
    cudaMallocHost:   34 / 26ms          vs  537 / 29ms           (-94%)
    End-to-end:       ~522ms             vs  ~585ms               (-11%)
