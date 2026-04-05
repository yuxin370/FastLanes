GALP CLI Quick Usage

Replace the placeholders below:
- <GALP_CLI> = /path/to/FastLanes/build/galp/tools/galp_cli
- <FLS_FILE> = /path/to/FastLanes/data/fls/galp-test/data.fls

1) Decompress full table to CSV
<GALP_CLI> read_table <FLS_FILE> /tmp/out.csv

2) Decompress one rowgroup only
<GALP_CLI> read_table <FLS_FILE> /tmp/out.csv --rowgroup 0

3) Benchmark (default: streaming mega-kernel, typed-batch launches)
<GALP_CLI> benchmark <FLS_FILE> --samples 100

4) Benchmark per-rowgroup (non-mega path)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --no-mega-kernel

5) Benchmark with GPU-side mixed dispatch (one kernel launch per sample)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --gpu-dispatch-kernel

6) Benchmark with global write-back enabled
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --gpu-dispatch-kernel --write-back

7) Benchmark with launch-overhead estimate
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --estimate-launch --launch-iters 10000

8) Benchmark one rowgroup with launch-overhead estimate
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --estimate-launch --launch-iters 10000 --rowgroup 0

9) Benchmark with zero-copy parse path
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --zero-copy-parse

10) Benchmark without streaming (synchronous chunked execution)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --mega-kernel-no-stream

11) Benchmark with custom streaming chunk thresholds
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --stream-target-work-items 131072 --stream-max-rowgroups 4

12) Frequency: branchless patcher (extended + PrefetchAllBranchless)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --gpu-dispatch-kernel --freq-prefetch-all-branchless

13) Frequency: hybrid selection (stateful/branchless by exception density)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --gpu-dispatch-kernel --freq-prefetch-all-branchless --freq-hybrid-patcher --freq-branchless-threshold 6

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
  --mega-kernel-no-stream           Keep whole-table aggregation but use synchronous chunked execution
                                    instead of the default double-buffered streaming pipeline.
  --gpu-dispatch-kernel             Use GPU-side mixed dispatch: packs all types into a single kernel
                                    launch per sample.  Without this flag, benchmark defaults to
                                    typed-batch launches (one launch per active type per sample).
  --write-back                      Force benchmark kernels to write decompressed outputs to global memory.
  --zero-copy-parse                 Use zero-copy rowgroup materialization path; avoids host-side copies
                                    when the source layout supports it (read_table and benchmark modes).
  --stream-target-work-items N      Chunk flush threshold by work_items in whole-table streaming
                                    benchmark (default: 262144).
  --stream-max-rowgroups N          Chunk flush threshold by rowgroup count in whole-table streaming
                                    benchmark (default: 8, 0 disables rowgroup-cap flushing).
  --freq-prefetch-all-branchless    Use FREQ extended format with PrefetchAllBranchless patcher.
  --freq-hybrid-patcher             Enable per-column selection between stateful and branchless FREQ
                                    patcher. Requires --freq-prefetch-all-branchless.
  --freq-branchless-threshold N     Hybrid cutoff: average exceptions per vector (default: 6).
                                    Requires both --freq-prefetch-all-branchless and --freq-hybrid-patcher.

Notes
- Default whole-table benchmark uses typed-batch launches (one kernel per active type per sample)
  with a streaming double-buffer pipeline: rowgroups are grouped into chunks (by
  --stream-target-work-items / --stream-max-rowgroups), each chunk is uploaded and launched
  asynchronously while the next chunk is prepared on the host.
- `--gpu-dispatch-kernel` switches to GPU-side mixed dispatch (one kernel launch per sample).
- `--mega-kernel-no-stream` disables the streaming overlap but still groups rowgroups into chunks.
- `--no-mega-kernel` processes each rowgroup as an independent workset (no cross-rowgroup batching).
- Note: read_table mode always uses mixed dispatch internally.
- `--write-back` forces benchmark kernels to write decompressed outputs to global memory.
- `--zero-copy-parse` enables `read_rowgroup_zero_copy_materialized` path: the rowgroup is read into a
  single backing buffer and host column structs point directly into it (with alignment fallback copies).
- Frequency patcher defaults to the stateful path. Pass `--freq-prefetch-all-branchless` to switch to
  branchless, and additionally `--freq-hybrid-patcher` to let the engine pick per-column based on
  exception density vs `--freq-branchless-threshold`.
