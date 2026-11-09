# Native command reference

For the standalone JPEG dataset commands (`python -m fastlens convert`,
`inspect`, and `annotations build/inspect`), see
[conversion](../docs/guides/CONVERSION.md), [Store format](../docs/spec/STORE.md)
and [annotation formats and premixing](../docs/guides/ANNOTATIONS.md).
The commands below document the lower-level native tools.

## Command examples

```text
Replace the placeholders below:
- <FASTLENS_CLI> = /path/to/FastLanes/build/fastlens/tools/fastlens_cli
- <FLS_FILE> = /path/to/FastLanes/data/fls/fastlens-test/data.fls

1) Decompress full table to CSV (all optimizations enabled by default)
<FASTLENS_CLI> read_table <FLS_FILE> /tmp/out.csv

2) Decompress one rowgroup only
<FASTLENS_CLI> read_table <FLS_FILE> /tmp/out.csv --rowgroup 0

3) Benchmark (default: write-back-free streaming + mixed-dispatch + zero-copy rowgroup parsing)
<FASTLENS_CLI> benchmark <FLS_FILE> --samples 5

4) Benchmark per-rowgroup worksets
<FASTLENS_CLI> benchmark <FLS_FILE> --samples 5 --per-rowgroup-workset

5) Benchmark with typed-batch launches instead of mixed-dispatch
<FASTLENS_CLI> benchmark <FLS_FILE> --samples 5 --no-mixed-dispatch

6) Benchmark including host materialization
<FASTLENS_CLI> benchmark <FLS_FILE> --samples 5 --include-materialize

7) Benchmark with custom streaming chunk thresholds
<FASTLENS_CLI> benchmark <FLS_FILE> --samples 5 --stream-target-work-items 131072 --stream-max-rowgroups 4

8) Frequency: branchless patcher (extended + PrefetchAllBranchless)
<FASTLENS_CLI> benchmark <FLS_FILE> --samples 5 --freq-patcher branchless

9) Frequency: hybrid selection (stateful/branchless by exception density)
<FASTLENS_CLI> benchmark <FLS_FILE> --samples 5 --freq-patcher hybrid:6

10) Write per-rowgroup pipeline timeline while benchmarking
  FASTLENS_ROWGROUP_TIMELINE_CSV=/tmp/fastlens_timeline.csv \
    <FASTLENS_CLI> benchmark <FLS_FILE> --samples 1 --prefetch-workers 0 --prefetch-depth 4 --stream-max-rowgroups 1

11) JPEG DCT pipeline benchmark, compare pushdown with full-decode-then-crop
<FASTLENS_CLI> pipeline_benchmark <manifest.bin> --crop 64 64 512 512 --mode compare

12) JPEG DCT pushdown only
<FASTLENS_CLI> pipeline_benchmark <manifest.bin> --crop 64 64 512 512 --mode pushdown

13) JPEG DCT baseline only: full decode, then crop the decoded DCT blocks
<FASTLENS_CLI> pipeline_benchmark <manifest.bin> --crop 64 64 512 512 --mode baseline

14) JPEG DCT auto policy: choose pushdown or full-then-crop per window
<FASTLENS_CLI> pipeline_benchmark <manifest.bin> --crop 64 64 512 512 --mode auto

15) JPEG DCT coefficient pushdown: compare selected DCT coefficients only
<FASTLENS_CLI> pipeline_benchmark <manifest.bin> --crop 64 64 512 512 --dct-coeffs list:0,2,5 --mode compare

16) JPEG DCT coefficient pushdown benchmark: keep crop pushdown fixed, compare against post-decode coefficient selection
<FASTLENS_CLI> pipeline_benchmark <manifest.bin> --crop 64 64 512 512 --dct-coeffs first:8 --mode dct-compare
```

## JPEG DCT pushdown validation matrix

Use `compare` for correctness: it runs vector-level crop pushdown and
full-decode-then-crop over the same windows and returns a non-zero exit code
only on coefficient mismatch. With `--dct-coeffs`, compare mode validates the
selected output against the corresponding coefficient subset from the full
64-coefficient baseline. Keep verification enabled for correctness runs; use
`--no-verify` only for timing-only sweeps after correctness has already been
checked for the same dataset, crop, window size, coefficient selection, and
cache setting.

Recommended GPU validation commands:

```bash
# Large-image or medium-image benefit check: pushdown should reduce decode work.
<FASTLENS_CLI> pipeline_benchmark <large-manifest.bin> \
  --crop 30 40 224 224 \
  --window-images 128 \
  --cache-capacity-mib 1024 \
  --mode compare

# Small-image fixed-overhead pressure check: auto may choose full-then-crop.
<FASTLENS_CLI> pipeline_benchmark <small-manifest.bin> \
  --crop 0 0 16 16 \
  --window-images 128 \
  --cache-capacity-mib 1024 \
  --mode compare

# Runtime policy check: verify the per-window decision is not dataset-name based.
<FASTLENS_CLI> pipeline_benchmark <manifest.bin> \
  --crop 0 0 16 16 \
  --window-images 128 \
  --cache-capacity-mib 1024 \
  --mode auto

# DCT coefficient pushdown: crop plus selected coefficient columns.
<FASTLENS_CLI> pipeline_benchmark <manifest.bin> \
  --crop 30 40 224 224 \
  --window-images 128 \
  --dct-coeffs list:0,2,5 \
  --mode compare

# DCT coefficient pushdown without crop: full-image selected coefficient columns.
<FASTLENS_CLI> pipeline_benchmark <manifest.bin> \
  --window-images 128 \
  --dct-coeffs list:0,2,5 \
  --mode compare

# Isolated DCT coefficient pushdown benchmark: both sides use crop pushdown.
<FASTLENS_CLI> pipeline_benchmark <manifest.bin> \
  --crop 30 40 224 224 \
  --window-images 128 \
  --dct-coeffs first:8 \
  --mode dct-compare
```

Native pipeline commands emit their own metrics. Retain stdout when comparing runs;
see the [diagnostic metrics reference](DIAGNOSTICS.md) for interpretation.


Record at least the following fields for every reported dataset/crop:

| dataset | image size | crop size | mode | outputs_match | dct coeffs | pushdown_selected_vector_ratio | full_then_crop_selected_vector_ratio | pushdown_total_ms | full_then_crop_total_ms | pushdown_speedup_vs_full_then_crop | pushdown_saved_ms_vs_full_then_crop | pushdown_plan_ms | pushdown_read_decode_ms | pushdown_decode_ms | pushdown_gather_ms | pushdown_decoded_gather_ms | pushdown_cached_gather_ms | pushdown_sync_rowgroup_read_ms | pushdown_prefetch_queue_start_ms | pushdown_prefetch_wait_ms | pushdown_prefetch_depth_block_ms | pushdown_prefetch_rowgroup_read_ms | pushdown_prefetch_ready_ahead_ms | pushdown_prefetch_initial_cache_hit_rowgroup_count | pushdown_prefetch_candidate_rowgroup_count | pushdown_prefetch_active_shard_count | pushdown_prefetch_config_disabled_shard_count | pushdown_prefetch_all_hit_shard_count | pushdown_prefetch_small_batch_disabled_shard_count | pushdown_prefetch_selected_vector_disabled_shard_count | pushdown_prefetch_selected_vector_miss_rowgroup_count | pushdown_prefetch_initial_hit_runtime_miss_count | pushdown_prefetch_skipped_repeated_runtime_miss_count | pushdown_prefetched_rowgroup_count | pushdown_prefetch_consumed_as_hit_count | pushdown_prefetch_skipped_repeated_rowgroup_count | pushdown_prefetch_consumed_as_hit_read_ms | pushdown_prefetch_consumed_as_hit_wait_ms | pushdown_workset_count | pushdown_decode_kernel_launch_count | pushdown_gather_kernel_launch_count | pushdown_scratch_allocation_count | pushdown_internal_sync_count | pushdown_runtime_policy_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

For `auto` runs also report `auto_pushdown_windows`,
`auto_full_then_crop_windows`, `auto_no_dct_pushdown_windows`,
`auto_policy_selected_vector_ratio`,
`auto_policy_estimated_pushdown_worksets`,
`auto_policy_estimated_full_worksets`, `auto_policy_coefficient_pushdown_windows`,
and `auto_policy_reason`. The default summary includes the full auto policy
field set. The policy is expected to make decisions from selected/full vectors,
block ratio, rowgroup ratio, estimated worksets, and estimated gather items; it
should not depend on dataset names.


## Options, environment and defaults

```text
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
  --crop x y w h                    Pixel-space JPEG source crop for pipeline_benchmark. In
                                    --mode pushdown, the crop is pushed into JPEG DCT/FastLanes
                                    planning and workset scheduling. In --mode baseline
                                    (full-then-crop), the benchmark decodes full images first and
                                    then selects the same cropped DCT blocks from the full output.
  --window-images N                 Images per pipeline_benchmark window (default: 256).
  --cache-capacity-mib N            Decoded rowgroup cache capacity for JPEG DCT pipeline.
  --decode-batch-rowgroups N        Rowgroups per JPEG DCT decode workset (default: 64).
                                    Larger values can reduce small workset launches and batch-end
                                    syncs when the window has many tiny touched rowgroups.
  --no-jpeg-device-rowgroup-prefetch
                                    Disable JPEG DCT device rowgroup prefetch.
  --jpeg-device-prefetch-depth N    JPEG DCT device rowgroup prefetch queue depth (default: 4).
  --jpeg-device-prefetch-workers N  JPEG DCT device prefetch worker threads (default: 1).
  --jpeg-device-prefetch-min-batches N
                                    Minimum miss decode batches before device prefetch starts
                                    (default: 2).
  --dct-coeffs SPEC                 DCT coefficient pushdown for pipeline_benchmark:
                                    all, first:N, or list:0,1,...
  --mode MODE                       pipeline_benchmark mode:
                                    compare runs pushdown and full-then-crop and verifies matching
                                    cropped DCT coefficients;
                                    pushdown runs only crop-pushed decode;
                                    baseline/full-then-crop runs only full decode followed by
                                    cropped block selection;
                                    auto chooses pushdown or full-then-crop per window from
                                    CPU crop/full prepared plans, keeping small windows on
                                    full-then-crop to avoid fixed pushdown overhead.
  --no-verify                       Skip coefficient equality validation in compare mode.
Environment Variables
  FASTLENS_DISABLE_ASYNC_H2D=1          Disable dedicated h2d_stream entirely; GPU allocation and
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
    FASTLENS_DISABLE_ASYNC_H2D=1 \
    <FASTLENS_CLI> benchmark <FLS_FILE> --samples 5 --kernel-samples 100 \
      --per-rowgroup-workset --no-mixed-dispatch
```
