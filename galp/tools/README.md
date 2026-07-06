# GALP CLI Quick Usage

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

11) JPEG DCT pipeline benchmark, compare pushdown with full-decode-then-crop
<GALP_CLI> pipeline_benchmark <manifest.bin> --crop 64 64 512 512 --mode compare

12) JPEG DCT pushdown only
<GALP_CLI> pipeline_benchmark <manifest.bin> --crop 64 64 512 512 --mode pushdown

13) JPEG DCT baseline only: full decode, then crop the decoded DCT blocks
<GALP_CLI> pipeline_benchmark <manifest.bin> --crop 64 64 512 512 --mode baseline

14) JPEG DCT auto policy: choose pushdown or full-then-crop per window
<GALP_CLI> pipeline_benchmark <manifest.bin> --crop 64 64 512 512 --mode auto

15) JPEG DCT coefficient pushdown: compare selected DCT coefficients only
<GALP_CLI> pipeline_benchmark <manifest.bin> --crop 64 64 512 512 --dct-coeffs list:0,2,5 --mode compare

16) JPEG DCT coefficient pushdown benchmark: keep crop pushdown fixed, compare against post-decode coefficient selection
<GALP_CLI> pipeline_benchmark <manifest.bin> --crop 64 64 512 512 --dct-coeffs first:8 --mode dct-compare

JPEG DCT pushdown validation matrix

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
<GALP_CLI> pipeline_benchmark <large-manifest.bin> \
  --crop 30 40 224 224 \
  --window-images 128 \
  --cache-capacity-mib 1024 \
  --mode compare

# Small-image fixed-overhead pressure check: auto may choose full-then-crop.
<GALP_CLI> pipeline_benchmark <small-manifest.bin> \
  --crop 0 0 16 16 \
  --window-images 128 \
  --cache-capacity-mib 1024 \
  --mode compare

# Runtime policy check: verify the per-window decision is not dataset-name based.
<GALP_CLI> pipeline_benchmark <manifest.bin> \
  --crop 0 0 16 16 \
  --window-images 128 \
  --cache-capacity-mib 1024 \
  --mode auto

# DCT coefficient pushdown: crop plus selected coefficient columns.
<GALP_CLI> pipeline_benchmark <manifest.bin> \
  --crop 30 40 224 224 \
  --window-images 128 \
  --dct-coeffs list:0,2,5 \
  --mode compare

# DCT coefficient pushdown without crop: full-image selected coefficient columns.
<GALP_CLI> pipeline_benchmark <manifest.bin> \
  --window-images 128 \
  --dct-coeffs list:0,2,5 \
  --mode compare

# Isolated DCT coefficient pushdown benchmark: both sides use crop pushdown.
<GALP_CLI> pipeline_benchmark <manifest.bin> \
  --crop 30 40 224 224 \
  --window-images 128 \
  --dct-coeffs first:8 \
  --mode dct-compare
```

Save raw output and generate the reporting table with:

```bash
<GALP_CLI> pipeline_benchmark <manifest.bin> \
  --crop 30 40 224 224 \
  --window-images 128 \
  --cache-capacity-mib 1024 \
  --mode compare | tee pipeline_benchmark.log

python3 scripts/my_tool/bench_pipeline_summary.py \
  --dataset <dataset> --image-size <width>x<height-or-varies> \
  --require-match --require-default-fields \
  pipeline_benchmark.log
```

For JPEG DCT device rowgroup prefetch work, collect an on/off matrix with:

```bash
python3 scripts/my_tool/bench_jpeg_prefetch.py \
  --input data/flower_photos \
  --work-dir /tmp/galp_jpeg_prefetch \
  --mode compare \
  --repeats 3
```

The script generates or reuses a sharded JPEG DCT manifest, runs small and large
window benchmarks with prefetch enabled and disabled, and writes a CSV summary
that includes the prefetch overhead and readiness metrics. `--repeats` controls
independent samples per on/off case; keep the raw samples when judging whether a
small workload regressed or a large miss workload improved. The runner also
writes `jpeg_device_prefetch_pairs.csv`, a paired on/off delta table with wall
time delta, off/on speedup, and the key prefetch wait/readiness/waste metrics.
If raw benchmark logs were already summarized, regenerate only the paired table
with `--pair-summary-only <summary.csv> --pairs-out <pairs.csv>`.
Use `--dry-run` first on machines without a CUDA driver or dataset access to
inspect the exact manifest, benchmark, and summary commands without executing
them.

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
  file_read_ms                    Aggregated read/setup stage before rowgroup build.
  rowgroup_build_ms               Aggregated host-side rowgroup/column build time after IO.
  pinned_acquire_ms               Pinned rowgroup buffer lease time before file IO.
  pread_ms                        Time spent in File::ReadRangeUnchecked / pread.
  pread_wall_ms                   Wall span from first pread start to last pread end.
  zero_copy_view_setup_ms         Zero-copy RowgroupView/descriptor setup after pread.
  read_wall_ms                    Wall span from first rowgroup load start to last rowgroup build end.
  file_read_wall_ms               Wall span of the file_read_ms stage.
  storage_bytes                   Sum of rowgroup descriptor m_size bytes read from the FLS file.
  storage_read_gbps               storage_bytes / pread_wall_ms, using decimal GB/s.
  append_expr_ms / upload_workset_ms / ...
                                   Remaining host-side stage timings accumulated by stage.
  output_arena_bytes              Total device output bytes reserved across chunks/worksets.
  write_back / include_materialize
                                  1 when --include-materialize is enabled.
  consume_only / write_back_free  1 only for the default no-materialize benchmark path.
  *_planned_selected_vector_count FastLanes vectors selected by the crop before runtime policy.
  *_selected_vector_count         FastLanes vectors actually scheduled by the JPEG DCT pipeline.
                                  Dense-cache hits do not contribute here because they skip decode.
  *_full_vector_count             Full-rowgroup vector count used as the denominator for pushdown.
  *_planned_saved_vector_count    full_vector_count - planned_selected_vector_count.
  *_actual_saved_vector_count     full_vector_count - selected_vector_count.
                                  This includes both runtime pushdown fallback decisions and
                                  dense-cache hits that skipped decode.
  *_rowgroup_count                Touched rowgroups in the device batch windows.
  *_planned_selected_vector_ratio planned_selected_vector_count / full_vector_count.
  *_selected_vector_ratio         selected_vector_count / full_vector_count.
                                  With a dense cache this is the scheduled decode ratio, not only
                                  the crop-selected vector ratio.
  *_workset_count                 Number of FastLanes worksets submitted by the JPEG DCT pipeline.
                                  Newly decoded rowgroups are batched up to
                                  --decode-batch-rowgroups per workset (default 64).
                                  JPEG scratch preserves the decode workset's stream, events,
                                  output arena, and chunk arena capacity across these submissions.
  *_decode_kernel_launch_count    FastLanes decode kernel launches.
  *_gather_kernel_launch_count    DCT block-major gather launches. Newly decoded rowgroups use
                                  fused projection materialization; gather remains for dense
                                  decoded-rowgroup cache hits.
  *_cached_gather_kernel_launch_count
                                  Gather launches sourced only from dense decoded-rowgroup cache hits.
  *_materialize_kernel_launch_count
                                  Fused projection and dense decoded-rowgroup cache
                                  materialization launches.
  *_gather_item_count             DCT blocks handled by cached gather kernels.
  *_decoded_gather_item_count     Legacy decoded gather items; expected to stay zero for the
                                  fused projection path.
  *_cached_gather_item_count      Gather items sourced from dense decoded-rowgroup cache hits.
  *_projection_item_count         Compact DCT projection entries materialized directly from
                                  newly decoded FastLanes worksets.
  *_decoded_projection_item_count Projection entries sourced from newly decoded FastLanes
                                  worksets.
  *_workset_upload_count          Workset metadata uploads.
  *_scratch_upload_count          Device scratch metadata uploads. Projection coefficient
                                  pointers, source tags, and projection entries are packed into
                                  reusable scratch arrays.
  *_scratch_allocation_count      Reader-owned reusable device scratch capacity growth events.
                                  Host staging vectors, including pending rowgroup work, are
                                  reused but not counted as device allocations. The reusable
                                  decode workset has its own preserved arenas and is counted by
                                  workset submissions rather than scratch allocation events.
                                  JPEG DCT metadata scratch grows geometrically from a small
                                  initial capacity floor to avoid repeated tiny cudaMalloc/free.
  *_internal_sync_count           Stream-local synchronization points inside the JPEG DCT device path.
                                  Dense-cache-hit gathers can hand off to a following decoded
                                  batch via CUDA event instead of synchronizing immediately;
                                  cached-hit gather items are queued across shard boundaries
                                  within a device batch.
  *_cached_gather_sync_count      Cached-gather completion waits that could not be handed off
                                  to a following decoded batch stream. Consecutive cached gathers
                                  on the cache-hit stream do not wait unless the reusable item
                                  buffer must grow.
  *_decoded_batch_sync_count      Decoded-batch completion waits before reusing rowgroup metadata,
                                  workset arenas, and scratch buffers.
  *_cached_gather_event_handoff_count
                                  Cached gather completions waited by a following decoded batch stream.
  *_sparse_vector_cache_hits/misses
                                  Reserved for a future sparse vector cache; currently 0.
  *_dense_cache_hits/misses       Aliases for the decoded dense rowgroup cache hit/miss counters.
  *_runtime_policy_decision       selected-vector, full-rowgroup, mixed, or none.
  *_runtime_policy_reason         Counts for selected/full/tail/ratio/low-saving policy outcomes.
                                  The current policy uses selected/full ratio < 0.75 and at least
                                  4 saved vectors to choose selected-vector decode.
  *_device_planning_ms            Crop-to-rowgroup/vector planning time inside the JPEG DCT reader.
  *_workset_build_ms / *_workset_upload_ms / *_decode_ms / *_gather_ms
                                  Internal JPEG DCT device-stage timings.
  *_decoded_gather_ms / *_cached_gather_ms
                                  Gather time split by legacy decoded-rowgroup path and dense-cache-hit path.
                                  The fused decoded path should report projection time instead.
  *_projection_ms / *_decoded_projection_ms
                                  Fused compact DCT projection materialization time.
  *_sync_rowgroup_read_ms         Synchronous JPEG DCT rowgroup read + materialization time.
  *_prefetch_queue_start_ms       Time to create JPEG DCT rowgroup prefetch queues.
  *_prefetch_wait_ms              Time the device batch consumer waited for prefetched rowgroups.
  *_prefetch_depth_block_ms       Time JPEG DCT prefetch workers spent blocked by depth back-pressure.
  *_prefetch_rowgroup_read_ms     Background rowgroup read + materialization time for consumed
                                  prefetched rowgroups.
  *_prefetch_ready_ahead_ms       Time prefetched rowgroups were ready before the consumer asked
                                  for them; high values indicate the queue ran ahead.
  *_prefetch_initial_cache_hit_rowgroup_count
                                  Rowgroups that were already present in the decoded cache during
                                  prefetch planning.
  *_prefetch_candidate_rowgroup_count
                                  Initial cache misses considered by the JPEG DCT prefetch policy.
  *_prefetch_active_shard_count   Shards where JPEG DCT rowgroup prefetch was started.
  *_prefetch_config_disabled_shard_count
                                  Shards where prefetch was disabled by configuration.
  *_prefetch_all_hit_shard_count  Shards where every rowgroup was already cached.
  *_prefetch_small_batch_disabled_shard_count
                                  Shards where miss candidates did not span enough decode batches
                                  to offer structural overlap.
  *_prefetch_selected_vector_disabled_shard_count
                                  Shards where prefetch had no full-rowgroup miss candidates because
                                  all misses used selected-vector decode.
  *_prefetch_selected_vector_miss_rowgroup_count
                                  Initial cache misses excluded from JPEG DCT rowgroup prefetch because
                                  the runtime policy chose selected-vector decode.
  *_prefetch_initial_hit_runtime_miss_count
                                  Rowgroups that were cache hits during prefetch planning but became
                                  runtime misses before consumption.
  *_prefetch_skipped_repeated_runtime_miss_count
                                  Repeated rowgroups skipped by prefetch planning but still read
                                  synchronously because the expected dense cache reuse was unavailable.
  *_prefetched_rowgroup_count     Rowgroups scheduled into JPEG DCT device prefetch queues.
  *_prefetch_consumed_as_hit_count
                                  Prefetched rowgroups discarded because cache service became
                                  available before consumption.
  *_prefetch_skipped_repeated_rowgroup_count
                                  Repeated rowgroups left out of the prefetch schedule because
                                  a prior full-rowgroup decode can populate the dense cache.
  *_prefetch_consumed_as_hit_read_ms
                                  Background read + materialization time spent on rowgroups
                                  later discarded because the cache became a hit.
  *_prefetch_consumed_as_hit_wait_ms
                                  Consumer wait time spent before discarding prefetched
                                  rowgroups that had become cache hits.
  *_plan_ms / *_read_decode_ms    Pipeline benchmark plan time and prepared-plan execution time.
                                  Device planning is counted once through *_device_planning_ms;
                                  read_decode excludes prepared-plan construction.
  pushdown_speedup_vs_full_then_crop
                                  full_then_crop_total_ms / pushdown_total_ms when both stages run;
                                  0 for one-sided modes.
  pushdown_saved_ms_vs_full_then_crop
                                  full_then_crop_total_ms - pushdown_total_ms when both stages run;
                                  0 for one-sided modes.
  auto_pushdown_windows / auto_full_then_crop_windows / auto_no_dct_pushdown_windows
                                  Window-level decisions made by pipeline_benchmark --mode auto.
                                  auto_no_dct_pushdown means crop remains pushed down and only
                                  DCT coefficient pushdown is rejected.
  auto_policy_ms / auto_total_ms   CPU prepared-plan policy time and aggregate selected-path time
                                  in auto mode. The selected path reuses the prepared plan instead
                                  of planning again; no-crop auto windows prepare only the full
                                  candidate. auto_total_ms includes auto policy/planning time once.
  auto_policy_reason               Last auto decision reason, including prepared-plan ratios,
                                  coefficient counts, reuse candidates, materialization/sync
                                  estimates, and decoded/output byte estimates.
  auto_policy_selected_blocks / auto_policy_full_blocks
                                  Aggregate crop/full prepared-plan block counts used by auto mode.
  auto_policy_selected_block_ratio Aggregate selected/full prepared-plan block ratio.
  auto_policy_selected_vectors / auto_policy_full_vectors
                                  Aggregate runtime-policy estimated crop/full prepared-plan FastLanes
                                  vector counts used by auto mode. This accounts for rowgroups that
                                  the device runtime would decode fully because selected chunks are not
                                  worth pushing down.
  auto_policy_selected_vector_ratio
                                  Aggregate estimated selected/full prepared-plan vector ratio. Auto mode
                                  uses this ratio when available to estimate decode work saving.
  auto_policy_touched_rowgroups / auto_policy_full_rowgroups
                                  Aggregate crop/full prepared-plan rowgroup counts used by auto mode.
  auto_policy_estimated_pushdown_worksets / auto_policy_estimated_full_worksets
                                  Aggregate prepared-plan workset estimates grouped by shard and
                                  the JPEG DCT decode batch rowgroup limit.
  auto_policy_estimated_pushdown_gather_items / auto_policy_estimated_full_gather_items
                                  Aggregate prepared-plan gather item estimates for cropped pushdown
                                  output and full-image decode output.
  auto_policy_pushdown_reuse_candidate_rowgroups / auto_policy_full_reuse_candidate_rowgroups
                                  Prepared-plan rowgroups that appeared in earlier auto windows.
                                  This is a cheap cache-reuse proxy, not a measured cache hit.
  auto_policy_touched_rowgroup_ratio
                                  Aggregate touched/full prepared-plan rowgroup ratio.
  auto_policy_pushdown_reuse_candidate_ratio / auto_policy_full_reuse_candidate_ratio
                                  Reuse-candidate rowgroups divided by prepared-plan rowgroups for each
                                  auto candidate path.
  auto_policy_avg_full_blocks_per_rowgroup
                                  Aggregate full prepared-plan block density per rowgroup.
  auto_policy_*_windows            Per-reason auto policy window counts, useful for checking
                                  whether small-window, workset-overhead, touched-rowgroup,
                                  or gather-output guards are active.
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
- `pipeline_benchmark --mode baseline --crop ...` is valid. It measures the
  full-decode baseline while reporting cropped output cardinality and transform time. Use
  `--mode compare` when you also want pushdown-vs-baseline coefficient validation.

H2D Transfer Architecture
  A workset uses a shared DeviceArena to aggregate all appended expressions into one
  staged upload. Each column packs its sub-arrays via copy_to_device(arena, out), and
  arena.upload() performs one aggregated allocation/upload for the workset. Execution
  metadata (DeviceExpression arrays, work items, mixed slots) is packed into the same
  arena so the entire workset is transferred in a single H2D operation. Allocation sizes are rounded
  to power-of-2 buckets (min 64KB) for DevicePool cache efficiency.
