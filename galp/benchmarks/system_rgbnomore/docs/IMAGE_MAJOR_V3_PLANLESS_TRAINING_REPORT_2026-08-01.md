# Image-major manifest-v3 planless training repair — 2026-08-01

## Status

The compact planner, reader, staged-I/O pipeline, diagnostics, stable arena
planning, training instrumentation, CPU/native semantic tests, controlled
physical-read probe, and target-device semantic matrix are implemented.  The
final12 acceptance report has no blocking failures and no unverified blocking
gates.  Its status is **`pass_with_layout_followup`**: the planless and reader
repair is accepted, while the independently measured physical-layout
amplification remains a non-blocking follow-up.

The authoritative final artifacts are:

- `/tmp/galp-v3-acceptance-final12-20260804.json`;
- `/tmp/galp-v3-arena-planning-gpu-tests-final12.xml`;
- `/tmp/galp-v2-planless-observability-gpu-final12-20260804`;
- `/tmp/galp-v3-planless-observability-gpu-final12-20260804`.

The v3 training run records the hashes of 31 runtime source files.  A final
workspace audit matched every recorded hash to the current file (zero
mismatches), so the measurements below correspond to the current runtime
implementation rather than an older binary/source state.

## Original benchmark and timing boundary

The supplied immutable baseline is
`/tmp/galp-training-v3-1k-semantic-rerun-20260801`.  Its runtime measured
region contains 3 warmup and 10 measured batches of 64 images.

| Metric | GALP | DALI |
| --- | ---: | ---: |
| End-to-end throughput | 306.499 img/s | 1156.947 img/s |
| End-to-end step | 208.810 ms | 55.318 ms |
| Exposed loader wait | 140.908 ms/batch, 67.48% | separately recorded in its pipeline artifact |
| Model compute event total | 58.755 ms/batch | 45.667 ms/batch |
| Compute-only upper bound | 1089.273 img/s | 1401.452 img/s |
| Native physical read | 119.497 ms/batch | not applicable |
| Native decode | 0.166 ms/batch | not applicable |
| Native fixed transform | 11.080 ms/batch legacy aggregate | not applicable |
| Native planning | 193.474 ms/batch legacy aggregate | not applicable |

The baseline native aggregate predates phase separation and covers all 13
consumed warmup+measured batches; its latest planning snapshot is 227.613 ms.
Read/decode values above use measured stage distributions where available.

Native planner/read/decode/transform stages overlap with asynchronous data
production and GPU model work.  They must not be summed to reconstruct the
208.810 ms end-to-end step.  GALP and DALI also use different input domains and
model paths, so their end-to-end difference is not a pure codec comparison.

## Confirmed root causes

The old v3 path failed the formal planless contract and entered the generic
transformed-grid planner.  That path constructed one host transform item per
source block, built complete source lists, and globally sorted/grouped the
items.  The baseline recorded 481,112 expanded and sorted items and 75,264
source lists in its latest batch.

The old compact reader then issued rowgroup-oriented synchronous reads and
rebuilt temporary views and payload storage.  Increasing Python workers or
queue depth did not speed up that single slow producer.  Training also folded
native statistics before asynchronous batches completed, and process-global
allocator snapshots were later summed as though they were per-batch deltas.
That made allocation stability impossible to audit correctly.

FastLanes decode was not the primary bottleneck: the baseline decode stage was
about 0.166 ms/batch while planning, physical read, and exposed loader wait
were two to three orders of magnitude larger.

The code-level ownership of the repair is explicit:

- `galp/src/jpeg/jpeg_dct_planner.cpp` owns formal v3 admission and compact
  image/component/vector-rowgroup plan construction in
  `try_planless_device_batch`;
- `galp/src/jpeg/jpeg_dct_device.cu` owns pre-submission compact staging,
  linear column binding, capacity planning, workset execution, and detailed
  runtime counters;
- `galp/src/format/reader.cu` owns compact scatter/selected-column range
  compilation, physical coalescing, `preadv`, and caller-owned zero-copy views;
- `galp/torch/direct_dct_torch.cpp` and the training adapter own asynchronous
  preparation/staging, ordered CUDA submission, lifetime handoff, and queue
  telemetry.

## Compact v3 planner design

`try_planless_device_batch` now admits v3 only when the manifest formally
declares compact descriptors, independent one-vector rowgroups,
`image-major-vector-rowgroups`, the configured FastLanes vector size, and one
vector per rowgroup.  It is not enabled by merely deleting a version check.

The plan expresses:

- one descriptor per requested image, including component geometry, crop,
  sampling phase, resize rational programs, flip, and output ownership;
- an image-local vector binding range mapping local vector indices to physical
  `(shard,rowgroup)` sources;
- deduplicated selected physical rowgroups and bounded axis-phase programs;
- one logical mixed-shard workset when it fits the configured resident budget.

The host does not create per-source-block transform items or output-block
source lists.  The GPU planless transform resolves the compact image-local
bindings while producing the transformed Y/CbCr grids.  Component crop mapping
uses exact image dimensions with floor(begin)/ceil(end), independently for each
sampling plane.  Unsupported geometry returns to the existing correct path
rather than weakening semantics.

## Compact reader and overlap design

The compact reader now provides batched full-rowgroup scatter and
selected-column operations, including caller-owned `*_into` variants.  It
sorts physical offsets, merges exactly adjacent payloads, performs scatter I/O,
and restores caller order.  Compact views use the validated shared schema plan
and parallel view construction.  For canonical v3 flags, the batched hot path
now decodes only rowgroup-varying segment geometry into lightweight views and
reuses normalized schema pointers from the descriptor mapping.  It therefore
does not unpack 64 native column descriptors or rebuild a FlatBuffer per
one-vector rowgroup.  The public single-rowgroup compatibility API and
non-canonical descriptors retain the existing full-descriptor fallback.

The device path groups compact rowgroups by physical reader/shard, distributes
read groups over native workers, and leases owner-affine arenas from a pooled
pinned-host buffer.  Capacity grows to the high-water mark, then reuses the
arena.  Recoverable pinned allocation failure has an explicitly counted
pageable fallback.

Runtime dependencies continue to use the existing streams, events, ordered
reader submission chain, and batch completion fence.  Statistics expose read
groups/workers, coalesced runs, `preadv`, handoffs, internal sync reasons,
buffer acquisition/growth/reuse/capacity/high-water, and pageable fallback.

`StagePreparedDeviceDctBatchIo` executes compact scatter reads and zero-copy
rowgroup materialization before ordered CUDA submission.  The training adapter
keeps an ordered bounded queue of native asynchronous handles, so the next
batch can plan and stage physical I/O while the current batch is transformed
and consumed by model training.  Only ordered device submission is serialized;
the prepared plan owns the staged shard arenas until execution consumes them.

Training artifacts now separate `native_execution_stats_by_phase.warmup` and
`.measured`.  Process-global `galp_native_*` allocation counters use high-water
snapshots rather than sums.  `native_allocation_stability` compares the warmup
high-water with the total consumed-batch high-water and conservatively requires
zero measured compact-arena growth and pageable fallback.

Output and chunk workset arenas now have independent requested/capacity/growth
diagnostics.  A batch-size-derived capacity plan reserves both arenas once
before measured execution (128 MiB each for batch 64 in final12).  Column
binding is built by one linear pass over the materialized expression batches
and rowgroups instead of rescanning every expression list for each rowgroup.

## Same-workload planner evidence

Both standalone probes use the same exact 1,000-image JPEG population, images
0–63, batch 64, center-half crop, alternating flip, 5 warmups, 2,000 measured
iterations, 129,500 source blocks, 75,264 output blocks, and 452/486 selected
and full vectors.

| Metric | v2 | v3 |
| --- | ---: | ---: |
| Planning mean | 0.072416 ms/batch | 0.081517 ms/batch |
| Planning mean | 0.001132 ms/image | 0.001274 ms/image |
| v3/v2 | — | 1.125679× |
| Image descriptors | planless v2 representation | 64 |
| Compact runtime plan | — | 16,944 bytes |
| Expanded transform items | 0 | 0 |
| Output-block source lists | 0 | 0 |
| Global transform sort items | 0 | 0 |

This proves the standalone `<=1.2×` planner gate and the absence of host
per-block expansion.  It does not substitute for the required `<=20 ms/batch`
planning evidence inside the final GPU training run.

Artifacts:

- `/tmp/galp-planner-probe-v2-strict1k-scalarpolicy-final-20260801.json`
- `/tmp/galp-planner-probe-v3-strict1k-scalarpolicy-final-20260801.json`

## Physical crop pushdown evidence

The controlled v3 reader probe uses the same 64-image center-half crop,
alternating flip, and all 64 coefficients.  Page cache was not controlled, so
byte and syscall counters are authoritative; wall time is diagnostic.

| Metric | Selected crop | Full image |
| --- | ---: | ---: |
| Vector-rowgroups | 452 | 486 |
| Compressed bytes | 11,008,143 | 11,392,259 |
| Physical read runs / `preadv` calls | 34 | 1 |
| Reported pread time | 7.197 ms | 6.560 ms |
| Reader call wall time | 27.615 ms | 25.805 ms |

The crop skips 34 rowgroups and 384,116 compressed bytes: vector ratio
0.930041 and byte ratio 0.966283.  Planned decode vectors fall from 486 to 452,
but source/output block amplification is 1.72061×, above the 1–1.3× target.
`rowgroup_full_read_ratio == 1` is expected for training because
`require_all_coefficients=True`; no coefficient-prefix benefit is claimed.

The mechanism is working, but the current image-major spatial organization
intersperses the 34 skipped rowgroups and turns one full read into 34 selected
runs.  Consequently the modest 3.37% byte saving does not improve this cached
reader wall time.  This is a layout-locality limitation, not a missing range
API, decode-vector-selection failure, or reason to revert the compact planner.

Artifact:
`/tmp/galp-reader-probe-v3-strict1k-schemafast-final-20260801.json`.

## Optimization progression and final12 result

The final6 v2 and v3 runs used the same train/validation JSON, image population,
augmentation sequence, batch size 64, worker count 4, prefetch depth 2, three
warmup steps, ten measured steps, runtime timing boundary, device, and model.
Only the manifest version, physical layout, and corresponding payload identity
differed.

| Measured metric | v2 | v3 final6 |
| --- | ---: | ---: |
| End-to-end throughput | 1697.869 img/s | 781.799 img/s |
| End-to-end step | 37.694 ms | 81.863 ms |
| Exposed loader wait | 0.040 ms, 0.107% | 27.398 ms, 33.468% |
| Native planning | 5.193 ms/batch | 3.838 ms/batch |
| Physical read | 0.204 ms/batch | 26.109 ms/batch |
| Decode CUDA-event stage | 0.237 ms/batch | 0.137 ms/batch |
| Fixed transform stage | 1.721 ms/batch | 22.461 ms/batch |
| Planless transform GPU kernel | not separately required | 1.539 ms/batch |
| Internal native syncs | 1.0/batch | 1.0/batch |
| Coalesced runs / `preadv` | 0 | 78.5/batch |

Stage timings can overlap and must not be summed.  The 1.539 ms GPU transform
kernel versus the 22.461 ms transform-stage observation shows that host setup,
uploads, and the completion boundary—not transform arithmetic alone—dominated
that final6 stage.  The final6 allocation failure was also narrow and exact:
one 8,192-byte device allocation and one 783,360-byte pinned allocation after
the warmup high-water snapshot.  The latter is the largest measured compact-v3
column-binding staging buffer (`510 * 64 * 24` bytes).

Subsequent changes addressed these observations without changing the public
training interface or creating a v2-only training branch: canonical compact
rowgroups use the direct geometry path, compact-v3 column bindings use
persistent power-of-two staging and linear binding construction, compact I/O
is staged before ordered submission, and output/chunk workset arenas use an
explicit batch capacity plan.

The final12 v2 and v3 runs use the same train/validation JSON, image population,
sample and augmentation sequence, model and initial state, batch size 64,
worker count 4, prefetch depth 2, five warmup batches, thirty measured batches,
three fresh repeats, runtime timing boundary, device, and optimizer recipe.
Only the manifest version, declared physical layout, and corresponding payload
identity differ.  The acceptance check also requires at least three repeats
and comparable compute-only conditions.

| Measured three-repeat mean | v2 | v3 final12 |
| --- | ---: | ---: |
| End-to-end throughput | 1446.021 img/s | 1497.070 img/s |
| End-to-end step | 44.715 ms | 42.812 ms |
| Compute-only upper bound | 1629.060 img/s | 1678.222 img/s |
| Compute CV | 9.681% | 2.921% |
| Exposed loader wait | 0.111% | 0.119% |
| Native planning | 3.757 ms/batch | 2.291 ms/batch |
| Native planning | 0.05871 ms/image | 0.03580 ms/image |
| Producer I/O staging | 4.769 ms/batch | 28.058 ms/batch |
| Physical read | 0.250 ms/batch | 0.175 ms/batch |
| Decode CUDA-event stage | 0.182 ms/batch | 0.145 ms/batch |
| Fixed transform stage | 1.585 ms/batch | 1.622 ms/batch |
| Column binding | 0.031 ms/batch | 0.327 ms/batch |
| Internal native syncs | 1.0/batch | 1.0/batch |

The compute-only means differ by 2.929%, within the 10% comparability gate;
both CVs also pass the 10% gate.  V3 is 3.530% faster end to end than v2 and
comfortably exceeds the 900 img/s target.  Stage timings overlap and must not
be summed to reconstruct the end-to-end step.  All 90 measured v3 batches are
queue hits (100% hit rate), so the 28.058 ms/batch compact read plus zero-copy
materialization stage is hidden from the consumer; measured exposed wait is
about 0.051 ms/batch.  The v2 compute CV is close to the 10% acceptance limit,
so these results establish the defined gate rather than a noise-free
microbenchmark ranking.

The v3 measured region records a 64-image capacity plan of 134,217,728 bytes
for each output and chunk arena.  Observed requested high water is 34,921,472
bytes for output and 36,830,944 bytes for chunks.  Across all three v3 repeats,
measured output-arena growth, chunk-arena growth, compact-buffer growth,
pageable fallback, new CUDA allocation count/bytes, and new pinned allocation
count/bytes are all zero.  The 60 per-repeat device allocation *requests* are
served without a new CUDA allocation and therefore do not contradict stable
post-warmup reuse.

## Semantic and boundary evidence

CPU/native tests cover compact descriptor bounds, legal zero-payload constant
and alias rowgroups, exact scatter order, selected-column ranges, ragged image
and component geometry, shard boundaries, image-local vector mapping,
different sampling shapes/phases, crop/resize/flip descriptors, and mixed image
sizes.  The strict 1K metadata-only integration verifies six legal empty
payload rowgroups from the real v3 canary.

The final12 target-device artifact passed, without skips:

- `JpegDct.DirectDctTrainingFlipMatchesDctRuleAndPreservesProvenance`;
- `JpegDct.ManifestV3PlanlessMatchesLegacyAcrossRaggedShardsAndSampling`;
- `JpegDct.PlanlessDeviceMatchesLegacyAcrossGeneralityMatrix`.

Final v2/v3 training artifacts have identical fresh-clone first-step
sample IDs, labels, augmentation decisions, Y/CbCr input hashes, logits, loss,
and reproducibility hashes.  `v3_acceptance.py` enforces both this training
comparison and a non-skipped GoogleTest XML result.

Final verification:

- `_galp_direct_dct` and `galp_tests` build successfully;
- the current Python training benchmark suite succeeds with 49 passes and one
  environment-skipped CUDA-only formal-step test; the final12 GPU runtime
  artifacts independently execute real optimizer steps on the target device;
- the compact scatter, selected-column, and metadata-only reader tests pass;
- compact format tests and focused planless structural CPU tests pass;
- the strict 1K zero-payload integration passes when given its FLS shard;
- all three final12 target-device semantic tests pass;
- all final12 blocking acceptance gates pass;
- `git diff --check` passes.

## Final acceptance disposition

`/tmp/galp-v3-acceptance-final12-20260804.json` reports:

- `blocking_failures: []`;
- `blocking_unverified: []`;
- `overall_status: pass_with_layout_followup`.

The final runtime evidence records `uses_planless_fixed_transform=true`, a
compact plan of 18,624 bytes, and zero host expanded transform items, output
source lists, and global sort items.  Merged range statistics are observable
(82.633 coalesced runs and `preadv` calls per v3 measured batch), while loader
wait, planning, allocation stability, and synchronization all pass their
blocking gates.

## Layout decision

Keep image-major v3 on the repaired training mainline.  Its compact planning,
staged reader, and direct rowgroup addressing are accepted, but the controlled
crop evidence does not meet the 1–1.3× locality target.

Continue block-major as an isolated follow-up candidate for spatial locality.
Do not migrate training until its public reader API, semantic matrix, empty and
ragged cases, memory bounds, and independent same-condition benchmark are all
stable.  The present evidence supports further evaluation, not migration.
