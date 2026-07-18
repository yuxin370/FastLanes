# Planless Direct-DCT Execution RFC

Status: Milestone 3 general compact planner and CPU structural gates implemented; target-GPU gates pending  
Scope: image-major Direct-DCT transformed-grid execution  
Last updated: 2026-07-18

## 1. Decision

The canonical transformed-grid path must not build a host-side graph of source
transform items.  For each selected image the host submits one compact image
descriptor.  A GPU block owns one output DCT block and derives its bounded
source stencil, physical row, coefficient binding, and output address from the
descriptor and the output coordinate.

The implemented program covers every crop-contained reduced rational axis
relation whose numerator and denominator are at most 64.  Identity and down2
use the exact RGBNoMore reference operation graph.  Every other relation uses
one reader-cached, batch-deduplicated phase dictionary with exactly
`up + down - 1` 8x8 matrices; no matrix, source list, or transform item is
indexed by absolute output position.  Larger or non-contained geometries fall
back to the diagnostic legacy path.

## 2. Existing call chain and root cause

The pre-RFC cache-miss path is:

```text
ReadDeviceDctBatch
  -> PrepareDeviceDctBatch
    -> cached_device_batch_plan
      -> plan_device_batch
        -> shard_for_global_image
        -> effective_crop_box / fixed crop geometry
        -> locate_row_in_shard_for_plan for every source block
        -> append JpegDctDeviceGatherItem for every source block
        -> append JpegDctDeviceFixedTransformItem for every contribution
        -> selected_decode_vectors
        -> remap_fixed_transform_items_to_selected_vectors
        -> stable_sort OrderedFixedTransformItem
        -> build item permutation and group offsets
  -> execute_jpeg_dct_device_batch_plan
    -> prepare_decoded_rowgroup_work
    -> execute_decoded_rowgroup_batch
      -> project_transformed_dct_grid_batch
        -> rebuild and upload JpegDctDeviceFixedTransformBatchItem array
        -> transformed_dct_grid_grouped_kernel
```

For batch size 50 the measured graph contains about 235,200 source transform
items and 58,800 output blocks.  `JpegDctDeviceFixedTransformItem` is 64 bytes
on the current ABI, so its vectors alone represent about 15.1 MB per batch,
before vector capacities, gather items, block metadata, the sortable item array,
the permutation, group offsets, and the second device-batch-item array are
counted.  The full 50K pass constructs 235.2 million transform items.

The expensive functions are not performing discovery.  They expand the
deterministic relation between output coordinates and source coordinates, then
sort that expansion back into output order.  Output ownership makes both steps
unnecessary.

## 3. Information classification

| Information | Lifetime | Representation |
| --- | --- | --- |
| shard path and image -> rowgroup relation | dataset-static | formula-derived shard index for verified uniform ranges, compact 16-bit fallback otherwise, plus direct local-image=rowgroup formula |
| component dimensions and sampling | image-static | interned 64-byte layout record shared by equal images |
| component row offset | image-static, derivable | prefix sum captured once in the interned layout record |
| spatial order | shard-static | one enum in the image descriptor |
| quantization values | image/profile-static | reader-resident value-deduplicated dictionary; only referenced tables enter a batch |
| output dimensions and numeric clamp | profile-static | shared `JpegDctGridTransformSpec` |
| down2 conversion matrix | profile-static | device constant |
| general rational axis program | relation-static | `up + down - 1` shared 8x8 phase matrices per reduced relation |
| image IDs and request order | batch-dynamic | one descriptor index/request index per image |
| crop origin and rational factors | batch/image-dynamic | component descriptor fields |
| source block coordinate | derivable per output | device formula; never stored |
| physical row in rowgroup | derivable per source | component prefix + spatial-order rank |
| output block address | derivable per output | request index + component + output coordinate |
| permutation and group offsets | redundant | removed on the compact path |

## 4. Compact execution representation

### Dataset/image description

No new persistent sidecar is required.  At reader open, the implementation
compiles existing image-major v2 metadata into a 16-byte per-image locator, a
formula-derived image-to-shard lookup for verified uniform shard ranges (with a
compact 16-bit per-image fallback for irregular ranges), one 32-byte compact
descriptor per shard, interned 64-byte layout records, and a value-deduplicated
quantization-table dictionary.  A verified canonical shard uses
`local_image_index == fls_rowgroup_index` and `row_start == 0`, so planning does
not touch the per-image locator at all.  The coefficient payload and all on-disk
files are unchanged.

### Shared transform program

`JpegDctGridTransformSpec` remains the generic profile-level program.  The
implemented device program supports identity/down2 independently on X and Y
and preserves the existing float32 RGBNoMore operation graph for those axes.
For every other reduced `(up, down)` relation, the relative phase
`source * up - output * down` ranges from `-(up - 1)` through `down - 1`.
The host therefore builds exactly `up + down - 1` matrices once, caches them by
relation, and deduplicates equal X/Y/component programs in the batch.

### Batch launch packet

`JpegDctDevicePlanlessImageDescriptor` contains:

- request/output index;
- rowgroup-local row start;
- coefficient binding base assigned at execution;
- zigzag and spatial-order classes;
- at most three component descriptors.

Each component descriptor contains its component row prefix, dimensions, crop,
reduced rational factors, X/Y phase-dictionary bases, quantization-table index,
and presence bit.  The current ABI is 172 bytes per image, or about 8.6 KB for
a batch of 50.  It scales with images plus distinct reduced axis relations, not
source or output blocks.

## 5. Device algorithm

For each image and output block:

1. Decode the linear launch block into component and output `(x, y)`.
2. Derive the bounded source interval on each axis directly from the reduced
   rational factors.
3. Add crop origin to obtain component coordinates.
4. Compute the dense row rank using an O(1) raster/tiled-raster formula or an
   O(log(max(width,height))) dense Morton rank for Z orders.
5. Add component and rowgroup prefixes to obtain the decoded row.
6. Load/dequantize the 64 coefficients for each stencil block.
7. Apply the same composed operation graph for identity/down2, or look up the
   two relative-phase matrices for a general rational contribution.
8. Write the exclusively owned output block; no global sort, permutation,
   group offsets, atomics, or global schedule buffer are required.

Mapping is fused into the transform kernel.  `device_mapping_ms` is therefore
zero as a separate stage and `device_mapping_fused` explicitly records that
interpretation.  Fixed-transform time includes the mapping instructions.

## 6. Complexity

| Stage | Expanded path | Compact path |
| --- | --- | --- |
| host planning | O(transform items log transform items) | O(batch images + selected rowgroups) |
| host dynamic execution metadata | O(transform items + output blocks) | O(batch images) |
| device launch metadata | O(transform items + output blocks) | O(batch images) |
| device mapping | pre-expanded global graph | bounded formula per output block |
| deterministic grouping | host stable sort + permutation | canonical output ownership |

The compact path deliberately decodes the selected image rowgroup in full.
This removes selected-vector source lists and makes planning independent of
crop block count.  An FLS rowgroup is the storage and decompression atom for the
current image-major layout, so a smaller host-selected vector list cannot reduce
physical bytes read; it only reintroduces per-source expansion.  This decision
is final for the current format.  A future format with a smaller independently
addressable storage atom may revisit it, while target-GPU decode cost and native
allocation remain benchmark gates for this implementation.

The batch creates one logical workset containing a flat rowgroup-locator list in
request order.  Each locator names its physical shard directly; no per-shard
request bucket or host regrouping is built.  Same-shard sequential batches keep
the existing prefetch executor, while mixed-shard batches consume the flat list
through the unified executor.  Memory and work therefore remain bounded by the
current batch and its selected rowgroups.

## 7. Native proof counters

The following counters are exposed through C++ and Python:

```text
host_expanded_transform_items_created
host_output_block_source_lists_created
host_global_transform_sort_items
planless_image_descriptor_count
planless_transform_output_block_count
planless_axis_program_count
planless_axis_phase_matrix_count
planless_axis_program_bytes
rowgroup_storage_bytes_read
device_mapping_ms
device_mapping_fused
exact_batch_plan_cache_enabled
decoded_rowgroup_cache_enabled_batches
galp_native_device_in_use_bytes
galp_native_device_peak_in_use_bytes
galp_native_device_cached_bytes
galp_native_device_allocation_requests
galp_native_device_cuda_allocation_count
galp_native_device_cuda_allocation_bytes
```

Plan previews also expose the byte capacities of the reader's compact image
locator, optional shard-index fallback, shard descriptors, layout dictionary,
quantization dictionary, their total, and whether the shard index is derived.
The storage audit queries exact FLS rowgroup record sizes through a separate
diagnostic API, so this accounting does not add storage lookups to the online
planner.

The canonical benchmark validator requires all three host counters and
`fixed_transform_items` to be zero, one planless descriptor per processed
image, fused mapping for every batch, zero generic projection items, and both
the exact-batch plan cache and decoded-rowgroup cache to be disabled.  The
compact path bypasses the key builder and cache lookup even when a caller
supplies a nonzero historical capacity, ignores and retires decoded-cache state for
transformed production execution, and retires legacy expanded plans already
resident in that reader.

## 8. Correctness and structural invariants

The compact path preserves:

- one logical workset for a canonical batch when `decode_batch_rowgroups`
  covers its rowgroups;
- one decode launch and the existing single decoded-batch synchronization;
- request-major output addressing across shards and shuffled input order;
- raster, tiled-raster-32, Z-order, and tiled-Z-32 row formulas;
- grayscale presence semantics (missing chroma outputs remain zero);
- 4:4:4 and 4:2:0 component geometry for all supported reduced rational axes;
- variable image shapes and cross-shard shuffled request order;
- all 64 coefficient selection and existing dequantize/clamp/round behavior.

The CPU structural test
`CanonicalImageMajorFixedGridUsesCompactPlanlessDescriptors` proves that a
cross-shard canonical batch creates two descriptors and zero expanded objects.
It also prepares the same compact batch twice with nonzero exact-plan and
decoded-cache capacities and proves that both caches are disabled.
`PlanlessRationalProgramsCoverSamplingShapesShardsAndSpatialOrders` exercises
all four spatial orders, three variable shapes in three shards, shuffled input,
grayscale/4:4:4/4:2:0 sampling, and 7/5 plus 3/2 relations.  It proves the batch
uses two relation programs, exactly 15 phase matrices (3,840 bytes), and zero
expanded objects across deterministic repeats.  The compiled target-device
test `PlanlessDeviceMatchesLegacyAcrossGeneralityMatrix` compares exact int16
Y/CbCr outputs against the legacy path for that matrix and the canonical
identity/down2 path; it is gated by `GALP_RUN_GPU_TESTS=1` so CPU-only runs do
not probe CUDA.  The same test requires one logical workset, one decode launch,
exactly one decoded-batch/internal synchronization, and zero cached-gather
synchronizations for every compact case.
The CPU planning audit on the fixed 50K view records median and p95 per-batch
latencies for sequential and seeded-shuffled traces at both 1K and 50K images.
It hard-gates median at 2 ms, p95 at 3 ms, and both the sequential/shuffled and
1K/50K median differences at 10%, exactly matching KR5, independently in five
repeats.  The final recorded 20 trace/repeat cells span 0.013362--0.014072 ms
median and 0.013775--0.020550 ms p95; the worst trace-order difference is
2.176% and the worst dataset-size difference is 3.946%.  CUDA semantic and throughput
gates still require execution on the target GPU.

The canonical benchmark artifact separates 640 strict input/logit samples from
a compact full-50K prediction audit.  The latter stores ordinals, labels, and
Top-1/Top-5 indices rather than multi-gigabyte input tensors, and gates 100%
Top-1 agreement plus equal full-dataset Top-1/Top-5 correct counts.

## 9. Storage, I/O, and allocation accounting

The first slice changes no on-disk format and adds no persistent files.  The
50K audit reports 39,321,600,000 raw DCT bytes, 3,589,345,422 compressed
rowgroup-record bytes, 1,572,935,602 index bytes (19,553,458 JPEG image-index
bytes plus 1,553,382,144 inline FLS table-descriptor bytes), a 797-byte
manifest, 336 FLS framing bytes, zero persistent execution metadata, and
5,162,282,157 total persistent bytes: exactly 1.0x the baseline and a 7.617x
raw-to-total compression ratio.  Each cold sequential or seeded-shuffled trace
reads all 3,589,345,422 rowgroup bytes plus 1,572,936,735 metadata/index/framing
bytes, or 5,162,282,157 total (103,245.6 bytes/image), also exactly 1.0x
baseline.

The reader-resident compact structures occupy 800,544 native bytes: 800,000
image-locator bytes, zero shard-index bytes because the canonical ranges are
formula-derived, seven 32-byte shard descriptors, one 64-byte shared layout,
and 256 quantization-dictionary bytes.  That is 0.0223% of the compressed
rowgroup payload.  Reader open is separately reported (152.3 ms in the recorded
CPU run); the whole-process RSS delta is 40.2 MB and intentionally includes the
pre-existing full JPEG metadata objects.  Actual execution uses
`rowgroup_storage_bytes_read` to
confirm the schedule-derived byte count.  Target-GPU peak native allocation and
measured execution-counter evidence remain pending.  The native device counters
come from GALP's own device pool and are aggregated as process-lifetime gauges,
not added once per batch; they complement rather than alias Torch's allocator-
only peak counters.

The e2e harness now records native per-batch planning, mapping, and fused
mapping-plus-transform distributions.  Its `galp_legacy` diagnostic pipeline
uses the same manifest, payload, native binary, checkpoint, input order, and
cache-off settings, changing only `enable_planless_execution`; strict full-50K
prediction and sampled input/logit equivalence gates make it a controlled A/B.
The hot aggregate is repeats 1--4 and directly gates GALP median >= 1.10x
same-round DALI median, GALP hot minimum > DALI median, and <=5% population CV
for both production pipelines.  Actual runs snapshot GPU
temperature/clocks/power/utilization, CPU utilization, memory, and block-I/O
counters before and after each pipeline, and fingerprint the native extension.
Every repeat also records the main pipeline process's Linux `VmRSS` before and
after measurement and lifetime `VmHWM`; the scope explicitly includes Python,
Torch, and native pipeline allocations while excluding loader worker processes.
Canonical e2e execution refuses to start unless every declared benchmark
runtime file matches its repository `HEAD`.  Validation recomputes the same
runtime hashes and commit IDs after all pipelines, so unrelated user-owned
untracked files remain visible in the general status without invalidating a
clean, committed benchmark source set.

## 10. Remaining work to Definition of Done

1. Run the compiled compact-versus-legacy device generality matrix on the
   target GPU and archive its exact-output evidence.
2. Complete target-GPU native allocation accounting and attach measured
   execution read counters to the passing persistent/read-schedule audit.
3. Record fused mapping + transform distributions on the target GPU and archive
   the already-passing CPU planning audit with the final evidence bundle.
4. Run the fixed 50K, batch-50, FP32, cache-off five-repeat
   GALP/legacy/RGBNoMore/DALI benchmark on the same GPU and archive hashes,
   environment, correctness, architecture A/B, CV, and hot min/median evidence.
5. Run seeded-shuffled compact and legacy full-population loader traces and join
   them with the sequential pipeline counters in the final actual-I/O audit.

Until those items and every Phase-2 gate pass, this RFC describes an aligned
milestone, not completion of the overall goal.
