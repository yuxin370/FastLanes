# Block-major compact/planless design

> **Historical design and measurement record (2026-08-01).** The descriptor
> design remains relevant, but the CLI, fallback policy and tuning commands in
> this document are not the current production interface. Production now uses
> only `dct_major_pushdown` with native profile
> `block-major-p4-scheduled-bounded-110-v1`. See
> [README](../README.md), [run guide](RUN_GUIDE.md) and
> [complete suite](COMPLETE_TEST.md) for executable commands.

## Status and boundary

This document is the design and implementation record for the existing manifest-v1
`spatial-major-image-minor` dataset. It does not describe the unrelated
Compact-v3 image-major FLS descriptor work. The block-major design keeps every
existing `.fls` byte unchanged and adds only checked, mmap-able index sidecars
under `galp/` tooling. A missing or incompatible sidecar selects the current
legacy planner explicitly.

No file under the repository-root `src/` is part of this design.

## Implementation status (2026-08-01)

The descriptor and compact planning phases are implemented. The real 50K
descriptor generation produced 62,997,320 bytes including the companion index,
which is a 0.446581% storage increase. Exhaustive validation
covered all 9,251,484 groups and all 603,019 rank cells without changing or
re-encoding any `.fls` payload.

The production reader now discovers a validated companion descriptor, builds a
compact block-major plan, and selects the output-owned planless CUDA path. A
missing companion index retains the explicit legacy fallback. Current 5,000
sequential planning evidence is:

| Quantity | Current result |
| --- | ---: |
| Planning wall time | 237.92 ms |
| Legacy planning wall time | 11,280.83 ms |
| Planning speedup | 47.4x |
| Compact plan bytes | 22,134,502 |
| Compact plan peak bytes | 30,814,194 |
| Touched groups | 376,038 |
| Selected rowgroups | 900 |
| Selected vector runs | 3,860 |
| Selected vectors | 52,835 |
| Expanded transform items | 0 |
| Global transform sort items | 0 |

The exact real-data legacy comparison selected the same 52,835 vectors while
removing all 16,978,735 expanded/sorted transform items. The formal 5K workload
uses approximately 1K-image segments, putting planning in the
tens-of-milliseconds range per segment. End-to-end GPU evidence is still
required before this is reported as final performance.

A fresh post-cache-audit 5K CPU run produced 224.26 ms planning, the same
52,835 vectors, 22,134,502-byte resident compact plan, 30,814,194-byte measured
builder peak, and zero expanded/sort items. The paired 237.92 ms value in the
table remains the directly comparable run used for the 47.4x legacy ratio.

Reader construction no longer opens all 30 descriptor mmaps or eagerly parses
all shard metadata. It reads and CRC-validates the fixed-size companion index,
checks every descriptor file's existence and indexed byte size, then retains a
thread-safe per-shard lazy state. The first request touching a shard opens that
descriptor exactly once, checks its byte size and CRC against the companion,
validates source identity, and retains the mmap for the planner lifetime.
Metadata and its in-memory lookup index follow the same per-shard first-touch
lifetime. Legacy/non-planless calls still force the metadata they require.

On the real 50K manifest, the old eager reader constructor measured 3,638.14 ms
(metadata parse 2,747.62 ms, metadata index 809.20 ms, static profile 78.92 ms).
After full companion-index CRC validation, the lazy constructor measures
3.859 ms (native 3.829 ms, companion 2.479 ms) with zero metadata shards and
zero descriptor mmaps resident; binding import measured separately at
382.699 ms. A fresh 1K first plan measured 300.279 ms, with a prior decomposed
run attributing
one touched-shard metadata parse/index (121.22/35.86 ms) and descriptor
open/source validation (4.98/57.01 ms); same-process repeats measured 75.849 ms
and 67.095 ms. These are CPU startup/planning observations, not GPU throughput.

The CUDA path compiles and uses compressed touched rank cells rather than
uploading the 63 MB descriptor. It now executes rowgroups in byte-bounded
worksets. The default 512 MiB total budget also bounds a two-buffer prefetch:
each buffer receives at most half the budget, and overlap is permitted only
when the adjacent estimated resident byte sum remains within the total. A
single oversized rowgroup executes alone and is counted. Full-segment pinned
staging is disabled for this path. Worksets upload only their resident group
bindings and contribute deterministic linear partials in fixed rowgroup order;
there are no floating-point atomics or per-source partial buffers.

The block-major compile-I/O path also bypasses the historical persistent FLS
reader and sparse-plan LRUs, which are limited by entry count. Readers are
deduplicated only inside the current compact batch and are released with that
batch; compiled sparse plans are owned by their selected-rowgroup records.
Likewise, planless rational axis dictionaries are built once into the current
compact plan and do not enter the legacy thread-local transform caches. Formal
validation requires all legacy sparse-vector, resize-weight and conversion-
matrix cache hit/miss counters to remain zero and exports those counters to the
result CSV.

Five real-data GPU preflight gates passed on 2026-07-31. They cover
sequential requests, seeded random order, repeated IDs with distinct explicit
crops, a shard-0/shard-1 boundary request, and Y-only grayscale image 239. All
five selected the planless kernel, matched the eager legacy selected vectors,
repeated deterministically, stayed within one integer DCT level of legacy,
used zero expanded/source-list/sort items, stayed inside the 512 MiB bounded
double-buffer budget, staged zero complete rowgroups, and reported zero exact-
plan, decoded-rowgroup, sparse-vector, resize-weight, and conversion-matrix
cache activity.

The small-gate measurements are diagnostic rather than publication throughput:

| Gate | Plan speedup | Cold execution | Warm execution | Decoded-vector reduction | Native GPU peak reduction | Pinned peak reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential, 2 images | 2.43x | 1.89x | 7.73x | 56.06% | 71.69% | 75.00% |
| Random, 8 images / 7 shards | 2.95x | 0.96x | 2.83x | 60.19% | 53.78% | 49.76% |
| Duplicate explicit crops, 8 requests | 6.14x | 5.82x | 21.11x | 45.57% | 85.66% | 88.89% |
| Cross-shard, 4 images | 6.40x | 1.34x | 4.36x | 56.57% | 56.06% | 50.00% |
| Grayscale, 1 image | 2.74x | 1.18x | 5.13x | 55.10% | 68.35% | 75.00% |

Every gate chose `BitmapExact` for nearly all rowgroups, so compressed bytes
read equaled full-rowgroup bytes and read amplification was 1.0 despite the
39.8%--60.2% decoded-vector reduction. The random gate's cold execution was
4.4% slower than legacy, while its repeat was 2.83x faster. Consequently these
gates prove semantics, determinism, bounded resources, and warm-path viability;
they do not prove the final no-shuffle model throughput or Host RSS condition.
A 1K six-pipeline feature/evaluation screening and then a separate-process
legacy/planless ABBA remain required before any formal 5K/50K claim.

The then-current suite's removed `--gates-only` mode packaged the five device
audits into a 23-request fail-fast run. The historical bundle
`/tmp/galp-dct-major-planless-gates-user-20260731-r1` reports `ok=true` and zero
model invocations, but predates the current binding and is provenance only. A
current run must use `run_suite.py --dry-run` to inspect the complete supported
matrix and then execute that matrix in a fresh output directory.

## Measured source geometry

The ImageNet validation manifest contains:

| Quantity | Value |
| --- | ---: |
| Images | 50,000 |
| Shards | 30 |
| Positive block groups | 9,251,484 |
| Physical coefficient rows | 486,618,392 |
| FLS bytes | 13,763,507,382 |
| Existing metadata bytes | 343,081,924 |
| FLS + metadata bytes | 14,106,589,306 |
| Hard 1% sidecar limit | 141,065,893 |
| 0.5% target | 70,532,946 |

A per-group bitmap/list design is not acceptable: even choosing the smallest
of fixed-u16 present IDs, missing IDs, and a bitmap independently for every
group is approximately 208 MB (1.48%) after structural records.

Presence depends only on an image component's block width and height. Across
all shards and semantic slots there are only 602,695 positive
`(width-threshold,height-threshold)` cells, while there are 9.25 million block
groups. Reusing one adaptive rank/select payload per threshold cell reduces a
conservative fixed-u16 descriptor estimate below the hard 1% limit. Phase 2
must report the real serialized size; delta-ULEB image IDs and exact structural
packing target 0.5%.

The estimate is reproducible without CUDA execution:

```bash
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/diagnostics/estimate_block_major_descriptor.py \
  galp/data/imagedataset_dct/ImageNet-val/manifest.bin \
  --output-json /tmp/galp-block-major-descriptor-estimate.json
```

## Persistent format

Each shard receives `shard_NNNNNN.block_major_access.bin`. A small companion
index beside the original manifest binds all sidecars to the exact source
manifest. The original manifest, metadata, FLS header/footer, coefficient
payload, and compressed rowgroups are not rewritten.

Every sidecar starts with a fixed 256-byte little-endian header containing:

- magic, format version, header bytes, flags, and complete descriptor bytes;
- source manifest CRC64, shard ID, first global image ID, and image count;
- source metadata size/CRC64 and FLS size/payload CRC64;
- semantic-slot, coordinate-position, positive-group, rowgroup, and rank-cell
  counts;
- offset/size pairs for every section;
- descriptor CRC64 with the checksum field zeroed while hashing.

All section bounds, counts, monotonic offsets, checksums, and source identities
are validated before a record is exposed. The descriptor range is mmap-able;
the loader never copies the complete sidecar to heap or GPU memory.

### Image and component sections

`ImageRecord` stores source pixel dimensions, local image ID, first component,
component count, present-slot mask, and flags. Present components have compact
records containing block/padded dimensions, sampling factors, semantic slot,
quant-dictionary ID, and layout/profile ID. Missing chroma is represented by
the image present mask and produces zero Cb/Cr according to the existing
RGB-no-more contract.

Quantization tables are deduplicated per shard. Component records refer to the
dictionary; coefficient values are never placed in the descriptor.

### Rank/select cells

For each semantic slot the descriptor stores sorted distinct component widths
and heights. A block coordinate maps by `upper_bound` to a shared threshold
cell. Each positive cell chooses the smallest actual encoding:

- `AllPresent`: no payload and `rank(image)=image`;
- `MissingList`: delta-ULEB128 missing local IDs;
- `BitmapRank`: image bitmap plus uint16 rank checkpoints;
- `SparseList`: delta-ULEB128 present local IDs.

The cell directory stores the present count, two-bit encoding, and bounded
payload checkpoints. Payload offsets are verified and never inferred past the
descriptor boundary. Rank/select returns the same image-minor row rank as
`locate_row_in_shard`.

### Group topology

Block coordinates retain the source shard's current Z-order. Empty coordinates
are omitted exactly as in legacy metadata. A checkpoint every bounded number
of coordinate positions records positive-group rank, cumulative physical row,
FLS rowgroup, and row-in-rowgroup. Resolving a coordinate scans at most one
checkpoint interval, obtains row count from its threshold cell, and reconstructs
a logical `BlockGroupRecord` containing group ID, rowgroup ID, group row start,
row count, and rank-cell ID.

Rowgroup-boundary bits plus bounded rank checkpoints avoid storing a redundant
fixed-width rowgroup ID for all 9.25 million groups. The builder must prove that
every reconstructed record equals legacy `block_group_index` before publishing
the staged sidecar generation.

## Compact runtime plan

`BlockMajorBatchPlan` contains only:

- request records and original output slots;
- a duplicate fan-out table and per-shard unique request bitmap;
- per-request crop rectangles and deduplicated deterministic axis programs;
- touched-group bindings;
- selected rowgroups and selected-vector runs;
- rowgroup-to-decoded-binding records and logical-to-compact vector runs;
- the exact rank-cell/topology pages needed by this segment.

It never contains source-contribution transform items, ordered transform items,
global item-order arrays, or materialized output-group source lists. All plan
and cache counters are byte counters in addition to entry counts.

For a 5,000-image segment spanning at most a few shards, a conservative budget
is:

| Runtime object | Budget |
| --- | ---: |
| Requests, output slots, duplicates | 1 MiB |
| Crop/layout/axis dictionaries | 4 MiB |
| Touched-group bindings/topology pages | 24 MiB |
| Rowgroup/vector-run/remap records | 4 MiB |
| Rank-cell payload pages | 16 MiB |
| Alignment and temporary builder state | 8 MiB |
| **Compact plan host peak** | **57 MiB** |

The plan has a hard configurable byte budget and reports current/peak bytes.
Rank/topology pages uploaded for one segment share the same bounded budget;
the persistent descriptor is never uploaded wholesale. Pinned staging and
native GPU allocations must remain no higher than the measured legacy path.

## Sequential and generic planning

Requests first become stable `(shard,local_image,output_slot)` records. Sorting
is only over request records, never source contributions. Duplicate image IDs
share physical selection but retain all output slots.

For each component the planner derives one source-support rectangle per request
from the fixed or explicit crop and its axis program. A rectangle sweep over
block coordinates produces the active request set for each touched group. It
ANDs that set with group presence and converts it through rank/select into
physical row intervals. Arbitrary order retains request/output indirection and
duplicate fan-out. The implemented fast path maintains active image IDs as a counted bitset, preserving
duplicate IDs and different explicit crops. Since an image-minor group normally
crosses only one to three 1,024-row vectors, it applies select only at internal
vector boundaries and intersects those image-ID ranges with the active bitset.
This remains exact while avoiding a rank decode for every fragmented active
run. Adjacent selected vectors become vector runs, then rowgroup bindings.

This makes planner memory
`O(batch + touched_groups + selected_rowgroups + selected_vector_runs)`.
The rectangle sweep and interval coalescing do not materialize one record per
source contribution.

The implemented storage policy maps the three candidates to observable runtime
behavior. `RunIntervalExact` means sparse physical interval reads plus exact
selected-vector decode. `BitmapExact` means exact selected-vector decode after
a complete physical rowgroup read. `FullRowgroup` reads and decodes the entire
rowgroup. Automatic mode first prices selected versus full decode, then builds
the exact sparse read plan and compares its physical bytes, pread count and
logical-materialization traffic against the complete rowgroup. The final
strategy counts, rejected sparse candidates, estimated nanoseconds, actual
physical bytes and read amplification are exported in execution statistics.

## Planless GPU transform

The new kernel is output-owned: one deterministic work unit owns one output DCT
block and its 64 coefficient lanes. It obtains the request/component axis
program, walks source taps in a fixed order, resolves the touched-group binding,
uses rank/select and the selected-vector run remap to find the decoded source
row, applies quantization and separable transform weights, and writes one final
value. It does not use floating-point atomics or a group-major partial buffer.

The existing image-major planless axis dictionaries and output-block kernel are
reused where their contracts match. Block-major adds group/rank/vector binding
inputs; it does not fall back to `JpegDctDeviceFixedTransformItem`.

Two launch policies may share the format:

- contiguous/high-density: interval rank and dense touched-group binding pages;
- generic: bitmap/list rank plus stable request/output-slot indirection.

Both must produce stable tensor hashes, logits, and Top-1 across repeated runs.

### One-time active-output workset schedule

The runtime first freezes the actual rowgroup worksets, including half-budget
chunks selected by bounded double buffering. It then builds exactly one
workset-major active-output schedule for the segment. This replaces the former
`for each workset -> scan every output block` construction.

The builder creates two bounded ownership maps: `(shard,rowgroup)->workset` and
`(shard,semantic-slot,x,y)->workset`. It enumerates each logical output block
once, uses the same inverse rational/down2 source interval as the CUDA kernel,
and visits only that output's requested source contributions. The small owner
set for one output is generation-deduplicated and sorted by workset ID. Counts
are prefix-summed into `offsets[workset+1]`, then the output IDs are flattened
once into a contiguous array. Each workset uploads/launches only its
`[offsets[w], offsets[w+1])` slice; the flattened device index is uploaded once
per segment.

For `L` logical output blocks, `C` requested source contributions, `A` active
output/workset ownership entries and `W` worksets, build time is
`O(L + C + A)` apart from bounded hash lookups and sorting the tiny owner set
of one output. Persistent schedule memory is `4*A + 8*(W+1)` bytes. Temporary
memory consists of the two touched ownership maps, one vector per workset, and
generation/owner scratch; its measured/conservative peak is exported. Output
IDs inside every slice are strictly increasing and unique, offsets are
monotonic and terminal-exact, and workset order preserves deterministic source
accumulation. Empty worksets and partial tails have explicit empty/suffix
slices.

Formal validation requires one build per segment, `W` schedule slices for `W`
executed worksets, `active + skipped == L*W`, exact index/offset byte
accounting, and valid offsets. CPU schedule time and CUDA kernel time are
reported separately as
`planless_transform_active_output_planning_ms` and
`planless_transform_gpu_kernel_ms`; neither is inferred from whole-operation
wall time.

## Cache and prefetch

Allowed caches are descriptor pages, rowgroup metadata, axis/layout templates,
and compact plans. Expanded plans are never cacheable. The block-major path
bypasses the legacy exact-batch plan cache and disables the decoded-rowgroup
cache until those objects can be addressed without retaining segment-scale
state. Its active decode workset has an explicit byte capacity and reports the
configured capacity, maximum estimated resident bytes, oversized rowgroups,
double-buffer workset count and overlapped peak estimate. Full-segment pinned
staging is forbidden. Async rowgroup preparation is restricted to the next
half-budget workset and back-pressures before the combined estimate exceeds the
single total budget. The formal contract records
`decode_workset_capacity_mib` (512 by default), and runtime validation requires
an exact MiB-to-byte match.

Decoded-rowgroup cache insertion now evicts before retaining a new entry, so
resident bytes never temporarily exceed capacity. It reports current/peak bytes
and current/peak rowgroup entry counts. Formal planless runs set this cache to
zero and require all hit/miss/insert/eviction counters to remain zero. Enabling
the legacy count-bounded exact-plan cache on the planless path is also a formal
validation failure. The persistent reader/sparse-plan and thread-local
transform caches are bypassed as described above; plan-local reader sharing is
runtime-plan state, not a retained cache, and its lifetime is bounded by one
batch.

The descriptor cache is planner-owned rather than process-global. Its hard byte
bound is the sum of indexed descriptor sizes, while current bytes/count include
only shards actually opened. Per-shard mutexes make concurrent first touch a
single open/validation and keep returned shared descriptor lifetimes valid.
Initialization diagnostics distinguish companion-index load, eager/lazy shard
metadata counts, descriptor open/validation time, and current/bounded bytes.

## Phase gates

Phase 2 cannot complete until all descriptor images/groups agree with legacy
mapping and the serialized 50K sidecars are at most 1% of FLS+metadata bytes.
Phase 3 requires sequential, shuffled, repeated-ID, cross-shard, explicit-crop,
and grayscale selected-row/vector equality. Phase 4 requires zero expanded and
sort items plus a positive planless launch count. Phase 5 requires independent
compact-plan, RSS, pinned, and native-GPU peaks no higher than legacy. Only then
does Phase 6 run the formal 5K/50K model matrix.
