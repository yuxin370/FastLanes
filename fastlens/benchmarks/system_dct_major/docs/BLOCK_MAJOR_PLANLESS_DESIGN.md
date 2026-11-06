# Block-major compact/planless design

This document describes access sidecars, compact planning and output-owned GPU
transforms for manifest-v1 `spatial-major-image-minor` data. The descriptor
indexes existing `.fls` payloads; generating it does not re-encode coefficients.
See the [binary format](../../../src/jpeg/BLOCK_MAJOR_ACCESS_V1.md) for the
serialized layout and integrity rules.

The benchmark uses `dct_major_pushdown` with native profile
`block-major-p4-scheduled-bounded-110-v1`. Follow the [run guide](RUN_GUIDE.md)
for preparation and execution.

## Persistent format

Each segment receives `geometry/NNNNNN.bin`. The companion
`geometry/index.bin` binds all descriptors to the exact source manifest. The original manifest, metadata, FLS header/footer, coefficient
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

The plan has a configurable byte budget and reports current/peak bytes.
Rank/topology pages uploaded for one segment share that budget; uploads contain
only the descriptor pages needed by the segment.

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

The kernel is output-owned: one deterministic work unit owns one output DCT
block and its 64 coefficient lanes. It obtains the request/component axis
program, walks source taps in a fixed order, resolves the touched-group binding,
uses rank/select and the selected-vector run remap to find the decoded source
row, applies quantization and separable transform weights, and writes one final
value. It does not use floating-point atomics or a group-major partial buffer.

The existing image-major planless axis dictionaries and output-block kernel are
reused where their contracts match. Block-major adds group/rank/vector binding
inputs; it does not fall back to `JpegDctDeviceFixedTransformItem`.

The design distinguishes two launch policies:

- contiguous/high-density: interval rank and dense touched-group binding pages;
- generic: bitmap/list rank plus stable request/output-slot indirection.

### One-time active-output workset schedule

The runtime first freezes the actual rowgroup worksets, including half-budget
chunks selected by bounded double buffering. It then builds exactly one
workset-major active-output schedule for the segment.

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

Each segment has one schedule build and one slice per executed workset, with
`active + skipped == L*W`. CPU schedule time and CUDA kernel time are reported as
`planless_transform_active_output_planning_ms` and
`planless_transform_gpu_kernel_ms`.

## Cache and prefetch

Caches hold descriptor pages, rowgroup metadata, axis/layout templates and
compact plans. The block-major path disables exact-batch and decoded-rowgroup
caching to keep segment data scoped to the active request.
Its active decode workset has an explicit byte capacity and reports the
configured capacity, maximum estimated resident bytes, oversized rowgroups,
double-buffer workset count and overlapped peak estimate. Async rowgroup
preparation stages the next half-budget workset and applies backpressure before
the combined estimate exceeds `decode_workset_capacity_mib` (512 by default).

Decoded-rowgroup cache insertion evicts before retaining a new entry, so
resident bytes never temporarily exceed capacity. It reports current/peak bytes
and current/peak rowgroup entry counts. With this cache disabled, all
hit/miss/insert/eviction counters remain zero. The block-major path also bypasses
persistent reader/sparse-plan and thread-local transform caches; shared readers
within a plan live for one batch.

The descriptor cache is planner-owned rather than process-global. Its hard byte
bound is the sum of indexed descriptor sizes, while current bytes/count include
only shards actually opened. Per-shard mutexes make concurrent first touch a
single open/validation and keep returned shared descriptor lifetimes valid.
Initialization diagnostics distinguish companion-index load, eager/lazy shard
metadata counts, descriptor open/validation time, and current/bounded bytes.
