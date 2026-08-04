# GALP Compact Descriptor v3

Compact v3 is a GALP-only replacement for the redundant FastLanes
`TableDescriptor`. It does not change the CPU FastLanes encoder, the FLS file
header, any compressed rowgroup byte, or the size/layout of the FLS footer.
The footer's descriptor offset still begins immediately after the payload and
its descriptor size names the compact descriptor.

The JPEG-DCT manifest-v3 extension persists the exact contract values
`physical_layout=image-major-vector-rowgroups`,
`descriptor_kind=galp-compact-v1`, `vector_size=1024`, and
`spatial_order=tiled-z32`. Compact-v3 generation fixes the image-major spatial
order to tiled Z-order with 32-block tiles; the manifest reader rejects any
other value before opening a shard.

All integers below are little-endian. File and descriptor offsets are
`uint64_t`; offsets within one vector rowgroup are `uint32_t`. Every compact
rowgroup contains exactly one FastLanes vector (`1024` rows, except the final
real-row count may be smaller).

## File layout

```text
FastLanes FileHeader (unchanged)
FastLanes compressed rowgroup payloads (byte-for-byte unchanged)
GALP CompactDescriptorV3
FastLanes FileFooter (unchanged size; points at CompactDescriptorV3)
```

The 256-byte descriptor header is:

| Offset | Type | Meaning |
|---:|---:|---|
| 0 | byte[8] | `GALPCV3\0` |
| 8 | u16 | format version, `3` |
| 10 | u16 | header bytes, `256` |
| 12 | u32 | flags: little-endian, CRC64-ECMA, bounded scalar-page compression, omitted diagnostic expression sizes, dense segment geometry, and dense coefficient ranges |
| 16 | u64 | complete descriptor bytes |
| 24 | u64 | payload bytes after the FLS header |
| 32 | u64 | payload CRC64-ECMA |
| 40 | u64 | rowgroup count |
| 48 | u32 | coefficient/column count |
| 52 | u32 | vector size, `1024` for JPEG DCT v3 |
| 56 | u32 | spatial-order enum |
| 60 | u32 | shared-schema count |
| 64 | u32 | image count |
| 68 | u32 | component count |
| 72..167 | 6 × (u64 offset, u64 size) | schema, image, component, rowgroup, coefficient, and rowgroup-page sections |
| 168 | u64 | descriptor CRC64-ECMA; this field is zero while hashing |
| 176..255 | bytes | reserved, zero |

Sections are 8-byte aligned. The parser bounds-checks every section before it
dereferences a record and mmaps only the descriptor range (plus the required
leading host page alignment).

## Directories

The schema section begins with `schema_count + 1` section-relative
`u64` offsets. Each blob is a verified FlatBuffers `ColumnDescriptor` containing
only normalized shared structure: data type, names, RPN/operator shape, child
shape, segment types, and other invariant fields. It is not a
`TableDescriptor`.

Each 32-byte image record contains `first_rowgroup`, `rowgroup_count`,
`real_row_count`, `first_component`, `component_count`, `spatial_order`, and
`first_physical_row`. Each 32-byte component record contains semantic slot,
real and padded block-grid dimensions, image-relative row offset, and component
index.

Each 48-byte rowgroup record contains:

| Offset | Type | Meaning |
|---:|---:|---|
| 0 | u64 | absolute payload offset |
| 8 | u32 | payload size |
| 12 | u32 | real row count |
| 16 | u64 | rowgroup-page section offset |
| 24 | u32 | rowgroup-page size |
| 28 | u32 | local image index |
| 32 | u32 | image-local vector index |
| 36 | u32 | reserved |
| 40 | u64 | rowgroup payload CRC64-ECMA |

With the dense-coefficient flag, the coefficient directory is a
`rowgroup_count × coefficient_count` array of 4-byte `u32 size` records.
Offsets are exact prefix sums within each rowgroup; the compactor rejects a
source descriptor unless the columns densely and exactly partition the
rowgroup payload. Readers also accept the earlier 8-byte `(u32 offset, u32
size)` representation when the flag is absent. In either case the exposed
ranges are exact compressed byte ranges, not estimates or post-decode filters.

The rowgroup-page section stores the per-rowgroup values removed from shared
schemas. Every page starts with a codec byte and ULEB128 decoded size. Codec 0
is raw; codec 1 is a bounded LZ stream with 128-byte literals and 3--130-byte,
16-bit-distance matches. The parser rejects pages whose decoded size exceeds
64 MiB, invalid distances, truncated tokens, or trailing bytes. For every root
column the current decoded page stores a ULEB128 schema ID followed,
depth-first, by ULEB128 null count, fixed-width maximum bytes, and one ULEB128
data size per segment. Encoder candidate expression sizes are diagnostic-only
and reconstruct as zero; their operator types remain in the shared schema.
Because v3 has exactly one vector per rowgroup, each segment entrypoint size is
fixed by its schema type, while entrypoint/data offsets are prefix sums. The
compactor validates that this derived geometry exactly equals the source
descriptor before omitting it. Readers retain the earlier full-variable page
path when these feature flags are absent. Nested children additionally store
their ULEB128 column offset/size. Root column offset/size come from the exact
coefficient directory and are intentionally not duplicated. Reconstructing
one rowgroup descriptor is therefore bounded
by one rowgroup page and its referenced shared schema entries.

The optimized writer accepts the root-only column trees emitted by the
JPEG-DCT `int16` table. It rejects structured/nested columns instead of
guessing their parent/child byte interleave; the compatibility parser can
still read earlier full-variable compact pages.

## Reader and pushdown rules

`FlsReader` detects the descriptor magic before attempting to load a standard
FastLanes descriptor. Compact shards never load a `TableDescriptor`, never
build `SparseDatasetAccessIndex`, and never consult an `.svb` sidecar.

Crop planning selects vector rowgroups through the image/component directory.
Coefficient-prefix planning selects exact coefficient ranges, recursively adds
physical dependency columns (including equality and dictionary references),
and coalesces overlapping, adjacent, or same-page ranges. Reads populate those
ranges at their original rowgroup-relative offsets in a zeroed full-size
backing so the existing GALP decompressor sees unchanged descriptor geometry.

Reported I/O separates:

- logical requested coefficient bytes;
- actual bytes passed to `pread` after range coalescing;
- distinct 4 KiB pages covered by the selected ranges;
- coalesced read-run count, selected-vector ratio, and selected-coefficient ratio.

For Full-All, the JPEG runtime classifies decoded-cache misses first, groups
them by physical compact shard, sorts their payload ranges, and uses `preadv`
to scatter each contiguous run into rowgroup-owned backings. This keeps one
vector per independently addressable rowgroup without issuing one syscall per
rowgroup. Prefix requests compile one shard-level coefficient-range plan, so
their logical bytes and distinct 4 KiB coverage are deduplicated across the
whole request while each `pread` remains restricted to an exact coalesced
coefficient range.

## Integrity and conversion

Opening validates the descriptor checksum and structure without scanning the
payload. The explicit payload audit recomputes the whole-payload CRC and every
rowgroup CRC. The offline compactor copies the complete FLS header/payload
prefix and writes only a new descriptor/footer. The expander reconstructs a
standard inline FastLanes `TableDescriptor` without recompressing payload data.
The reconstructed descriptor preserves every runtime decoding field and exact
compressed range; omitted encoder-candidate diagnostic sizes remain zero.

The `galp_jpeg_dct_tool` commands are:

```text
--compact-v3 input.fls --compact-output output.fls
--expand-compact-v3 input.fls --expanded-output output.fls
```

Both print aggregate and per-rowgroup payload hashes; a checksum mismatch is a
non-zero exit.

## Acceptance probes

`galp_compact_v3_runtime_audit` is intentionally a fresh-process probe rather
than an in-process benchmark helper. Memory mode reports RSS/PSS before and
after manifest/metadata loading, first planning, and bounded compact-descriptor
touches, including the actually resident `.fls` mmap pages:

```text
--manifest manifest.bin --memory-probe --batch-size 64 --output memory.json
```

I/O mode executes exactly one request class per process and reports logical
bytes, range bytes, 4 KiB coverage, `/proc/self/io` block reads, native
pread/preadv and coalesced-run counts, vector/coefficient ratios, pinned
memory, and native device memory:

```text
--manifest manifest.bin --workload full-all --coefficients 64
--manifest manifest.bin --workload crop-all --coefficients 64
--manifest manifest.bin --workload full-prefix --coefficients 8
--manifest manifest.bin --workload crop-prefix --coefficients 8
```

`compact_v3_acceptance.py run-runtime-matrix` launches the full K=1/4/8/16/32/64
matrix and the descriptor/index memory probe in independent processes. It
writes the complete evidence JSON before returning a non-zero status for any
failed automatic memory, accounting, `preadv`, or two-dimensional locality
gate. It reports operating-system block reads honestly; formal cold results
still require the caller to control and record page-cache state.

`compact_v3_acceptance.py run-ab` requires at least ten alternating legs, so
each variant has at least five fresh processes. It verifies that repeat zero is
included, warmup is zero, baseline source revisions are clean and hashed, and
the hardware, dataset, sample order, model, and execution configuration match.
The adapter and A/B runner use the same canonical contract names as the system
runner: `galp_planless` and `galp_fixed_items`. The old `galp` and
`galp_legacy` spellings are accepted only when consuming historical contracts;
adapted contracts are always written with canonical names.
Its GPU gate counts Torch-reserved plus GALP-native device memory, while the
process-tree PSS gate includes worker processes and pinned host allocations.
Both variants require passing semantic validator JSON bound to the identical
canonical benchmark contract. Formal page-cache control remains an explicit
unverified field rather than an implicit pass.
