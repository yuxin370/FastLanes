# FastLens Image-major

Normative. Image-major has two supported encodings: one image per rowgroup and
tiled vector rowgroups. These are choices within one physical-layout family;
their manifest versions identify the encodings rather than a performance ranking.
See the [format index](README.md) for writer, Reader and CLI compatibility.
The [Image-major measurements](../../benchmarks/development/image_major_comparison/README.md)
record the pipeline and crop evidence behind retaining both encodings.

## Generate the layout

Run from the repository root with the native JPEG tool built. For one image per
rowgroup, which is also the native `random-access` preset's default:

```bash
build/fastlens/tools/jpeg_dct/fastlens_jpeg_dct_tool \
  --shard --out-dir fastlens/data/example-image-major \
  --physical-layout image-major \
  --metadata-profile reconstruct --shard-images 1024 \
  fastlens/data/example-jpegs
```

For the tiled encoding used in the paper's Image-major comparison:

```bash
build/fastlens/tools/jpeg_dct/fastlens_jpeg_dct_tool \
  --shard --out-dir fastlens/data/example-image-major-tiled \
  --physical-layout image-major-vector-rowgroups --spatial-order tiled-z-32 \
  --metadata-profile reconstruct --shard-images 1024 \
  fastlens/data/example-jpegs
```

Open the resulting native `manifest.bin` with `DirectDctReader`. Public
`fastlens inspect` and annotation commands accept the Published Block-major Store
contract, not these native generation directories.

## Image rowgroups

The `image-major` writer emits manifest version 2 and a standard FastLanes
`TableDescriptor`. Each image occupies one independently addressable rowgroup;
the writer increases rowgroup capacity when necessary to fit the largest image.
The [manifest](MANIFEST.md) maps global image IDs to partitions. The
[image metadata](IMAGE_METADATA.md) image-group directory records each local
image's row start, row count, FLS rowgroup and row offset. Component geometry and
the metadata's spatial-order flags determine the ordering within that image.
FLS headers, footers and descriptor schema are defined by the
[bundled FastLanes format](README.md#fastlanes-coefficient-files).

## Tiled vector rowgroups

The tiled encoding uses Compact Descriptor version 3 in place of the standard
FastLanes `TableDescriptor`. Replacing the descriptor does not change the CPU
FastLanes encoder, FLS header, compressed rowgroup bytes or FLS footer layout.
The footer's descriptor offset begins immediately after the payload and its
descriptor size names the compact descriptor.

The manifest version-3 extension persists the exact values
`physical_layout=image-major-vector-rowgroups`,
`descriptor_kind=galp-compact-v1`, `vector_size=1024`, and
`spatial_order=tiled-z32`. The writer fixes the spatial order to tiled Z-order with
32-block tiles; the manifest reader rejects other values before opening a shard.

Each image stores components consecutively; within a
component, 32×32-block tiles follow raster order and blocks inside each tile follow
Z-order, clipped to the component's real grid. Vector rowgroups do not mix images.

All integers below are little-endian. File and descriptor offsets are
`uint64_t`; offsets within one vector rowgroup are `uint32_t`. Every compact
rowgroup contains exactly one FastLanes vector (`1024` rows, except the final
real-row count may be smaller).

## File layout

```text
FastLanes FileHeader (unchanged)
FastLanes compressed rowgroup payloads (byte-for-byte unchanged)
FASTLENS CompactDescriptorV3
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
| 176..255 | byte[80] | reserved, zero |

Sections are 8-byte aligned. The parser bounds-checks every section before it
dereferences a record and mmaps only the descriptor range (plus the required
leading host page alignment).

The directory pairs start at offsets 72, 88, 104, 120, 136 and 152. Offsets are
descriptor-relative. Flag bits 0–5 match the six features listed in the header;
bits 0–2 are required, the current writer emits `0x3f`, and unknown bits are errors.
CRC64-ECMA uses polynomial `0x42f0e1eba9ea3693`, initial value zero, no reflection
and no final xor. Only the descriptor checksum zeroes bytes 168–175; payload and
rowgroup checksums cover their respective compressed bytes without modification.

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

Image record offsets: `u32 first_rowgroup` at 0, `u32 rowgroup_count` at 4,
`u32 real_row_count` at 8, `u32 first_component` at 12, `u16 component_count` at 16,
`u16 reserved` at 18, `u32 spatial_order` at 20 and `u64 first_physical_row` at 24.
Component record offsets: `u32 semantic_slot` at 0, real width/height at 4/8,
padded width/height at 12/16, image-relative row offset at 20, component index at 24,
and a reserved `u32` at 28. All dimensions are in blocks; reserved fields are zero.

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
backing so the existing FASTLENS decompressor sees unchanged descriptor geometry.

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

The `fastlens_jpeg_dct_tool` commands are:

```text
--compact-v3 input.fls --compact-output output.fls
--expand-compact-v3 input.fls --expanded-output output.fls
```

Both print aggregate and per-rowgroup payload hashes; a checksum mismatch is a
non-zero exit.

Memory, I/O and comparison commands are documented in the
[runtime diagnostic guide](../../benchmarks/development/README.md#compact-format-diagnostics).

## One image per rowgroup and tiled comparison

Both encodings remain supported. Native `image-major` and the implicit
`random-access` preset continue to select image rowgroups; tiled vector rowgroups
are an explicit choice. Matched-input measurements did not justify replacing the
former. The paper's §6.5.1 compares Block-major with tiled Image-major, not the two
Image-major encodings.

The end-to-end comparison used five independent processes per encoding, each with two
16,384-image repeats, batch size 64 and the ViT-Ti DCT validation transform with
all 64 coefficients. Semantic checks passed, but the tiled encoding did not pass
the throughput and variability gates. The runner's `cold` metrics describe the
first repeat in each process, not a controlled cold filesystem cache.

A separate RTX 4090 warm-cache comparison used the same 16,384 images, batch size
64 and all coefficients, with five independent processes per encoding and request.
It measured synchronous raw DCT tensor reads without resizing or model inference.
For 128×128 crops inside one tile or at random positions, tiled rowgroups reduced
payload reads and decoded vectors, but image rowgroups still had higher throughput.
All measured block-device read counts were zero. These results do not establish
the ranking for cold storage or frequency-selected workloads.

Crop position matters as well as size. On a 512×512 4:2:0 image, a centered
128×128 crop crosses all four 32×32-block luma tiles and both chroma vectors;
it therefore skips no vector. The same-sized crop contained within one luma tile
touches three of the image's six vectors. Measure actual decoded vectors and
payload bytes against a separate full-image request for the same samples.
`full_compressed_payload_bytes` in a cropped request can describe only its touched
rowgroups, so that counter alone is not a full-image baseline.

For an end-to-end comparison, prepare both datasets from the same ordered JPEG
list, matching model/preprocessing contracts and passing validation records:

```bash
python fastlens/tools/jpeg_dct/compact_v3_acceptance.py run-ab \
  --contract-a /path/to/image-rowgroup-contract.json \
  --contract-b /path/to/tiled-contract.json \
  --validation-a /path/to/image-rowgroup-validation.json \
  --validation-b /path/to/tiled-validation.json \
  --output-dir "$HOME/tmp/fastlens-image-major-ab" --legs 10
```

This command requires prepared inputs and a working GPU. It alternates independent
processes using the validated preprocessing profile; it does not independently
select arbitrary crop windows. Keep input-only crop timings separate from model
pipeline timings. An unavailable latency metric is unverified, not a pass.
Neither Image-major encoding has a general speedup guarantee over the other.
