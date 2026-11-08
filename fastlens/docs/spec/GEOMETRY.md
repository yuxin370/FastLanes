# Geometry access specification

Normative. Binary descriptor and dataset-index version: 1.
See the [format index](README.md) for layout and entry-point compatibility.

`geometry` is a read-only sidecar for manifest-v1
`spatial-major-image-minor` shards. It does not replace the FLS table
descriptor and does not contain coefficient data. Building it leaves the
manifest, metadata, FLS header/footer, compressed rowgroups, and coefficient
payload unchanged.

The builder reports the 0.5% and 1% storage targets as measurements. Exceeding
them does not invalidate an index; small datasets often have larger relative
sidecar overhead.

The dataset builder publishes one
`NNNNNN.bin` per partition plus
`index.bin` in a separate output directory. Published Converter stores require the index and every descriptor. Missing or
incompatible files are errors; the standard store path never falls back to a
planner without geometry indexes. Binary descriptor version 1 is unchanged.

## Byte order and integrity

All integers are little-endian. Each partition descriptor is independently mmap-able
and starts with a 256-byte header. The dataset index has a separate 64-byte header.
CRCs use CRC64-ECMA polynomial
`0x42f0e1eba9ea3693`, initial value zero, no reflection, no final xor. The descriptor checksum
field is treated as eight zero bytes while hashing.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | byte[8] | `GALPBMA\0` |
| 8 | u16 | format version, currently 1 |
| 10 | u16 | header bytes, 256 |
| 12 | u32 | required format flags |
| 16 | u64 | complete sidecar bytes |
| 24 | u64 | complete sidecar CRC64 |
| 32 | u64 | source manifest CRC64 |
| 40 | u64 | source manifest bytes |
| 48 | u64 | source shard metadata CRC64 |
| 56 | u64 | source shard metadata bytes |
| 64 | u64 | source FLS payload CRC64 when available, otherwise zero |
| 72 | u64 | source FLS bytes |
| 80 | u64 | first global image index |
| 88 | u32 | shard id |
| 92 | u32 | local image count |
| 96 | u32 | vectors per source rowgroup |
| 100 | u32 | FastLanes vector size, currently 1024 |
| 104 | u32 | source rowgroup count |
| 108 | u32 | semantic-slot count |
| 112 | u32 | present-component record count |
| 116 | u32 | quant dictionary count |
| 120 | u32 | positive block-group count |
| 124 | u32 | dense coordinate-position count |
| 128 | u32 | shared threshold-cell count |
| 132 | u32 | topology-checkpoint count |
| 136 | u16 | bitmap rank-checkpoint image stride |
| 138 | u16 | topology coordinate stride |
| 140 | byte[4] | reserved, zero |
| 144..200 | u64[8] | offsets of slots, thresholds, cells, rank payload, topology, images, components, and quant tables |
| 208 | byte[48] | reserved, zero |

Section offsets are monotonic. Fixed-record section byte counts must agree
exactly with header counts. The loader validates the descriptor checksum,
source identities, section bounds, cell encodings, threshold monotonicity, and
all referenced ranges before exposing records.

Flags are bits 0–3: little-endian, CRC64-ECMA, shared threshold/rank cells, and
topology checkpoints. All four are required (`0x0f`); other bits are rejected.
Offsets are descriptor-relative. The last section ends at complete sidecar bytes.
Threshold and rank-payload sections can end with zero padding to 8-byte alignment;
fixed-record sections have exactly `count * record_bytes` bytes.

## Dataset index

`geometry/index.bin` has a **64-byte header**, followed by one **32-byte record**
per manifest partition in the same order. It contains no paths. Descriptor names
are the partition ID as at least six decimal digits followed by `.bin`.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | byte[8] | `GALPBMI\0` |
| 8 | u16 | version, 1 |
| 10 | u16 | header bytes, 64 |
| 12 | u32 | flags, `0x03` (little-endian and CRC64-ECMA) |
| 16 | u64 | complete manifest CRC64 |
| 24 | u64 | complete manifest bytes |
| 32 | u32 | partition count |
| 36 | byte[4] | reserved, zero |
| 40 | u64 | complete index bytes, `64 + 32 * partition_count` |
| 48 | u64 | index CRC64; bytes 48–55 are zero while hashing |
| 56 | byte[8] | reserved, zero |

Partition record offsets are relative to the record start:

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | u32 | partition ID |
| 4 | u32 | local image count |
| 8 | u64 | first global physical image ID |
| 16 | u64 | descriptor bytes |
| 24 | u64 | descriptor checksum from its header at byte 24 |

The last checksum is the descriptor CRC calculated with its checksum field zeroed,
not the CRC of the already-checksummed file. Index length, record order, counts,
manifest identity, descriptor lengths and checksums must agree before access.

## Shared presence/rank cells

For one semantic slot, let `W` and `H` be the sorted distinct positive component
block widths and heights in the shard. Coordinate `(x,y)` maps to
`lower_bound(W,x+1) * |H| + lower_bound(H,y+1)`. Coordinates with the same width
and height thresholds share one presence/rank payload even when many physical
block groups use it.

Each 8-byte cell record contains `u16 present_count`, `u8 encoding`, one reserved
byte, and a shard-payload-relative `u32 offset`. The builder measures actual
encoded bytes and chooses the smallest of:

- `Empty` or `AllPresent`, with no payload;
- `SparseList`, sorted present local-image IDs encoded as delta ULEB128;
- `MissingList`, sorted missing local-image IDs encoded as delta ULEB128;
- `BitmapRank`, one image bit plus a u16 prefix rank at the configured image
  stride.

Sparse/missing lists are selected only when their complete delta stream is
smaller than the bitmap representation. Rank is the number of present local
image IDs strictly below the query image. Select returns the local image at a
present rank. This is exactly the source layout's image-minor row offset.

Encoding IDs are Empty=0, AllPresent=1, SparseList=2, MissingList=3 and BitmapRank=4.
Cell offsets are relative to the rank-payload section. Delta lists encode the first
ID directly and then strictly positive differences, using unsigned LEB128. Their
element counts are respectively `present_count` and `image_count-present_count`.
Bitmap payload starts with `ceil(image_count/8)` bytes; bit `i % 8` of byte `i / 8`
represents image i. Then `ceil(image_count/stride)` little-endian `u16` checkpoints
store the number of set bits strictly before each stride boundary. Empty and
AllPresent consume no payload.

## Group topology

Each 64-byte semantic-slot record stores maximum block dimensions, threshold
ranges, cell ranges, coordinate/group ranges, topology-checkpoint ranges, and
the raster/Morton order flag. Zero-presence coordinates are omitted exactly as
they are in legacy metadata.

A 24-byte checkpoint stores, before one bounded coordinate interval, the
positive-group rank, cumulative physical row, FLS rowgroup, and row within that
rowgroup. The reader computes a coordinate's dense raster/Morton rank and scans
at most one checkpoint stride. Per-coordinate row counts come from shared rank
cells. Applying the original block-group-aligned rowgroup capacity reconstructs
`group_id`, `row_start`, `row_count`, `fls_rowgroup_index`, and
`row_start_in_rowgroup` without a per-group record array.

The source metadata's legacy `z_order_index` can contain gaps inherited from a
global maximum grid. It is not persisted: coordinate order and every physical
row/rowgroup field are validated directly.

## Image, component, and quant records

The descriptor stores one 24-byte image record and one 24-byte record for each
present component. Missing chroma appears only as a missing bit/record, allowing
the runtime to fill Cb/Cr with zero while still reading Y for grayscale JPEGs.
Component records contain block and padded dimensions, sampling factors,
semantic slot, component/profile IDs, quant table number, and a quant dictionary
reference. Quant tables are deduplicated per shard; each record is an 8-byte
fingerprint followed by 64 u16 values.

## Build an index

Fixed record fields are packed in this order:

| Section | Record fields | Bytes |
| --- | --- | ---: |
| Slots | `u32 semantic_slot,max_width,max_height`; `u16 width_count,height_count`; `u32 first_width,first_height,first_cell,cell_count,coordinate_count,first_group,group_count,first_topology,topology_count,z_order`; 8 reserved zero bytes | 64 |
| Thresholds | `u16` widths/heights, referenced by element index from each slot | 2 |
| Cells | `u16 present_count`; `u8 encoding,reserved`; `u32 payload_offset` | 8 |
| Topology | `u32 positive_before,reserved`; `u64 row_start`; `u32 rowgroup,row_in_rowgroup` | 24 |
| Images | `u32 width,height,first_component`; `u16 component_count,present_slot_mask`; `u8 precision,reserved`; `i16 color_space`; `u32 reserved` | 24 |
| Components | `u16 slot,local_component,width,height,padded_width,padded_height`; `u8 h_sampling,v_sampling`; `u16 quant_dictionary_id`; `u32 profile_id`; `i16 component_id,quant_table_number` | 24 |
| Quant tables | `u64 fingerprint`; `u16 values[64]` in natural order | 136 |

Slot `z_order` is 0 for raster and 1 for Morton order. Quant dictionary ID 65535
means no dictionary reference; other IDs index the quant section. Every reserved
field written by the builder is zero.

```bash
./build/fastlens/tools/jpeg_dct/fastlens_block_major_access_tool \
  /path/to/manifest.bin \
  --output-dir "$HOME/tmp/block-major-access" \
  --output-json "$HOME/tmp/block-major-access.json" \
  --exhaustive-rank-validation
```

The report contains descriptor sizes and topology validation results. See the
[diagnostic guide](../../benchmarks/development/README.md#geometry-index-footprint)
for an ImageNet-50K measurement.
