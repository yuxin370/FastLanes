# GALP JPEG DCT block-major access descriptor v1

`block_major_access_v1` is a read-only sidecar for manifest-v1
`spatial-major-image-minor` shards. It does not replace the FLS table
descriptor and does not contain coefficient data. Building it leaves the
manifest, metadata, FLS header/footer, compressed rowgroups, and coefficient
payload unchanged.

The dataset builder publishes one
`shard_NNNNNN.block_major_access.bin` per shard plus
`manifest.block_major_access.bin` in a separate output directory. Readers must
fall back explicitly to the legacy planner when the index or a compatible
sidecar is absent.

## Byte order and integrity

All integers are little-endian. Every sidecar is independently mmap-able and
starts with a 256-byte header. CRCs use CRC64-ECMA polynomial
`0x42f0e1eba9ea3693`, initial value zero, no final xor. The descriptor checksum
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
| 144..200 | u64[8] | offsets of slots, thresholds, cells, rank payload, topology, images, components, and quant tables |

Section offsets are monotonic. Fixed-record section byte counts must agree
exactly with header counts. The loader validates the descriptor checksum,
source identities, section bounds, cell encodings, threshold monotonicity, and
all referenced ranges before exposing records.

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

## Measured ImageNet-50K footprint

For `galp/data/imagedataset_dct/ImageNet-val/manifest.bin`, the v1 builder
produced 30 sidecars totaling 62,996,296 bytes plus a 1,024-byte companion
index. The unchanged FLS plus metadata dataset is 14,106,589,306 bytes, so the
measured growth is 0.446580804427%. All 9,251,484 positive block groups matched
the legacy physical topology. This passes both the hard 1% gate and the 0.5%
target.

Reproduce the build and emit a machine-readable report with:

```bash
./build/galp/tools/jpeg_dct/galp_block_major_access_tool \
  galp/data/imagedataset_dct/ImageNet-val/manifest.bin \
  --output-dir /tmp/galp-block-major-access-v1-real \
  --output-json /tmp/galp-block-major-access-v1-real.json \
  --exhaustive-rank-validation
```
