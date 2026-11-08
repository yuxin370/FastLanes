# Image metadata specification

Normative for sectioned metadata, binary version 3. [Format index](README.md).
This metadata is used by both Block-major and Image-major. All integers are packed
little-endian. A Published Store names the files `partitions/NNNNNN.images.bin`;
native generations commonly use `.meta.bin`.

## Header

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | byte[8] | `GJDCTMD3` |
| 8 | u16 | metadata version, 3 |
| 10 | u16 | profile: 0 DCT dataset, 1 reconstructable JPEG, 2 original markers |
| 12 | u16 | row ordering: 0 single image, 1 Block-major, 2 Image-major |
| 14 | u16 | ragged validation mode, 2 |
| 16 | u16 | layout flags |
| 18 | u64 | image count |
| 26 | u32 | semantic component count |
| 30 | u32 | encoding-profile count |
| 34 | u16 | partition policy: 0 semantic slot, 1 encoding profile |
| 36 | u16 | coefficient encoding; 0 dense64 FastLanes in the public Converter |

Layout flag bit 0 means zigzag coefficient columns, bit 1 legacy global Z-order,
bit 2 an explicit Image-major spatial order, and bits 3–4 its ID: 0 raster,
1 tiled raster with 32-block tiles, 2 global Z-order, 3 tiled Z-order with
32-block tiles. Other flag bits are unsupported. Explicit spatial order is
written for Image-major. Native coefficient encodings 1–4 respectively identify
cross-RLE int16, DC-delta dense64, DC-delta AC sparse-RLE, and JPEG-like run-length;
they are outside the public dense64 Store contract.

## Sections

Sections follow byte 38 until EOF. Each is `u16 section_id`, `u64 payload_bytes`,
then the payload. Length excludes the 10-byte section header. There is no padding
between sections. Empty directory sections have zero payload length. The writer
emits sections 1, 5, 6, 7, then 2 when images are present, then 3/4 as required by
the profile. Readers must use section IDs and lengths, not assume this order.

The field sequences below are packed, in listed order; array counts and byte sizes
determine record boundaries. Signed values use two's complement.

| ID | Name | Payload |
| ---: | --- | --- |
| 1 | Component grid | `component_count` × 28-byte records: `u32 semantic_slot`, `u32 component_index`, `i32 component_id`, `u32 width_blocks`, `u32 height_blocks`, `u32 padded_width_blocks`, `u32 padded_height_blocks` |
| 5 | Encoding profiles | `profile_count` × 152-byte records: `u32 profile_id`, `i32 h_sampling`, `i32 v_sampling`, `i32 quant_table_id`, `u64 quant_fingerprint`, `u16 quant_values[64]` |
| 6 | Block-group directory | 36-byte records: `u32 semantic_slot`, `u32 z_order_index`, `u32 block_x`, `u32 block_y`, `u64 row_start`, `u32 row_count`, `u32 fls_rowgroup`, `u32 row_start_in_rowgroup` |
| 7 | Image-group directory | 24-byte records: `u32 local_image`, `u64 row_start`, `u32 row_count`, `u32 fls_rowgroup`, `u32 row_start_in_rowgroup` |
| 2 | Per-image grid | One image record per image, described below |
| 3 | Reconstruction information | `u64 image_count`, then reconstruction records |
| 4 | Original JPEG markers | `u64 image_count`, then marker records |

Section 2 image records begin with `u32 width_pixels`, `u32 height_pixels`,
`u8 precision`, `i32 jpeg_color_space`, `u32 warning_count` (17 bytes). Follow with
exactly `component_count` 25-byte records: `u8 present`, `u32 semantic_slot`,
`u32 local_component_index`, `i32 component_id`, `u32 width_blocks`,
`u32 height_blocks`, `u32 encoding_profile_id`. Missing slots have `present=0`;
their remaining fields are still serialized. Present components obtain sampling,
quantization and padded dimensions through their profiles and grid geometry.
Semantic slots 0/1/2 identify Y/Cb/Cr; grayscale omits chroma.

Section 3 records contain `u32 width_pixels`, `u32 height_pixels`, `u8 precision`,
`i32 jpeg_color_space`, `u32 quant_table_count`; each quant table is `u8 table_id`
and 64 `u16` values in natural order (129 bytes). Profiles 1 and 2 write this
section. Quantization does not reproduce the original JPEG byte stream.

Section 4 records contain `u32 marker_count`; each marker is `u8 marker`,
`u64 payload_length`, then that many bytes. Only profile 2 writes original markers.
JPEG color-space IDs use libjpeg's `J_COLOR_SPACE` (0 unknown, 1 grayscale,
2 RGB, 3 YCbCr, 4 CMYK, 5 YCCK). Quantization fingerprints use the existing
64-bit FNV-style calculation initialized to 1469598103934665603: for each `u16`
quant value, xor/multiply its low then high byte by 1099511628211 modulo 2^64.

Partial headers, sections extending beyond EOF, inconsistent counts and invalid
enums are errors. Metadata has no embedded whole-file checksum; the Block-major
geometry descriptor binds its complete bytes and CRC64.

## Legacy metadata

The native reader also accepts `GJDCTMD1` with metadata version 1. This is not
the public Converter's output and must not be inferred from manifest version 1.
Its layout remains implemented in
[jpeg_dct_metadata.cpp](../../src/jpeg/jpeg_dct_metadata.cpp); the retained
Image-major compatibility path does not change the sectioned encoding above.
