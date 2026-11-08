# Manifest specification

Normative. [Format index](README.md). The entry file is `manifest.bin`.
Integers are little-endian, packed without alignment. A string is a `u32` byte
length followed by exactly that many bytes, without a NUL terminator. Public
conversion records paths as UTF-8. Native serialization preserves the path bytes.

## Common header

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | byte[8] | `GJDCTSH1` (also used by versions 2 and 3) |
| 8 | u32 | manifest version: 1 Block-major, 2 image-rowgroups, 3 vector-rowgroups |
| 12 | u16 | ragged validation mode, 2 |
| 14 | u32 | maximum vectors per source rowgroup |
| 18 | u32 | rowgroups per shard |
| 22 | u64 | total image count |
| 30 | u32 | partition record count |

## Partition records

Starting at byte 34, repeat the following packed record in manifest order. Offsets
in this table are relative to the start of each record. String lengths make records
variable-sized; the next record starts immediately after the metadata path.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | u32 | partition ID |
| 4 | u64 | first global physical image ID |
| 12 | u32 | local image count |
| 16 | u64 | real coefficient rows |
| 24 | u64 | padding rows |
| 32 | u64 | physical rows |
| 40 | u32 | rowgroup count |
| 44 | u32 | block-group count |
| 48 | u64 | complete FLS file bytes |
| 56 | u64 | complete image metadata file bytes |
| 64 | string | relative FLS path |
| variable | string | relative image metadata path |

Both paths resolve against the manifest directory. Absolute paths, `..`, and
symlinks escaping the dataset root are rejected. Native generations can have
different names; the Published Store additionally requires contiguous partition
IDs, `partitions/NNNNNN.fls`, `partitions/NNNNNN.images.bin`, and consecutive,
nonempty image intervals covering the total. See [Store](STORE.md).

## Image-major tiled extension

The current version-3 writer appends this extension immediately after the last
partition. The reader requires it for the tiled compact layout.

| Order | Type | Meaning |
| ---: | --- | --- |
| 1 | byte[8] | `GJDCCV31` |
| 2 | string | physical layout, `image-major-vector-rowgroups` |
| 3 | string | descriptor kind, `galp-compact-v1` |
| 4 | u32 | vector size, 1024 |
| 5 | string | spatial order, `tiled-z32` |
| 6 | u16 | spatial order ID, 3 |
| 7 | u32 | extension partition count, equal to common header count |

Each extension partition is 36 bytes: `u32 partition_id`, then `u64 payload_size`,
`u64 payload_crc64`, `u64 compact_descriptor_size`, `u64 source_descriptor_size`.
Payload starts after the FLS header and ends before the descriptor. Its CRC uses
CRC64-ECMA with initial value zero and no reflection or final xor.
The descriptor-kind string is a historical identifier, not the descriptor's
numeric version: the descriptor header records version 3.

Versions 1 and 2 have no compact extension. Unknown versions, malformed lengths,
inconsistent extension values and trailing bytes are errors. A rename or rewrite
changes the manifest bytes and invalidates geometry and annotation bindings;
copying a complete Store to another directory does not.
