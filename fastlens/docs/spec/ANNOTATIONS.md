# Annotation format

Normative. Current schema identifier: `fastlens-samples-v2`.
[Format index](README.md); [annotation commands](../guides/ANNOTATIONS.md).

`annotations/samples.jsonl` is UTF-8 JSON Lines. Each row has `image_id` (consecutive
integer starting at zero, in physical image order), a unique nonempty string
`sample_id`, and `label` (a finite JSON value, including explicit null). Sample IDs
are logical identities, normally source-root-relative POSIX paths, not absolute
source paths. Arrays, objects, strings and numeric labels are supported; CSV and
folder-label inputs use nonnegative integer class IDs.

`annotations/metadata.json` is a JSON object:

| Field | Meaning |
| --- | --- |
| `schema_version` | `fastlens-samples-v2` |
| `image_count` | Number of samples, equal to manifest image count |
| `manifest_sha256` | SHA-256 of complete `manifest.bin` bytes |
| `source_order_sha256` | Value retained in `geometry/source_order.sha256` |
| `samples_sha256` | SHA-256 of complete `samples.jsonl` bytes |
| `sample_order_sha256` | Ordered sample-ID digest defined below |
| `label_source` | `folders`, `csv`, `jsonl`, or null for unlabelled premixing |
| `shuffle` | null, or `{"algorithm":"fastlens-shuffle-v1","seed":N}` |
| `class_mapping` | `class_to_idx.json` for folder labels, otherwise null |
| `class_mapping_sha256` | Complete class-map file SHA-256, otherwise null |

Hashes are lowercase hexadecimal SHA-256. The sample-order digest concatenates,
for each sample ID, its UTF-8 byte length as unsigned 64-bit **big-endian** followed
by those bytes, and hashes the resulting stream. The source-order digest instead
hashes the exact UTF-8 absolute JPEG path list, one path plus `\n` per image in
physical order. These are different contracts. The retained source-order file
contains 64 lowercase hex characters, normally followed by a newline.

The optional class-map JSON object maps class names to distinct nonnegative
integer IDs. It may include classes absent from this Store. Existing class IDs
are preserved; source directories do not imply ImageNet checkpoint class numbers.

## Deterministic premixing

Algorithm `fastlens-shuffle-v1` is Fisher–Yates on complete
`(source_path, sample_id, label)` records before partitioning. For bounds N down to
2, draw the first eight bytes of
`SHA256(b"fastlens-shuffle-v1\0" + uint64_be(seed) + uint64_be(counter))`
as an unsigned big-endian integer. Counter starts at zero and increments on every
draw. Reject values at least `floor(2^64/bound)*bound`; otherwise swap positions
`bound-1` and `value % bound`. Seed is an unsigned 64-bit integer.

`shuffle: null` makes no claim about how an already ordered input was produced.
Reusing a physical-order list must not apply the shuffle again. See
[cross-machine reproduction](../guides/CONVERSION.md#cross-machine-reproduction-and-relocation).

Loading validates counts, consecutive image IDs, unique sample IDs and the file
and order hashes. This detects mismatched annotations, not incorrect externally
supplied labels. It needs neither source JPEGs nor `conversion/`. These hashes
are existing identity bindings, not payload checksums or signatures.
