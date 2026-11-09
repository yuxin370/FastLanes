# Published Store format

Normative for the published Block-major Store. [Format index](README.md).

```text
example-fastlens/
  manifest.bin
  partitions/
    000000.fls
    000000.images.bin
    ...
  geometry/
    index.bin
    source_order.sha256
    fls_crc64.txt                    # optional complete FLS file CRC64
    000000.bin
    ...
  annotations/                      # when labels or premixing are requested
    samples.jsonl
    metadata.json
    class_to_idx.json               # folder-label mode only
  conversion/
    sources.txt
    report.json
```

| File or directory | Contents and purpose |
| --- | --- |
| `manifest.bin` | Dataset entry point: global image intervals and relative coefficient/metadata paths for each partition. Native binary format remains manifest v1. |
| `partitions/000000.fls` | FastLanes-compressed quantized DCT coefficients for partition 0. These files are not JPEGs or model tensors. |
| `partitions/000000.images.bin` | Complete JPEG-DCT metadata for partition 0: image/component geometry, sampling, quantization tables, encoding profiles and block-group bookkeeping. The existing binary metadata format is unchanged. |
| `geometry/index.bin` | Dataset-level geometry index directory and integrity bindings. |
| `geometry/source_order.sha256` | SHA-256 of the native input path list in physical image order, retained with the runtime data even when `conversion/` is removed. |
| `geometry/000000.bin` | Optimized geometry index for partition 0, mapping requested image/component/block positions to rows, vectors and lanes. It is derived from the partition metadata, not another copy of its coefficients. |
| `geometry/fls_crc64.txt` | Optional complete FLS file checksums, recorded by `convert --record-payload-crc` and checked only by `inspect --check-payload`. |
| `annotations/samples.jsonl` | One physical-order record per image: `image_id`, stable `sample_id` and task `label`. Input CSV/JSONL files are not needed to read labels. |
| `annotations/metadata.json` | Annotation schema, count, source kind, shuffle provenance and hashes binding samples to the manifest, retained physical source order and sample order. |
| `annotations/class_to_idx.json` | Class-name-to-integer mapping, emitted only for folder labels. |
| `conversion/sources.txt` | Original absolute JPEG paths in final physical order. Required for adding annotations by source path or re-verifying against the source JPEGs. |
| `conversion/report.json` | Conversion parameters, stage times, byte counts, verification result, source-list hash and tool/source versions. This is a historical report, not a runtime manifest. |

The matching numeric names identify the same partition; annotation `image_id` is
always the manifest's global physical image ID, not a partition-local offset or an
original input-directory position. Native code still uses `shard` for this object
in API names, options and JSON fields. These dataset partitions are distinct from
FastLanes' internal segments.

Keep `manifest.bin`, `partitions/` and `geometry/` together. Manifest members must
be relative paths inside the store: absolute paths, `..` and escaping symlinks
are rejected. The store can be moved without rewriting its files. Readers discover
`geometry/` automatically; unset `FASTLENS_BLOCK_MAJOR_ACCESS_DIR` if it points at
another dataset. Missing or incompatible geometry indexes fail; a published store
does not silently switch to a planner without that index.

`annotations/` is needed only when consuming labels or sample identities.
`conversion/` can be removed for deployment without affecting Reader, Pipeline or
annotation loading. Removing it loses the recorded source mapping, automatic
source-based annotation building and conversion history. Exact re-verification
then requires a separately retained physical-order JPEG list.

This is the published Converter layout and annotations v2 contract. Older stores are
not automatically upgraded or loaded through legacy sidecar names; follow the
[order-preserving rebuild procedure](../guides/CONVERSION.md#rebuild-an-older-store-without-changing-sample-order).
Low-level native writer transactions still use generations.

## Standard Store and native manifest APIs

The public Converter produces this manifest-v1 block-major layout. Store inspection
and annotation CLI commands require partition IDs `0..N-1`, exactly
`partitions/NNNNNN.fls` and `partitions/NNNNNN.images.bin`, and nonempty, contiguous
image-ID ranges covering the manifest's image count. Filename numbers use at least
six decimal digits. The matching `geometry/NNNNNN.bin` describes that partition.

`fastlens inspect` additionally validates every geometry descriptor and the
source-order binding. It rejects incomplete Stores; rename-only changes to a
manifest invalidate its geometry and annotation bindings.

The native manifest parser, writer and manifest-based Reader APIs also serve
the Image-major layouts and complete native generation outputs. `DirectDctReader` accepts these native manifests subject to its profile
and geometry requirements. Opening one does not certify it as a published Store.
The generic parser retains its safe-path checks; it does not enforce Store naming.
Use `python -m fastlens inspect STORE` when the published Store contract is needed.
The Converter's `--finalize-store` step is for unpublished staging, not migration.

## Physical source-order binding

`geometry/source_order.sha256` is an identity binding, not a geometry descriptor.
It contains 64 lowercase hexadecimal SHA-256 characters, normally followed by a
newline. The converter hashes the exact UTF-8 bytes of its final native input
list: one absolute source path per line, in physical image-ID order. It writes
this digest even when no annotations are requested.

Annotations v2 compares its `source_order_sha256` with this retained value. Thus
copying annotations from a differently ordered conversion fails even if its
manifest has identical names, sizes and counts. The file contains no source paths
and must remain with the runtime files when `conversion/` is removed.

`fastlens inspect` requires a valid digest; if `conversion/sources.txt` exists,
it also checks that file against the digest. Inspection never requires the source
JPEGs. This is a recorded source-order binding, not a checksum of FLS contents or
a guarantee that externally supplied labels are correct. See [Annotations](ANNOTATIONS.md)
for sample-ID, label and annotation metadata contracts.

## Inspection and storage accounting

`python -m fastlens inspect STORE --json` gives machine-readable native manifest and
index details plus current file sizes. Inspection validates sidecar checksums and
source identities without decoding coefficients or recomputing full FLS payload
checksums. `index.compatible=true` does not establish payload integrity: same-size
FLS corruption can remain undetected. Missing or incompatible geometry indexes
fail inspection. Source JPEG size is unknown for stores
without a conversion report. The report's `total_output_bytes` is the original
publication size; inspection's `total_stored_bytes` is the current size of all
files. Inspection also reports `core_storage_bytes` (manifest-referenced partition
files plus required geometry indexes and the source-order binding), `annotation_bytes`, and `conversion_record_bytes`.
Adding annotations or removing conversion records never rewrites the historical
report. Unrelated files contribute only to the current total, not core storage.

The core count is the sum of `manifest.bin`, its referenced FLS and image metadata,
`geometry/index.bin`, one descriptor per partition, and `geometry/source_order.sha256`.
Missing required files fail inspection. Extra files such as `geometry/build.json`
and runtime schedule caches contribute to `total_stored_bytes`, not
`core_storage_bytes`. Annotations and conversion records have separate byte counts.
The paper storage runner uses this same core count; experiment label maps and PLS
mappings are additional prepared-input overhead, reported separately from the core.

Full source verification, publication timing and report fields are documented in
[Conversion](../guides/CONVERSION.md). The binary geometry descriptor remains documented in
[Geometry](GEOMETRY.md); this document does
not redefine its binary format or the native image-metadata payload.

## Optional complete FLS checksums

`geometry/fls_crc64.txt` is ASCII. Its first line is exactly
`fastlens-fls-crc64-ecma-1`. Each following line is a decimal partition ID, one
space, and a 16-digit lowercase hexadecimal CRC64. Records are in manifest order,
one per partition, with no missing, duplicate or extra IDs. The path is derived
from the Published Store's `partitions/NNNNNN.fls` naming contract.

CRC64-ECMA uses polynomial `0x42f0e1eba9ea3693`, initial value zero, no reflection
and no final xor, over the **complete FLS file**, including header, payload,
descriptor and footer. No bytes are zeroed. This differs from descriptor CRCs and
Image-major's payload-only CRCs. Check value for ASCII `123456789` is
`6c40df5f0b497347`.

The converter writes this optional file before the conversion report and atomic
publication. Its bytes contribute to `total_output_bytes` and `total_stored_bytes`,
not `core_storage_bytes`. It remains usable after removing `conversion/` and source
JPEGs. Default inspection and Reader do not scan coefficient data or require it.
Explicit payload checking fails if the file is absent, malformed, incomplete, or
a checksum differs. It does not create a baseline from the data being checked.

CRCs detect accidental corruption, not malicious replacement or authenticity.
The optional file is not signed. Existing SHA-256 order bindings remain unchanged;
no SHA-256 payload list is added. Image-major tiled retains its existing native
payload and rowgroup CRC audit, described in [Image-major](IMAGE_MAJOR.md).
