# FastLens format specifications

[Documentation index](../README.md)

FastLens exposes two physical layouts: **Block-major** and **Image-major**.
Layout names describe coefficient organization. The manifest, metadata and access
descriptor have independent binary versions; a larger number is not a performance
ranking or a replacement for the other layout.

These documents are normative for the encodings described. A disk-format change
requires an explicit version or documented feature flag. Magic bytes and existing
version values are not renamed to match presentation labels.

| Layout / encoding | Manifest | Image metadata | Access descriptor | Writer | Reader | Public inspect / annotations CLI |
| --- | --- | --- | --- | --- | --- | --- |
| [Block-major](BLOCK_MAJOR.md) | 1 | 3 (`reconstruct` profile) | Geometry 1 | `python -m fastlens convert`; native writer | `DirectDctReader` with geometry | Yes, for the [Published Store](STORE.md) directory contract |
| [Image-major, one image per rowgroup](IMAGE_MAJOR.md#image-rowgroups) | 2 | 3; legacy metadata 1 remains readable | Standard FastLanes table descriptor | Native `--physical-layout image-major`; implicit `random-access` preset | Native Reader / `DirectDctReader`, subject to profile requirements | No |
| [Image-major tiled](IMAGE_MAJOR.md) | 3 | 3 | Compact Descriptor 3 | Native `--physical-layout image-major-vector-rowgroups --spatial-order tiled-z-32` | `DirectDctReader` | No |

Block-major remains the public Converter default. Native `random-access` continues
to select one image per rowgroup; tiled Image-major is an explicit choice.
Both Image-major encodings remain supported. Matched-input comparisons validated
their outputs but did not justify replacing image rowgroups with tiled rowgroups.
The published paper's §6.5.1 compares Block-major with Image-major tiled, not the
two Image-major encodings. See
[Image-major comparison](IMAGE_MAJOR.md#one-image-per-rowgroup-and-tiled-comparison).
Neither native Image-major encoding is a Published Store merely because its
manifest is readable. Tensor profiles can impose narrower geometry requirements
than the storage reader. See [API support](../reference/API_STABILITY.md).

| Specification | Scope |
| --- | --- |
| [Store](STORE.md) | Files, bindings, relocation and optional file CRC64 |
| [Manifest](MANIFEST.md) | Partition directory, strings and layout extension |
| [Image metadata](IMAGE_METADATA.md) | Image geometry, quantization and section records |
| [Geometry](GEOMETRY.md) | Block-major partition descriptor and dataset index |
| [Annotations](ANNOTATIONS.md) | Labels, sample identities and order binding |
| [Test vectors](TEST_VECTORS.md) | Small binary examples and expected field values |

## FastLanes coefficient files

The `.fls` payload uses the bundled FastLanes implementation. Its
[FileHeader](../../../third_party/fastlanes/src/include/fls/file/file_header.hpp),
[FileFooter](../../../third_party/fastlanes/src/include/fls/file/file_footer.hpp),
and [descriptor schema bindings](../../../third_party/fastlanes/src/include/fls/footer/table_descriptor_generated.h)
are part of the decoding dependency. Paths above are relative to this repository's
vendored revision. FastLens's metadata specifications do not replace the FastLanes
codec or FlatBuffers schema. The Image-major compact descriptor replaces the table
descriptor, while retaining the compressed coefficient payload and FLS framing.

All FastLens integers are little-endian unless an individual text/hash contract
explicitly specifies otherwise. `uN` / `iN` mean unsigned / signed N-bit integers.
Byte offsets are zero-based, with no implicit C/C++ struct padding. In Magic
strings, `\0` denotes one zero byte, not two printable characters.
