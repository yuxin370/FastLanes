# JPEG dataset conversion

[Documentation index](../../README.md#documentation)

Run commands from the repository root.

## Convert and inspect

Four [bundled example photographs](../data/example-jpegs/README.md) are ready in
`fastlens/data/example-jpegs/`: 512×512 RGB JPEGs with 4:2:0 chroma subsampling.
No external dataset preparation is needed for this example.

From the source checkout, convert and inspect the complete dataset:

```bash
python -m fastlens convert fastlens/data/example-jpegs \
  --output fastlens/data/example-fastlens --layout block-major \
  --partition-images 1024 --workers auto --verify full
python -m fastlens inspect fastlens/data/example-fastlens
python -m fastlens convert --help
```

The CLI currently runs from a **source checkout** with `python -m fastlens`;
a pip/wheel distribution and installed console script are not provided.
No PyTorch import or CUDA device is needed for conversion, inspection, or CLI unit
tests. Building the existing native targets still requires the CUDA toolkit.
If you only need to rebuild the converter's tools after configuring CMake:

```bash
cmake --build build --target fastlens_jpeg_dct_tool fastlens_block_major_access_tool --parallel
```

Tool discovery uses `--build-dir`, then `FASTLENS_BUILD_DIR`, then `build/`,
`build-native/`, and sorted `cmake-build-*` directories. An explicitly selected
build must contain both tools; the CLI does not silently switch builds.

FastLens stores **original JPEG quantized DCT coefficients in FastLanes format**,
plus quantization tables and spatial geometry. Conversion does not decode to RGB,
resize, or re-encode JPEGs. The goal is selective, high-throughput execution;
FastLens storage is **not guaranteed to be smaller than JPEG**. Reconstructable
metadata preserves the coefficient interpretation, not the original JPEG byte
stream or EXIF/APP markers.

The converter accepts libjpeg-supported JPEGs, including progressive and grayscale
images and differing dimensions/sampling grids. Extensions are `.jpg`, `.jpeg`,
and `.jpe`, case-insensitively. Malformed files, including files that libjpeg could
recover with corruption warnings, fail conversion even with `--verify none`.
Model profiles can impose stricter geometry requirements: the example uses
512×512 RGB, 4:2:0 JPEGs. Prepare that geometry explicitly before conversion
if needed, and keep source files unchanged throughout conversion and verification.

Directory discovery is recursive, without traversing directory symlinks. It sorts
relative POSIX paths lexicographically by Unicode code point, case-sensitively.
Image IDs are consecutive, starting at zero in that order unless optional
`--shuffle-seed` premixing is requested (see [annotations and premixing](ANNOTATIONS.md)). To choose the IDs:

```bash
python -m fastlens convert --input-list ordered_paths.txt --output ./test_store
```

The UTF-8 list contains one unquoted JPEG path per line, in the exact desired order;
relative paths resolve against the list's directory. Blank lines are ignored, CRLF
is accepted, and spaces are preserved. Paths containing newlines are unsupported.
Empty datasets, missing files, and duplicate resolved paths (including symlink
aliases) are rejected. `conversion/sources.txt` records the resulting absolute paths in
image-ID order. It is useful for re-verification; the online reader does not need
the original JPEGs or this list.

`block-major` is the public Converter layout: native `--shard --physical-layout
spatial-major --metadata-profile reconstruct`, manifest v1. Corresponding spatial
blocks are grouped across images, with DCT frequencies stored as columns.
The [Image-major native writer](spec/IMAGE_MAJOR.md#generate-the-layout) is a
separate entry point with two supported encodings: one image per rowgroup, selected
by native `random-access`, and explicitly selected tiled vector rowgroups. The
public Converter's `balanced`, `throughput` and `crop-latency` presets all write
Block-major; `random-access` is a native-only preset.

Default `--verify full` invokes the native exact
DCT verifier for every image before publication. Verification compares all 64
coefficients, image dimensions, component geometry and sampling, and quantization
table IDs and values against the source JPEGs. `--verify none` skips only that
source comparison; index construction and compatibility checks still run.

## Rebuild an older Store without changing sample order

For relocation across machines, first read
[Cross-machine reproduction and relocation](#cross-machine-reproduction-and-relocation).

Keep the old Store and rebuild into a new directory from the original JPEGs and
annotation input. Old generation/index directory names and annotations v1 are
not migrated by renaming files: the final manifest, geometry bindings and
annotations v2 must be generated together.

The old `source_paths.txt` already contains the **final physical order**, including
any offline shuffle. Use it as `--input-list` and **omit `--shuffle-seed`**. Applying
the old seed to that list would shuffle it a second time. For a current Store,
the equivalent list is `conversion/sources.txt`.

```bash
OLD_STORE=/datasets/old_store
NEW_STORE=/datasets/rebuilt_store
IMAGE_ROOT=/datasets/imagenet512
LABELS_CSV=/datasets/indexbase_val.csv

python -m fastlens convert --input-list "$OLD_STORE/source_paths.txt" \
  --image-root "$IMAGE_ROOT" --labels-csv "$LABELS_CSV" \
  --output "$NEW_STORE" --partition-images 1024 --preset balanced \
  --workers auto --verify full
python -m fastlens inspect "$NEW_STORE"
python -m fastlens annotations inspect "$NEW_STORE" --json
```

Use the original image root and label convention so logical sample IDs and class
IDs remain unchanged. The example's partition size and preset must also match the
intended experiment. Folder labels require the original fixed class map; JSONL
inputs require the original `path,label[,sample_id]` records. The generated
`annotations/samples.jsonl` is an output format, not a `--labels-jsonl` input.

Compare the old and new `(image_id, sample_id, label)` sequences before switching
the application's manifest path. With an existing sample sidecar, this check
does not need to load the old annotation schema:

```python
import json
from pathlib import Path
from fastlens.annotations import load_annotations

old = Path("/datasets/old_store")
new = Path("/datasets/rebuilt_store")
old_samples = [json.loads(line) for line in (old / "annotations/samples.jsonl").read_text().splitlines()]
new_samples, metadata = load_annotations(new)
assert new_samples == old_samples
assert (new / "conversion/sources.txt").read_bytes() == (old / "source_paths.txt").read_bytes()
```

The new metadata records `shuffle: null` because this conversion consumes an
already ordered list; it does not apply a new shuffle. The sample-order hash and
sample/label sequence remain unchanged. Retain the old report for the original
shuffle provenance. Reusing an original *pre-shuffle* list with the original seed
is another option, but requires the same membership, initial order and algorithm.

After source verification and Reader/label checks pass, point the downstream
application to `NEW_STORE/manifest.bin`; keep the old Store until those checks
complete. PLS experiments additionally require matching physical partition
boundaries and their dedicated mapping/contract hashes; a generic reconversion
does not regenerate those experiment contracts.

## Publication and reports

Optional complete FLS file checksums can be recorded before publication:

```bash
python -m fastlens convert fastlens/data/example-jpegs \
  --output fastlens/data/example-fastlens-crc --record-payload-crc --verify full
python -m fastlens inspect fastlens/data/example-fastlens-crc --check-payload
```

The extra scan uses CRC64-ECMA and writes `geometry/fls_crc64.txt` before the
conversion report. Explicit inspection streams complete `.fls` files and fails
on missing checksums or mismatches. Default inspection stays structural; Reader
does not use this file. Checksums remain usable after removing source JPEGs and
`conversion/`. They detect accidental damage, not malicious replacement. See the
[checksum format](spec/STORE.md#optional-complete-fls-checksums).

Conversion uses a sibling `.OUTPUT.fastlens-staging-<id>` directory and preserves
the native writer's transactional generations within it. After compression, the
Converter finalizes `partitions/` and rewrites the manifest with the native
serializer. It then builds the geometry index against that final manifest,
performs the requested exact verification, and writes annotations and the report.
Only after all phases pass is the directory atomically renamed to the destination. A nonempty destination or destination symlink is
refused; use a new path. Ordinary errors and Ctrl-C clean staging data and preserve
any existing destination. SIGKILL, machine failure, or abrupt termination may leave
the named staging directory. Once the converter and its native child have stopped,
remove that specific staging directory and rerun; there is no resume operation.
Publication provides process-level atomic visibility, not power-loss durability.

`--dry-run` checks discovery, paths, tools, and parallel bounds without creating
the destination or decoding JPEGs. `--report /existing/parent/report.json` also saves
the report at a new external path; existing report files are protected. The report
contains observed source/file sizes, chosen options, wall times, verification
status, source revision, and native executable paths/sizes/timestamps. It defines
source/output as original JPEG bytes divided by all stored file bytes (including
the report and source list). `compressed_payload_bytes` counts complete FLS files,
including their headers and footers. Timings cover discovery, compression, layout
finalization, index construction/inspection, annotations and verification; total time ends before report writing
and directory publication. No estimated throughput is substituted. Index storage
targets of 0.5%/1% are informational; small stores can exceed them.

For the required files, inspection checks and storage accounting, see
[Store format](STORE_FORMAT.md). For labels and premixing, see
[Annotations](ANNOTATIONS.md).

## Verification and parallelism

`inspect` checks structure and identity bindings. `inspect --check-payload` also
checks recorded complete-file CRCs. Source-exact verification below compares the
stored DCT interpretation with original JPEGs. None substitutes for the others.

For a later exact check against the original files:

```bash
build/fastlens/tools/jpeg_dct/fastlens_jpeg_dct_tool \
  --verify-manifest ./test_store/manifest.bin --verify-workers 4 \
  --input-list ./test_store/conversion/sources.txt
```

Parallel options affect different phases:

| Option | Default and resource effect |
| --- | --- |
| `--workers auto\|N` | Budget used to derive defaults. Auto uses half the CPUs visible through process affinity, at least 1 and at most 32. This is conservative, not optimal tuning. `N` is 1–256. Explicit phase options override derived values. |
| `--layout-threads N` | Header/layout scan before shard processing; defaults to the budget. More threads increase concurrent file access. |
| `--shard-workers N` | Concurrent resident shards; defaults to `min(4, max(1, budget // 4))`. Increases CPU use and peak memory substantially. |
| `--shard-decode-threads N` | Coefficient decoders per active shard. Default `min(4, max(1, budget // shard_workers))`. |
| `--encoding-workers-per-shard N` | FastLanes rowgroup encoders per active shard; same derived default as decode. Additional workers hold additional encoder buffers. |
| `--verify-workers N` | Independent shard verifiers; default `min(4, budget)`, maximum 32. Each worker holds decompressed shard/source state. |
| `--partition-images N` | Maximum images per shard, default 1024, range 1–65535. Native rowgroup capacity may split earlier. Larger values increase per-shard memory. |
| `--preset balanced\|throughput\|crop-latency` | Existing native presets, with 128/256/64 vectors per rowgroup respectively. Larger rowgroups generally require more memory. Encoder algorithms/defaults are unchanged. |

Layout scan and shard processing are sequential. Within a shard, JPEG decode and
FastLanes encoding are sequential, so their thread counts do not multiply. Native
bounds remain `layout_threads <= 256` and
`shard_workers * max(shard_decode_threads, encoding_workers_per_shard) <= 256`.
The CLI rejects violations rather than silently clamping explicit options. Lower
partition size and shard/verification concurrency on memory-constrained machines;
automatic CPU selection does not measure available RAM or tune to workload size.

## Cross-machine reproduction and relocation

Moving a published Store means copying `manifest.bin`, `partitions/`, `geometry/`
and any needed annotations together. Relative member paths and recorded identities
remain valid; do not rewrite or recalculate them. Optional `fls_crc64.txt` moves
with geometry and remains usable. `conversion/` can be removed for deployment.

Rebuilding consumes the original physical-order JPEG list and matching sample IDs
and labels. Preserve that list, source mapping and class map separately if the
conversion records will be deleted. A physical-order list is already shuffled;
omit `--shuffle-seed` when replaying it.

On another machine, resolve each original source path to its new location while
retaining the same logical sample ID and input position. Identical JPEG content
under different absolute paths can produce a different `source_order_sha256`:
that digest hashes the original absolute UTF-8 path list, not image content.
The sample-order digest can remain identical when sample IDs and order are retained,
while source provenance changes. Exact compressed bytes are not promised across
encoder/build changes.

If content-level dataset identity is required, retain an independent stable sample
list and content digest in the experiment's existing records. This does not change
FastLens's source-order algorithm or reinterpret existing hashes.
