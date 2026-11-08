# Binary test vectors

The [format fixtures](../../tests/fixtures/formats) contain a generated 16×16 RGB,
4:2:0 JPEG and two actual native writer outputs:

| Directory | Expected interpretation |
| --- | --- |
| `block-major/` | One image, manifest 1, image metadata 3, geometry 1; includes dataset index and retained source-order binding |
| `image-major/` | One image, manifest 3, image metadata 3, Compact Descriptor 3; tiled-z32, vector size 1024 |

Each directory includes `expected.json` with all documented header fields,
metadata section IDs/lengths, image and rowgroup counts. These are field-level
test vectors, not benchmark data. `source.jpg` was generated from a 16×16 RGB
gradient `(13*x, 11*y, 7*(x+y))`, JPEG quality 90 and 4:2:0. Its pixels have no
third-party source. The four real Quickstart photographs are separate.

Run the independent standard-library parser without CUDA or a native build:

```bash
python -B -m unittest fastlens.tests.test_binary_spec -v
```

It reads offsets and scalar types directly from the normative Markdown tables,
then parses the actual files, checks section sizes and recalculates CRCs with a
bitwise reference implementation. It does not call production parsers. The native
CLI integration suite also runs this parser on fresh writer outputs. These checks
validate the FastLens directories/metadata; they do not implement a second FLS
coefficient decoder. Source-exact native verification covers coefficients.

To generate fresh outputs, use a new output directory:

```bash
python -m fastlens convert --input-list /path/to/one-source-jpeg.txt \
  --output "$HOME/tmp/fastlens-vector-block" --verify full
build/fastlens/tools/jpeg_dct/fastlens_jpeg_dct_tool \
  --shard --out-dir "$HOME/tmp/fastlens-vector-image" \
  --physical-layout image-major-vector-rowgroups --spatial-order tiled-z-32 \
  --metadata-profile reconstruct fastlens/tests/fixtures/formats/source.jpg
```

The ordered input list must point to the bundled `source.jpg`. Native generation
directory names and absolute source-path hashes can differ between runs. Compare
decoded fields and bindings rather than assuming those incidental names are
byte-identical. The distributed Block-major vector omits removable conversion
history and retains the required source-order digest. Future fixture changes must
be accompanied by reviewed expected field values.
