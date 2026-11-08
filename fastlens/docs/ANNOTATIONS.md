# Task annotations and offline premixing

[Documentation index](../../README.md#documentation)

Run commands from the repository root.

## Add annotations to a Store

An unannotated store produced by the current Converter can acquire annotations
without recompressing images, provided `conversion/sources.txt` is retained:

```bash
python -m fastlens annotations build ./train_store \
  --image-root /datasets/imagenet/train --labels-from-folders \
  --class-map ./imagenet_class_to_idx.json
python -m fastlens annotations inspect ./train_store --json
```

Only the manifest, retained source-order digest, source path list and supplied
annotations are read. Source JPEGs, coefficient files and binary geometry
descriptors are not needed for this operation.
`conversion/sources.txt` must be the original converter-produced list: row N denotes
physical `image_id=N`. Building annotations never changes that order.
`--image-root` names the source root recorded in that list; it need not still
contain the JPEGs. Moving the store does not require rewriting source paths.

Choose exactly one annotation source:

| Option | Input and interpretation |
| --- | --- |
| `--labels-from-folders` | `ROOT/CLASS/...` paths; class is the first directory below `--image-root`. |
| `--labels-jsonl FILE` | One JSON object with `path`, `label`, and optional `sample_id` per line. Relative paths resolve against the JSONL file's directory. |
| `--labels-csv FILE` | `Filepath,Label` columns, with optional `sample_id`. Relative paths resolve against `--image-root`, or the CSV directory when no root is given. Labels are nonnegative integers, preserved verbatim. |

For folder labels, a supplied `--class-map` is a JSON object such as
`{"n01440764": 0, "n01443537": 1}`. Values must be distinct nonnegative integers.
It may include classes absent from this split. Reuse the same map for train and
validation, including subsets. Without a map, class names present in the source
list are sorted by Unicode code point and numbered from zero. **These inferred
IDs describe this dataset, not necessarily ImageNet-1K or a checkpoint's output
classes.** Flat `ILSVRC2017_test_*.JPEG` filenames do not supply class labels;
they require a matching external annotation table.

CSV is optional: use folder labels directly for an ImageFolder dataset. Reusing
a baseline's CSV preserves its image-to-label assignments and numeric class
convention; it does not change the FastLanes storage format or automatically
reproduce the baseline's preprocessing or sample order. Directory names alone do not define a universal integer class order. See the
[RGB-no-more annotation example](../benchmarks/baselines/system_rgbnomore/README.md#imagenet-labels-and-conversion)
when reproducing that checkpoint's class numbering.

## CSV and JSONL contracts

The CSV contract is UTF-8 text with a comma-separated header. Column names are
case-sensitive:

| Column | Required | Meaning |
| --- | --- | --- |
| `Filepath` | Yes | Nonempty absolute or relative JPEG path. Relative paths resolve against the effective image root: `--image-root`, or the input directory during directory conversion. Without an image root, they resolve against the CSV's directory. |
| `Label` | Yes | Nonnegative integer class ID, preserved without renumbering. Classification applications must ensure it is within their model's output class range. Blank values and JSON `null` are not integer CSV labels; use JSONL for explicitly unlabelled or structured targets. |
| `sample_id` | No | Unique, nonempty logical identity. If omitted, the JPEG path relative to the image root is used; supply this column when a root-relative identity cannot be derived. It stays attached to the image during shuffle. |

Use standard CSV quoting for fields containing commas or quotes. Each selected
image must occur exactly once; extra images are rejected as unknown paths. CSV
rows are joined by path, so their order does not set physical image IDs. Use
`--input-list` to choose an input order and `--shuffle-seed` to premix it.

For a generic dataset rooted at `./images`, a CSV can contain:

```csv
Filepath,Label,sample_id
class-a/image1.jpg,0,example-1
class-b/image2.jpg,1,example-2
```

Use `--image-root ./images` when adding these labels to an existing Store.

JSON labels can be integers, arrays, objects, strings or explicit `null`:

```json
{"path":"images/a.jpg","sample_id":"sample-a","label":{"classes":[2,5]}}
{"path":"images/b.jpg","sample_id":"sample-b","label":null}
```

All selected images must have exactly one annotation. Unknown paths, duplicate
paths (including resolved aliases), missing records/labels and duplicate sample
IDs are errors. A full-dataset index must be explicitly filtered to the store's
subset before use. Explicit `null` means intentionally unlabelled; it does not
permit missing rows. NaN, Infinity and non-finite numbers are rejected.

Default `sample_id` is the relative POSIX path under `--image-root`. For sources
outside that root or with no common root, provide unique nonempty `sample_id`
values in JSONL/CSV. Absolute filesystem paths are not used as default identities.
The table's row order never determines image IDs.

## Annotation files and validation

The following directory is published by one rename after validation:

```text
STORE/annotations/
  samples.jsonl
  metadata.json
  class_to_idx.json    # folder-label mode only
```

The [annotation specification](spec/ANNOTATIONS.md) defines sample records, metadata
fields, hashes and validation. Loading needs neither `conversion/` nor source JPEGs.
Existing annotation directories, including symlinks, are never replaced.

`annotations build --dry-run` validates all supplied annotations without writing.
There is no UDF execution or import; a UDF interface is deferred. Adding annotations
does not rewrite `conversion/report.json`: its sizes remain the original conversion
measurements, while `fastlens inspect` reports current stored bytes.

## Conversion and deterministic premixing

The same source options are available during conversion. Directory conversion
defaults `--image-root` to its input directory; input-list conversion needs an
explicit root or explicit table sample IDs when annotations/premixing are requested.

```bash
python -m fastlens convert /datasets/imagenet/train --output ./train_store \
  --labels-from-folders --class-map ./imagenet_class_to_idx.json --shuffle-seed 42
python -m fastlens convert ./images --output ./flat_store \
  --labels-jsonl ./annotations.jsonl --shuffle-seed 42
```

`--shuffle-seed` is optional and accepts 0 through 2^64-1. It mixes entire
`(source_path, sample_id, label)` records once, before partitioning; the resulting
`conversion/sources.txt`, native writer/verifier and samples sidecar all share the same
physical order. Premixing without labels produces explicit null labels. With
neither labels nor premixing, conversion produces no annotation sidecar.

The [premixing specification](spec/ANNOTATIONS.md#deterministic-premixing) defines
the cross-version permutation. Identical initial order and seed reproduce the same
permutation; this does not promise byte-identical compressed
files. Source discovery or membership changes can change the permutation.

Conversion publishes annotations together with the verified dataset; annotation
failure also prevents dataset publication. `--dry-run` validates the preparation
plan without compression. Offline premixing does not replace epoch-time sampling.
In particular, these sidecars **do not implement the PLS training mapping
contract**: `TrainingPolicy.mapping` still requires its dedicated CSV with actual
shard membership and PLS fields. General JSON task annotations are interpreted by
the application, not the JPEG storage layer.

For model-side label lookup and the classification demo, see
[Direct-DCT data loading](DIRECT_DCT.md#labels-and-classification).
