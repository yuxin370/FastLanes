# §6.5.1 — Block-major locality (Figure 16)

Defaults: both GPUs, four 1,024-image cohorts, seed 11997733, batches 64/1,024, shared/correlated/independent crops, one warmup/correctness pass plus three measured trials. Requires aligned block-major and image-major-tiled validation layouts; tensor equality is checked before the paired trials.

## Prepare and validate

Follow [validation JPEG and tiled Image-major preparation](../../REBUILDING_REFERENCES.md#prepare-the-validation-source)
and [native Block-major preparation](../../REBUILDING_REFERENCES.md#prepare-native-source-dct-inference)
on the same 50,000-image source. The current worker reads these repository-relative
directories under `fastlens/data/compressed/`:

- `imagenet512_val_compact_v3/`: tiled Image-major manifest, FLS/metadata and
  `index.csv` from the preparation tool's `--index-file` input.
- `imagenet512_val_block_major/`: Block-major manifest, FLS/metadata, geometry,
  and physical-order `samples.json`.

The Image-major command must explicitly select `image-major-vector-rowgroups`;
the `random-access` preset alone selects the older one-image-per-rowgroup encoding.
Use the full validation population: the four measured cohorts begin at physical
IDs 0, 16,384, 32,768 and 48,128. The worker checks logical sample identities and
compares projected tensors exactly before the three timed trials. This is the
paper's Block-major/Image-major comparison, not an Image-major v2/v3 comparison.

## Run and outputs

```bash
python -m fastlens.benchmarks.paper.section_6_5_1.block_major_locality.run --help
python -m fastlens.benchmarks.paper.section_6_5_1.block_major_locality.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_5_1.block_major_locality.run plot
```

Published records (repository-relative, unprefixed batch names are under `results/`):

`results/paper_rerun_20261007/summary.csv`

New measurements are written to the selected `--output`, by GPU under `locality/`,
with per-trial records and layout counters. `plot` uses the retained published
records described in [results](../../results/README.md).

See [the paper reproduction guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.
