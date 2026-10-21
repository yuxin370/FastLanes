# §6.5.1 — Block-major locality (Figure 16)

Defaults: both GPUs, four 1,024-image cohorts, seed 11997733, batches 64/1,024, shared/correlated/independent crops, one warmup/correctness pass plus three measured trials. Requires aligned block-major and image-major-tiled validation layouts; tensor equality is checked before the paired trials.

```bash
python -m fastlens.benchmarks.paper.section_6_5_1.block_major_locality.run --help
python -m fastlens.benchmarks.paper.section_6_5_1.block_major_locality.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_5_1.block_major_locality.run plot
```

Published records (repository-relative, unprefixed batch names are under `results/`):

`results/paper_rerun_20261007/summary.csv`

See [the reviewer guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.
