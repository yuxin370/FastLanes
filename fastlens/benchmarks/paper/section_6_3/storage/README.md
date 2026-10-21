# §6.3 — Storage costs (Table 3)

Actions: `run`, `table`. `table` prints all eight representations in the final
PDF's Table 3 at its published precision. `run` stats existing JPEG/FastLens/FFCV
files; it does not convert datasets. The retained raw storage measurements cover
these representations, not every Table 3 row. IC/FS sizes are recorded by
`baselines.dct_baselines` build, and L3 uses `baselines.l3.storage`; their full
train/val conversion artifacts remain required to remeasure the complete table.
Override `--ffcv-beton` for the measured FFCV file.

```bash
python -m fastlens.benchmarks.paper.section_6_3.storage.run --help
python -m fastlens.benchmarks.paper.section_6_3.storage.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_3.storage.run table
```

Published records (repository-relative, unprefixed batch names are under `results/`):

See [the complete published table](../../results/section_6_3/storage_table3.csv)
and [byte-level measurements](../../results/section_6_3/storage_measurements.csv).
The published FFCV-JPEG validation size includes the external sample mapping;
the older byte table counts only `.beton`. The
[result notes](../../results/README.md#storage-evidence-and-source-boundaries)
explain the 2.157 GB versus 2.180 GB accounting.

See [the reviewer guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.
