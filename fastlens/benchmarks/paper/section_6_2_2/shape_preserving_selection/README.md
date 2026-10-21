# §6.2.2 — Shape-preserving selection (Table 2)

Actions: `prepare`, `run`, `table`. Run `paper.prepare_sources` first. The H100 defaults cover three models × D4/D6/D8 × five paths. IC uses fixed-frequency stores; FS and FastLens use full-frequency stores. Verification is regenerated before timing.

```bash
python -m fastlens.benchmarks.paper.section_6_2_2.shape_preserving_selection.run --help
python -m fastlens.benchmarks.paper.section_6_2_2.shape_preserving_selection.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_2_2.shape_preserving_selection.run table
```

Published records (repository-relative, unprefixed batch names are under `results/`):

[`paper/results/section_6_2_2/selected_inference.csv`](../../results/section_6_2_2/selected_inference.csv)

See [the reviewer guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.
