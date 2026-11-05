# §6.2.2 — Compact input (Figure 10)

Run `paper.prepare_sources` first. Defaults use the 8,194-image training population,
D3/D15, three models, five paths and both GPUs. Repeat counts follow the published
configuration; use `--repeats` to override them.

```bash
python -m fastlens.benchmarks.paper.section_6_2_2.compact_input.run --help
python -m fastlens.benchmarks.paper.section_6_2_2.compact_input.run prepare
python -m fastlens.benchmarks.paper.section_6_2_2.compact_input.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_2_2.compact_input.run run
python -m fastlens.benchmarks.paper.section_6_2_2.compact_input.run plot
```

Published results: [compact_training.csv](../../results/section_6_2_2/compact_training.csv).

See [the paper reproduction guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.
