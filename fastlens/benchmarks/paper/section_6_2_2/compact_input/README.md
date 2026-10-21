# §6.2.2 — Compact input (Figure 10)

Actions: `prepare`, `run`, `plot`. Run `paper.prepare_sources` first. Defaults use the 8,194-image training population, D3/D15, three models, five paths and both GPUs. `measured_repeats()` in run.py preserves the published single/three-repeat distinction; `--repeats` overrides it.

```bash
python -m fastlens.benchmarks.paper.section_6_2_2.compact_input.run --help
python -m fastlens.benchmarks.paper.section_6_2_2.compact_input.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_2_2.compact_input.run plot
```

Published records (repository-relative, unprefixed batch names are under `results/`):

[`paper/results/section_6_2_2/compact_training.csv`](../../results/section_6_2_2/compact_training.csv)

See [the reviewer guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.

## Current validation limitation

A newly prepared 9-image fixture passes input/label checks and a ViT training pass
for CPU-DCT, both IC layouts and FastLens. The FS prefetch path has failed the existing
1e-5 input gate: 2,182 of 190,512 elements differ, with maximum absolute difference
about 0.00610. Direct FS reads and a later standalone CPU check-only run on the same fixture
pass; the intermittent difference is unresolved. The gate remains enforced, and a failed check produces no throughput
result. This path is not yet validated for a fresh complete paper rerun.
