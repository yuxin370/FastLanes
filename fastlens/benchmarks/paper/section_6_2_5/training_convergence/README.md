# §6.2.5 — Training convergence (Figure 12)

Defaults: SwinV2 and MobileNet24, both crop policies, 300 epochs, 8 processes, BF16, microbatch 64/global batch 1,024, seed 11997733, 4 segments per pool, pread. This requires the A800-scale training inputs and validation layouts. Historical shared MobileNet training changed GPU count during continuation; a fresh 8-GPU run follows the recipe, not the historical interrupted execution timeline.

```bash
python -m fastlens.benchmarks.paper.section_6_2_5.training_convergence.run --help
python -m fastlens.benchmarks.paper.section_6_2_5.training_convergence.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_2_5.training_convergence.run plot
```

Published records (repository-relative, unprefixed batch names are under `results/`):

`a800_convergence_20261003/figures/convergence_observed.csv; a800_swin_independent_20261012/remote/independent/metrics.jsonl; a800_crop_policy_20261010/remote/independent/metrics.jsonl`

See [the reviewer guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.
