# §6.2.3 — Deployment performance (Figure 11a)

Five DCT profiles; six RGB baselines. The two MobileNet DCT profiles share one RGB model. Training/inference and model filters use the same protocol as matched DCT, so no duplicate measurement implementation is maintained.

```bash
python -m fastlens.benchmarks.paper.section_6_2_3.deployment.run --help
python -m fastlens.benchmarks.paper.section_6_2_3.deployment.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_2_3.deployment.run plot
```

Published records (repository-relative, unprefixed batch names are under `results/`):

`resident_study_20261011_fixed, remaining_baselines_20261012, swin_inference_20261011_fixed_after_training`

See [the reviewer guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.
