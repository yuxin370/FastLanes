# §6.2.1 — Matched DCT performance (Figure 9)

Uses seven models and both GPUs by default. `--models`, `--gpus`, `--phases` and `--repeats` select a subset. Training checks precede three interleaved restored-state trials. Inference takes fresh prediction/contract inputs through `--references FILE`; see the [rebuilding guide](../../REBUILDING_REFERENCES.md#connect-fresh-references-to-the-chapter-entry-points).

```bash
python -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run --help
python -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run plot
```

Published records (repository-relative, unprefixed batch names are under `results/`):

`resident_study_20261011_fixed, remaining_baselines_20261012, swin_inference_20261011_fixed_after_training, resnet_matched_20261013`

See [the reviewer guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.
