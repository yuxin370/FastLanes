# §6.2.4 — GPU-resident training reference (Figure 11b)

Training only. End-to-end windows consume 65,536 distinct images. Resident references repeat eight real batches (512 images) with the same model state; this is a throughput reference, not an accuracy experiment. All six RGB paths are measured for the best-RGB comparison.

```bash
python -m fastlens.benchmarks.paper.section_6_2_4.resident_training.run --help
python -m fastlens.benchmarks.paper.section_6_2_4.resident_training.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_2_4.resident_training.run plot
```

Published records (repository-relative, unprefixed batch names are under `results/`):

`resident_study_20261011_fixed, remaining_baselines_20261012`

See [the reviewer guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.
