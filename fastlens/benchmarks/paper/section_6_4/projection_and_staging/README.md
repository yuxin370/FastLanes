# §6.4 — Projection pushdown and compact staging (Figure 13)

Defaults reproduce the RTX 4090 projection point at K=8 and the separate H100 staging controls (16,384 images, crops 448/160, three trials). `--experiments` selects one; `--gpus` overrides the published GPU choice. The full frequency sweep remains available in the development driver. Requires the block-major validation data; staging also needs build/fastlens/benchmarks/libinput_boundary_controls.so, built by paper.reproduce build.

```bash
python -m fastlens.benchmarks.paper.section_6_4.projection_and_staging.run --help
python -m fastlens.benchmarks.paper.section_6_4.projection_and_staging.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_4.projection_and_staging.run plot
```

Published records (repository-relative, unprefixed batch names are under `results/`):

`results/paper_rerun_20261007/remaining/H100/staging/summary.csv; results/paper_rerun_20261007/summary.csv`

See [the reviewer guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.

Published report tables: [projection](../../results/section_6_4/projection.csv) (one point) and [staging](../../results/section_6_4/staging.csv) (six trial rows).
