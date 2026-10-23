# §6.3 — Host-to-device traffic (Figure 14)

Defaults: RTX 4090, ViT-Ti/MobileNet24, training and inference. H2D capture uses
Nsight Systems and counts CUPTI copy bytes separately from timing trials.
For a fresh checkout, follow the
[reference rebuilding guide](../../REBUILDING_REFERENCES.md#capture-h2d-traffic)
to generate contracts and profile a newly constructed workload.

The chapter entry point profiles newly completed trials under `--run-dir`
(default: `results/reproduced`). It reads their `run.command.json` and
`result.json`, selects one trial per workload and checks the replay against that
trial. Correctness checks and resident-only controls are excluded.

```bash
python -m fastlens.benchmarks.paper.section_6_3.h2d_traffic.run --help
python -m fastlens.benchmarks.paper.section_6_3.h2d_traffic.run run --run-dir results/reproduced --dry-run
python -m fastlens.benchmarks.paper.section_6_3.h2d_traffic.run run --run-dir results/reproduced
python -m fastlens.benchmarks.paper.section_6_3.h2d_traffic.run plot
```

Published records (repository-relative, unprefixed batch names are under `results/`):

`results/section42_resources_20261008/measurements.csv; results/ffcv_h2d_20261014/measurements.csv`

See [the reviewer guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.

The [published traffic table](../../results/section_6_3/h2d_traffic.csv) contains the 28 points used in Fig. 14. Memory-diagnostic passes and unused models remain in the historical source CSV rather than the reported table.
