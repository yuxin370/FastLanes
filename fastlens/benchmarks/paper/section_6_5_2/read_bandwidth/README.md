# §6.5.2 — Input read bandwidth

Defaults: both GPUs, crops 448/160, 125/500/2,000 decimal MB/s, three trials, batch 64, 16,384 images. The limiter is shared by all reader threads and paces application read completion on warm source pages; it does not simulate network latency or a cold disk. Requires the input-boundary control library from the paper build.

```bash
python -m fastlens.benchmarks.paper.section_6_5_2.read_bandwidth.run --help
python -m fastlens.benchmarks.paper.section_6_5_2.read_bandwidth.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_5_2.read_bandwidth.run plot
```

Published records (repository-relative, unprefixed batch names are under `results/`):

`results/paper_rerun_20261007/remaining_continue/{4090,H100}/bandwidth/summary.csv; results/bandwidth_recheck_20261012_2000/H100/bandwidth/summary.csv`

See [the reviewer guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.
