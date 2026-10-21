# §6.4 — Compressed transfer and direct tensors (Figure 15)

Defaults: MobileNet24, 16,384 images, batch 64, crop 448, three paired trials, separate correctness/warmup/Nsight/memory passes. --timing-only omits diagnostics. Requires results/preparation/raw_int16_phase2_16384/preparation.json and both raw-int16 and compressed control datasets.

```bash
python -m fastlens.benchmarks.paper.section_6_4.compressed_transfer_and_tensor.run --help
python -m fastlens.benchmarks.paper.section_6_4.compressed_transfer_and_tensor.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_4.compressed_transfer_and_tensor.run plot
```

Published records (repository-relative, unprefixed batch names are under `results/`):

`results/paper_rerun_20261007/summary.csv; results/paper_rerun_20261006/{4090,H100}/core/*/diagnostic/summary.json`

See [the reviewer guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.
