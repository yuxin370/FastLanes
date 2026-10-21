# FASTLENS benchmarks

Choose the entry point by purpose:

| Directory | Purpose |
| --- | --- |
| [`paper/`](paper/README.md) | Reviewer reproduction of `fastlens.pdf`, organized by section and experiment, with measured defaults |
| [`development/`](development/README.md) | Supporting microbenchmarks, diagnostics and CPU regression checks |
| [`baselines/`](baselines/README.md) | RGB/IC/FS/L3/nvCOMP comparison implementations and their tests |
| [`paper/results/`](paper/results/README.md) | Complete published result tables, trial records and plotting inputs |
| Other packages below | Shared data readers, models, training recipes and benchmark implementations used by those entry points |

Run Python commands from the repository root. Paper measurements and development
checks are separate commands:

```bash
python -m fastlens.benchmarks.development.check
python -m fastlens.benchmarks.paper.reproduce figures
python -m fastlens.benchmarks.paper.section_6_2_2.compact_input.run --help
```

## Shared implementations

| Package | Role |
| --- | --- |
| `baselines/system_rgbnomore/` | RGB-no-more, native DCT, DALI and PyTorch system workloads |
| `system_dct_major/` | DCT-major crop/coefficient pushdown and common pipeline machinery |
| `training_pls/` | PLS recipes, optimizer, training and distributed convergence workloads |
| `dct_models/` | DCTNet/eFUN adapters, CNN inference and training workloads |
| `baselines/dct_baselines/` | IC/FS stores, shape-preserving selection and compact-input training |
| `baselines/l3/` | L3 storage and model input baseline |
| `baselines/ffcv/` | FFCV input preparation and verification |
| `paper/shared/` | Timing workers and accepted prediction references reused by several paper sections |
| `execution/` | Process execution, logging, resource guards and existing source snapshot checks |
| `profiling/`, `transfer_pipeline.py` | Instrumentation and transfer helpers |
| `include/`, `baselines/nvcomp/` | Shared C++ benchmark support and nvCOMP comparisons |
| `common.py`, `training_audit_policy.py` | Common source identity and existing training contracts |

Development-only studies (`dct_retokenization/`, `coefficient_mask_evaluator/`,
`metadata/`, `model_only_training_calibration.py`) are
not final-paper experiment entries.
Unit tests remain beside the implementations they test; `development.check`
collects the CPU suites without launching paper measurements.

See each package README for low-level APIs and external dependencies. Data,
checkpoints, profiler captures, historical queues and development reports are
excluded from the submission. Published measurements are in `paper/results/`.
Reviewers should use the chapter entries in `paper/`.

## What is not required to run a benchmark?

`__pycache__/`, historical `runs/`, profiler captures, generated figures and
old reports are outputs or reference material. They are not executable
dependencies. The tests, CMake file, shared headers, data readers, models and
training recipes are needed for building, validating or running their respective
workloads; a file is not redundant merely because it is not an entry point.
