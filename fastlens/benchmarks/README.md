# FASTLENS benchmarks

Choose the entry point by purpose:

| Directory | Purpose |
| --- | --- |
| [`paper/`](paper/README.md) | Reproduction of `fastlens.pdf`, organized by section and experiment, with measured defaults |
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

Benchmark implementations, published measurements and reproducibility
instructions are organized separately. Start with the
[paper reproduction guide](paper/README.md) for evaluated workloads, and use
package READMEs for lower-level APIs and dependencies. Published measurements
and their source references are in [paper/results/](paper/results/README.md).

Datasets, checkpoints and profiler captures are external inputs or generated
outputs. Each experiment guide describes how to prepare its inputs. Write new
runs to a separate output directory; the published tables are reference data.
