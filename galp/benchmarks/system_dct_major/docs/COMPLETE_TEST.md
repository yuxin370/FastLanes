# Complete DCT-major production-profile test

Last updated: 2026-08-14

This is the current publication suite for the no-shuffle DCT-major benchmark.
Runtime details such as segment policy, selected-decode mode, bounded reads,
workset capacity, double buffering, stream priority and kernel launch shape are
owned by runtime policy `block-major-p4-scheduled-bounded-110-v1`; they are not
suite options.

The supported comparison matrix is exactly:

- `dct_major_pushdown`;
- `rgbnomore`;
- `dali`;
- `pytorch`.

Old planless/legacy/full/image-major aliases and the tuning switches recorded
in dated reports are intentionally rejected by the current runner.

## Inspect the complete contract

`--dry-run` creates `suite_plan.json` and prints every command without starting
GPU work:

```bash
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run_suite.py \
  --block-major-access-dir /path/to/block-major-access \
  --output-dir /tmp/galp-dct-major-complete-dryrun \
  --dry-run
```

## Run and resume

Run in a fresh directory:

```bash
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run_suite.py \
  --block-major-access-dir /path/to/block-major-access \
  --output-dir /tmp/galp-dct-major-complete
```

After an interruption, use the identical arguments plus `--resume`. Completed
phases are skipped only when their recorded command matches. An incomplete
phase directory is never overwritten; inspect or move that one directory
before resuming.

## Current phase matrix

| Phase | Workload | Pipelines / domain |
| --- | --- | --- |
| `01_feature_smoke` | feature extraction | all four pipelines |
| `02_evaluation_smoke` | evaluation | all four pipelines |
| `06_formal_feature_extraction` | feature extraction | all four pipelines |
| `07_formal_evaluation` | evaluation | all four pipelines |
| `08_model_ceiling_0_*` through `08_model_ceiling_3_*` | synthetic model ceiling | DCT/RGB × feature/evaluation |

With the defaults, the exact accounting is:

| Class | Model invocations |
| --- | ---: |
| Two smokes | 32 |
| Two formal runs (`4 × 50,000 × 5` each pair) | 2,000,000 |
| Four model ceilings (`4 × 50 × 300`) | 60,000 |
| **Total** | **2,060,032** |

## Acceptance boundary

Every `run.py` phase validates the immutable input snapshot, exact no-shuffle
sample trace, semantic artifacts, current runtime policy identity, native
resource/I/O counters, repeat stability and source fingerprints. The suite
stops at the first failed phase and writes a failure marker next to its log.

The primary outputs are `suite_plan.json`, per-phase `results.json` and the
final `suite_results.json`. DCT-major versus RGB pipelines is deployment-level
context because the model/input domains differ; semantic claims are enforced
only for comparisons with a valid common contract.

The fixed center crop for this suite is a benchmark profile for 512×512 source
images. It must not be treated as a universal crop rule. The ordinary
RGB-no-more validation profile derives its logical 32-block reference grid
from the source/crop relation and produces the 28-block model grid.
