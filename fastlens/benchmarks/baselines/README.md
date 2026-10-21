# Baseline implementations

Run modules from the repository root. Baseline code and its tests live here;
the final-paper experiment entry points and measured defaults are in
[`../paper/`](../paper/README.md).

| Package | Contents | Used by the final PDF |
| --- | --- | --- |
| [`dct_baselines/`](dct_baselines/README.md) | IC/FS stores, JPEG-DCT reference readers, selected-frequency and compact-input comparisons | Yes |
| [`system_rgbnomore/`](system_rgbnomore/README.md) | PyTorch/RGB-no-more and DALI readers, RGB/DCT comparison runners, shared model and training support | Yes |
| [`l3/`](l3/README.md) | L3 codec, DALI operator, storage accounting and model-input baseline | Yes |
| [`ffcv/`](ffcv/README.md) | FFCV input preparation and verification | Yes |
| `nvcomp/` | nvCOMP implementations and their C++ headers | Development compression comparisons |

The RGB package also contains the native FastLens comparison arm because its
runners share input and timing contracts. Model adapters remain in
[`../dct_models/`](../dct_models/README.md), and PLS recipes remain in
[`../training_pls/`](../training_pls/). These are shared workload implementations.
CoorDL and FFCV adapters are in
[`system_rgbnomore/shared/paper_adapters.py`](system_rgbnomore/shared/paper_adapters.py),
with the measured training inputs and FFCV crop decoder under
`system_rgbnomore/training/`. Final-paper scheduling and result collection are
in `../paper/`; historical run queues are excluded from this submission.

Examples:

```bash
python -m fastlens.benchmarks.baselines.dct_baselines --help
python -m fastlens.benchmarks.baselines.system_rgbnomore.inference.run --help
python -m fastlens.benchmarks.baselines.l3.run --help
python -m fastlens.benchmarks.development.check
```

Tests remain alongside the implementations. Historical result files retain the
measured values and source digests; source paths follow the current layout.
The dated DCT shell launchers are development continuation/tuning tools and may
require their original prepared inputs and accepted verification records.
Reviewers should use the section-specific `paper/` commands.
