# DCTNet and eFUN workloads

This directory contains model adapters and reproducible inference/training
workloads. Shared training recipes, augmentation, audit policy and the canonical
PLS schedule live in `galp.benchmarks.training_pls`; RGB and DALI adapters live in
`galp.benchmarks.system_rgbnomore.training`.

Run commands from the repository root. See [benchmark setup](../README.md) for
Python dependencies, module paths and the CMake storage bridge. DCTNet and FUN
source trees and their official checkpoints are external, ignored inputs.
`backend.py` selects DCTNet with `DCTNET_PROFILE`; `efun.py` installs the eFUN
backend for the same workload modules. See [EFUN.md](EFUN.md) for its upstream
recipe and checkpoint setup.

| Entry | Purpose |
| --- | --- |
| `online_crop.py`, `online_crop_suite.py` | Online source-DCT crop, projection and native scheduling comparisons |
| `evaluate.py`, `evaluate_rgb.py`, `evaluate_shards.py` | DCT reference, stored-DCT geometry, RGB and native inference routes |
| `dct_geometry.py` | Shared DCT geometry adapter for stored source grids |
| `training_pls.py` | A0 JPEG-DCT, B6 native and RGB/DALI training with shared recipes |
| `cnn_training_baselines.sh` | One matrix launcher for all training backends |
| `training_model_only.py`, `training_model_only_matrix.sh` | Model-only training calibration |
| `generate.py`, `validate.py`, `storage.py` | Offline target-grid data construction and validation using `galp_dct_storage` |
| `capture.py`, `profile.sh`, `analyze_*.py` | Explicit profiling and trace analysis |

## Online inference

Use the full-frequency source representation for online crop experiments. Input
paths and preparation are described in [ONLINE_CROP.md](ONLINE_CROP.md) and the
[data inventory](../../docs/DATASETS.md).

```bash
python -m galp.benchmarks.dct_models.online_crop_suite \
  --source-data /path/to/source-dct \
  --output-dir "$HOME/tmp/galp-online-crop"
```

The offline target grids under `backup/offline_model_input` are useful for
isolating projection and coefficient selection. They do not measure online
source crop. The stored-DCT adapter also has different geometry from the official
pixel-resize/re-encode reference; compare its tensors and accuracy explicitly.

For the offline comparison across all four DCTNet configurations:

```bash
export CUDA_VISIBLE_DEVICES=GPU-REPLACE_WITH_YOUR_UUID
export PYTHON=python
bash galp/benchmarks/dct_models/run_inference.sh "$CUDA_VISIBLE_DEVICES" inference_run
```

Use `MEASURE_PROCESS_MEMORY=1` to collect process memory alongside that matrix.
Select a fresh result name when changing measurement parameters.

## Training

The matrix accepts `probe`, `full`, `data`, `profile` or `plan`, a required result
name, and space-separated `model:backend` pairs. Models are `mobilenet24`,
`mobilenet32`, `resnet24`, `resnet64` and `efun`; backends are `native`, `jpeg`,
`rgb_pytorch`, `rgb_d2` and `rgb_d3`.

```bash
bash galp/benchmarks/dct_models/cnn_training_baselines.sh plan training_run \
  'mobilenet24:native mobilenet24:jpeg mobilenet24:rgb_pytorch'
bash galp/benchmarks/dct_models/cnn_training_baselines.sh full training_run \
  'mobilenet24:native mobilenet24:jpeg mobilenet24:rgb_pytorch'
bash galp/benchmarks/dct_models/training_model_only_matrix.sh calibration_run
```

The launcher reads the premixed training representation's existing
`materialization_contract.json`. Use `python -m
galp.benchmarks.dct_models.training_pls --help` for direct manifest, mapping,
checkpoint-resume and scheduling arguments. This model-specific driver reuses the
shared PLS schedule; it provides DCTNet/eFUN geometry and model execution.

A0 and B6 compare independent crop/global shuffle with grouped crop/closed-pool
shuffle. Keep initialization, sample population, optimizer, precision and
validation preprocessing fixed. RGB/DALI are system baselines with different
input semantics. Model-only calibration excludes I/O, augmentation and H2D.

## Tests and results

```bash
python -m unittest galp.benchmarks.dct_models.tests.test_online_crop
python -m unittest galp.benchmarks.dct_models.tests.test_efun
```

These integration tests require their upstream sources and dependencies. GPU
throughput and full training are explicit runs. Raw results, checkpoints and
traces remain ignored; historical performance reports record the hardware,
inputs and measurement conditions of their original runs.
