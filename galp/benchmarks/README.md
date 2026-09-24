# GALP benchmarks and experiments

All benchmark and research source lives in this directory. Run Python modules
from the repository root, using the environment that provides the Torch binding:

```bash
export PYTHONPATH="$PWD:$PWD/build/galp/torch${PYTHONPATH:+:$PYTHONPATH}"
export RGBNOMORE_ROOT="$HOME/RGB-no-more"
```

| Directory / entry | Purpose |
| --- | --- |
| `micro_bench.cu`, `run_delta_microbenchmark.py` | Codec throughput and DELTA resource measurements |
| `compressor_bench.cu`, `nvcomp/` | Compression comparisons; requires `GALP_WITH_NVCOMP` |
| `metadata/` | Metadata access, decoding and storage breakdown |
| `system_rgbnomore/` | RGB-no-more, GALP, DALI and PyTorch system comparisons |
| `system_dct_major/` | DCT-major crop and coefficient pushdown; six-pipeline suite with optional CoorDL baseline |
| `training_pls/` | Shared training recipes, canonical PLS schedule, training and reports |
| `dct_models/` | DCTNet/eFUN adapters, inference and training workloads |
| `coefficient_mask_evaluator/` | Raw-JPEG mask accuracy and reusable reference preprocessing |
| `dct_retokenization/` | Token merge, super-patch and delayed-merge experiments |
| `profiling/`, `transfer_*` | Profiling and transfer-contention diagnostics |
| `common.py` | Shared file hashing, JPEG sampling metadata and upstream path defaults |

## Build and run

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
  -DGALP_BUILD_BENCHMARKS=ON -DGALP_WITH_NVCOMP=OFF
cmake --build build --target micro_bench galp_dct_storage -j2

python -m galp.benchmarks.system_rgbnomore.inference.run --help
python -m galp.benchmarks.system_dct_major.run_suite --help
python -m galp.benchmarks.training_pls.run_matrix --help
python -m galp.benchmarks.coefficient_mask_evaluator.evaluate --help
python -m galp.benchmarks.dct_retokenization.evaluate --help
```

Use the generator of an existing build directory when reconfiguring it.
The C++ storage bridge is built as `galp_dct_storage`; set `GALP_DCT_STORAGE`
when using a build directory other than `build`.

Python system workloads require NumPy, Pillow, PyTorch and torchvision;
training/profiling also use einops and psutil. PLS layout/report workloads use
PyArrow and matplotlib. DALI, Nsight, RGB-no-more, DCTNet and FUN are external,
workload-specific dependencies described in the corresponding READMEs.

## Inputs and results

Pass data, checkpoints and output directories explicitly for published runs.
In particular, DCT-major `first:32` evaluation requires
`--raw-mask-oracle-dir` pointing at a completed coefficient-mask evaluation.
Its required files are `per_sample_top1.csv.gz`, `prefix_accuracy_curve.csv` and
`run_metadata.json`. No dated local run is selected automatically.

The reusable PLS entry is `training_pls.run_matrix`; training and reports use
`runs/CONDITION/seed_SEED/run_manifest.json` as the sole resolved contract.
The short RGB-no-more runner retains its independent correctness workload;
PLS crop/shuffle experiments use the canonical PLS runner.

Keep datasets, checkpoints, predictions and profiler captures out of Git.
Existing ignored `runs/` directories are local evidence, not test fixtures.
Use a caller-selected output directory (for example `$HOME/tmp/galp-runs`).
Dated performance reports preserve the measurement conditions of their runs;
current commands are documented by each workload's README and `--help`.

## Tests

CPU suites require the Python dependencies above, but do not run full GPU
benchmarks or ImageNet training:

```bash
python -m unittest discover -s galp/benchmarks/system_dct_major/tests -t .
python -m unittest galp.benchmarks.system_rgbnomore.tests.test_system_benchmark \
  galp.benchmarks.system_rgbnomore.tests.test_training_benchmark \
  galp.benchmarks.system_rgbnomore.tests.test_equal_image_epoch_benchmark \
  galp.benchmarks.system_rgbnomore.tests.test_prepare_imagenet512_v3_train \
  galp.benchmarks.system_rgbnomore.tests.test_rgbnomore_fingerprint_cache
python -m unittest discover -s galp/benchmarks/training_pls/tests -t .
python -m unittest discover -s galp/benchmarks/dct_retokenization/tests -t .
python -m unittest discover -s galp/benchmarks/coefficient_mask_evaluator/tests -t .
python -m unittest discover -s galp/benchmarks/profiling/tests -t .
```

CMake registers the CPU suites when benchmarks, tests and Torch bindings are
enabled. The real-GPU training step has a separate `gpu` label. DCTNet/FUN
integration tests remain explicit because they load upstream code and checkpoints. Full GPU suites and performance runs are separate from
CPU correctness checks.
