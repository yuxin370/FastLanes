# Development benchmarks and checks

These programs support codec and pipeline development. They do not reproduce a
paper figure. Reviewer entry points are in [`../paper/`](../paper/README.md).

| Source / command | Purpose |
| --- | --- |
| `micro_bench.cu` | Codec decode microbenchmarks |
| `compressor_bench.cu` | nvCOMP compression comparisons |
| `run_delta_microbenchmark.py`, `delta_register_resource_probe.cu` | DELTA throughput and compiler resource measurements |
| `transfer_contention.cpp` | CUDA transfer/contention diagnostics |
| `python -m fastlens.benchmarks.development.check` | Parallel CPU regression suites; no performance measurements |
| `tests/` | Regression tests moved from the former root `experiments/` |
| `diagnostics/` | Profiling, correctness investigations and exploratory controls |
| `historical/` | Dated experiment queues, retries and continuation scripts |
| `reports/` | Historical analysis, result selection and report generation |

Historical scripts retain their original experiment-specific defaults and may
require old runtime snapshots, artifacts or source files that are no longer
available. They are retained for traceability, not as reviewer reproduction
commands. Use `paper/` for the final PDF; report generators can select results
for other manuscript versions and must not replace `paper/results/`.
Some old snapshot builders still name `p0_adapters.py`, `training_ffcv_crop.py`
and `remaining_baseline_inputs.py`; those historical files were already absent
before this reorganization. Final-paper entries do not import these files.

The CMake target names and binary locations are unchanged:

```bash
cmake --build build --target micro_bench fastlens_transfer_contention --parallel
python -m fastlens.benchmarks.development.run_delta_microbenchmark --help
python -m fastlens.benchmarks.development.check
```

Configure with `FLS_ENABLE_FASTLENS_TESTING_AND_BENCHMARKING=ON` and
`FASTLENS_BUILD_BENCHMARKS=ON`; `compressor_bench` additionally requires installed
nvCOMP and `FASTLENS_WITH_NVCOMP=ON`. The paper build command is also a convenient
way to configure native/Torch dependencies before building these targets.

CPU tests are kept with their shared implementation packages; the former
root experiment tests are collected from `tests/`. `check --jobs N` limits parallel suites and hides CUDA in each
child process. Use the environment with PyTorch/torchvision, NumPy, Pillow,
einops and psutil; PCR coefficient tests additionally require `jpegtran` on PATH.
GPU integration and model/checkpoint tests remain explicit, including:

```bash
ctest --test-dir build -L gpu --output-on-failure
```

The known native lifetime smoke failure and its exact manifest requirement are
recorded in the [paper verification notes](../paper/README.md#核查结论).
