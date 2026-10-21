# Supporting benchmarks and checks

Reviewer experiment entry points are in [`../paper/`](../paper/README.md).
This directory retains source needed by the build and correctness checks;
historical run queues, report generators and their tests are not distributed.

| Source / command | Purpose |
| --- | --- |
| `micro_bench.cu` | Codec decode microbenchmarks |
| `compressor_bench.cu` | Optional nvCOMP comparisons |
| `run_delta_microbenchmark.py`, `delta_register_resource_probe.cu` | DELTA throughput and compiler resource measurements |
| `transfer_contention.cpp` | CUDA transfer diagnostics |
| `python -m fastlens.benchmarks.development.check` | Parallel CPU regression suites; no performance measurements |
| `diagnostics/` | Profiling and input-correctness helpers |

Build with `FASTLENS_BUILD_TESTS=ON` and `FASTLENS_BUILD_BENCHMARKS=ON`;
`compressor_bench` also requires nvCOMP and `FASTLENS_WITH_NVCOMP=ON`.

```bash
cmake --build build --target micro_bench fastlens_transfer_contention --parallel
python -m fastlens.benchmarks.development.check
```

`check --jobs N` limits parallel suites and hides CUDA in each child process.
Use an environment with PyTorch/torchvision, NumPy, Pillow, einops and psutil;
coefficient checks also require `jpegtran` on PATH. GPU tests remain explicit:

```bash
ctest --test-dir build -L gpu --output-on-failure
```

The [paper guide](../paper/README.md) documents external input requirements and
known runtime limitations. Test outputs are not published measurement tables.
