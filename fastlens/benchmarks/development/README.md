# Supporting benchmarks and checks

Paper reproduction entry points are in [`../paper/`](../paper/README.md).
This directory contains microbenchmarks, profiling helpers and correctness checks.

| Source / command | Purpose |
| --- | --- |
| `micro_bench.cu` | Codec decode microbenchmarks |
| `compressor_bench.cu` | Optional nvCOMP comparisons |
| `run_delta_microbenchmark.py`, `delta_register_resource_probe.cu` | DELTA throughput and compiler resource measurements |
| `transfer_contention.cpp` | CUDA transfer diagnostics |
| `python -m fastlens.benchmarks.development.check` | Parallel CPU regression suites; no performance measurements |
| `diagnostics/` | Profiling and input-correctness helpers |
| [Image-major comparison](image_major_comparison/README.md) | Retained pipeline and crop measurements for the two Image-major encodings |

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

## Geometry index footprint

For `fastlens/data/compressed/imagenet_original_val_fls_v1_rg64/manifest.bin`,
the v1 builder produced 30 descriptors totaling 62,996,296 bytes and a 1,024-byte
companion index. Relative to 14,106,589,306 bytes of FLS and metadata, the index
adds 0.447%. All 9,251,484 positive block groups matched the source topology.
The [format guide](../../src/jpeg/BLOCK_MAJOR_ACCESS_V1.md#build-an-index)
provides the build command and report options.

## Compact format diagnostics

Build `fastlens_compact_v3_runtime_audit` with the command-line tools. Its memory
probe reports RSS/PSS before and after metadata loading, planning and descriptor
access, including resident `.fls` mmap pages:

```bash
build/fastlens/tools/jpeg_dct/fastlens_compact_v3_runtime_audit \
  --manifest /path/to/manifest.bin --memory-probe --batch-size 64 \
  --output "$HOME/tmp/compact-memory.json"
```

For I/O measurements, run one workload per process. Supported workloads are
`full-all`, `crop-all`, `full-prefix` and `crop-prefix`:

```bash
build/fastlens/tools/jpeg_dct/fastlens_compact_v3_runtime_audit \
  --manifest /path/to/manifest.bin --workload crop-prefix --coefficients 8
```

The output reports logical and physical bytes, 4 KiB coverage, `/proc/self/io`
block reads, `pread`/`preadv` counts, coefficient/vector ratios, pinned memory
and device memory. Cold-cache measurements require external page-cache control.

`fastlens/tools/jpeg_dct/compact_v3_acceptance.py run-runtime-matrix` runs
K=1/4/8/16/32/64 and memory probes in separate processes. It writes a JSON report
and returns nonzero if memory, accounting, I/O or locality checks fail.

The same script's `run-ab` command compares at least five fresh processes per
variant in alternating order, with zero warmup and repeat zero included. Both
variants require matching hardware, data, sample order and model configuration,
clean source revisions, and passing semantic validation for the same benchmark
contract. GPU memory includes Torch-reserved and FastLens-native allocations;
process-tree PSS includes workers and pinned allocations. The report records
whether page-cache control was verified. Use `fastlens` as the system name.
