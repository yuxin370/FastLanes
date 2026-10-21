# FastLens

FastLens is the research system accompanying the paper in
[`fastlens-final.pdf`](fastlens/docs/paper/outputs/fastlens-final.pdf).
It provides compressed JPEG DCT storage, GPU decoding and selection, and a
PyTorch input pipeline. This repository contains the implementation, tests,
published result tables, and experiment entry points organized by paper section.

## Build and smoke test

The native implementation requires CMake 3.22+, a C++20 compiler (Clang 13+ or
GCC 11+), a CUDA toolkit with Thrust/CCCL, and libjpeg development headers and
library. The Torch binding also requires Python development headers and a
CUDA-enabled PyTorch installation compatible with the toolkit. Tests use
GoogleTest v1.15.2, downloaded by CMake; FastLanes itself is included in this tree.
For an offline build, pass `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest`.

Activate the Python/CUDA environment, then run from this directory:

```bash
./scripts/build.sh -DCMAKE_CUDA_ARCHITECTURES='89;90'
./scripts/smoke_test.sh
ctest --test-dir build -L fastlens-cpu --output-on-failure --parallel 8
```

Architectures 89 and 90 correspond to the paper's RTX 4090 and H100. Use the
architecture of your GPU on other hardware. Set `CUDACXX` and `CUDAHOSTCXX` if
the CUDA compiler and its host compiler are not selected automatically.
`FASTLENS_BUILD_DIR` selects a different build directory. Additional arguments
to `build.sh` are passed to CMake; `CMAKE_BUILD_PARALLEL_LEVEL` controls build jobs.
The smoke test checks the public C++ API, package consumption, and Torch import;
the full GPU/data tests have their own input requirements.

For the C++ library and tools without Torch or test dependencies:

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DFASTLENS_BUILD_TORCH=OFF
cmake --build build-native --parallel
```

## Reproduce the paper

Start with the [paper reproduction guide](fastlens/benchmarks/paper/README.md).
The [published tables](fastlens/benchmarks/paper/results/README.md) retain the
measurements used by the final PDF, including the documented aggregation
difference for Fig. 11b. New measurements are written separately.

```bash
python -m fastlens.benchmarks.paper.reproduce figures
python -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run run --dry-run
```

Plotting uses the included tables and requires NumPy and matplotlib; it does not
require a GPU or ImageNet. Fresh measurements require ImageNet, checkpoints and
the selected baseline dependencies. Historical inference/H2D replay also needs
the original contract/prediction bundle; those large artifacts are not included.
The reproduction guide identifies these inputs and existing runtime limitations.

## Source organization

| Path | Contents |
| --- | --- |
| `fastlens/src/`, `fastlens/include/fastlens/` | FastLens implementation and public C++ API |
| `fastlens/torch/`, `fastlens/profiles/` | Torch runtime and execution profiles |
| `fastlens/tools/`, `fastlens/tests/` | Dataset tools and implementation tests |
| `fastlens/benchmarks/paper/` | Experiments by paper section and published result tables |
| `fastlens/benchmarks/baselines/` | Baseline implementations and preparation |
| `fastlens/benchmarks/development/` | Development diagnostics and historical drivers |
| `third_party/fastlanes/` | Vendored FastLanes library, required build files and license |

FastLens links the modified FastLanes library through `FastLanes::core`.
See [dependency changes and provenance](third_party/fastlanes/CHANGES.md),
the [original license](third_party/fastlanes/LICENSE), and the
[FastLens API documentation](fastlens/README.md).
Upstream provenance and copyright notices are retained as attribution.
