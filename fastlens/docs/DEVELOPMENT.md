# Build, test and develop

[Documentation index](../../README.md#documentation)

Run commands from the repository root.

## Dependencies

Make sure you have Conda and an NVIDIA driver compatible with CUDA 12.8.

Install the system build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y build-essential clang cmake libjpeg-dev numactl
```

FastLens requires CMake 3.22+, a C++20 compiler (Clang 13+ or GCC 11+), Make or Ninja, and libjpeg development headers.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate fastlens
```

[`environment.yml`](../../environment.yml) installs Python 3.11 and the CUDA 12.8.1 development toolkit, including `nvcc` and Thrust/CCCL. It also installs the packages pinned in [`requirements.txt`](../../requirements.txt): PyTorch 2.11.0 and torchvision 0.26.0 for CUDA 12.8, plus the common image, model, benchmark and plotting dependencies. The paper uses CUDA 12.8 and PyTorch 2.11.

If you already have Python 3.11 and a compatible CUDA development toolkit, you can install the Python dependencies separately:

```bash
python -m pip install -r requirements.txt
```

## Build and smoke test

With the `fastlens` environment active, build and run the smoke tests:

```bash
./scripts/build.sh \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_ARCHITECTURES='89;90'
./scripts/smoke_test.sh
```

For a toolkit installed outside Conda, set `CMAKE_CUDA_COMPILER` to its `nvcc` path.
Select a CUDA-supported host compiler with `CMAKE_CUDA_HOST_COMPILER`.
Architectures 89 and 90 correspond to RTX 4090 and H100; select the architecture of your GPU on other hardware.

For a C++-only build without PyTorch or tests:

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DFASTLENS_BUILD_TESTS=OFF -DFASTLENS_BUILD_TORCH=OFF
cmake --build build-native --parallel
```

Additional build options:

- `FASTLENS_BUILD_DIR` selects a different build directory used by `build.sh`; documentation examples use `build/`.
- `CMAKE_BUILD_PARALLEL_LEVEL` controls parallel build jobs.
- `FASTLENS_WITH_NVCOMP=ON` enables optional nvCOMP comparisons; the FastLens input pipeline does not depend on nvCOMP.
- CMake downloads GoogleTest v1.15.2 for tests. For an offline build, pass `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest`.

DALI, FFCV, CoorDL, L3 and the RGB-no-more `dct_manip` extension have additional setup steps in the [benchmark guides](../benchmarks/README.md).

## Converter and Reader tests

Converter checks (native tests require Pillow and the built tools; CPU tests do not):

```bash
python -B -m unittest fastlens.tests.test_cli fastlens.tests.test_annotations -v
FASTLENS_REQUIRE_NATIVE_TESTS=1 python -B -m unittest fastlens.tests.test_cli_native.NativeCliTests -v
ctest --test-dir build --output-on-failure -LE nightly -R 'FastLensConverter|FastLensConvertedStore|JpegDctExactVerifier|JpegDctBlockMajorAccess'
```

The converter suites do not require PyTorch. `FastLensConvertedStoreTorchPython`
is registered separately when `FASTLENS_BUILD_TORCH=ON` and the native tools are
enabled. It requires the built extension and skips GPU execution when no CUDA
device is available (CTest reports it as skipped). GPU CI must set the existing
`FASTLENS_REQUIRE_CUDA_SMOKE=1` strict mode so missing CUDA fails the test. The
test uses distinct constant Y/Cb/Cr images with different quantizers and an
independent analytic DCT reference to check tensor values and image-ID ordering,
including cross-shard batches.

The native suite includes 17 mixed-geometry images with partitions of 8. A separate
nightly-labelled canary converts 1025 images with the default partition size of
1024 and checks the full source comparison, indexes, and boundary image reads:

```bash
ctest --test-dir build --output-on-failure -R '^FastLensConverterDefaultPartitionCanary$'
FASTLENS_REQUIRE_CUDA_SMOKE=1 ctest --test-dir build --output-on-failure -R '^FastLensConvertedStoreTorchPython$'
```

`FastLensPipelineFacadePython` covers the Python pipeline contract without a GPU
or the Torch extension. `FastLensDirectDctLifetimeTorchPython` uses a small generated
JPEG store to check tensor ownership, pending consumer streams, legacy lifetime
support, and metrics completion. It uses
the private `_fastlens_test_consumer` module, which is built only for testing.
The same strict CUDA setting applies to this suite.

```bash
FASTLENS_REQUIRE_CUDA_SMOKE=1 ctest --test-dir build --output-on-failure \
  -R 'FastLensPipelineFacadePython|FastLensDirectDctLifetimeTorchPython'
```

Benchmark tests retain the `benchmark` label; use `-LE 'benchmark|nightly'` for
regular functional checks, or `-L benchmark` for experiment tooling checks.
The optional native scheduler comparison probe can be built with
`cmake --build build --target _fastlens_phase2_native_ab`.
The runtime no-D2H check also enforces asynchronous submission; it scans native
runtime paths, leaving test reference calculations free to inspect CPU values.

## Test labels

The smoke script checks the public API, package consumption and Torch import. The default build also provides the following test groups:

```bash
ctest --test-dir build -L public-api --output-on-failure
ctest --test-dir build -L fastlens-cpu --output-on-failure --parallel 8
ctest --test-dir build -L fastlens-gpu --output-on-failure
```

GPU tests require a working CUDA device. Some integration tests also require local datasets, as documented in the corresponding test setup.

The CUDA toolkit is required to configure the project, including when only CPU tests are run.

## Documentation checks

Run the documentation checks without CUDA or third-party Python packages:

```bash
python -B -m unittest fastlens.tests.test_documentation -v
```

`FastLensDocumentationPython` runs the same checks through CTest. It validates
relative Markdown links against publishable files and directories, heading
anchors, introductory Python syntax, and specific internal-session phrases,
machine GPU UUIDs and dated temporary paths. Dated `history/` records and
manuscript build notes retain provenance; their relative links are still checked.
External URLs are not fetched by this check.

`FastLensConvertedStoreTorchPython` also executes the reader examples extracted
from the root README, architecture guide and Direct-DCT guide against a generated
four-image Store. Only data and extension paths are replaced for the fixture.
Use the strict CUDA command above to validate these examples on a GPU.

## C++ interface

FastLens also provides native C++ APIs:

- `<fastlens/stable.hpp>` for reading and decompressing compressed tables (`FastLens::core`).
- `<fastlens/advanced/direct_dct.hpp>` for GPU-resident DCT batches (`FastLens::jpeg_dct_device`).

```cpp
#include <fastlens/stable.hpp>

int main() {
    fastlens::Reader reader("fastlens/data/example-fastlens/partitions/000000.fls");
    fastlens::DecompressOptions options;
    options.write_output = true;
    options.scope = fastlens::TableDecompressionScope::WholeTable;
    fastlens::Table table = reader.decompress(options);
    return table.empty() ? 1 : 0;
}
```

The materialized table API supports `int8_t` and `int16_t` columns. Rowgroup and column views refer to storage owned by the `Table`, so the table must remain alive while these views are in use.

For DCT batches that remain on the GPU, use the Direct-DCT interface instead of materializing a CPU table.

See [fastlens/examples/](../examples/) for complete CMake examples.

### API compatibility

The compatibility headers `<fastlens/fastlens.hpp>` and
`<fastlens/direct_dct.hpp>` remain available for existing integrations.
See [API boundaries](../DIRECT_DCT_API_BOUNDARIES.md) for stable, advanced and
diagnostic interfaces.
