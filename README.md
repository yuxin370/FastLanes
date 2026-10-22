# FastLens

Research artifact for **[FastLens: FastLanes for High-Throughput AI Image Pipelines](fastlens/docs/paper/outputs/fastlens.pdf)**.

FastLens is a GPU-oriented image input system that combines storage-level spatial
and frequency selection, compressed host-to-device transfer, GPU-native decoding,
and direct construction of model-ready tensors. It stores JPEG DCT coefficients
in a compressed columnar representation and supplies DCT-domain training and
inference workloads through a PyTorch-compatible pipeline.

The FastLens implementation lives in `fastlens/`, with
a modified FastLanes library included as an internal dependency in
`third_party/fastlanes/`. The root CMake project builds both. This repository
contains the implementation, usage examples, tests, published result tables and
experiment entry points organized by paper section.

## System design

The implementation follows the paper's three main components:

- **Compressed image format and projection pushdown (§4.1).** A cross-image
  block-major layout groups corresponding spatial blocks across images and stores
  DCT frequencies as columns. Dependency-aware planning maps spatial and frequency
  requests to compressed vectors, including the referenced columns and auxiliary
  data required for decoding.
- **GPU-native compressed processing (§4.2).** The CPU plans and stages selected
  compressed data. A mixed-dispatch FastLanes decoder handles heterogeneous
  encodings on the GPU, preserving compression throughout host-to-device transfer.
- **Selection-preserving input pipeline (§4.3 and §5).** Locality-aware asynchronous
  scheduling separates physical reads from model-visible sample order. GPU kernels
  directly construct model-ready tensors from selected coefficients, with the
  geometry and augmentation required by the model profile.

```text
Offline: JPEG images -> quantized DCT coefficients + metadata
                     -> cross-image block-major compressed storage

Online:  sample IDs + spatial/frequency requests
         -> dependency-aware projection -> selective compressed reads
         -> compressed H2D transfer -> mixed-dispatch GPU decoding
         -> direct tensor construction -> PyTorch model
```

Selection operates at vector granularity. Source-frequency selection and output
channel selection are distinct: resizing can require source frequencies that do
not appear in the final model tensor. The selected representation must match the
model's preprocessing contract.

## Results reported in the paper

Across seven ImageNet-1K workloads, the matched DCT comparison uses the same
source data, DCT model and numerical semantics for FastLens and the PyTorch DCT
baseline. The paper reports the following average speedups (§6.2.1):

| GPU | Training | Inference |
| --- | ---: | ---: |
| NVIDIA RTX 4090 | 3.80× | 6.17× |
| NVIDIA H100 | 4.20× | 7.95× |

FastLens reaches 99.1% of the GPU-resident training throughput reference on
average (§6.2.4). The evaluated training store occupies 2.12× the source JPEG
size (§6.3), and compressed transfer reduces H2D traffic by 87.7% relative to raw
`int16` transfer for the same workload (§6.4).

The [published tables](fastlens/benchmarks/paper/results/README.md) retain the
measurements used by the paper and document their aggregation conventions.
RGB deployment comparisons use different input representations and are reported
separately from the matched DCT comparison.

## Requirements and build

Run all commands below from the **FastLens repository root**. The build targets
Linux with an NVIDIA GPU and requires:

- CMake 3.22 or newer, with Make or Ninja.
- A C++20 compiler: Clang 13+ (the default) or GCC 11+, plus a CUDA-supported
  host compiler.
- A CUDA development toolkit with `nvcc` and Thrust/CCCL, and a compatible driver.
- The libjpeg development API: `jpeglib.h` and `libjpeg`, such as those provided by
  `libjpeg-dev` or `libjpeg-turbo8-dev`.
- Python 3.10+ with development headers and CUDA-enabled PyTorch for the Python
  interface. The paper uses CUDA 12.8 and PyTorch 2.11.

Activate the Python/CUDA environment, then build and run the smoke tests:

```bash
./scripts/build.sh -DCMAKE_CUDA_ARCHITECTURES='89;90'
./scripts/smoke_test.sh
```

The build script enables the tools, Torch extension, benchmarks and tests.
Architectures 89 and 90 correspond to RTX 4090 and H100; select the architecture
of your GPU on other hardware. Set `CUDACXX` and `CUDAHOSTCXX` when the CUDA
compiler and its host compiler need explicit selection.

The script uses Python from the active environment. For a CUDA toolkit installed
in that environment with Conda, select the compilers explicitly:

```bash
./scripts/build.sh \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_ARCHITECTURES='89;90'
```

For other CUDA installations, use the full path to that toolkit's `nvcc`.

`FASTLENS_BUILD_DIR` selects a different build directory; the examples and paper
runners below use `build/`. Additional arguments to `build.sh` go to CMake, and
`CMAKE_BUILD_PARALLEL_LEVEL` controls parallel build jobs. CMake downloads
GoogleTest v1.15.2 for tests. For an offline build, pass
`-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest`.

For the C++ library and tools without Torch or tests:

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DFASTLENS_BUILD_TESTS=OFF -DFASTLENS_BUILD_TORCH=OFF
cmake --build build-native --parallel
```

Optional nvCOMP comparisons require `FASTLENS_WITH_NVCOMP=ON`; the FastLens input
pipeline does not depend on nvCOMP.

## Prepare a small image dataset

Place at least four 512×512 RGB JPEG images with 4:2:0 chroma subsampling in
`fastlens/data/example-jpegs/`. This matches the source geometry of the model profile used
below. Dataset resizing and JPEG preparation precede conversion.

```bash
build/fastlens/tools/jpeg_dct/fastlens_jpeg_dct_tool \
  --shard --out-dir fastlens/data/example-fastlens \
  --preset balanced --physical-layout spatial-major \
  --shard-images 1024 --metadata-profile reconstruct \
  fastlens/data/example-jpegs

build/fastlens/tools/jpeg_dct/fastlens_block_major_access_tool \
  fastlens/data/example-fastlens/manifest.bin \
  --output-dir fastlens/data/example-fastlens/block_major_access_v1

build/fastlens/tools/jpeg_dct/fastlens_jpeg_dct_tool \
  --verify-manifest fastlens/data/example-fastlens/manifest.bin fastlens/data/example-jpegs
```

The CLI name `spatial-major` denotes the cross-image block-major layout. Conversion
writes `manifest.bin`, compressed `shard_*.fls` files and `shard_*.meta.bin`
metadata. The access tool adds the indexes used by the block-major reader. Keep
these files together. For explicit image ID assignment, replace the input
directory with `--input-list ordered_paths.txt`, containing one JPEG path per
line in the intended order.

Verification checks exact preservation of source DCT coefficients. The small
dataset exercises functionality; paper measurements use the full populations
specified in the reproduction guide.

## PyTorch interface

The public API is `fastlens.torch`. A profile specifies model-facing geometry,
normalization and tensor representation. For a source build, `module_path`
points to the compiled extension:

```python
from fastlens.profiles.rgbnomore import VALIDATION_CENTER_CROP_512
from fastlens.torch import DirectDctReader

reader = DirectDctReader(
    "fastlens/data/example-fastlens/manifest.bin",
    module_path="build/fastlens/torch",
)

with reader.iter_batches(
    [[0, 1], [2, 3]],
    profile=VALIDATION_CENTER_CROP_512,
    coefficients=None,
) as batches:
    for batch in batches:
        print(batch.sample_ids, batch.y.shape, batch.cbcr.shape)
        # A compatible DCT model consumes batch.y and batch.cbcr on CUDA.
```

`coefficients=None` requests all 64 source coefficients. An explicit iterable,
such as `[5, 0, 2]`, preserves the requested order; the profile controls the model's
output representation. For CNNs, `DctModelProfile` specifies output channels,
geometry and normalization, and `DirectDctPipeline` exposes the projected NCHW
tensor through `batch.projected`. Training uses `TrainingPolicy` with a dataset
mapping and `pipeline.start_epoch(epoch)`, exposing `batch.targets`.

Returned tensors retain native storage. Before consuming a batch on another CUDA
stream, call `batch.record_stream(consumer_stream)`. Profiles, labels and model
checkpoints must describe the same workload.

A complete example with a small CUDA classifier is included:

```bash
python3 fastlens/examples/direct_dct_pipeline_demo.py \
  fastlens/data/example-fastlens/manifest.bin \
  --module-path build/fastlens/torch \
  --profile validation-center-crop-512 --batch-size 2 --steps 3
```

See the [architecture and API guide](fastlens/ARCHITECTURE.md) and
[DCT augmentation semantics](fastlens/DCT_DOMAIN_AUGMENTATIONS.md) for details.

## C++ interface

Use `<fastlens/fastlens.hpp>` for table decompression, or
`<fastlens/direct_dct.hpp>` for GPU-resident DCT batches. The corresponding public
CMake targets are `FastLens::core` and `FastLens::jpeg_dct_device`.

```cpp
#include <fastlens/fastlens.hpp>

int main() {
    fastlens::Reader reader("fastlens/data/example-fastlens/shard_000000.fls");
    fastlens::DecompressOptions options;
    options.write_output = true;
    options.scope = fastlens::TableDecompressionScope::WholeTable;
    fastlens::Table table = reader.decompress(options);
    return table.empty() ? 1 : 0;
}
```

The materialized table API supports `int8_t` and `int16_t` columns. Rowgroup and
column views borrow storage from their owning `Table`. Use the Direct-DCT
interface for model inputs that stay on the GPU. Complete CMake consumers are
in [fastlens/examples/](fastlens/examples/).

## Reproduce the paper

Start with the [paper reproduction guide](fastlens/benchmarks/paper/README.md).
Experiment entry points follow §§6.2–6.5, and published tables live in
`fastlens/benchmarks/paper/results/`.

To redraw the paper's ten plots from the included measurements:

```bash
python3 -m fastlens.benchmarks.paper.reproduce figures
```

Plotting requires NumPy and Matplotlib and runs without a GPU, dataset, native
build or checkpoint. Outputs go to `results/reproduced/paper/figures/`.

To build the native benchmark targets and inspect an experiment's commands:

```bash
python3 -m fastlens.benchmarks.paper.reproduce build
python3 -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run --help
python3 -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run run --dry-run
```

| Paper section | Experiment directories under `fastlens/benchmarks/paper/` |
| --- | --- |
| §6.2.1: matched DCT training and inference | `section_6_2_1/matched_dct/` |
| §6.2.2: selected frequencies and compact inputs | `section_6_2_2/shape_preserving_selection/`, `section_6_2_2/compact_input/` |
| §6.2.3–6.2.5: deployment, resident reference, convergence | `section_6_2_3/deployment/`, `section_6_2_4/resident_training/`, `section_6_2_5/training_convergence/` |
| §6.3: storage and H2D traffic | `section_6_3/storage/`, `section_6_3/h2d_traffic/` |
| §6.4: projection, staging, transfer and tensor construction | `section_6_4/projection_and_staging/`, `section_6_4/compressed_transfer_and_tensor/` |
| §6.5: locality and constrained read bandwidth | `section_6_5_1/block_major_locality/`, `section_6_5_2/read_bandwidth/` |

Fresh measurements require ImageNet, prepared stores, model checkpoints and the
selected baseline dependencies. See [benchmark setup](fastlens/benchmarks/README.md)
for Python packages and the DALI, FFCV, CoorDL and L3 comparisons. JPEG-B8
measurements use RGB-no-more's `dct_manip` extension, and H2D captures use Nsight
Systems.

The [reference rebuilding guide](fastlens/benchmarks/paper/REBUILDING_REFERENCES.md)
explains how to prepare public model code and checkpoints, generate inference
contracts and predictions, and capture H2D traffic from newly generated workloads.
Keep new measurements separate from the published tables and run timed
experiments on an otherwise idle GPU with the selected experiment's protocol.

## Tests

The smoke script checks the public API, package consumption and Torch import.
The default build also provides the following test groups:

```bash
ctest --test-dir build -L public-api --output-on-failure
ctest --test-dir build -L fastlens-cpu --output-on-failure --parallel 8
ctest --test-dir build -L fastlens-gpu --output-on-failure
```

GPU tests require a working CUDA device; dataset-dependent integration tests
also require their documented local inputs. The CUDA toolkit is needed to
configure the project even when running only CPU tests.

## Source layout and dependency

| Path | Purpose |
| --- | --- |
| `fastlens/include/fastlens/` | Public C++ API |
| `fastlens/src/format/`, `fastlens/src/core/` | Storage descriptors, readers and expression planning |
| `fastlens/src/engine/`, `fastlens/src/codecs/`, `fastlens/src/cuda/` | Worksets, codecs, GPU execution and memory management |
| `fastlens/src/direct_dct/` | Direct-DCT runtime and native pipeline |
| `fastlens/torch/`, `fastlens/profiles/` | PyTorch interface and model input profiles |
| `fastlens/tools/`, `fastlens/examples/`, `fastlens/tests/` | Conversion tools, examples and correctness checks |
| `fastlens/benchmarks/paper/` | Paper reproduction entry points and published results |
| `fastlens/benchmarks/baselines/` | Baseline implementations and preparation |
| `scripts/` | Repository build and smoke-test entry points |
| `third_party/fastlanes/` | Modified FastLanes dependency and its license |

FastLens links the vendored FastLanes library through `FastLanes::core`. See
[changes and upstream provenance](third_party/fastlanes/CHANGES.md) and the
[FastLanes license](third_party/fastlanes/LICENSE). Upstream copyright and license
notices are retained.
