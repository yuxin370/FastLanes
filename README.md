# FastLens

Code and experiments for **[FastLens: FastLanes for High-Throughput AI Image Pipelines](fastlens/docs/paper/outputs/fastlens.pdf)**.

FastLens is a GPU-oriented image input system that combines storage-level spatial and frequency selection, compressed host-to-device transfer, GPU-native decoding, and direct construction of model-ready tensors. It stores JPEG DCT coefficients in a compressed columnar representation and supplies DCT-domain training and inference workloads through a PyTorch-compatible pipeline.

## Performance

The paper evaluates FastLens against a PyTorch DCT baseline across seven ImageNet-1K workloads, using matched source data, models, and numerical semantics. The table reports average speedups across these workloads (§6.2.1).

| GPU | Training speedup | Inference speedup |
| --- | ---: | ---: |
| NVIDIA RTX 4090 | 3.80× | 6.17× |
| NVIDIA H100 | 4.20× | 7.95× |

FastLens also achieves 99.1% of GPU-resident training throughput on average. The evaluated training store uses 2.12× the space of the source JPEGs, while compressed transfer reduces H2D traffic by 87.7% compared with transferring raw `int16` coefficients.

These results are from the matched DCT evaluation. RGB deployment comparisons use different input representations and are reported separately.

See the [published tables](fastlens/benchmarks/paper/results/README.md) for individual measurements and aggregation details.

## How it works

FastLens combines three components:

- **Compressed image format and projection pushdown (§4.1).** A cross-image block-major layout groups corresponding spatial blocks across images and stores DCT frequencies as columns. Dependency-aware planning maps spatial and frequency requests to compressed vectors, including the referenced columns and auxiliary data required for decoding.
- **GPU-native compressed processing (§4.2).** The CPU plans and stages selected compressed data. A mixed-dispatch FastLanes decoder handles heterogeneous encodings on the GPU, preserving compression throughout host-to-device transfer.
- **Selection-preserving input pipeline (§4.3 and §5).** Locality-aware asynchronous scheduling separates physical reads from model-visible sample order. GPU kernels directly construct model-ready tensors from selected coefficients, with the geometry and augmentation required by the model profile.

```text
Offline: JPEG images -> quantized DCT coefficients + metadata
                     -> cross-image block-major compressed storage

Online:  sample IDs + spatial/frequency requests
         -> dependency-aware projection -> selective compressed reads
         -> compressed H2D transfer -> mixed-dispatch GPU decoding
         -> direct tensor construction -> PyTorch model
```

The CPU handles selection planning and data staging, while GPU kernels handle decompression and tensor construction.

## Quick start

Run all commands below from the **FastLens repository root**. This setup uses Ubuntu 22.04/24.04 with an NVIDIA GPU, Python 3.11, and CUDA 12.8.

### 1. Install dependencies

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

[`environment.yml`](environment.yml) installs Python 3.11 and the CUDA 12.8.1 development toolkit, including `nvcc` and Thrust/CCCL. It also installs the packages pinned in [`requirements.txt`](requirements.txt): PyTorch 2.11.0 and torchvision 0.26.0 for CUDA 12.8, plus the common image, model, benchmark and plotting dependencies. The paper uses CUDA 12.8 and PyTorch 2.11.

If you already have Python 3.11 and a compatible CUDA development toolkit, you can install the Python dependencies separately:

```bash
python -m pip install -r requirements.txt
```

### 2. Build and smoke test

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

- `FASTLENS_BUILD_DIR` selects a different build directory used by `build.sh`; the examples and paper runners below use `build/`.
- `CMAKE_BUILD_PARALLEL_LEVEL` controls parallel build jobs.
- `FASTLENS_WITH_NVCOMP=ON` enables optional nvCOMP comparisons; the FastLens input pipeline does not depend on nvCOMP.
- CMake downloads GoogleTest v1.15.2 for tests. For an offline build, pass `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest`.

DALI, FFCV, CoorDL, L3 and the RGB-no-more `dct_manip` extension have additional setup steps in the [benchmark guides](fastlens/benchmarks/README.md).

## Prepare a small image dataset

To try FastLens without a full ImageNet dataset, prepare at least four 512×512 RGB JPEG images with 4:2:0 chroma subsampling and place them in `fastlens/data/example-jpegs/`.
These images match the source geometry expected by the example model profile. Resize and prepare the JPEGs before conversion.

Convert the images into FastLens storage:

```bash
build/fastlens/tools/jpeg_dct/fastlens_jpeg_dct_tool \
  --shard --out-dir fastlens/data/example-fastlens \
  --preset balanced --physical-layout spatial-major \
  --shard-images 1024 --metadata-profile reconstruct \
  fastlens/data/example-jpegs
```

Build the block-major access indexes:

```bash
build/fastlens/tools/jpeg_dct/fastlens_block_major_access_tool \
  fastlens/data/example-fastlens/manifest.bin \
  --output-dir fastlens/data/example-fastlens/block_major_access_v1
```

Verify that the original DCT coefficients are preserved:

```bash
build/fastlens/tools/jpeg_dct/fastlens_jpeg_dct_tool \
  --verify-manifest fastlens/data/example-fastlens/manifest.bin fastlens/data/example-jpegs
```

The `spatial-major` option selects the cross-image block-major layout. Conversion produces a `manifest.bin`, compressed `shard_*.fls` files, and `shard_*.meta.bin` metadata. The access tool generates the indexes used by the block-major reader. Keep these files together.

By default, image IDs are assigned during conversion. To control their order, pass `--input-list ordered_paths.txt` instead of the input directory, with one JPEG path per line.

This small dataset is intended for checking the pipeline. The paper benchmarks use the full datasets described in the reproduction guide.

## PyTorch interface

The Python API is available through `fastlens.torch`. The example below reads DCT coefficients from the dataset prepared above.

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

`module_path` points to the compiled PyTorch extension. The profile determines the input geometry, normalization, and tensor representation expected by the model.

By default, `coefficients=None` reads all 64 source DCT coefficients. You can also request specific frequencies:

```python
coefficients=[5, 0, 2]
```

For CNN workloads, `DctModelProfile` defines the output channels, geometry, and normalization. `DirectDctPipeline` then exposes the projected NCHW tensor as `batch.projected`.

Training uses `TrainingPolicy` with a dataset mapping. Call `pipeline.start_epoch(epoch)` to start an epoch; training batches provide labels through `batch.targets`.

Batches retain their underlying CUDA storage. If a batch is consumed on another CUDA stream, call `batch.record_stream(consumer_stream)` before using it there.

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

FastLens also provides native C++ APIs:

- `<fastlens/fastlens.hpp>` for reading and decompressing compressed tables (`FastLens::core`).
- `<fastlens/direct_dct.hpp>` for GPU-resident DCT batches (`FastLens::jpeg_dct_device`).

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

The materialized table API supports `int8_t` and `int16_t` columns. Rowgroup and column views refer to storage owned by the `Table`, so the table must remain alive while these views are in use.

For DCT batches that remain on the GPU, use the Direct-DCT interface instead of materializing a CPU table.

See [fastlens/examples/](fastlens/examples/) for complete CMake examples.

## Reproduce the paper

The experiments are organized by paper section under `fastlens/benchmarks/paper/`. Detailed instructions, required datasets, and experiment configurations are in the [paper reproduction guide](fastlens/benchmarks/paper/README.md).

Build the native benchmark targets:

```bash
python3 -m fastlens.benchmarks.paper.reproduce build
```

Each experiment has its own runner. For example, to inspect the matched DCT training and inference benchmark (§6.2.1):

```bash
python3 -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run --help
python3 -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run run --dry-run
```

Experiment directories follow the paper's evaluation sections:

| Paper section | Experiment directories under `fastlens/benchmarks/paper/` |
| --- | --- |
| §6.2.1: matched DCT training and inference | `section_6_2_1/matched_dct/` |
| §6.2.2: selected frequencies and compact inputs | `section_6_2_2/shape_preserving_selection/`, `section_6_2_2/compact_input/` |
| §6.2.3–6.2.5: deployment, resident reference, convergence | `section_6_2_3/deployment/`, `section_6_2_4/resident_training/`, `section_6_2_5/training_convergence/` |
| §6.3: storage and H2D traffic | `section_6_3/storage/`, `section_6_3/h2d_traffic/` |
| §6.4: projection, staging, transfer and tensor construction | `section_6_4/projection_and_staging/`, `section_6_4/compressed_transfer_and_tensor/` |
| §6.5: locality and constrained read bandwidth | `section_6_5_1/block_major_locality/`, `section_6_5_2/read_bandwidth/` |

Running the full experiments requires ImageNet, prepared FastLens stores, model checkpoints, and the dependencies of the baselines being compared. The [benchmark setup guide](fastlens/benchmarks/README.md) covers DALI, FFCV, CoorDL, L3, and other benchmark-specific requirements. JPEG-B8 experiments use the RGB-no-more `dct_manip` extension. H2D traffic measurements require Nsight Systems.

The [reference rebuilding guide](fastlens/benchmarks/paper/REBUILDING_REFERENCES.md) gives the preparation order: public model code and checkpoints, ImageNet stores, pipeline contracts, then full-population reference predictions. Once its reference configuration is ready:

```bash
python3 -m fastlens.benchmarks.paper.prepare_references \
  --references results/rebuilt/references.json
python3 -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run run \
  --references results/rebuilt/references.json
```

Training also requires the prepared training split described in the guide. Use `--phases inference` for an inference-only run. To capture H2D traffic from completed chapter trials, run `python3 -m fastlens.benchmarks.paper.section_6_3.h2d_traffic.run run --run-dir results/reproduced` with Nsight Systems installed.

To redraw the paper's ten plots from the included measurements:

```bash
python3 -m fastlens.benchmarks.paper.reproduce figures
```

Plotting requires NumPy and Matplotlib and runs without a GPU, dataset, native build or checkpoint. Outputs go to `results/reproduced/paper/figures/`.

## Tests

The smoke script checks the public API, package consumption and Torch import. The default build also provides the following test groups:

```bash
ctest --test-dir build -L public-api --output-on-failure
ctest --test-dir build -L fastlens-cpu --output-on-failure --parallel 8
ctest --test-dir build -L fastlens-gpu --output-on-failure
```

GPU tests require a working CUDA device. Some integration tests also require local datasets, as documented in the corresponding test setup.

The CUDA toolkit is required to configure the project, including when only CPU tests are run.

## Repository layout

| Path                                    | Contents                                               |
| --------------------------------------- | ------------------------------------------------------ |
| `fastlens/include/fastlens/`            | Public C++ headers                                     |
| `fastlens/src/`                         | Storage format, readers, codecs, and CPU/GPU execution |
| `fastlens/src/direct_dct/`              | Direct-DCT runtime and pipeline                        |
| `fastlens/torch/`, `fastlens/profiles/` | PyTorch interface and model profiles                   |
| `fastlens/tools/`, `fastlens/examples/` | Conversion tools and usage examples                    |
| `fastlens/tests/`                       | Correctness and integration tests                      |
| `fastlens/benchmarks/`                  | Paper experiments, baselines, and results              |
| `scripts/`                              | Build and smoke-test scripts                           |
| `third_party/fastlanes/`                | Modified FastLanes dependency                          |

FastLens includes a modified copy of FastLanes under `third_party/fastlanes/`, linked through `FastLanes::core`. See [CHANGES.md](third_party/fastlanes/CHANGES.md) for the modifications and upstream provenance, and [LICENSE](third_party/fastlanes/LICENSE) for the original license. Upstream copyright and license notices are retained.
