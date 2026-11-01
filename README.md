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
These images work with the example model profile. Resize and prepare the JPEGs before conversion if needed.

From the source checkout, convert and inspect the complete dataset:

```bash
python -m fastlens convert fastlens/data/example-jpegs \
  --output fastlens/data/example-fastlens --layout block-major \
  --partition-images 1024 --workers auto --verify full
python -m fastlens inspect fastlens/data/example-fastlens
python -m fastlens convert --help
```

P0-1 runs from a **source checkout**; pip/wheel distribution and an installed
`fastlens` console script are deferred to P0-3. Both commands use `fastlens.cli.main`.
No PyTorch import or CUDA device is needed for conversion, inspection, or CLI unit
tests. Building the existing native targets still requires the CUDA toolkit.
If you only need to rebuild the converter's tools after configuring CMake:

```bash
cmake --build build --target fastlens_jpeg_dct_tool fastlens_block_major_access_tool --parallel
```

Tool discovery uses `--build-dir`, then `FASTLENS_BUILD_DIR`, then `build/`,
`build-native/`, and sorted `cmake-build-*` directories. An explicitly selected
build must contain both tools; the CLI does not silently switch builds.

FastLens stores **original JPEG quantized DCT coefficients in FastLanes format**,
plus quantization tables and spatial geometry. Conversion does not decode to RGB,
resize, or re-encode JPEGs. The goal is selective, high-throughput execution;
FastLens storage is **not guaranteed to be smaller than JPEG**. Reconstructable
metadata preserves the coefficient interpretation, not the original JPEG byte
stream or EXIF/APP markers.

The converter accepts libjpeg-supported JPEGs, including progressive and grayscale
images and differing dimensions/sampling grids. Extensions are `.jpg`, `.jpeg`,
and `.jpe`, case-insensitively. Malformed files, including files that libjpeg could
recover with corruption warnings, fail conversion even with `--verify none`.
Model profiles can impose stricter geometry requirements: the example uses
512×512 RGB, 4:2:0 JPEGs. Prepare that geometry explicitly before conversion
if needed, and keep source files unchanged throughout conversion and verification.

Directory discovery is recursive, without traversing directory symlinks. It sorts
relative POSIX paths lexicographically by Unicode code point, case-sensitively.
Image IDs are consecutive, starting at zero in that order. To choose the IDs:

```bash
python -m fastlens convert --input-list ordered_paths.txt --output ./test_store
```

The UTF-8 list contains one unquoted JPEG path per line, in the exact desired order;
relative paths resolve against the list's directory. Blank lines are ignored, CRLF
is accepted, and spaces are preserved. Paths containing newlines are unsupported.
Empty datasets, missing files, and duplicate resolved paths (including symlink
aliases) are rejected. `source_paths.txt` records the resulting absolute paths in
image-ID order. It is useful for re-verification; the online reader does not need
the original JPEGs or this list.

`block-major` is the only public layout: native `--shard --physical-layout
spatial-major --metadata-profile reconstruct`, manifest v1. Corresponding spatial
blocks are grouped across images, with DCT frequencies stored as columns. It does
not select image-major Compact-v3. Default `--verify full` invokes the native exact
DCT verifier for every image before publication. `--verify none` skips only that
source comparison; index construction and compatibility checks still run.

```text
example-fastlens/
  manifest.bin
  generation_<id>/
    shard_000000.fls
    shard_000000.meta.bin
    ...
  block_major_access_v1/
    manifest.block_major_access.bin
    shard_000000.block_major_access.bin
    ...
  source_paths.txt
  conversion_report.json
```

Keep the whole directory together. Manifest references are relative to the store;
index sidecars identify the manifest and metadata by content. The complete store
can be moved without rewriting either. The reader discovers `block_major_access_v1`
automatically; unset `FASTLENS_BLOCK_MAJOR_ACCESS_DIR` if it points at another store.

Conversion uses a sibling `.OUTPUT.fastlens-staging-<id>` directory and preserves
the native writer's transactional generations within it. Only after compression,
index validation, and requested verification pass is the directory atomically
renamed to the destination. A nonempty destination or destination symlink is
refused; use a new path. Ordinary errors and Ctrl-C clean staging data and preserve
any existing destination. SIGKILL, machine failure, or abrupt termination may leave
the named staging directory. Once the converter and its native child have stopped,
remove that specific staging directory and rerun; there is no resume operation.
Publication provides process-level atomic visibility, not power-loss durability.

`--dry-run` checks discovery, paths, tools, and parallel bounds without creating
the destination or decoding JPEGs. `--report /existing/parent/report.json` also saves
the report at a new external path; existing report files are protected. The report
contains observed source/file sizes, chosen options, wall times, verification
status, source revision, and native executable paths/sizes/timestamps. It defines
source/output as original JPEG bytes divided by all stored file bytes (including
the report and source list). `compressed_payload_bytes` counts complete FLS files,
including their headers and footers. Timings cover discovery, compression, index
construction/inspection, and verification; total time ends before report writing
and directory publication. No estimated throughput is substituted. Index storage
targets of 0.5%/1% are informational; small stores can exceed them.

`python -m fastlens inspect STORE --json` gives machine-readable native manifest and
index details plus current file sizes. Inspection validates sidecar checksums and
source identities without decoding coefficients. It fails on incompatible indexes;
an absent index is reported as absent. Source JPEG size is unknown for stores
without a conversion report. For a later exact check against the original files:

```bash
build/fastlens/tools/jpeg_dct/fastlens_jpeg_dct_tool \
  --verify-manifest ./test_store/manifest.bin --verify-workers 4 \
  --input-list ./test_store/source_paths.txt
```

Parallel options affect different phases:

| Option | Default and resource effect |
| --- | --- |
| `--workers auto\|N` | Budget used to derive defaults. Auto uses half the CPUs visible through process affinity, at least 1 and at most 32. This is conservative, not optimal tuning. `N` is 1–256. Explicit phase options override derived values. |
| `--layout-threads N` | Header/layout scan before shard processing; defaults to the budget. More threads increase concurrent file access. |
| `--shard-workers N` | Concurrent resident shards; defaults to `min(4, max(1, budget // 4))`. Increases CPU use and peak memory substantially. |
| `--shard-decode-threads N` | Coefficient decoders per active shard. Default `min(4, max(1, budget // shard_workers))`. |
| `--encoding-workers-per-shard N` | FastLanes rowgroup encoders per active shard; same derived default as decode. Additional workers hold additional encoder buffers. |
| `--verify-workers N` | Independent shard verifiers; default `min(4, budget)`, maximum 32. Each worker holds decompressed shard/source state. |
| `--partition-images N` | Maximum images per shard, default 1024, range 1–65535. Native rowgroup capacity may split earlier. Larger values increase per-shard memory. |
| `--preset balanced\|throughput\|crop-latency` | Existing native presets, with 128/256/64 vectors per rowgroup respectively. Larger rowgroups generally require more memory. Encoder algorithms/defaults are unchanged. |

Layout scan and shard processing are sequential. Within a shard, JPEG decode and
FastLanes encoding are sequential, so their thread counts do not multiply. Native
bounds remain `layout_threads <= 256` and
`shard_workers * max(shard_decode_threads, encoding_workers_per_shard) <= 256`.
The CLI rejects violations rather than silently clamping explicit options. Lower
partition size and shard/verification concurrency on memory-constrained machines;
automatic CPU selection does not measure available RAM or tune to workload size.

Converter checks (native tests require Pillow and the built tools; CPU tests do not):

```bash
python -B -m unittest fastlens.tests.test_cli -v
FASTLENS_REQUIRE_NATIVE_TESTS=1 python -B -m unittest fastlens.tests.test_cli_native -v
ctest --test-dir build --output-on-failure -R 'FastLensConverter|FastLensConvertedStore|JpegDctExactVerifier|JpegDctBlockMajorAccess'
```

The converter suites do not require PyTorch. `FastLensConvertedStoreTorchPython`
is registered separately when `FASTLENS_BUILD_TORCH=ON` and the native tools are
enabled. It requires the built extension and skips GPU execution when no CUDA
device is available.

This small dataset is intended for checking the pipeline. The paper benchmarks use the full datasets described in the reproduction guide.

## PyTorch interface

The Python API is available through `fastlens.torch`. The example below reads DCT coefficients from the dataset prepared above.

```python
from fastlens.profiles.rgbnomore import VALIDATION
from fastlens.torch import DirectDctReader

reader = DirectDctReader(
    "fastlens/data/example-fastlens/manifest.bin",
    module_path="build/fastlens/torch",
)

with reader.iter_batches(
    [[0, 1], [2, 3]],
    profile=VALIDATION,
    coefficients=None,
) as batches:
    for batch in batches:
        print(batch.sample_ids, batch.y.shape, batch.cbcr.shape)
        # A compatible DCT model consumes batch.y and batch.cbcr on CUDA.
```

`module_path` points to the compiled PyTorch extension. The profile determines the input geometry, normalization, and tensor representation expected by the model.

The existing `VALIDATION_CENTER_CROP_512` scheduled reader currently requires one
complete shard per submitted batch, with image IDs in ascending order; subsets,
reordered IDs, or batches crossing shards can fail with a canonical-plan-digest
error. The general `VALIDATION` profile used here supports arbitrary sample-ID
batches. This converter does not change the reader's scheduling implementation.

By default, `coefficients=None` reads all 64 source DCT coefficients. You can also request specific frequencies:

```python
coefficients=[5, 0, 2]
```

For CNN workloads, `DctModelProfile` defines the output channels, geometry, and normalization. `DirectDctPipeline` then exposes the projected NCHW tensor as `batch.projected`.

The same store also supports the public asynchronous pipeline with a Direct-DCT profile:

```python
from fastlens.torch import DirectDctPipeline

with DirectDctPipeline(
    "fastlens/data/example-fastlens/manifest.bin",
    profile=VALIDATION,
    batch_size=2,
    module_path="build/fastlens/torch",
) as pipeline:
    for batch in pipeline.start([[0, 1], [2, 3]]):
        print(batch.sample_ids, batch.y.shape, batch.cbcr.shape)
```

Training uses `TrainingPolicy` with a dataset mapping. Call `pipeline.start_epoch(epoch)` to start an epoch; training batches provide labels through `batch.targets`.

Batches retain their underlying CUDA storage. If a batch is consumed on another CUDA stream, call `batch.record_stream(consumer_stream)` before using it there.

A complete example with a small CUDA classifier is included:

```bash
python3 fastlens/examples/direct_dct_pipeline_demo.py \
  fastlens/data/example-fastlens/manifest.bin \
  --module-path build/fastlens/torch \
  --profile validation --batch-size 2 --steps 3
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
