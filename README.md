# FastLens

Code and experiments for **[FastLens: FastLanes for High-Throughput AI Image Pipelines](fastlens/docs/paper/outputs/fastlens.pdf)**.

FastLens is a GPU-oriented image input system that combines storage-level spatial and frequency selection, compressed host-to-device transfer, GPU-native decoding, and direct construction of model-ready tensors. It stores JPEG DCT coefficients in a compressed columnar representation and supplies DCT-domain training and inference workloads through a PyTorch-compatible pipeline.

## Key results

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

FastLens supports **Block-major** and **Image-major** physical layouts.
The public Converter writes Block-major. Native Image-major provides both
one image per rowgroup (the `random-access` preset's default) and tiled vector
rowgroups, selected explicitly for the paper's Image-major experiments.
Both Image-major encodings remain supported; see the
[layout and entry-point matrix](fastlens/docs/spec/README.md).

## Requirements and installation

Run commands from the repository root. FastLens currently runs from a source
checkout; a pip/wheel distribution is not provided.

The tested setup uses Ubuntu 22.04/24.04, Python 3.11, CUDA 12.8 and PyTorch 2.11.
Building requires CMake 3.22+, a C++20 compiler and libjpeg development headers.
Conversion and inspection need the native tools but do not import PyTorch or
require a CUDA device. The CUDA toolkit is required to configure the project.
Reading GPU tensors additionally requires the Torch extension and a working GPU.

```bash
sudo apt-get update
sudo apt-get install -y build-essential clang cmake libjpeg-dev numactl
conda env create -f environment.yml
conda activate fastlens
./scripts/build.sh \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_ARCHITECTURES='89;90'
./scripts/smoke_test.sh
```

Architectures 89 and 90 target RTX 4090 and H100. Select the architecture of your
GPU and a CUDA-supported host compiler. See [Development](fastlens/docs/DEVELOPMENT.md)
for native-only builds, alternate toolkits, build options and offline dependencies.

## Quick start

### Convert JPEGs

Four [bundled photographs](fastlens/data/example-jpegs/README.md) already meet the
512×512 RGB, 4:2:0 requirements of the tensor example. Run from the repository root.

```bash
python -m fastlens convert fastlens/data/example-jpegs \
  --output fastlens/data/example-fastlens --verify full
python -m fastlens convert --help
```

Conversion preserves original quantized DCT coefficients and their metadata.
It does not resize images or re-encode JPEGs, and the Store may be larger than
its source JPEGs. Directory inputs use sorted relative paths by default.
`--verify full` compares every image's coefficients, geometry, sampling and
quantization tables with the source before publication.

### Inspect the Store

```bash
python -m fastlens inspect fastlens/data/example-fastlens
```

A published Store contains `manifest.bin`, `partitions/` and `geometry/`, with
optional `annotations/` and removable `conversion/` records. Inspection checks
structure, metadata, indexes and source-order bindings; it does not checksum
all FLS payload bytes. Details are in [Store format](fastlens/docs/spec/STORE.md).

### Read DCT tensors

```python
from fastlens.profiles.rgbnomore import VALIDATION
from fastlens.torch import DirectDctReader

reader = DirectDctReader(
    "fastlens/data/example-fastlens/manifest.bin",
    module_path="./build/fastlens/torch",
)
with reader.iter_batches([[0, 1]], profile=VALIDATION) as batches:
    for batch in batches:
        print(batch.sample_ids, batch.y.shape, batch.cbcr.shape)
```

This profile requires compatible JPEG geometry and produces CUDA inputs for a
compatible DCT model. It is not an automatic input adapter for arbitrary RGB
models. See [Direct-DCT data loading](fastlens/docs/guides/DIRECT_DCT.md) for pipelines,
profiles, labels and CUDA stream ownership.

## Documentation

The [documentation index](fastlens/docs/README.md) provides reading paths for
using FastLens, understanding layouts, integrating APIs, modifying the runtime
and reproducing experiments. Continue from this Quickstart with
[Conversion](fastlens/docs/guides/CONVERSION.md) and
[Direct-DCT data loading](fastlens/docs/guides/DIRECT_DCT.md).

## Reproduce the paper

Start with the [paper reproduction guide](fastlens/benchmarks/paper/README.md)
and [reference preparation](fastlens/benchmarks/paper/REBUILDING_REFERENCES.md).
They define the datasets, class mappings, checkpoints, baseline dependencies and
measurement protocols. Published tables are linked under [Key results](#key-results).

```bash
python -m fastlens.benchmarks.paper.reproduce build
python -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run --help
```

Full experiments require prepared ImageNet data and model checkpoints. To redraw
the published plots using included measurements, without a GPU or dataset:

```bash
python -m fastlens.benchmarks.paper.reproduce figures
```

## Testing

[Development](fastlens/docs/DEVELOPMENT.md#converter-and-reader-tests) describes CPU/native checks,
strict GPU tests and the Quickstart integration tests.

## Repository layout

The [architecture guide](fastlens/ARCHITECTURE.md#source-organization) maps the
implementation packages; the [benchmark index](fastlens/benchmarks/README.md)
separates paper experiments, baseline implementations and development tools.

## License

See [LICENSE](LICENSE) for the project license. FastLens includes a modified
FastLanes dependency under `third_party/fastlanes/`, linked through
`FastLanes::core`. Its [changes and provenance](third_party/fastlanes/CHANGES.md)
and [upstream license](third_party/fastlanes/LICENSE) are retained with the
original copyright notices.
