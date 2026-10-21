# FastLens

**Selective Compressed Input for GPU Image Pipelines**

FastLens is a GPU-oriented image input system that combines selective compressed
storage, compressed host-to-device transfer, GPU decoding, and direct construction
of model-ready PyTorch tensors. It targets DCT-domain image training and inference,
where a model can consume representations derived from JPEG coefficients without
reconstructing RGB pixels.

This is the anonymous research artifact accompanying the
[FastLens paper](docs/paper/outputs/fastlens-final.pdf). FastLens is an independent
project. Its implementation is in `fastlens/`, and a modified FastLanes library
is included as an internal dependency in `third_party/fastlanes/`. The root
`CMakeLists.txt` builds FastLens and its dependencies together. All commands
below run from the **FastLens repository root**.

## How it works

```text
Offline preparation
  JPEG images -> quantized DCT coefficients + image metadata
              -> cross-image block-major FastLanes storage

Online input pipeline
  sample IDs + spatial transforms + source-frequency selection
    -> dependency-aware selection of compressed vectors
    -> selective reads and compressed host-to-device transfer
    -> mixed-dispatch GPU decoding
    -> direct tensor construction in logical sample order
    -> PyTorch model
```

- **Select before reading and decoding.** A cross-image block-major layout groups
  corresponding spatial blocks across images and stores DCT frequencies as
  columns. The planner maps spatial and frequency requests to compressed vectors,
  including the auxiliary data and referenced columns needed to decode them.
- **Keep transfers compressed.** The CPU plans and stages selected data; CUDA
  kernels decode heterogeneous FastLanes expressions through mixed dispatch.
- **Construct the requested tensors directly.** GPU kernels apply profile-specific
  geometry and numerical processing to decoded coefficients, avoiding full RGB
  reconstruction and unnecessary intermediate grids.
- **Preserve training locality.** Asynchronous preparation separates physical read
  order from model-visible sample order. The training policy shuffles partitions
  and samples within preparation pools, with partition-shared crops and
  per-sample augmentation.

Selection operates at the format's vector granularity, so physical reads can
include values beyond the logical request. Source-frequency selection is also
separate from output-channel selection: resizing can require source frequencies
that are absent from the final model tensor. Reducing source frequencies can
change model inputs and must match the model's preprocessing contract.

## Results reported in the paper

The matched DCT comparison uses the same source data, DCT model, and numerical
semantics for FastLens and the PyTorch DCT baseline. Across seven workloads on
ImageNet-1K, the paper reports these average speedups (§6.2.1):

| GPU | Training | Inference |
| --- | ---: | ---: |
| NVIDIA RTX 4090 | 3.80× | 6.17× |
| NVIDIA H100 | 4.20× | 7.95× |

The evaluation uses CUDA 12.8 and PyTorch 2.11. These are workload-specific
results, not expected speedups for every model or machine. The RGB deployment
comparisons use different input representations and should be interpreted
separately from the matched DCT comparison.

The layout trades storage space for selective access and parallel execution:
the evaluated training store occupies 2.12× the source JPEG size (§6.3). In the
compressed-transfer ablation, H2D traffic falls by 87.7% relative to raw `int16`
transfer for the same workload (§6.4).

## Requirements and build

The instructions below target Linux with an NVIDIA GPU.

- CMake 3.22 or newer and Ninja.
- A C++20 compiler; the root build defaults to Clang and requires Clang 13 or
  newer. It also accepts GCC 11 or newer. Use a CUDA-supported host compiler.
- A CUDA development toolkit, including `nvcc`, Thrust/CCCL, and a compatible
  NVIDIA driver. CUDA 12.8 is the paper's reference version.
- The libjpeg-compatible development API (`jpeglib.h` and `libjpeg`), for example
  `libjpeg-dev` or `libjpeg-turbo8-dev`. The TurboJPEG API alone is insufficient.
- Python 3.10 or newer and CUDA-enabled PyTorch for the Python interface;
  PyTorch 2.11 is the paper's reference version.

Configure the image tools and PyTorch extension in an environment containing
those dependencies:

```bash
cmake -S . -B build-fastlens -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DFASTLENS_BUILD_TESTS=OFF \
  -DFASTLENS_BUILD_TOOLS=ON \
  -DFASTLENS_WITH_JPEG_DCT=ON \
  -DFASTLENS_BUILD_TORCH=ON \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DCMAKE_CUDA_ARCHITECTURES="89;90"

cmake --build build-fastlens --parallel --target \
  fastlens_cli fastlens_jpeg_dct_tool fastlens_block_major_access_tool \
  _fastlens_direct_dct
```

`89;90` builds for RTX 4090 and H100; set the architecture list for the target
GPU. CMake discovers PyTorch through the selected Python interpreter. If libjpeg
is missing, configuration disables JPEG DCT support and the requested PyTorch
extension cannot be built.

For a C++-only build, set `FASTLENS_BUILD_TORCH=OFF` and omit
`_fastlens_direct_dct` from the build targets. Optional nvCOMP comparisons require
`FASTLENS_WITH_NVCOMP=ON`; nvCOMP is not needed for the input pipeline.

## Prepare a small image dataset

Place at least four 512×512 RGB JPEG images with 4:2:0 chroma subsampling in
`data/example-jpegs/` for the examples below. This matches the source geometry of
the demonstrated model profile. Dataset resizing and JPEG preparation are
separate from the conversion command.

```bash
build-fastlens/fastlens/tools/jpeg_dct/fastlens_jpeg_dct_tool \
  --shard --out-dir data/example-fastlens \
  --preset balanced --physical-layout spatial-major \
  --shard-images 1024 --metadata-profile reconstruct \
  data/example-jpegs

build-fastlens/fastlens/tools/jpeg_dct/fastlens_block_major_access_tool \
  data/example-fastlens/manifest.bin \
  --output-dir data/example-fastlens/block_major_access_v1
```

The CLI name `spatial-major` denotes the cross-image block-major organization.
Conversion extracts the JPEG's quantized DCT coefficients and writes
`manifest.bin`, compressed `shard_*.fls` files, and `shard_*.meta.bin` metadata.
The second command builds the access indexes used by the block-major reader.
Keep these files together. To control image ID assignment, replace the input
directory with `--input-list ordered_paths.txt`, containing one JPEG path per
line in the intended order.

Check that conversion preserves the source coefficients:

```bash
build-fastlens/fastlens/tools/jpeg_dct/fastlens_jpeg_dct_tool \
  --verify-manifest data/example-fastlens/manifest.bin data/example-jpegs
```

This verifies DCT data, not byte-for-byte reconstruction of the original JPEG
container. A small dataset exercises functionality; it does not reproduce the
paper's throughput measurements.

## Use the PyTorch interface

The public API is `fastlens.torch`. A profile specifies model-facing geometry,
normalization, and tensor representation. For a source build, `module_path`
points to the directory containing the compiled extension:

```python
from fastlens.profiles.rgbnomore import VALIDATION_CENTER_CROP_512
from fastlens.torch import DirectDctReader

reader = DirectDctReader(
    "data/example-fastlens/manifest.bin",
    module_path="build-fastlens/fastlens/torch",
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
such as `[5, 0, 2]`, preserves the requested coefficient order. The model profile
controls the output representation; this selection does not imply an arbitrary
change to the model's input shape.

The repository also provides a complete example with a small CUDA classifier:

```bash
python3 fastlens/examples/direct_dct_pipeline_demo.py \
  data/example-fastlens/manifest.bin \
  --module-path build-fastlens/fastlens/torch \
  --profile validation-center-crop-512 --batch-size 2 --steps 3
```

For CNNs, `fastlens.profiles.DctModelProfile` defines output channels, geometry,
and normalization; `DirectDctPipeline` returns the projected NCHW tensor through
`batch.projected`. Training uses `TrainingPolicy` with a dataset mapping and
`pipeline.start_epoch(epoch)`, exposing `batch.targets`. The model profile and
labels must match the chosen dataset and checkpoint. The paper's
`FastLensPipeline` listing is illustrative; the implemented Python class is
`DirectDctPipeline`.

Returned tensors retain native storage. Before consuming a batch on a different
CUDA stream, call `batch.record_stream(consumer_stream)`. The Direct-DCT path
produces DCT-derived tensors rather than a general-purpose RGB DataLoader.

## C++ interface

Use `<fastlens/fastlens.hpp>` for table decompression, or
`<fastlens/direct_dct.hpp>` for GPU-resident DCT batches. The public CMake targets
are `FastLens::core` and `FastLens::jpeg_dct_device`, respectively.

```cpp
#include <fastlens/fastlens.hpp>

int main() {
    fastlens::Reader reader("data/example-fastlens/shard_000000.fls");
    fastlens::DecompressOptions options;
    options.write_output = true;
    options.scope = fastlens::TableDecompressionScope::WholeTable;
    fastlens::Table table = reader.decompress(options);
    return table.empty() ? 1 : 0;
}
```

The materialized table API supports `int8_t` and `int16_t` columns. Rowgroup and
column views borrow storage from their owning `Table`. This API materializes
host-accessible results; use the Direct-DCT interface for model inputs that stay
on the GPU. Complete CMake consumers are in `fastlens/examples/`.

## Reproduce the paper

The experiment entry points in `fastlens/benchmarks/paper/` follow §§6.2–6.5 of
`fastlens-final.pdf`. Retained CSV/JSON results are under
`fastlens/benchmarks/paper/results/`.

To redraw the paper's ten plots from the retained results:

```bash
python3 -m fastlens.benchmarks.paper.reproduce figures
```

This requires the plotting dependencies, including NumPy and Matplotlib, but no
GPU, dataset, native build, or model checkpoint. Outputs go to
`results/reproduced/paper/figures/`. It redraws recorded measurements; it does
not collect new measurements.

For fresh experiments, the runner builds the native benchmark targets and
provides per-experiment help and command previews:

```bash
python3 -m fastlens.benchmarks.paper.reproduce build
python3 -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run --help
python3 -m fastlens.benchmarks.paper.section_6_2_1.matched_dct.run run --dry-run
```

| Paper section | Experiment directory under `fastlens/benchmarks/paper/` |
| --- | --- |
| §6.2.1: matched DCT training and inference | `section_6_2_1/matched_dct/` |
| §6.2.2: selected frequencies and compact inputs | `section_6_2_2/shape_preserving_selection/`, `section_6_2_2/compact_input/` |
| §6.2.3–6.2.5: deployment, resident reference, convergence | `section_6_2_3/deployment/`, `section_6_2_4/resident_training/`, `section_6_2_5/training_convergence/` |
| §6.3: storage and transfer volume | `section_6_3/storage/`, `section_6_3/h2d_traffic/` |
| §6.4: projection, staging, transfer, tensor construction | `section_6_4/projection_and_staging/`, `section_6_4/compressed_transfer_and_tensor/` |
| §6.5: locality and constrained read bandwidth | `section_6_5_1/block_major_locality/`, `section_6_5_2/read_bandwidth/` |

Each experiment directory has a `run.py` module with `--help`. Fresh measurements
require ImageNet data, prepared stores, model checkpoints, and the dependencies
for the selected baselines. The benchmark Python environment also uses
`torchvision`, NumPy, Pillow, `einops`, `psutil`, and Matplotlib. DALI, FFCV, CoorDL,
and L3 are needed for their respective comparisons; JPEG-B8 measurements require
the RGB-no-more `dct_manip` extension, and transfer diagnostics use Nsight Systems.

For fresh full-population inference and H2D captures, follow the
[reference rebuilding guide](benchmarks/paper/REBUILDING_REFERENCES.md). It
describes how to prepare public model code and checkpoints, generate contracts
and predictions from the ImageNet inputs, and profile a newly generated command.
Run timed measurements on an otherwise idle GPU and retain the protocol of the
selected experiment.

## Tests

For a separate test build, enable FastLens testing through the root option,
which also arranges the GoogleTest dependency:

```bash
cmake -S . -B build-fastlens-tests -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFASTLENS_BUILD_TESTS=ON \
  -DBUILD_TESTING=ON \
  -DFASTLENS_BUILD_BENCHMARKS=OFF \
  -DFASTLENS_BUILD_TOOLS=ON \
  -DFASTLENS_WITH_JPEG_DCT=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89;90"
cmake --build build-fastlens-tests --parallel
ctest --test-dir build-fastlens-tests -L public-api --output-on-failure
ctest --test-dir build-fastlens-tests -L fastlens-cpu --output-on-failure
ctest --test-dir build-fastlens-tests -L fastlens-gpu --output-on-failure
```

CUDA tests need a working device, and some reader/integration tests also need
local datasets. Missing prerequisites can cause tests to skip; a skipped test
does not establish GPU correctness. The CUDA toolkit is required to configure
this component even when only CPU tests are selected.

## Source layout

| Path within `fastlens/` | Purpose |
| --- | --- |
| `include/fastlens/` | Public C++ API |
| `src/format/`, `src/core/` | Storage descriptors, readers, expression planning |
| `src/engine/`, `src/codecs/`, `src/cuda/` | Worksets, codecs, GPU execution, memory management |
| `src/direct_dct/` | Direct-DCT runtime and native pipeline |
| `torch/`, `profiles/` | PyTorch interface and model input profiles |
| `tools/` | Dataset conversion, access indexes, CLI |
| `examples/`, `tests/` | Usage examples and correctness checks |
| `benchmarks/paper/` | Paper reproduction entry points and retained results |
