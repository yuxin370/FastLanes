# FastLens architecture and interfaces

FastLens is a GPU image-input system. Its implementation is in
`fastlens/`; a modified FastLanes library is an internal dependency under
`third_party/fastlanes/`. The root CMake project builds both. Dependency
provenance and modifications are documented in `third_party/fastlanes/CHANGES.md`.
See [README.md](../README.md) for build and usage examples.

## Execution path

```text
Sample IDs, model profile, spatial and frequency requests
    -> native logical-batch scheduling
    -> physical shard and compressed-vector planning
    -> asynchronous reads and compressed H2D transfers
    -> CUDA decoding and tensor construction
    -> PyTorch tensors in logical sample order
```

Python expresses requests and exposes tensors. Native code owns scheduling,
physical segmentation, bounded prefetch, storage lifetime and metric aggregation.
Benchmark runners select workloads and record measurements.

## Source organization

Paths are relative to the FastLens repository root.

| Path | Responsibility |
| --- | --- |
| `CMakeLists.txt`, `scripts/` | Standalone build, installation and smoke checks |
| `fastlens/include/fastlens/` | Public C++ headers |
| `fastlens/src/api/` | Table, Direct-DCT and PLS entry points |
| `fastlens/src/core/`, `fastlens/src/format/` | Data types, descriptors and file readers |
| `fastlens/src/engine/`, `fastlens/src/codecs/` | Execution worksets and compression primitives |
| `fastlens/src/cuda/` | Streams, events, allocation and resource management |
| `fastlens/src/jpeg/` | JPEG-DCT metadata, planning, decoding and transforms |
| `fastlens/src/direct_dct/` | Logical batches, physical layouts, lifetime and metrics |
| `fastlens/torch/`, `fastlens/profiles/` | PyTorch bindings and model input profiles |
| `fastlens/tools/`, `fastlens/examples/` | Data conversion and runnable consumers |
| `fastlens/benchmarks/paper/` | Section-based experiments and published tables |
| `third_party/fastlanes/` | Vendored FastLanes dependency |

## Python interface

Import the reader and batch interfaces from `fastlens.torch`.
`DirectDctReader` opens a manifest; a profile specifies geometry, normalization
and the model-facing representation. `read` processes a single logical batch;
`iter_batches` and `pipeline` expose repeated reads.

```python
from fastlens.profiles.rgbnomore import VALIDATION_CENTER_CROP_512
from fastlens.torch import DirectDctReader

reader = DirectDctReader("data/example-fastlens/manifest.bin",
                         module_path="build/fastlens/torch")
with reader.iter_batches([[0, 1], [2, 3]],
                         profile=VALIDATION_CENTER_CROP_512,
                         coefficients=None) as batches:
    for batch in batches:
        y, cbcr = batch.tensors
```

`coefficients=None` selects all source coefficients. Explicit selection refers
to storage columns; profile output geometry is a separate contract. CNN
profiles can expose a projected NCHW tensor through `batch.projected`.
Training combines a dataset mapping and `TrainingPolicy` with an epoch schedule.
The interfaces under `fastlens.torch.experimental` expose PLS pools and
microbatches. See the [PLS guide](benchmarks/training_pls/README.md) and
[DCT augmentation semantics](DCT_DOMAIN_AUGMENTATIONS.md).

## Ownership and CUDA streams

Returned tensors retain their native storage. Before consuming a batch on
another CUDA stream, call `batch.record_stream(consumer_stream)`. Native
completion tracks producer work and registered consumers before reclaiming
storage. Close pipelines explicitly or use their context managers.

Metric snapshots can include pending asynchronous work; finalized measurements
belong to completed runs. Diagnostics live in `fastlens.diagnostics` and are
separate from the model input interface.

## C++ interface and tests

Use `<fastlens/fastlens.hpp>` with `FastLens::core` for table decompression, or
`<fastlens/direct_dct.hpp>` with `FastLens::jpeg_dct_device` for device-resident
DCT inputs. Rowgroup and column views borrow their owning table's storage.
Installed-package consumers are under `fastlens/examples/`.

Native tests live in `fastlens/tests/`; Torch checks live in `fastlens/torch/`.
Benchmark tests remain beside their implementations. Run
`./scripts/smoke_test.sh` for public build/import checks and consult the root
README for the CPU and GPU test labels.
