# FastLens architecture

FastLens is a GPU image-input system. Its implementation is in
`fastlens/`; a modified FastLanes library is an internal dependency under
`third_party/fastlanes/`. The root CMake project builds both. Dependency
provenance and modifications are documented in `third_party/fastlanes/CHANGES.md`.
See the [documentation index](docs/README.md) for user guides and
[Development](docs/DEVELOPMENT.md) for builds and tests.

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

`fastlens.torch` translates sample IDs and semantic profiles into native logical
batch requests. The reader and pipeline share native scheduling and lifetime
management; Python does not maintain a second producer queue. Profiles define
the model-facing geometry and normalization separately from storage-column
selection.

The [Direct-DCT guide](docs/guides/DIRECT_DCT.md) owns usage examples and profile
requirements. [API boundaries](docs/reference/API_BOUNDARIES.md) identifies
public and advanced entry points; [augmentation semantics](docs/guides/DCT_AUGMENTATIONS.md)
defines the coefficient-domain transformations.

## Ownership and CUDA streams

Native batches retain storage through producer completion and registered consumer
events. Scheduling and reclamation live in `fastlens/src/direct_dct/`; Python
exposes the batch lifetime and stream-registration operations described in the
[Direct-DCT guide](docs/guides/DIRECT_DCT.md#reader-and-pipeline).

Metric snapshots can include pending asynchronous work. Finalized measurements
belong to completed runs; [diagnostic metrics](tools/DIAGNOSTICS.md) defines their
scope separately from the model input interface.

## C++ interface and tests

The public C++ headers expose table operations and opt-in device DCT primitives;
[API boundaries](docs/reference/API_BOUNDARIES.md) defines those surfaces.
Installed-package consumers live under `fastlens/examples/`. Native tests are in
`fastlens/tests/`, Torch checks in `fastlens/torch/`, and benchmark tests beside
their implementations. See [Development](docs/DEVELOPMENT.md#converter-and-reader-tests) for execution.
