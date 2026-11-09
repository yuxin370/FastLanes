# FastLens documentation

Start with the [project Quickstart](../../README.md#quick-start). All commands
assume the repository root as the working directory.

| Reader goal | Reading path |
| --- | --- |
| Use FastLens for the first time | [Quickstart](../../README.md#quick-start) → [Conversion](guides/CONVERSION.md) → [Direct-DCT](guides/DIRECT_DCT.md) |
| Understand the two physical layouts | [Format index](spec/README.md) → [Block-major](spec/BLOCK_MAJOR.md) / [Image-major](spec/IMAGE_MAJOR.md) |
| Integrate Python or C++ | [Direct-DCT](guides/DIRECT_DCT.md) → [API boundaries](reference/API_BOUNDARIES.md) → [API stability](reference/API_STABILITY.md) |
| Modify the implementation | [Architecture](../ARCHITECTURE.md) → [Development](DEVELOPMENT.md) → the relevant source package |
| Reproduce the paper | [Benchmarks](../benchmarks/README.md) → [Paper](../benchmarks/paper/README.md) → the corresponding §6.x experiment |
| Assess baseline implementations and comparisons | [Baselines](../benchmarks/baselines/README.md) → implementation conventions → protocol → results |

## User guides

- [Conversion](guides/CONVERSION.md): JPEG inputs, Store creation, verification and relocation.
- [Annotations](guides/ANNOTATIONS.md): labels, sample identities and offline premixing.
- [Direct-DCT](guides/DIRECT_DCT.md): profiles, tensors, pipelines and CUDA stream ownership.
- [DCT augmentations](guides/DCT_AUGMENTATIONS.md): coefficient-domain geometry, rounding and training transforms.

## References and specifications

- [API boundaries](reference/API_BOUNDARIES.md) identifies public, advanced and private interfaces.
- [API stability](reference/API_STABILITY.md) defines compatibility, deprecation and environment support.
- [Binary specifications](spec/README.md) define layouts, field encodings and Reader/Store compatibility.
- [Published Store](spec/STORE.md) defines the deployment directory and integrity checks.

## Experiments and implementation

[Paper reproduction](../benchmarks/paper/README.md) owns experiment preparation
and measured configurations; [published results](../benchmarks/paper/results/README.md)
owns retained measurements and aggregation rules. Baseline conventions and
comparison protocols live beside the [baseline implementations](../benchmarks/baselines/README.md).
[Development benchmarks](../benchmarks/development/README.md) cover diagnostics
and performance investigations outside the paper experiment entry points.

[Architecture](../ARCHITECTURE.md) explains the execution path and source
responsibilities. [Development](DEVELOPMENT.md) covers builds and tests, while
the [command reference](../tools/README.md), [native options](../tools/NATIVE_REFERENCE.md)
and [diagnostic metrics](../tools/DIAGNOSTICS.md) describe the lower-level tools.
