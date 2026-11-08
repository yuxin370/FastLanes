# API stability and supported environments

The stable Python surface is `fastlens.torch`: `DirectDctReader`,
`DirectDctPipeline`, `DirectDctBatch`, registered profiles and documented metrics.
The C++ stable source surface is `fastlens/stable.hpp`. Their documented arguments,
results and ownership semantics are the compatibility contract. Private modules,
implementation headers and diagnostic counters are outside it; the
[API boundaries](../DIRECT_DCT_API_BOUNDARIES.md) identify advanced opt-in headers.

## Source and binary compatibility

Compatible releases preserve documented source usage and behavior. A breaking
change is identified in release notes with its replacement and affected entry
points. An ordinary deprecation names a replacement and earliest removal release,
and remains available for at least one intervening published release. No removal
release has been assigned to the existing `fastlens/direct_dct.hpp` and
`fastlens/fastlens.hpp` adapters.

There is no stable C++ ABI or cross-version binary compatibility promise for the
compiled Torch extension. Rebuild against the application's compiler, PyTorch and
CUDA installation when those change. The source compatibility policy does not
promise compatibility between arbitrary CUDA/PyTorch releases.

## Environments

| Scope | Status |
| --- | --- |
| Linux x86-64, C++20, libjpeg, CUDA toolkit | Maintained build/runtime target |
| Ubuntu 22.04/24.04, Python 3.11, CUDA 12.8, PyTorch 2.11 | Recorded tested configuration |
| RTX 4090 (SM 89), H100 (SM 90) | Paper's tested inference/training GPUs |
| Other Linux/compiler/toolkit/PyTorch combinations | Require local build and relevant tests; no tested version range is implied |
| Windows, macOS, CPU tensor execution | No support commitment in this release |

Conversion and inspection run on CPU without initializing CUDA or importing
PyTorch, but the current native build requires a CUDA toolkit. Direct-DCT tensor
execution requires a working NVIDIA GPU. A skipped GPU job is not GPU validation.
The project distributes a source checkout, not a supported wheel/ABI matrix.

## Store compatibility

The [format matrix](spec/README.md) is the single authority for supported disk
encodings and writer/reader/CLI combinations. Binary versions are independent of
API releases and physical-layout names. Native Image-major manifests are readable
subject to profile requirements; they do not acquire Published Store inspection
or annotations support. Both Image-major encodings are supported: one image per
rowgroup (manifest 2) and tiled vector rowgroups (manifest 3). Native
`random-access` continues to select the former. Any future removal must explicitly
state that old data is no longer readable.
