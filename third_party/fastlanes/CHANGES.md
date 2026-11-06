# Modifications to FastLanes

Upstream: https://github.com/cwida/FastLanes

Upstream commit recorded in the source repository's merge history:
`896e92c7886b7da575143dc45cfad147f9ff90fb`.
This identifies the upstream parent recorded before the FastLens modifications;
the vendored snapshot is not an unmodified upstream release.

FastLens includes a modified version of FastLanes under its original MIT license.
Relative to the recorded upstream commit, the storage changes include:

- Memory-table ingestion and explicit rowgroup sizing for DCT inputs.
- Bounded parallel encoding and support for large null-free rowgroups.
- Scatter/batched reads and optional io_uring I/O support.
- Footer/column metadata needed by those storage and access paths.
- FSST/ALP corrections and related expression/type changes.

FastLanes builds as a subdirectory dependency, with compiler options and the ALP
rowgroup constant propagated to consumers of its headers. This repository includes
the library source, CMake files, dataset declarations and fixture generators
needed by FastLens. Upstream
benchmark records, sample datasets, standalone bindings, publications and other
optional projects are not distributed. Build this dependency through the FastLens
root project. Dataset declarations use source-relative paths.

The upstream snapshot also contained G-ALP/GPU code from which parts of FastLens
evolved. Original copyright and license notices in those files are preserved.
