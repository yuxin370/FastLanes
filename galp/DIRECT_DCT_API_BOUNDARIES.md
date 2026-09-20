# Direct-DCT API boundaries

The model-facing stable API is the `galp.torch` facade:

- `DirectDctReader`, `DirectDctPipeline`, and `DirectDctBatch`;
- registered semantic profiles and the small `DirectDctMetrics` schema;
- Pythonic `coefficients=` request selection and the thin
  `DirectDctReader.iter_batches()` convenience iterator;
- `batch.record_stream(actual_consumer_stream)` for cross-stream consumers.

`coefficients=None`, `range(...)`, and an explicitly ordered index iterable
normalize to the existing native coefficient-selection request. Legacy
`dct_coeffs` strings remain source-compatible. `iter_batches()` owns no Python
producer, queue, or scheduling policy: it delegates to the same
`DirectDctPipeline.start()` and native iterator used by explicit pipeline code.

The canonical `galp/stable.hpp` C++ umbrella intentionally does not expose raw
CUDA events, device descriptors, physical rowgroups, allocator state, or
rollback selectors.

Native C++ integrations that deliberately need device descriptors and runtime
primitives use `galp/advanced/direct_dct.hpp`. Raw, versioned execution
counters use `galp/diagnostics/direct_dct.hpp`. Implementation-only planning,
submission, lifetime, allocator, and rollback details remain under `galp/src`
or the private `_galp_direct_dct` extension.

`galp/direct_dct.hpp` remains a deprecated source-compatible adapter to the
advanced header. The historical `galp/galp.hpp` umbrella also remains source
compatible for one deprecation window; new code should include
`galp/stable.hpp` and opt into advanced Direct-DCT explicitly. The Phase 3,
Phase 4, and Phase 6 rollback paths are likewise
internal construction-time seams. They may be removed after one release cycle
with native scheduler, lifetime, and physical orchestration enabled by default,
provided downstream compatibility builds and rollback telemetry are clean.
