# Direct-DCT API boundaries

[Documentation index](../README.md)

See [API stability and environments](API_STABILITY.md) for the source
compatibility policy, deprecation rules, ABI limits and tested configurations.

The model-facing stable API is the `fastlens.torch` facade:

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

`fastlens.torch.experimental` exposes PLS pools and microbatches outside the stable
facade. The [PLS guide](../../benchmarks/training_pls/README.md) describes their use.

Use `fastlens/stable.hpp` for new C++ table-reading integrations. It exposes
readers, tables, decompression options and errors.

Native C++ integrations that deliberately need device descriptors and runtime
primitives use `fastlens/advanced/direct_dct.hpp`. Raw, versioned execution
counters use `fastlens/diagnostics/direct_dct.hpp`. Implementation-only planning,
scheduling, lifetime and allocation details remain under `fastlens/src`
or the private `_fastlens_direct_dct` extension.

## API compatibility

New code should include `fastlens/stable.hpp` and opt into advanced Direct-DCT
explicitly. The [source compatibility policy](API_STABILITY.md#source-and-binary-compatibility)
defines the treatment of existing adapters and their deprecation lifetime.
