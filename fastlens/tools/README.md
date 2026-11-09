# FastLens command reference

[Documentation index](../docs/README.md) · [Development](../docs/DEVELOPMENT.md)

Run commands from the repository root after building the required targets.

| Task | Supported entry point | Details |
| --- | --- | --- |
| Convert JPEGs | `python -m fastlens convert INPUT --output STORE` | [Conversion](../docs/guides/CONVERSION.md) |
| Inspect a published Store | `python -m fastlens inspect STORE` | [Store contract](../docs/spec/STORE.md) |
| Check recorded FLS CRC64 | `python -m fastlens inspect STORE --check-payload` | Requires conversion with `--record-payload-crc` |
| Build / inspect annotations | `python -m fastlens annotations build STORE` / `inspect STORE` | [Annotations](../docs/guides/ANNOTATIONS.md) |
| Write native Image-major | `fastlens_jpeg_dct_tool` | [Layout and invocation](../docs/spec/IMAGE_MAJOR.md#generate-the-layout) |
| Benchmark native table / pipeline work | `fastlens_cli` | [Native command reference](NATIVE_REFERENCE.md) |
| Reproduce paper experiments | Section-specific Python modules | [Paper guide](../benchmarks/paper/README.md) |
| Interpret diagnostics | Native benchmark output | [Metrics dictionary](DIAGNOSTICS.md) |

The public Python CLI accepts the Published Block-major Store contract. Native
writers, compact-format audits and benchmark tools have their own inputs and
support boundaries. They are documented separately from model-facing APIs.

```bash
python -m fastlens convert --help
python -m fastlens inspect --help
python -m fastlens annotations build --help
python -m fastlens annotations inspect --help
```

Native commands emit metrics directly; use the paper entry points above for
published comparisons and the diagnostic metrics reference for counter meanings.
