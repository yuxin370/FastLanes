# FLS metadata benchmark

Build the tools and run a baseline/candidate comparison with fixed ordering and seed:

```bash
cmake --build build --target galp_fls_metadata_tool galp_fls_metadata_benchmark -j2
python3 galp/benchmarks/metadata/run.py \
  --baseline /path/to/legacy.fls \
  --candidate /path/to/compacted.fls \
  --output-dir /tmp/galp-fls-metadata-benchmark \
  --repeats 5 --cache-state warm --cpu-list 0-3
```

`--cache-state` is provenance, not an instruction to mutate the OS page cache. A cold-cache run must be
prepared externally and recorded as such. The metadata phase measures descriptor open plus direct rowgroup/column
lookups. The decode phase range-reads one rowgroup and decodes all its vectors; each multithread worker uses its own
`Connection` and `TableReader`, matching the current API. Stage timing and process RSS are emitted as JSON.

Analyze and aggregate a complete canonical sharded dataset separately:

```bash
python3 galp/benchmarks/metadata/breakdown_dataset.py \
  --input-dir /path/to/dataset \
  --output-dir /tmp/galp-fls-metadata-breakdown
```
