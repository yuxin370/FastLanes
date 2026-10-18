# DCT baselines review pack

Start with `DCT_Baselines_Implementation_Review.md`. The uploaded implementation is preserved byte-for-byte under `review_20261009/dct_baselines/`; only macOS metadata and stale bytecode are excluded.

## Contents

- Chinese implementation review and prioritized fixes.
- Unmodified audited Python/Markdown files.
- Three isolated reproduction scripts and their captured JSON outputs.
- Original unittest import-failure log, runtime inventory and source hashes.
- The previous plan's draft result schema, used only by the grouping probe.

## Reproduce the isolated checks

From this archive's extracted root, with Python, NumPy and SciPy installed:

```bash
python -B review_20261009/probes_storage/probe_storage.py
python -B review_20261009/probes_semantics/probe_contracts.py
python -B review_20261009/probes_benchmark/probe_runner.py
```

The storage script supplies only the absent `galp.benchmarks.common.sha256_file` helper using standard SHA-256. It imports the original storage/native code unchanged. It uses synthetic coefficient arrays and an independent SciPy DCT oracle; it does not read real JPEG coefficients.

The semantics script executes selected pure functions extracted from the uploaded source AST. The runner script likewise executes original control flow while replacing unavailable external pipeline components with explicit stand-ins; schema validation is only a required-field check. These scripts do not prove model, Torch, CUDA, full schema or full pipeline correctness.

The runner script generates synthetic fixture files beneath its own directory. Values such as 100 and 900 images/s are deliberately invented input values for the aggregation regression. Never merge these fixtures or their passed flags into actual benchmark evidence.

Original tests need the complete parent repository, its modifications, model dependencies, native bindings and checkpoints. In the review environment both test modules fail during import because PyTorch is absent. The 25 original test bodies were not run.

`MANIFEST.sha256` hashes all other files in this review pack. `audit_evidence/source_file_hashes.json` hashes the exact source files as received. No source fixes have been applied.
