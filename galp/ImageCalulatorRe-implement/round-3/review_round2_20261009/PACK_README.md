# Round 2 review materials

Start with `DCT_Baselines_Round2_Review_and_IC_Search.md` at the archive root.
It contains the Chinese review, exact evidence limits, remaining matched gates,
and the proposed native Image Calculator search tasks.

## Contents

- `submission/integration.patch`: unchanged uploaded integration patch.
- `reconstructed/galp/benchmarks/dct_baselines`: complete new-file hunks only;
  existing parent-repository files have not been synthesized or patched locally.
- `submission/evidence-v2/{models,source,stores}` and `evidence-h100`: unchanged
  core CPU/H100 records, manifests, JPEG and NPZ fixtures.
- `probes_storage`: independent NumPy probes and observed results.
- `probes_integrity`: read-only reference/hash/store/schedule audit and results.
- `probes_integration/trace_window_audit.json`: independent pure-function trace
  window review.
- `review_manifest.json`: SHA-256 and byte size for each included file other than
  that manifest itself.

The original 12.9 MB submission is not duplicated. Its large repeated regression
fixture trees, initial model snapshots, external repositories, and research PDFs
are not bundled here. The core evidence JSON bytes are unchanged; their original
absolute provenance paths have not been rebased. The audit uses explicit path
prefix mappings only to locate the corresponding attached bytes.

## Re-run local probes

From a fresh extraction at the archive root, with NumPy installed:

```bash
python review_round2_20261009/probes_storage/probe_storage.py
python review_round2_20261009/probes_integrity/audit_evidence.py
```

The storage probe creates its own fixtures under `probes_storage/artifacts` and
requires that those fixture destinations do not already exist. Use a fresh copy
for another run. Neither probe changes the implementation or uploaded evidence.
Both load the original store/contracts modules with only the unavailable parent
package's ordinary SHA-256 helper supplied locally. Pure training functions are
extracted from the original source via AST; no Torch or GPU model is mocked and
reported as executed.

The GPU and full-repository evidence remains submitted evidence. To reproduce it,
use the submitted base revision, complete dependencies and host scripts described
in `submission/README.md`. This review pack is not a stand-alone FastLens build or
an implemented IC optimizer.
