# `e2e_v3` data retention and cleanup status

This document records the current split storage layout and the cleanup gates.
Never clean either data root while a compressor, the full preparation wrapper,
coefficient verification, or payload fingerprinting process is still active.

The official training Compact-v3 snapshot no longer lives below `e2e_v3`.
Both `training_manifests_official_v3/train.json` and the 2026-08-04 training
contracts point to the snapshot on `/mnt/nvme2`.

## Canonical storage roots

- repository-local JPEG and validation data:
  `/home/tangyuxin/gfastlanes/FastLanes/galp/data/system_rgbnomore/e2e_v3`;
- external official training snapshot:
  `/mnt/nvme2/home/tangyuxin/galp/compact_v3_tiled_z32_rgbnomore512_train_24x2_20260803`;
- external training and inference run artifacts:
  `/mnt/nvme2/home/tangyuxin/galp/{training_runs,inference_runs}`.

## Production data to retain

- `imagenet_512/train`: 1,281,167 RGB-no-more 512x512 training JPEGs;
- `imagenet_512/val`: 50,000 RGB-no-more 512x512 validation JPEGs;
- external snapshot `compact_v3_tiled_z32_rgbnomore512_train_24x2_20260803`:
  official training Compact-v3 data, manifest, labels, preparation evidence and
  frozen reader binaries;
- `compact_v3_tiled_z32_rgbnomore512`: official validation Compact-v3;
- `training_manifests_official_v3`: official train/validation benchmark JSON;
- `100-jpeg`: small semantic/debug fixture;
- `galp-v3-train-canary/train-1k-v3c` and `train-10k-v3c`: the latest canary
  evidence currently referenced by tests/docs.

## Required gates before moving anything

From the repository root:

```bash
pgrep -af '[p]repare_imagenet512_v3_train|[g]alp_jpeg_dct_tool|[v]erify-manifest|[p]ayload_fingerprint'

PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
E2E=/home/tangyuxin/gfastlanes/FastLanes/galp/data/system_rgbnomore/e2e_v3
TRAIN=/mnt/nvme2/home/tangyuxin/galp/compact_v3_tiled_z32_rgbnomore512_train_24x2_20260803

"$PY" -B galp/benchmarks/system_rgbnomore/training/manifest_preflight.py \
  "$TRAIN/dct/manifest.bin" \
  --expected-manifest-version 3 \
  --expected-physical-layout image-major-vector-rowgroups \
  --expected-spatial-order tiled-z32 \
  --expected-image-count 1281167

"$PY" -B galp/benchmarks/system_rgbnomore/training/manifest_preflight.py \
  "$E2E/compact_v3_tiled_z32_rgbnomore512/manifest.bin" \
  --expected-manifest-version 3 \
  --expected-physical-layout image-major-vector-rowgroups \
  --expected-spatial-order tiled-z32 \
  --expected-image-count 50000

test -f "$TRAIN/artifacts/COMPLETE.json"
jq -e '.coefficient_exact_verification == true and .expected_image_count == 1281167' \
  "$TRAIN/artifacts/COMPLETE.json"
jq -e '.ok == true and (.failures | length) == 0' \
  "$TRAIN/artifacts/reader_verify.json"
```

The first command must print no active data-preparation process. Both
preflights and both `jq` checks must exit zero. The current external training
snapshot contains 1,281,167 images, 939 shards and 1,878 fingerprinted payload
files. If the external snapshot moves, update `training_manifests_official_v3`
and regenerate benchmark contracts instead of leaving stale absolute paths.

## Current quarantine candidates

The following repository-local paths are not production inputs for the current
official 512x512 train/val pair:

| Path below `e2e_v3` | Approx. size | Reason |
| --- | ---: | --- |
| `compact_v3_tiled_z32` | 11 GB | superseded 50K dataset; keep only for historical reproduction |
| `compact_v3_tiled_z32_rgbnomore512_train_prepare.retry3-20260802` | 280 MB | failed/retried preparation snapshot |
| `compact_v3_tiled_z32_rgbnomore512_train_prepare{,.retry-*}` | negligible | old preparation contracts and retry logs |
| `runs` | 128 MB | diagnostic results; archive if their reports are still needed |

Do not move `train-1k-v3c` or `train-10k-v3c`: the current repository still
references them.

After all gates pass, create an inventory and move any selected candidates to
a new recoverable quarantine on the same filesystem. Resolve every target
explicitly; do not use a wildcard for retry directories.

```bash
E2E=/home/tangyuxin/gfastlanes/FastLanes/galp/data/system_rgbnomore/e2e_v3
QUAR=/home/tangyuxin/gfastlanes/FastLanes/galp/data/system_rgbnomore/.trash/e2e_v3-$(date +%Y%m%d-%H%M%S)
mkdir -p "$QUAR"

du -sb \
  "$E2E/compact_v3_tiled_z32" \
  "$E2E/compact_v3_tiled_z32_rgbnomore512_train_prepare.retry3-20260802" \
  "$E2E/runs" \
  | tee "$QUAR/inventory.du.txt"

mv "$E2E/compact_v3_tiled_z32" "$QUAR/"
mv "$E2E/compact_v3_tiled_z32_rgbnomore512_train_prepare.retry3-20260802" "$QUAR/"
mv "$E2E/runs" "$QUAR/"
```

Run the official four-pipeline smoke and validator after the move.  Keep the
quarantine until that smoke, a reader sample check, and at least one restart
from the official paths all pass. Only then remove the exact quarantine
directory.

## Cleanup history

On 2026-08-04, after the external official training snapshot had passed
manifest preflight, payload fingerprinting, metadata validation and exact
coefficient verification, the following repository-local data was permanently
removed:

- `.trash/e2e_v3-cleanup-20260802-144741`: approximately 12 GiB of already
  quarantined failed/superseded outputs;
- `e2e_v3/compact_v3_tiled_z32_rgbnomore512_train.incomplete-20260802-2229`:
  approximately 14 GiB of incomplete output with no final `manifest.bin` or
  `COMPLETE.json`.

The `.trash` parent directory remains available for future recoverable
quarantines.

## `.tmp` generation directories

A `.tmp` generation can be the active transactional output of a compressor;
its suffix alone never proves that it is stale.  After both compressors have
exited and the final manifests pass preflight, list them with:

```bash
find /home/tangyuxin/gfastlanes/FastLanes/galp/data/system_rgbnomore/e2e_v3 \
  -type d -name 'generation_*.tmp' -printf '%TY-%Tm-%Td %TH:%TM %s %p\n'
```

Move each confirmed unreferenced generation by its exact printed path into the
same quarantine.  Do not use a recursive wildcard deletion: current and failed
generations can coexist while a compressor is running.
