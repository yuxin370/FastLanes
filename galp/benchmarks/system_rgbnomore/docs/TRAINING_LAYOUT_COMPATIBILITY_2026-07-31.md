# Direct-DCT training layout compatibility — 2026-07-31

> **Historical compatibility record.** Layout conclusions remain useful, but
> commands must be checked against the current training and DCT-major guides.
> Low-level GALP runtime choices are now owned by native profiles.

## Decision

The training loop is shared by image-major v2 and
`image-major-vector-rowgroups` v3.  It must not branch on manifest version or
physical layout.  The only layout-aware training component is the manifest
preflight; data acquisition goes through `DirectDctTrainingReader`, which wraps
the public `_galp_direct_dct.DirectDctReader` batch API.

Do not add a separate v2 training implementation.  Keep v2 as a regression
control and migration fallback, but spend performance work on v3 after the v3
native reader passes the gates below.  Block-major/DCT-major is not accepted by
this training runner yet and must remain in `system_dct_major` until its public
reader surface is stable.

## Required preflight gates

A formal run must fail before model construction unless all requested fields
match:

- manifest magic `GJDCTSH1` and dataset kind `jpeg-dct-sharded`;
- v2 + `image-major`, or v3 + `image-major-vector-rowgroups`;
- requested physical image count;
- every declared FLS/metadata payload exists and has the declared size;
- the train/validation JSON files are disjoint and contain exactly the intended
  counts.

The run writes `manifest_preflight.json`.  Optional v3 extension fields are
JSON `null` when absent; they are not synthesized from planner or payload
internals.  `contract.json` also freezes the manifest, payload fingerprints,
Torch binding hash, seed, rank/world size, sample-order hash and transform hash.

## Clean v3 canaries

The intended persistent canaries are:

| Canary | Physical images | Train JSON | Validation JSON |
| --- | ---: | ---: | ---: |
| 1K | 1,000 | 900 | 100 |
| 10K | 10,000 | 9,000 | 1,000 |

Create each scale with the strict selector and a fresh tag; never reuse a
partial output directory.  The script selects only 4:4:4/4:2:0 JPEGs before
compression, freezes the writer and Torch binding, verifies every coefficient,
and asserts the exact logical split:

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
export CANARY_TAG=v3c
galp/benchmarks/system_rgbnomore/training/prepare_v3_training_canary_strict.sh 1k
galp/benchmarks/system_rgbnomore/training/prepare_v3_training_canary_strict.sh 10k
```

Before using the result, verify the exact contract rather than trusting the
directory name:

```bash
export REPO=/home/tangyuxin/gfastlanes/FastLanes
export PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
export C1="$REPO/galp/data/system_rgbnomore/e2e_v3/galp-v3-training-canary-strict/train-1k-$CANARY_TAG"
export C10="$REPO/galp/data/system_rgbnomore/e2e_v3/galp-v3-training-canary-strict/train-10k-$CANARY_TAG"

PYTHONPATH="$REPO/galp/benchmarks/system_rgbnomore" "$PY" -c '
import json, os
from pathlib import Path
from training.manifest_preflight import preflight_manifest
for root, images, train, val in (
    (Path(os.environ["C1"]), 1000, 900, 100),
    (Path(os.environ["C10"]), 10000, 9000, 1000),
):
    result = preflight_manifest(
        root / "dct/manifest.bin",
        expected_manifest_version=3,
        expected_physical_layout="image-major-vector-rowgroups",
        expected_image_count=images,
    )
    train_doc = json.loads((root / "training_manifests/train.json").read_text())
    val_doc = json.loads((root / "training_manifests/val.json").read_text())
    assert len(train_doc["samples"]) == train
    assert len(val_doc["samples"]) == val
    print(result.version, result.physical_layout, images, train, val)
'
```

The strict preparation exposed two independent native issues on 2026-07-31:

1. an older `jpeg_dct_tool.cpp` normalization revision reset an explicitly
   selected v3 layout when `--spatial-order tiled-z-32` was also present.  The
   current CLI preserves an explicitly selected image-major layout; a 2026-08-01
   built-binary smoke proved version 3, `image-major-vector-rowgroups` and
   `tiled-z32`.  Preparation scripts now pass both properties and preflight all
   three persisted values so this cannot regress silently;
2. on a strict coefficient-exact 1,000-image dataset containing exactly 900
   train and 100 validation samples, the first actual 64-image training step
   completed, but the formal repeat originally failed with
   `Compact v3 rowgroup payload is empty`.  This zero-payload reader bug is now
   repaired and covered by the native and acceptance tests listed below.  A
   host RTX 4090 subsequently completed every pipeline in the fixed-seed 3+10
   run.  That run exposed a separate RGBNoMore crop-to-component mapping bug.
   The Python adapter was repaired, and the fresh current-code GPU run described
   below now closes that release gate.

The older `run_v3_train_canary.sh` first-per-class selector produced 871/100
because 29 selected JPEGs were outside the training adapter's 4:4:4/4:2:0
sampling contract.  Do not use that output for formal training.  The strict
selector replaces each unsupported candidate within the same class and records
the complete selection plus content hashes.

### Observed strict 1K evidence

The constituent strict preparation commands were exercised on 2026-07-31 in
`/tmp/galp-training-v3-strict-1k-20260731`:

- exactly 1,000 classes and 1,000 images were selected: 803 4:4:4 and 197
  4:2:0 JPEGs; 18 4:2:2, 10 grayscale and one 4:4:0 first candidates were
  rejected and replaced within their original classes;
- selection SHA256:
  `11c136eef57f3fc6733a71b89887aa807596278cf5772eb347ba5e34e3165447`;
- `manifest.bin` SHA256:
  `2d1e303fb60538e4321b9225e9b2cbee39dec632b87ce70ef78e4f103a14ac32`;
- the preflight reports version 3, `image-major-vector-rowgroups`, 1,000
  images, `tiled-z32`, one vector per rowgroup and `galp-compact-v1`;
- coefficient verification covered 7,875,509 blocks with no missing, extra or
  mismatched coefficients and maximum absolute difference zero;
- train/validation manifests contain exactly 900/100 samples, with SHA256
  `1ae0403a444adeaee8a2aed9a50359569f0a3062469b09147c8af9805b800738`
  and `6c54f43e21d7ba43ccf83a313689cfd0a8c4e8ff76e2aa031363b2ab8155e059`.

Before the repair, on an RTX 4090, the first semantic training probe consumed a
real 64-image GALP batch, produced finite loss `6.915570259094238`, finite
gradients and a parameter update.  The subsequent formal 3-warmup + 10-measured
run stopped in the then-current public native reader with
`Compact v3 rowgroup payload is empty`.
The loaded Torch extension was byte-identical to that pre-repair build (SHA256
`7089d193d4847ebcb3c204dae6e277ff585d8a33cf5790d601064fb598709424`).
No throughput result should be reported from this incomplete run.

The native failure has since been localized without adding storage knowledge
to the training layer.  This canary contains six zero-payload rowgroups across
physical images 6, 106, 286 and 917.  They are legal constant-vector
rowgroups: `CompactDescriptorV3.AcceptsZeroPayloadConstantVectorRowgroup`
explicitly tests this representation.  The production read paths in
`reader.cu` previously rejected every descriptor record whose `payload_size`
was zero.  Under seed `11997733`, train image 6 is at epoch-0 position 180,
batch index 2 for batch size 64, so the first semantic batch succeeded and the
third warmup batch deterministically reached the contradictory check.

The public reader now materializes legal zero-payload constant-vector
rowgroups from descriptor/schema metadata.  The independent Compact-v3
acceptance parser verifies the zero-length storage partition and zero CRC;
native descriptor opening additionally proves that every expression is a
constant or an acyclic alias-to-constant.  The training preflight and training
loop remain unchanged and do not parse rowgroups.  The current rebuilt Torch
extension SHA256 is
`70802bfba432bc9291ef67e1c198e4e1846b01c731b8de690dfc44ed442f587c`.

The post-repair host run is preserved at
`/tmp/galp-training-v3-1k-zero-payload-fix-20260731`.  All four pipelines
completed their three warmup and ten measured steps, their per-pipeline
correctness and artifact-integrity gates passed, and GALP crossed the six
metadata-only rowgroups without a read or prefetch failure.  Its one-repeat
throughputs were GALP 296.79, RGBNoMore 8.47, DALI 1009.42 and PyTorch 637.29
images/s.  These values are diagnostic only: one repeat is not a stable
performance comparison, and the overall validation correctly failed the DCT
cross-pipeline semantic gate.  That artifact froze an earlier post-zero-fix
Torch extension (`ea9cdb3ab401d466acd4234d10821964dac62c2368edbfde06e223221cd2aeb9`),
so it must not be presented as validation of the current binding or Python
adapter.

The semantic failure was independent of `payload_size == 0`: sample identities,
labels and augmentation descriptors matched, while the old RGBNoMore adapter
mapped a source-pixel crop by dividing luma coordinates by eight and deriving
chroma from the luma ratio.  For JPEG dimensions not divisible by eight this
selected different boundary blocks from the native reader.  The adapter now
maps the source-pixel interval independently onto each component's padded block
grid using floor(begin) and ceil(end).  Replaying all 64 frozen first-step
tensors reduced maximum Y/C drift from 1.75784/0.615686 to
0.000980451/0.000980451, at most one normalized integer-coefficient step.  A
complete CPU model replay from the same frozen initial model/optimizer state
then passed the current DCT semantic gates: loss difference 0, maximum logit
drift `8.91e-7`, gradient cosine `0.9999999999997`, and first-update cosine
`0.9999999756`.  The evidence is stored in
`/tmp/galp-training-v3-1k-fixed-crop-offline-semantic-20260801.json`.
The default DCT absolute tolerance is therefore `1/1020 + 1e-7`; it does not
permit the old crop error.  This is strong CPU/offline semantic evidence, not a
substitute for regenerating all GPU run artifacts with the current Python
source hashes.

### Implemented zero-payload contract

`payload_size == 0` means that a rowgroup has logical tuples but all requested
values are represented by metadata-only encoding expressions.  For example,
`EXP_CONSTANT_I08/I16` stores the value in `ColumnDescriptor.max`; an
`EXP_EQUAL` column may refer to such a source column.  It does not mean an empty
image, a missing rowgroup, or an all-zero tensor.

The core repair is a first-class metadata-only read path, not merely removal of
the old exception:

1. classify and validate metadata-only rowgroups when opening the Compact-v3
   descriptor.  A zero payload is accepted only when the reconstructed schema,
   coefficient ranges and segment geometry require zero physical bytes;
2. construct a valid zero-copy rowgroup without issuing `pread`, acquiring a
   pinned-buffer slot, uploading a compressed payload, or performing pointer
   arithmetic on a null zero-length allocation;
3. make full-rowgroup, selected-column, selected-vector, mixed scatter and
   asynchronous-prefetch paths share that behavior while preserving request
   order.  Empty records must be excluded from physical scatter runs because
   adjacent legal zero-size records share the same file offset;
4. report zero storage bytes, zero physical-page bytes and zero preads for these
   records.  They still count as logical rowgroups/vectors and must still run
   the metadata-driven constant/alias materialization required for output;
5. reject malformed zero-payload descriptors early: nonzero coefficient
   ranges, payload-dependent operators/segments, invalid alias dependencies,
   zero logical row count, bad bounds, or CRC/descriptor inconsistencies;
6. keep v2 and nonzero-v3 paths unchanged.  The writer, C++ verifier and
   `compact_v3_acceptance.py` must agree on the same invariant.

The implemented native gates cover exact nonzero/signed constants,
constant-plus-alias dependencies, malformed payload-dependent descriptors,
single and mixed scatter reads, selected-column and selected-vector reads,
native prefetch, and the six metadata-only rowgroups for canary images 6, 106,
286 and 917.  The read timings assert zero storage bytes, zero physical-page
bytes and zero preads; the prefetch test asserts no pinned backing or pinned
acquire time.  The independent acceptance parser reports six zero-payload
rowgroups and accepts an all-metadata shard without division by zero.

Run these CPU/native regression gates against the current build:

```bash
./build/galp/tests/galp_compact_format_tests \
  --gtest_filter='CompactDescriptorV3.*ZeroPayload*'

./build/galp/tests/galp_tests \
  --gtest_filter='Reader.CompactV3MetadataOnlyRowgroupsUseValidatedZeroIoPaths:Reader.CompactV3FullRowgroupsUseScatterReadsAndPreserveCallerOrder:Reader.CompactV3CoefficientPrefixReadsExactPhysicalRanges'

GALP_COMPACT_V3_ZERO_PAYLOAD_TEST_FILE=/tmp/galp-training-v3-strict-1k-20260731/dct/generation_1785499577816995920_0/shard_000000.fls \
./build/galp/tests/galp_tests \
  --gtest_filter=Reader.CompactV3Strict1kMetadataOnlyIntegration

PYTHONPATH=galp/benchmarks/system_rgbnomore \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python -m unittest -v \
  galp.tests.test_compact_v3_acceptance \
  galp.tests.test_training_benchmark
```

ASan/UBSan and a multi-repeat nonzero-payload performance comparison remain
release-hardening work; they must not be inferred from the functional CPU tests
or this single-repeat smoke.  The required fresh fixed-seed 1K GPU 3+10 smoke
with the repaired crop mapping has completed and is recorded below.

Deliberately unacceptable workarounds include emitting a dummy payload byte,
filling the output with zeros, dropping the rowgroup/sample, catching the
exception and switching to JPEG/v2, or teaching training preflight to reject or
parse zero-payload rowgroups.

## v3 1K four-pipeline GPU smoke

Run this only after the preceding data assertions pass.  The first invocation
explicitly creates the payload-fingerprint cache; later measured runs omit
`--refresh-galp-payload-fingerprints` and reuse it.

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
export PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
export C1=/tmp/galp-training-v3-strict-1k-20260731
export BIND="$PWD/build/galp/torch"
export OUT=/tmp/galp-training-v3-1k-repro-20260801-r2

test ! -e "$OUT"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$BIND" "$PY" \
  galp/benchmarks/system_rgbnomore/training/run.py \
  --enabled-pipelines galp,rgbnomore,dali,pytorch \
  --required-comparison-groups dct,rgb \
  --phase smoke --execution-mode runtime \
  --train-manifest "$C1/training_manifests/train.json" \
  --val-manifest "$C1/training_manifests/val.json" \
  --galp-manifest "$C1/dct/manifest.bin" \
  --galp-torch-module-path "$BIND" \
  --expected-manifest-version 3 \
  --expected-physical-layout image-major-vector-rowgroups \
  --expected-image-count 1000 \
  --device cuda:0 --batch-size 64 --workers 4 --prefetch-depth 2 \
  --seed 11997733 --warmup-steps 3 --measured-steps 10 \
  --dct-semantic-atol 0.0009804921568627452 \
  --refresh-galp-payload-fingerprints \
  --output-dir "$OUT"

"$PY" galp/benchmarks/system_rgbnomore/training/validate.py "$OUT" --no-write
```

The runner executes the four pipelines sequentially on the selected GPU.  This
is the correct four-pipeline comparison command; it is not four-process DDP.
For distributed sampling tests, launch one isolated output directory per rank
with the same seed and `--distributed-world-size N --distributed-rank R`.

### Current-code 1K GPU result

The current binding and Python source completed this command on 2026-08-01.
The immutable run is stored at
`/tmp/galp-training-v3-1k-semantic-rerun-20260801`; its binding SHA256 is
`70802bfba432bc9291ef67e1c198e4e1846b01c731b8de690dfc44ed442f587c`.
The run was operator-interrupted while RGBNoMore was computing, then continued
with `--resume-run` from the same contract.  Resume reused the already-complete
GALP artifact and re-ran the incomplete pipeline from its frozen initial state;
all live runtime-file hashes still match the contract.

The independent validator reports `ok: true`, no failures, and valid artifact
hashes.  Every pipeline completed three warmup and ten measured optimizer
steps, covering 640 measured samples:

| Pipeline | images/s | Correctness | Same-domain semantic status |
| --- | ---: | --- | --- |
| GALP | 306.50 | passed | passed |
| RGBNoMore | 0.968 | passed | passed |
| DALI | 1156.95 | passed | warning |
| PyTorch | 619.46 | passed | warning |

The DCT pair has identical sample IDs, labels and augmentation descriptors;
first-step loss difference is zero, maximum input drift is
`0.000980451237410307`, gradient cosine is `0.9999999999999147`, and first
update cosine is `0.9999999863699378`.  The RGB pair has no failure: its maximum
decoder/interpolation drift is `0.21960780024528503`, above the `0.10` warning
threshold but below the `0.50` failure threshold; top-1 agreement is 1.0 and
gradient cosine is `0.9999998072088586`.

These one-repeat throughputs are smoke diagnostics, not stable performance
rankings.  In particular, RGBNoMore was CPU preprocessing-bound on this host;
the slow baseline does not enter the DCT semantic pass/fail decision.

## v3 10K command (do not run before 1K passes)

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
export PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
export C10="$PWD/galp/data/system_rgbnomore/e2e_v3/galp-v3-training-canary-strict/train-10k-v3c"
export OUT=/tmp/galp-training-v3-10k-four-pipeline-20260801

test ! -e "$OUT"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$C10/frozen/torch" "$PY" \
  galp/benchmarks/system_rgbnomore/training/run.py \
  --enabled-pipelines galp,rgbnomore,dali,pytorch \
  --required-comparison-groups dct,rgb \
  --phase smoke --execution-mode runtime \
  --train-manifest "$C10/training_manifests/train.json" \
  --val-manifest "$C10/training_manifests/val.json" \
  --galp-manifest "$C10/dct/manifest.bin" \
  --galp-torch-module-path "$C10/frozen/torch" \
  --expected-manifest-version 3 \
  --expected-physical-layout image-major-vector-rowgroups \
  --expected-image-count 10000 \
  --device cuda:0 --batch-size 64 --workers 4 --prefetch-depth 2 \
  --seed 11997733 --warmup-steps 10 --measured-steps 100 \
  --dct-semantic-atol 0.0009804921568627452 \
  --refresh-galp-payload-fingerprints \
  --output-dir "$OUT"

"$PY" galp/benchmarks/system_rgbnomore/training/validate.py "$OUT" --no-write
```

## v2 regression command

Use exactly the same runner and adapter.  Only the manifest expectation and
dataset paths differ:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_rgbnomore/training/run.py \
  --pipeline galp --phase smoke --execution-mode runtime \
  --train-manifest "$V2_TRAIN_JSON" --val-manifest "$V2_VAL_JSON" \
  --galp-manifest "$V2_MANIFEST" \
  --expected-manifest-version 2 \
  --expected-physical-layout image-major \
  --expected-image-count "$V2_IMAGE_COUNT" \
  --device cuda:0 --batch-size 64 --workers 4 --prefetch-depth 2 \
  --seed 11997733 --warmup-steps 3 --measured-steps 10 \
  --dct-semantic-atol 0.0009804921568627452 \
  --output-dir /tmp/galp-training-v2-regression-20260801
```

The repository's real 50,000-image v2 manifest and the 1,000-image v3 canary
were both opened through the current `DirectDctTrainingReader` constructor.
A real v2 preflight/contract dry-run is preserved at
`/tmp/galp-training-v2-real-contract-audit-20260801`: it records version 2,
`image-major`, 50,000 images, 14 manifest-declared FLS/metadata payloads and
seven explicitly identified legacy-v2 `.svb` reader dependencies.  V3 does
not infer undeclared `.svb` files; future v3 auxiliary inputs must be declared
by its manifest contract.

## Block-major hold point

Do not pass a block-major/spatial-major manifest to the training runner.  Its
preflight deliberately accepts only v2 and v3, and adding a layout branch in
the train loop would defeat the public-reader compatibility boundary.

Inspect the current isolated block-major suite without starting GPU work:

```bash
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run_suite.py \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-suite-dryrun \
  --dry-run
```

Training adoption becomes safe when block-major is exposed through the same
stable high-level Pipeline/Iterator semantic batch surface and
passes identity/order, transform provenance, missing/new stats, 3+10 GPU smoke,
and checkpoint-resume gates without a train-loop layout conditional.

## Change and overlap audit

The training-owned implementation is confined to
`benchmarks/system_rgbnomore/training`, its benchmark tests, canary scripts and
these documents.  The principal files are `manifest_preflight.py`,
`direct_dct_reader.py`, `generate_imagenet_manifests.py`, `sample_order.py`,
`pipeline.py`, `run.py`, `validate.py`, `select_imagenet_canary.py` and
`prepare_v3_training_canary_strict.sh`.

The later explicit request to repair legal zero-payload rowgroups expanded the
original no-core-edit boundary.  That native repair necessarily has file-level
overlap risk with the parallel v3 format/reader session in
`format/compact_descriptor_v3.cpp`, `format/reader.cu`,
`engine/pipeline/rowgroup_prefetch_queue.cuh`,
`engine/table/pipeline_state.cu`, `tests/reader_test.cu`,
`tests/compact_descriptor_v3_test.cpp` and `tests/CMakeLists.txt`.  It does not
add format knowledge to the training Python layer.

No block-major-specific access, plan, tool or `system_dct_major` file was
modified for the training implementation.  `reader.cu` and the shared CMake
targets remain integration surfaces, so merges with any session changing those
files require hunk-level review.  Because the worktree contains uncommitted
changes from several external sessions, git status alone cannot prove
per-session ownership; this is a file-overlap assessment, not an authorship
claim.

## Stable result fields

Every smoke/step repeat has a layout-independent `common_statistics` object:

```json
{
  "samples": 640,
  "batches": 10,
  "elapsed_seconds": 1.0,
  "samples_per_second": 640.0,
  "reader_wait_seconds": 0.1,
  "training_step_seconds": 0.8,
  "peak_host_queue_depth": 3,
  "peak_device_memory": 123456789,
  "loss_summary": {}
}
```

GALP may additionally emit `native_execution_stats`.  That namespace is
explicitly optional and has `correctness_dependency: false`; missing counters
or counters added by newer native binaries cannot fail correctness.
