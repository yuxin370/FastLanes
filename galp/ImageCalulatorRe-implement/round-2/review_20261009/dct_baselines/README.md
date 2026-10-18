# IC / Frequency-Store DCT baselines

Implementation of the format/reference portion of
[the supplied plan](../../ImageCalulatorRe-implement/fastlens_agent_pack/IMPLEMENTATION_PLAN.md).
Matched methods are **paper-guided reimplementations with a JPEG-DCT adapter**.
Native mechanisms use separate APIs and storage identities.

Run from the repository root with NumPy, Pillow, PyTorch and the existing
`dct_manip` coefficient extension. Model adapters additionally require the
existing RGB-no-more / DCTNet / FUN source trees and checkpoints. `report`
requires `jsonschema`. No downloads or native rebuild are needed for the
format implementation. The FastLens integer-source gate additionally uses
the existing `galp_dct_storage` target; rebuild it after this patch:
`cmake --build build --target galp_dct_storage -j2`.

```bash
export PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
"$PYTHON" -m unittest discover -s galp/benchmarks/dct_baselines/tests -t .
"$PYTHON" -m galp.benchmarks.dct_baselines --help
```

## Implemented scope

| Method | Physical organization |
| --- | --- |
| `ic_matched_shared` | One image per NPZ, int32 q and uint16 Q |
| `ic_matched_fixedk` | Same, retaining explicit profile source frequencies at every spatial position |
| `ic_matched_batched` | Default 256 images, one concatenated image-major q member plus Q |
| `fs_matched_b8` | Default group256, 15 independent diagonal NPZs, range-checked int8/int16, original JPEG q/Q |
| `fs_native` (Python API) | B32/Q50, 63 diagonals; independent native contract |

`dtype_policy=int16` is the common-width control. IC `safe` narrows over the
whole image/batch; FS `paper` and `safe` choose width over each complete
diagonal. Overflow raises an error. Store metadata records the policy and
each member's actual dtype. A failed build can leave a partial directory;
it has no usable `store.json` and is not silently resumed.

`StoreReader.plan/read` accepts component-specific natural frequency indices
and rectangular block regions. It reads and inflates whole image/batch or
group/diagonal units, gathers afterwards, and restores request order,
including duplicates. Decoded caching has an explicit byte budget (default
zero); arrays are shared within one preparation pool. Parallel NPZ decode
and CPU transforms use the configured worker threads. One next pool may
overlap model execution. CUDA submission is bounded to two preparation
pools, with completion waits at pool boundaries. There are no per-coefficient
Python decode loops.

Storage supports grayscale and ragged component grids. The seven model
adapters (`mobilenet24`, `mobilenet32`, `resnet24`, `resnet64`, `efun`,
`vitti`, `swinv2`) consume the existing source512 4:2:0 geometry. CNN/eFUN
outputs remain CHW; Transformer outputs remain separate Y and Cb/Cr grids.
Every resize profile explicitly requires all 64 source frequencies, so its
FixedK store currently retains all64. Output channel count never determines
source K.

## Freeze, build, verify and replay

The supplied draft YAML is a specification, not a runnable config. `prepare`
resolves a compact versioned `matched.json`, `source.json`, `profile.json`
and occurrence-level `trace.jsonl` from actual repository inputs. It hashes
source JPEGs and q, records component IDs/sampling/valid geometry/table IDs,
and freezes checkpoint, numerical operation order, output tensors and label
mapping. Existing Q tables are extracted directly through libjpeg. There
is no RGB round trip, Q50 requantization or baked online crop in a matched
store.

Provide `source-recipe.json` recording the actual common source preprocessing
(resize, encoder/version/settings, quality, subsampling and provenance).
Do not infer that recipe from JPEG q/Q or substitute the model's output
geometry. The local validation source's complete historical production
recipe still needs provenance confirmation before formal experiments.

```bash
run="$HOME/tmp/ic-fs-resnet24"
"$PYTHON" -m galp.benchmarks.dct_baselines prepare \
  --selection galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json \
  --source-recipe /path/to/source-recipe.json \
  --model resnet24 --upstream "$HOME/RGB-no-more" \
  --count 16 --batch-size 4 --pool-size 16 --workers 8 --torch-threads 1 \
  --seed 11997733 --output "$run/contract"

for method in ic_matched_shared ic_matched_fixedk ic_matched_batched fs_matched_b8; do
  "$PYTHON" -m galp.benchmarks.dct_baselines build \
    --config "$run/contract/matched.json" --method "$method" --output "$run/$method"
  "$PYTHON" -m galp.benchmarks.dct_baselines verify \
    --config "$run/contract/matched.json" --store "$run/$method" \
    --device cpu --check-model --training-steps 10 --output "$run/$method.verify.json"
  "$PYTHON" -m galp.benchmarks.dct_baselines bench \
    --config "$run/contract/matched.json" --store "$run/$method" \
    --evidence "$run/$method.verify.json" --device cpu --run-kind inference \
    --repeats 3 --output "$run/$method.results.jsonl"
done
"$PYTHON" -m galp.benchmarks.dct_baselines report \
  "$run"/*.results.jsonl --output "$run/summary.json"
```

Use a fresh directory for each `prepare/build`; results files also refuse
overwrite. Omit `--count` to import the complete selection. Batch/pool/worker
settings are caller choices, not claims about the papers' model batches.
Run different DCTNet model profiles in separate processes, as required by
the existing upstream imports. To check CUDA after driver recovery, replace
`--device cpu` with `--device cuda` in both `verify` and `bench` and use new
evidence/result paths. No GPU timing was validated in this implementation
session.

`verify` checks all stored q and Q against fresh independent JPEG extraction,
then all trace tensors against the existing CPU validation reference; it
optionally compares checkpoint logits and complete backward/SGD updates.
Tolerance is frozen during `prepare` (exact by default). Nonzero tolerances
require independent calibration; they are never relaxed on failure.
The CLI's SGD update check is a semantic smoke test, not the published
optimizer/scheduler recipe. Failed verification emits no passing evidence.
The benchmark rejects stale contracts, stores, implementation hashes,
devices or missing inference-logit verification.

The inference checkpoint command retains the existing validation crop.
The training commands below consume explicit crop/flip, RandAugment,
Mixup partners/weights and soft targets. Both retain all source frequencies
needed by the spectral resize; neither substitutes output channel count for K.

## Common training trace and numerical gates

`prepare --training` writes separate calibration and verification traces
with disjoint image identities. It freezes actual crop/flip and both
RandAugment operations, including component-specific Cutout centers/pads,
translation distances and ChromaDrop channel. Mixup stores both FP32 beta
weights and the partner occurrence within the actual microbatch, including
one-image tails. Repeated images remain distinct occurrences. No augmentation
RNG is sampled during replay.

`UnifiedReplay` connects `cpu_dct`, `fastlens`, `ic` and `fs` to those same
rows. CPU-DCT extracts q with JPEG2DCT and applies literal upstream geometry;
IC/FS use the original-q store adapter; FastLens uses the public native
pipeline with explicit crop/flip descriptors. CNN paths retain the complete
192-channel intermediate for augmentation before model projection. CNN
spatial augmentation scales the canonical 28-block parameters to its actual
grid; Transformer chroma uses half the luma grid. Training Swin uses the
existing **224px training architecture**, rather than the 256px inference
checkpoint architecture.

All paths share frozen augmentation and target decisions. The independent
oracle uses upstream scalar operations; the replay uses existing grouped
operations. The existing AdamW, independent decay, warmup/cosine update
scheduler, gradient clipping and sample-weighted accumulation are reused.
Accumulation closes at preparation-pool and epoch boundaries without dropping
tails. ViT uses FP32; CNN/eFUN/Swin use the existing BF16 autocast recipe.
These correctness runs use eager model execution, not performance timings.

`snapshot` records the from-scratch initial model, AdamW state, decay state,
scheduler and Python/NumPy/Torch/CUDA RNG. Its hash is shared by all methods.
The profile's pretrained checkpoint remains an inference-reference identity;
it does not replace this complete training snapshot.

Example from the repository root (supply the actual source production recipe):

```bash
run="$HOME/tmp/ic-fs-training-resnet24"
"$PYTHON" -m galp.benchmarks.dct_baselines prepare \
  --selection galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json \
  --source-recipe /path/to/source-recipe.json \
  --model resnet24 --upstream "$HOME/RGB-no-more" \
  --count 17 --batch-size 64 --pool-size 4096 --workers 8 --torch-threads 1 \
  --seed 11997733 --training --epochs 10 --calibration-count 4 \
  --output "$run/contract"
"$PYTHON" -m galp.benchmarks.dct_baselines snapshot \
  --config "$run/contract/matched.json" --output "$run/initial.pt"
"$PYTHON" -m galp.benchmarks.dct_baselines build \
  --config "$run/contract/matched.json" --method fs_matched_b8 --output "$run/fs"

"$PYTHON" -m galp.benchmarks.dct_baselines calibrate \
  --config "$run/contract/matched.json" --snapshot "$run/initial.pt" \
  --backend fs --root "$run/fs" --device cpu --updates 10 \
  --input-atol 0 --model-atol 0 --state-atol 0 --rtol 0 \
  --output "$run/fs.cpu.calibration.json"
"$PYTHON" -m galp.benchmarks.dct_baselines verify-training \
  --config "$run/contract/matched.json" --snapshot "$run/initial.pt" \
  --backend fs --root "$run/fs" --device cpu --updates 10 \
  --calibration "$run/fs.cpu.calibration.json" --output "$run/fs.cpu.verify.json"
```

Repeat with `--backend ic --root /path/to/ic-store`, or `--backend cpu_dct`
without a root. Each backend has its own calibration artifact. For FastLens
the source selection must contain the actual `galp_image_id` mapping. The
integer gate uses `read-source-manifest`, which reads the canonical block-major
source through the existing CPU dataset reader and restores natural frequency
and raster order without invoking the replay planner.

On the user's H100, set
`CUDA_VISIBLE_DEVICES=GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76` and use
`--device cuda:0`. FastLens also needs
`--backend fastlens --root galp/data/compressed/imagenet512_val_block_major/manifest.bin
--module-path build/galp/torch`.
Calibrate on that device before verifying; CPU calibration cannot authorize
GPU verification. Nonzero input/model/state limits must be explicitly chosen
and supported by the independent golden measurements. Spectral resize can
straddle integer rounding ties, so CNN tolerances must account for the actual
channel normalization. Limits are never increased after a verification failure.

G1 checks selected q/Q exactly against fresh JPEG extraction. G2/G3 compare
final inputs, exact soft targets, logits, loss, accumulated gradients, model
parameters/buffers and AdamW moment tensors after each update. Both executions
restore the same complete snapshot and RNG; decay, scheduler and resulting RNG
states must also match. Calibration and verification reject changed source,
profile, trace, checkpoint, implementation, snapshot, native binding or device
identities. Passing evidence records observed stage errors and the frozen
limits; it remains `headline_eligible=false`.

## Existing experiment integration

This package owns storage and numerical gates. Formal training uses the existing
experiment runners, models, schedules, checkpoints and result schemas; it does
not introduce a separate training suite. Both adapters replace coefficient
acquisition and preserve their established JPEG preprocessing and labels.

CNN/eFUN: `dct_models.training_pls --input-backend ic|fs --baseline-store DIR`
uses the existing JPEG training adapter and shared native GPU augmentation.
The store must cover the complete premixed mapping with identical logical IDs,
labels and source paths, retaining all64 frequencies. Native B6 versus default
IC/FS A0 retains the existing crop/order comparison; select the same condition
when making a strictly matched storage comparison. Store variant, source hash,
GPU identity and optional `--baseline-evidence` are saved in `training.json`.

```bash
export CUDA_VISIBLE_DEVICES=GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76
export CNN_IC_STORE=/path/to/full-train-ic-store
export CNN_FS_STORE=/path/to/full-train-fs-store
bash galp/benchmarks/dct_models/cnn_training_baselines.sh plan ic-fs-training \
  'mobilenet24:ic mobilenet24:fs resnet24:ic resnet24:fs efun:ic efun:fs'
# Change plan to probe/full/data/profile to use the established run modes.
"$PYTHON" -m galp.benchmarks.dct_models.report_cnn_suite \
  --output-dir "$HOME/tmp/cnn-with-ic-fs" \
  --additional-training /path/to/ic-run /path/to/fs-run
```

ViT-Ti/Swin: keep the existing `training_pls.run_matrix` arguments and change
`--execution-backend` to `ic-matched-dct` or `fs-matched-dct`, adding
`--baseline-store DIR`. Keep the existing train/val manifests, layout plan,
premixed mapping/hash, registered model, seed, condition and precision. The
original immutable run manifest now records the store identity. Epoch metrics,
validation, optimizer/accumulation and checkpoint/resume remain in the existing
runner. `report_equal_image_performance --dct-reference-run DIR` accepts a
JPEG/IC/FS reference and emits the same CSV/JSON, using the concrete store
variant as the row label. `report_convergence_reference --reference-run DIR`
also accepts these references. Existing paired source/model/recipe/GPU checks
still apply; archived runs with different identities cannot establish a matched
speedup merely by appending rows.

The initial formal adapters read whole NPZ members per sample without decoded
caching. Batched IC and FS therefore repeat group reads. They are diagnostic
integrations, not optimized throughput results, and stay
`headline_eligible=false`; the pooled replay's counters are not measurements of
these DataLoader adapters. GPU calibration, measurement validation and complete
epoch measurements remain required before using them in a speed headline.

## Native sanity API

```python
import numpy as np
from galp.benchmarks.dct_baselines.native import (
    NativeSpec, encode_ic, fs_source, mosaic, write_ic_archive, read_ic_archive,
)
from galp.benchmarks.dct_baselines.store import build_store, StoreReader
from galp.benchmarks.dct_baselines.contracts import ReadRequest

spec = NativeSpec(block_size=32, k=3, scalar_quantization=50)
expected = encode_ic(rgb_uint8_256x256, spec)  # int32 [3,24,24]
write_ic_archive("image.npz", [rgb_uint8_256x256], spec)
write_ic_archive("batch.npz", rgb_images, spec)  # explicit IC-b concatenation
build_store((fs_source(rgb, i) for i, rgb in enumerate(rgb_images)),
            "fs_native_store", "fs_native")
with StoreReader("fs_native_store") as reader:
    requests = [ReadRequest("0", (tuple(range(1024)),) * 3)]
    views, counters = reader.read(reader.plan(requests))
    actual = mosaic(np.stack(views[0].dense()), 3)
```

All native choices are explicit in `NativeSpec.contract()`. `candidates()`
enumerates 6048 candidates; it does not train or implement the self-design
optimizer. `fit_statistics(..., split="train")` computes per-plane population
statistics including structural zeros, and rejects validation/test data
and zero variance. Save the statistics as a shared hashed artifact before
using `normalize_mosaic`. Native inference never multiplies scalar Q back.

## Measurement and current limits

`bench` times physical planning, complete reads/inflate/gather, CPU transform,
batch organization, pinned H2D, optional inference, metric collection and
last CUDA completion. Initialization is outside the window and its manifest
read bytes are separately reported. No first measured pool is prepared
before the timer. No excluded warmup mutates model state. H2D includes the
final float32 DCT tensors and labels; this is explicitly a CPU transform
adapter. Input-only replay uses the same full path through tensor readiness.

Each raw row follows the supplied `contracts/result.schema.json`; `report`
validates it and groups only matching contracts/runtime settings. Adjacent
counter JSONs record logical requests, per-pool deduplicated q demand,
actual decoded q/bytes, file/read counts, decoded-cache hits/evictions and
decode amplification. Complete archive bytes include ZIP/NPY headers and
metadata. Physical disk bytes are **null**, not inferred from application
reads. Logical and allocated storage bytes include labels, tables, geometry
and manifests. Conversion includes JPEG coefficient extraction and store
encoding, excluding creation of the common resized/reencoded JPEG source.
Separate FixedK stores are separate copies and must be summed in a family
storage comparison.

All current timings are diagnostic, `headline_eligible=false`; OS cache
warmth is unverified. The common training replay and published-recipe numerical
gates are implemented. In this session MobileNet24, ViT-Ti and ResNet24 each
passed ten paired CPU updates on existing synthetic JPEGs, with zero observed
logit/loss/gradient/parameter/AdamW differences; all IC variants, FS and CPU-DCT
matched those source tensors and targets. Three real ImageNet validation images
also passed FastLens integer q/Q/geometry checks. These are bounded correctness
checks, not full ImageNet training. CUDA replay and its numerical calibration
remain unexecuted in the sandbox even though the user's host CUDA works.
Full temporary-file tests also require the host's writable `$HOME/tmp`.
Equal-budget multi-consumer runs, cache/resource calibration, complete
50K/epoch experiments and the second GPU remain unperformed. This implementation
does not mark the CUDA G3 gate, G4/G5 or the entire research plan complete.
See [integration map and assumptions](ASSUMPTIONS.md).
