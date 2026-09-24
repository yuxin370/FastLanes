# SwinV2-T DCT-native full training and baseline runbook

This runbook defines the active B6 evidence layers for SwinV2-T:

1. One native physical B6 convergence prefix, retained through epoch 75.
2. A paired standard RGB-no-more JPEG-to-DCT reference through epoch 15.
3. The equal-image two-epoch runtime matrix: Native physical B6, standard
   RGB-no-more JPEG-to-DCT, DALI D2, DALI D3, and PyTorch. DALI D3 is reported
   as a native-augmentation performance ceiling, so the headline comparison
   remains the four systems GALP, RGB-no-more, DALI, and PyTorch.

`A0`, `A1`, and `B2` are historical ViT-Ti factorial conditions. They are
deliberately excluded from this SwinV2 runbook and rejected by the SwinV2 model
registration.

The registered training model is `rgbnomore-swinv2-t-dct-224-v1`: 224 input,
window 7, DCT-native grouped/sub-block YCbCr stem, and BF16 autocast. This is
the apples-to-apples model for the existing 224-crop PLS layout. It is not the
official 256/window-8 inference checkpoint and makes no official pretrained
accuracy claim.

## Shared environment

```bash
cd /home/tangyuxin/gfastlanes/FastLanes

PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
TRAIN=$PWD/galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/train.json
VAL=$PWD/galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json
LAYOUT="$HOME/tmp/pls-layout-current/physical_layout_plan.json"
PHYSICAL="$PWD/galp/data/compressed/imagenet512_train_block_major_premixed/dct/manifest.bin"
MAPPING="$PWD/galp/data/compressed/imagenet512_train_block_major_premixed/ordered_mapping.csv"
MAPPING_SHA=98f77515e5886c098e46c23cddb41f57098b406ab32790509254c56356dc24ff
SUITE=/mnt/nvme2/home/tangyuxin/pls-experiments/swinv2-training-suite-4090-20260909-v2
B6_PREFIX=$SUITE/scientific_matrix/runs/B6/seed_11997733
REFERENCE_ROOT=/mnt/nvme2/home/tangyuxin/pls-experiments/swinv2-standard-dct-reference-e50-premixed-v2-4090-20260911
PERF_SUITE="$HOME/tmp/swinv2-training-performance-e2-current"
PERF_RGBNOMORE=$PERF_SUITE/rgbnomore_dct
MODEL=rgbnomore-swinv2-t-dct-224-v1

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/build/galp/torch${PYTHONPATH:+:$PYTHONPATH}"
```

Use a new output root after any training-runtime source edit. Contracts are
immutable and intentionally reject silent source changes.

Reserve the RTX 4090 for each timed run. Check
`nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv` before and during training, and
discard any epoch that overlaps another process on that GPU. Use the second
epoch for the warm-throughput comparison; the first includes cold-start work.

Generate the layout from the current training manifest before a new run. The
pre-migration layout has a different manifest hash even though its sample
mapping is unchanged:

```bash
"$PY" -m galp.benchmarks.training_pls.plan_layout \
  --train-manifest "$TRAIN" \
  --output-dir "$(dirname "$LAYOUT")" \
  --segment-images 1024 --organization-seed 20260810
```

## Completed paired convergence check through epoch 15

The single-seed convergence objective is already complete. At epoch 15, Native
B6 reaches 59.816% Top-1 versus 59.490% for the paired standard DCT reference;
the normalized Top-1 AUC difference is +0.3645 percentage points. This is enough
to show no observed convergence-speed regression for the requested diagnostic.
Do not resume either run merely to extend it to epoch 50.

The paired reference uses the mandatory premixed mapping to select the source
JPEG corresponding to every native physical position. It decodes those JPEGs
with RGB-no-more `dct_manip`, then applies the same DCT RandAugment, Mixup,
optimizer, scheduler, and sample order. Validation is the shared GALP
Direct-DCT path in both runs.

The older `swinv2-standard-dct-reference-e50-4090-20260910` directory used the
pre-premix manifest order and is invalid as a paired B6 reference. Preserve it
for diagnosis, but never resume it and never use it in the convergence report.

The checked report is stored under
`$SUITE/reports/convergence_reference_e15`. To reproduce the report without
training, run:

```bash
"$PY" -m galp.benchmarks.training_pls.report_convergence_reference \
  --native-run "$B6_PREFIX" \
  --reference-run "$REFERENCE_ROOT/runs/B6/seed_11997733" \
  --max-epoch 15 \
  --output-dir /path/to/new/convergence_reference_e15_report
```

The report emits the paired curves, endpoint deltas, and normalized Top-1 AUC.
It applies no ViT-derived threshold and performs no strategy selection. This is
a one-seed convergence-speed diagnostic, not an across-seed equivalence or
final-accuracy claim. It also records a source-identity caveat because the
reference backend was added after the native prefix was produced.

## Equal-image runtime baselines

Create a fresh Native B6 two-epoch observation. The recipe remains a
300-epoch recipe; `--stop-after-epoch 2` only pauses the run at a checkpoint:

```bash
"$PY" -m galp.benchmarks.training_pls.run_matrix \
  --train-manifest "$TRAIN" \
  --val-manifest "$VAL" \
  --layout-plan "$LAYOUT" \
  --conditions B6 \
  --seeds 11997733 \
  --model "$MODEL" \
  --epochs 300 \
  --stop-after-epoch 2 \
  --execution-backend native-physical-pls \
  --physical-galp-manifest "$PHYSICAL" \
  --premixed-mapping-csv "$MAPPING" \
  --expected-mapping-sha256 "$MAPPING_SHA" \
  --output-dir "$PERF_SUITE/galp_b6" \
  --device cuda:0 \
  --required-gpu-name-substring "RTX 4090" \
  --execute
```

Run the recipe-matched RGB-no-more JPEG-to-DCT baseline for the same two full
epochs. This is a fresh performance artifact; do not reuse the convergence
prefix because the unified report requires identical current source hashes:

```bash
"$PY" -m galp.benchmarks.training_pls.run_matrix \
  --train-manifest "$TRAIN" \
  --val-manifest "$VAL" \
  --layout-plan "$LAYOUT" \
  --conditions B6 \
  --seeds 11997733 \
  --model "$MODEL" \
  --epochs 300 \
  --stop-after-epoch 2 \
  --execution-backend standard-rgbnomore-dct \
  --premixed-mapping-csv "$MAPPING" \
  --expected-mapping-sha256 "$MAPPING_SHA" \
  --output-dir "$PERF_RGBNOMORE" \
  --device cuda:0 \
  --required-gpu-name-substring "RTX 4090" \
  --execute
```

Run DALI D2, DALI D3, and PyTorch in separate processes with the matching
SwinV2 RGB model, BF16 precision, seed, microbatch 64, accumulation 16, and
full-image tail:

```bash
for PIPELINE in d2 d3 pytorch; do
  "$PY" -m galp.benchmarks.system_rgbnomore.training.equal_image_epoch_benchmark \
    --train-manifest "$TRAIN" \
    --val-manifest "$VAL" \
    --output-dir "$PERF_SUITE/$PIPELINE" \
    --pipelines "$PIPELINE" \
    --model "$MODEL" \
    --device cuda:0 \
    --audit-mode runtime-first-100 \
    --required-gpu-name-substring "RTX 4090" \
    --execute
done
```

Generate the fail-closed unified comparison and breakdown:

```bash
"$PY" -m galp.benchmarks.training_pls.report_equal_image_performance \
  --d2-root "$PERF_SUITE/d2" \
  --d3-root "$PERF_SUITE/d3" \
  --pytorch-root "$PERF_SUITE/pytorch" \
  --native-run "$PERF_SUITE/galp_b6/runs/B6/seed_11997733" \
  --rgbnomore-run "$PERF_RGBNOMORE/runs/B6/seed_11997733" \
  --output-dir "$PERF_SUITE/report" \
  --requested-claim system-level
```

The JSON, CSV, and Markdown outputs include the direct recipe-matched GALP
versus RGB-no-more DCT comparison, the DALI D2 versus PyTorch RGB comparison,
the DALI D3 ceiling, warm epoch throughput ratios, model-only calibration,
preparation/exposed-wait/audit/sync fractions, DALI handoff and loader stages,
Native selected-vector fraction, compressed-read fraction, and the exact
fairness matrix. Stage times that may overlap are explicitly non-additive.

## ViT-Ti comparison boundary

For a B6-only model comparison, use the identical commands with
`MODEL=rgbnomore-vitti-dct-224-v1`, explicitly keep `--conditions B6`, and use
a separate `SUITE` root. Existing ViT-Ti artifacts may be retained as
historical evidence, but a strict model-to-model comparison requires matching
GPU UUID, audit policy, source hashes, seed, image count, batch/accumulation,
and each model's registered precision.
The current `pls-core-v2-20260811` directory has no epoch-300
`final_result.json`; status labels saying `running` do not prove a live process.
