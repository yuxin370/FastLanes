# PLS crop/shuffle core model experiment

This package implements the pre-registered 2×2 ImageNet model-effect experiment:

| condition | crop | sample order |
|---|---|---|
| A0 | per sample | global shuffle |
| A1 | per virtual PLS | the exact A0 global order |
| B2 | per sample | closed pool, `G=1024`, `M=4` |
| B6 | per virtual PLS | the exact B2 closed-pool order |

The supplemental `N6` control keeps the B6 crop policy and `M=4` resource
boundary but disables all epoch-time sample-order randomization:

| condition | crop | sample order |
|---|---|---|
| N6 | per virtual PLS | frozen premixed physical order; no PLS or pool shuffle |
| N2 | per sample | frozen premixed physical order; no PLS or pool shuffle |

`N2` and `N6` are supplemental controls, not additional cells in the original
registered 2x2. Compare `B6-N6` to measure closed-pool
shuffle versus no epoch shuffle under per-PLS crop, and `A1-N6` to compare
global shuffle with no epoch shuffle. Use `B2-N2` and `A0-N2` for the matching
per-sample-crop effects. The on-disk dataset was premixed once at
organization time, so this tests whether one frozen premix is sufficient; it
does not represent class-sorted raw ImageNet order. Crop, flip, RandAugment and
Mixup RNG remain epoch-aware. Fixed microbatch membership (and therefore
repeated Mixup partners) is an intentional consequence of disabling shuffle.

All four conditions use the same GALP Direct-DCT backend. The default formal
matrix uses the immutable `rgbnomore-vitti-dct-published-v1` recipe: ViT-Ti
DCT, 300 epochs, FP32,
microbatch 64, accumulation 16, effective batch 1024, LR 3e-3, 10,000-update
warmup plus epoch-aware cosine, independent RGB-no-more WeightDecay semantics,
gradient clipping 1, DCT Mixup 0.2, and the published two-op/magnitude-three DCT
RandAugment list. Validation uses `ResizedCenterCrop_DCT(32,28)`.
The fixed execution backend groups samples that have the same keyed
RandAugment operation/magnitude and compiles the fixed-shape ViT with
`torch.compile`/Inductor. The scalar RandAugment reference remains available
to the sanity check; grouping changes kernel dispatch only, not decisions or
tensor values.

The runner also has a model-independent DCT-native interface. A registered
model receives exactly two model-facing tensors from either backend:
`Y[N,1,28,28,8,8]` and `CbCr[N,2,14,14,8,8]`. Storage, Block-major B6
scheduling, crop pushdown, augmentation, validation, checkpointing and audit
logic do not import architecture-specific code. `model_registry.py` currently
registers:

| model id | DCT-native stem | execution precision | status |
|---|---|---|---|
| `rgbnomore-vitti-dct-224-v1` | RGB-no-more grouped DCT patch embedding | FP32 | default published ViT-Ti matrix |
| `rgbnomore-swinv2-t-dct-224-v1` | grouped/sub-block YCbCr DCT stem | BF16 autocast | active B6-only convergence and runtime baselines |

The SwinV2-T entry adapts the RGB-no-more DCT architecture to the existing
224-crop B6 contract (`window_size=7`). It is intentionally not compatible
with the project's 256-crop/window-8 checkpoint and therefore makes no
official pretrained-accuracy claim. It is suitable for measured throughput,
memory and from-scratch convergence comparisons under the same B6 data path.
The active SwinV2 B6 convergence suite, equal-image baselines, resume commands,
and breakdown reports are defined in
[`SWINV2_FULL_BASELINE_RUNBOOK.md`](SWINV2_FULL_BASELINE_RUNBOOK.md).

SwinV2 is intentionally registered for `B6` only. `A0`, `A1`, and `B2`
remain readable/executable only for the historical ViT-Ti factorial experiment;
they are not part of the active SwinV2 route.

The pre-registered model-effect matrix defaults to semantic emulation over a
frozen virtual physical mapping. A separate, fail-closed
`native-physical-pls` execution backend consumes the materialized premixed
block-major dataset through the advanced native C++/CUDA
`DirectDctPlsPipeline`; its Torch adapter remains explicitly experimental.
Physical and semantic checkpoints/contracts are intentionally incompatible;
one cannot be silently resumed as the other.

The third backend, `standard-rgbnomore-dct`, is the convergence reference. It
reads source JPEG coefficients with RGB-no-more `dct_manip`, while reusing the
registered model and the exact premixed B6 position-to-source-sample mapping,
plus the published DCT augmentation, optimizer, validation and checkpoint
lifecycle. The premixed mapping path and SHA-256 are mandatory, while a physical
GALP manifest is forbidden. It accepts any compatible registered DCT-native
model and carries no physical-FLS or storage-reduction claim. Use a 300-epoch
recipe with `--stop-after-epoch 50` for the current SwinV2 prefix comparison;
see the SwinV2 runbook for the exact command and paired report.

## Commands

From the repository root:

```bash
export PYTHONPATH="$PWD:$PWD/build/galp/torch${PYTHONPATH:+:$PYTHONPATH}"
export TRAIN_JSON="$PWD/galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/train.json"
export VAL_JSON="$PWD/galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json"
export PHYSICAL="$PWD/galp/data/compressed/imagenet512_train_block_major_premixed/dct/manifest.bin"
export MAPPING="$PWD/galp/data/compressed/imagenet512_train_block_major_premixed/ordered_mapping.csv"
export MAPPING_SHA=98f77515e5886c098e46c23cddb41f57098b406ab32790509254c56356dc24ff
```

Freeze the virtual physical layout:

```bash
python -m galp.benchmarks.training_pls.plan_layout \
  --train-manifest "$TRAIN_JSON" \
  --output-dir "$HOME/tmp/pls-layout-current" \
  --segment-images 1024 \
  --organization-seed 20260810
```

Create and inspect the 16-run plan (this never trains without `--execute`):

```bash
python -m galp.benchmarks.training_pls.run_matrix \
  --output-dir /tmp/pls-core \
  --train-manifest "$TRAIN_JSON" \
  --val-manifest "$VAL_JSON" \
  --layout-plan "$HOME/tmp/pls-layout-current/physical_layout_plan.json" \
  --conditions A0,A1,B2,B6 \
  --seeds 11997733,11997734,11997735,11997736 \
  --epochs 300 \
  --device cuda:0
```

Each planned run has one canonical resolved manifest at
`runs/CONDITION/seed_SEED/run_manifest.json`. It contains the frozen recipe,
condition, data/layout fingerprints, physical execution inputs and scoped
training-source identity. Training and reporting both consume this file;
launch commands use `--run-manifest`.

The manifest retains the full Git commit/status/diff provenance, but only the
explicitly scoped training runtime source tree is a blocking identity. Changes
to unrelated repository files are recorded in `environment.json` and no longer
invalidate a prepared run.

`run_manifest_hash` protects the complete JSON artifact. The separate
`condition_hash` covers only training/checkpoint compatibility fields, replacing
full Git provenance with the scoped runtime-source identity. Regenerating a
manifest after an unrelated repository change therefore preserves checkpoint
resume compatibility while retaining the new provenance record.

For a physical premixed B6 run, add:

```bash
  --conditions B6 \
  --execution-backend native-physical-pls \
  --physical-galp-manifest "$PHYSICAL" \
  --premixed-mapping-csv "$MAPPING" \
  --expected-mapping-sha256 "$MAPPING_SHA"
```

To run the same physical B6 path with the registered DCT-native SwinV2-T,
add only the model selector; its immutable recipe is inferred:

```bash
  --conditions B6 \
  --model rgbnomore-swinv2-t-dct-224-v1 \
  --execution-backend native-physical-pls \
  --physical-galp-manifest "$PHYSICAL" \
  --premixed-mapping-csv "$MAPPING" \
  --expected-mapping-sha256 "$MAPPING_SHA"
```

The resolved run manifest freezes the selected model id, complete input and
architecture configuration, parameter count, recipe hash, and SHA-256 hashes
of the external RGB-no-more model sources. A model/recipe mismatch or a model
source edit fails before training or resume.

For another DCT-native architecture (for example EfficientNet or MobileNetV3),
the extension boundary is one `DctModelSpec` plus its recipe. Its builder must
return `forward(y, cbcr) -> logits`, declare the input contract and expected
parameter count, and list every external source file. No B6/GALP pipeline or
training-loop branch is required.

For the native no-epoch-shuffle control, use the same command and physical
artifacts with `--conditions N6`. Planning remains the default; add `--execute`
only after inspecting the generated manifests and command.

For a fresh two-epoch operational test, keep the scientific 300-epoch recipe
and request an epoch-boundary pause directly from the matrix runner:

```bash
python -m galp.benchmarks.training_pls.run_matrix \
  --output-dir /tmp/native-pls-b6-e2 \
  --train-manifest "$TRAIN_JSON" \
  --val-manifest "$VAL_JSON" \
  --layout-plan "$HOME/tmp/pls-layout-current/physical_layout_plan.json" \
  --conditions B6 \
  --seeds 11997733 \
  --epochs 300 \
  --stop-after-epoch 2 \
  --device cuda:0 \
  --execution-backend native-physical-pls \
  --physical-galp-manifest "$PHYSICAL" \
  --premixed-mapping-csv "$MAPPING" \
  --expected-mapping-sha256 "$MAPPING_SHA" \
  --execute
```

`run_matrix` generates the manifest and immediately launches the child training
process, eliminating the manual plan/contract/train gap. A successful prefix
run ends with `state=paused-at-epoch-boundary` and `completed_epoch=2`; rerunning
the same command does not advance past an already completed requested boundary.

Once N6 has validation records, overlay the matched shuffle controls with:

```bash
python -m galp.benchmarks.training_pls.plot_premixed_progress \
  --experiment-root /mnt/nvme2/home/tangyuxin/pls-experiments/pls-core-v2-20260811 \
  --seed 11997733 \
  --conditions A1,B6,N6 \
  --run N6=/mnt/nvme2/home/tangyuxin/pls-experiments/pls-core-v2-20260811/explorations/premixed_m4_no_shuffle_n6/runs/N6/seed_11997733 \
  --x-axis processed-images
```

In this mode, Python does not construct crop descriptors, PLS membership,
closed-pool sample order, RandAugment decisions or Mixup decisions. It only
consumes model-ready native tensors, performs forward/backward, accumulation,
optimizer/validation and epoch checkpointing. Per-pool native vector/block/
compressed-byte counters are aggregated into `metrics.jsonl`.

To assign whole paired seed blocks to different GPUs while keeping all four
conditions for a seed on one device, add for example:

```bash
  --seed-devices 11997733=cuda:1,11997734=cuda:0,11997735=cuda:1,11997736=cuda:0
```

After inspecting `matrix_execution_plan.json`, repeat the command with
`--execute`. Runs are issued in the fixed balanced order, save an atomic
`latest.pt` at every epoch boundary, and resume with the same run manifest.

Run the required 1024-sample A0 reference check:

```bash
python -m galp.benchmarks.training_pls.sanity_check \
  --train-manifest "$TRAIN_JSON" \
  --layout-plan "$HOME/tmp/pls-layout-current/physical_layout_plan.json" \
  --output /tmp/pls-core/a0_reference_sanity.json \
  --seed 11997733 \
  --device cuda:0
```

Aggregate completed, failed, and still-missing runs without inventing results:

```bash
python -m galp.benchmarks.training_pls.report \
  --runs-root /tmp/pls-core \
  --output-dir /tmp/pls-core-report \
  --conditions A0,A1,B2,B6
```

Refresh the live single-seed premixed validation plots without requiring all
runs to reach the same epoch or to have a `final_result.json`:

```bash
python -m galp.benchmarks.training_pls.plot_premixed_progress \
  --experiment-root /mnt/nvme2/home/tangyuxin/pls-experiments/pls-core-v2-20260811 \
  --seed 11997733 \
  --x-axis epoch
```

The command rereads each `metrics.jsonl` on every invocation and atomically
refreshes `premixed_validation_curves.png`, the separate top-1/top-5/loss
plots, a CSV containing the plotted points, and a JSON source summary. Use
`--x-axis processed-images` for the primary scientific convergence axis. If a
future run has a different directory layout, override any source with repeated
`--run CONDITION=/absolute/run/directory` arguments.

After fresh-process historical revalidation, explicitly select the canonical
CSV so superseded inline validation records are not used:

```bash
python -m galp.benchmarks.training_pls.plot_premixed_progress \
  --experiment-root /mnt/nvme2/home/tangyuxin/pls-experiments/pls-core-v2-20260811 \
  --seed 11997733 \
  --canonical-csv /mnt/nvme2/home/tangyuxin/pls-experiments/pls-core-v2-20260811/stages/seed_11997733_historical_revalidation/canonical_historical_validation.csv \
  --x-axis epoch
```

Canonical mode requires unique condition/epoch rows marked
`canonical_fresh_process=True`. The generated CSV and summary preserve the
canonical input path, SHA-256, checkpoint hash, and validation GPU provenance;
plot titles are labelled `fresh-process canonical validation`.

Audit the registered formal matrix and an optional live comparison without
trusting stale `run_status.json` values as proof of an active process:

```bash
python -m galp.benchmarks.training_pls.audit_goal \
  --experiment-root /mnt/nvme2/home/tangyuxin/pls-experiments/pls-core-v2-20260811 \
  --layout-plan /mnt/nvme2/home/tangyuxin/pls-experiments/pls-layout-20260811/physical_layout_plan.json \
  --formal-report-dir /mnt/nvme2/home/tangyuxin/pls-experiments/pls-core-v2-20260811/formal_report \
  --output-dir /mnt/nvme2/home/tangyuxin/pls-experiments/pls-core-v2-20260811/audits/current \
  --comparison-id premixed-layout-M4-seed-11997733 \
  --comparison-seed 11997733 \
  --comparison-run A0=/absolute/A0/run \
  --comparison-run A1=/absolute/A1/run \
  --comparison-run B2=/absolute/B2/run \
  --comparison-run B6=/absolute/B6/run
```

The audit writes `goal_completion_audit.json` and
`goal_completion_audit.md` atomically. It requires an actual epoch-300
`final_result.json` for formal completion, reports exact checkpoint and
validation coverage at every milestone, and labels a single-seed Premixed
comparison as exploratory rather than as the pre-registered four-seed result.

Generate a strict common-epoch Premixed single-seed report (the command
rejects missing validation points or permanent checkpoints):

```bash
python -m galp.benchmarks.training_pls.report_premixed_milestone \
  --experiment-root /mnt/nvme2/home/tangyuxin/pls-experiments/pls-core-v2-20260811 \
  --seed 11997733 \
  --epoch 50 \
  --output-dir /mnt/nvme2/home/tangyuxin/pls-experiments/pls-core-v2-20260811/premixed_milestones/epoch_050
```

This report provides exact fixed-budget metrics, single-seed 2×2 contrasts,
and partial normalized AUC through the requested epoch. It deliberately does
not calculate a confidence interval from one seed.

## Implementation map

- `layout.py`, `plan_layout.py`, `parquet_helper.py`: immutable layout sidecars.
- `recipe.py`, `published_augmentation.py`, `published_optimizer.py`: locked recipe.
- `model_registry.py`: architecture-independent DCT input contract, model
  builders, parameter guards, recipe binding and external-source provenance.
- `core_schedule.py`: epoch-local global, closed-pool, and frozen-physical-order
  streams plus stable keys.
- `contracts.py`: per-seed condition whitelist and hashes.
- `train.py`: full epoch-aware runner, accumulation, validation, metrics, resume.
- `run_matrix.py`: plan-first balanced matrix orchestration.
- `sanity_check.py`: one-update GALP/reference semantic comparison.
- `report.py`: curves, final metrics, paired/factorial effects, Student-t CIs.
- `report_convergence_reference.py`: threshold-free paired Native-B6 versus
  standard-DCT prefix curves, endpoint deltas, and normalized Top-1 AUC.
- `audit_goal.py`: read-only formal-matrix, live-process, and evidence-boundary audit.
- `report_premixed_milestone.py`: strict common-epoch single-seed Premixed report.
- `galp/benchmarks/system_rgbnomore/training/equal_image_epoch_benchmark.py`: GPU-identity-checked
  D2/D3/PyTorch full-ImageNet E1/E2 runner with microbatch 64, accumulation 16,
  exact tail, shared RGB initialization, and epoch-boundary resume. D2 is the
  canonical-order planned-augmentation native DALI baseline; D3 is the
  DALI-native shuffle/augmentation performance ceiling.
- `report_equal_image_performance.py`: fail-closed equal-image table combining
  Native physical B6 with the D2/D3/PyTorch E1/E2 artifacts.

The report includes every condition and seed. Throughput, memory, loader timing,
and class composition are explanatory only and never select a condition.
