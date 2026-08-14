# PLS crop/shuffle core model experiment

This package implements the pre-registered 2×2 ImageNet model-effect experiment:

| condition | crop | sample order |
|---|---|---|
| A0 | per sample | global shuffle |
| A1 | per virtual PLS | the exact A0 global order |
| B2 | per sample | closed pool, `G=1024`, `M=4` |
| B6 | per virtual PLS | the exact B2 closed-pool order |

All four conditions use the same GALP Direct-DCT backend and the immutable
`rgbnomore-vitti-dct-published-v1` recipe: ViT-Ti DCT, 300 epochs, FP32,
microbatch 64, accumulation 16, effective batch 1024, LR 3e-3, 10,000-update
warmup plus epoch-aware cosine, independent RGB-no-more WeightDecay semantics,
gradient clipping 1, DCT Mixup 0.2, and the published two-op/magnitude-three DCT
RandAugment list. Validation uses `ResizedCenterCrop_DCT(32,28)`.
The fixed execution backend groups samples that have the same keyed
RandAugment operation/magnitude and compiles the fixed-shape ViT with
`torch.compile`/Inductor. The scalar RandAugment reference remains available
to the sanity check; grouping changes kernel dispatch only, not decisions or
tensor values.

The formal path is semantic emulation over a frozen virtual physical mapping.
It does not claim that the full dataset has been rewritten as a physical
block-major FLS, that a GPU pool was physically resident, or that bytes read
were reduced.

## Commands

From the repository root:

```bash
export PYTHONPATH="$PWD/galp/benchmarks/system_dct_major${PYTHONPATH:+:$PYTHONPATH}"
export TRAIN_JSON="$PWD/galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/train.json"
export VAL_JSON="$PWD/galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json"
```

Freeze the virtual physical layout:

```bash
python -m training_pls.plan_layout \
  --train-manifest "$TRAIN_JSON" \
  --output-dir /tmp/pls-layout \
  --segment-images 1024 \
  --organization-seed 20260810
```

Create and inspect the 16-run plan (this never trains without `--execute`):

```bash
python -m training_pls.run_matrix \
  --output-dir /tmp/pls-core \
  --train-manifest "$TRAIN_JSON" \
  --val-manifest "$VAL_JSON" \
  --layout-plan /tmp/pls-layout/physical_layout_plan.json \
  --conditions A0,A1,B2,B6 \
  --seeds 11997733,11997734,11997735,11997736 \
  --epochs 300 \
  --device cuda:0
```

To assign whole paired seed blocks to different GPUs while keeping all four
conditions for a seed on one device, add for example:

```bash
  --seed-devices 11997733=cuda:1,11997734=cuda:0,11997735=cuda:1,11997736=cuda:0
```

After inspecting `matrix_execution_plan.json`, repeat the command with
`--execute`. Runs are issued in the fixed balanced order, save an atomic
`latest.pt` at every epoch boundary, and resume with the same contract.

Run the required 1024-sample A0 reference check:

```bash
python -m training_pls.sanity_check \
  --train-manifest "$TRAIN_JSON" \
  --layout-plan /tmp/pls-layout/physical_layout_plan.json \
  --output /tmp/pls-core/a0_reference_sanity.json \
  --seed 11997733 \
  --device cuda:0
```

Aggregate completed, failed, and still-missing runs without inventing results:

```bash
python -m training_pls.report \
  --runs-root /tmp/pls-core \
  --output-dir /tmp/pls-core-report \
  --conditions A0,A1,B2,B6
```

Refresh the live single-seed premixed validation plots without requiring all
runs to reach the same epoch or to have a `final_result.json`:

```bash
python -m training_pls.plot_premixed_progress \
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

## Implementation map

- `layout.py`, `plan_layout.py`, `parquet_helper.py`: immutable layout sidecars.
- `recipe.py`, `published_augmentation.py`, `published_optimizer.py`: locked recipe.
- `core_schedule.py`: epoch-local global and closed-pool streams plus stable keys.
- `contracts.py`: per-seed condition whitelist and hashes.
- `train.py`: full epoch-aware runner, accumulation, validation, metrics, resume.
- `run_matrix.py`: plan-first balanced matrix orchestration.
- `sanity_check.py`: one-update GALP/reference semantic comparison.
- `report.py`: curves, final metrics, paired/factorial effects, Student-t CIs.

The report includes every condition and seed. Throughput, memory, loader timing,
and class composition are explanatory only and never select a condition.
