# Four-pipeline training guide

This runner checks and measures `fastlens`, `rgbnomore`, `dali` and `pytorch`.
Final-paper protocols and the PLS convergence experiments are described in
[paper/](../../../paper/README.md) and [training_pls/](../../../training_pls/README.md).

## Prepare datasets

Training and validation use separate JSON manifests with `split` set to `train`
or `val`, zero-based ImageNet labels and stable logical sample IDs. Relative
image paths are resolved against the manifest directory. Each sample records
`logical_sample_id`, `path`, `label`, `width`, `height`, an optional
`payload_sha256`, and the manifest-local `fastlens_image_id` for native reads.
The runner checks hashes, duplicate IDs, label ranges and train/validation overlap.

Prepare the official source splits as described in the
[reference rebuilding guide](../../../paper/REBUILDING_REFERENCES.md). The
Compact-v3 training preparation entry point is:

```bash
python -m fastlens.benchmarks.baselines.system_rgbnomore.training.prepare_imagenet512_v3_train \
  --phase all --resize-workers 32 --layout-threads 32 \
  --shard-decode-threads 4 --shard-workers 16 \
  --encoding-workers-per-shard 2 --verify-workers 16
python -m fastlens.benchmarks.baselines.system_rgbnomore.training.prepare_imagenet512_v3_train \
  --phase manifests
```

Use `--help` to set source paths for a new installation. Official manifests
contain 1,281,167 training and 50,000 validation images, with separate native
stores. Preparation records its effective parallelism. A held-out subset of one
store is suitable for functional checks, not official ImageNet validation.

## Contract and runtime smoke

```bash
E2E="$PWD/fastlens/data/system_rgbnomore/e2e_v3"
TRAIN="$E2E/training_manifests_official_v3/train.json"
VAL="$E2E/training_manifests_official_v3/val.json"
FASTLENS_TRAIN="$PWD/fastlens/data/compressed/imagenet512_train_compact_v3/dct/manifest.bin"
FASTLENS_VAL="$PWD/fastlens/data/compressed/imagenet512_val_compact_v3/manifest.bin"

CUDA_VISIBLE_DEVICES=0 python \
  -m fastlens.benchmarks.baselines.system_rgbnomore.training.run \
  --enabled-pipelines fastlens,rgbnomore,dali,pytorch \
  --required-comparison-groups dct,rgb \
  --execution-mode runtime --phase smoke \
  --train-manifest "$TRAIN" --val-manifest "$VAL" \
  --fastlens-manifest "$FASTLENS_TRAIN" --fastlens-validation-manifest "$FASTLENS_VAL" \
  --expected-manifest-version 3 \
  --expected-physical-layout image-major-vector-rowgroups \
  --expected-spatial-order tiled-z32 \
  --expected-image-count 1281167 --expected-validation-image-count 50000 \
  --device cuda:0 --batch-size 64 --workers 4 \
  --warmup-steps 3 --measured-steps 10 --repeats 3 \
  --output-dir "$HOME/tmp/fastlens-training-runtime-smoke"
```

Add `--dry-run-contract` to create only the contract. Use a new output directory
for each independent run. `--execution-mode audit` retains stage synchronization
and per-step checks; `runtime` performs the first-step correctness probe outside
the measured interval, then synchronizes at region boundaries. `--workers`
controls outer data workers, while the native profile owns prefetch and scheduling.

The default recipe uses crop/flip, FP32, batch 64, 500 warmup steps and AdamW.
It is a paired training benchmark; the full 300-epoch PLS recipe additionally
uses its published augmentation, Mixup, accumulation and scheduler configuration.
Step runs use five repeats and aggregate hot repeats 1-4. Each repeat restores
model, optimizer, scheduler, RNG, augmentation state and sample cursor.

## Initialization and resume

- `--init-mode random` initializes without a checkpoint.
- `--init-mode weights` strictly loads the supplied RGB/DCT model weights.
- `--init-mode full-checkpoint` restores the matching model, optimizer, scheduler,
  RNG and completed-batch cursor, checking the full recipe contract.
- `--resume-run DIR` resumes an incomplete run using its saved contract and
  initial-state files. Execution modes must match.

Create or refresh payload fingerprints outside timing using
`--refresh-fastlens-payload-fingerprints`. Sample augmentation is derived from
`(seed, epoch, logical_sample_id)`. DCT crops align to 16 source pixels;
normalized coefficient comparison uses the existing `1/1020` tolerance.

## Equal-image training

The equal-image runner compares `d2`, `d3` and `pytorch` over complete epochs,
with microbatch 64, accumulation 16 and the partial tail retained. D2 and PyTorch
share order and augmentation decisions. D3 uses DALI-native shuffle/crop and is
reported as a separate performance reference. Epoch 1 is the cold observation;
epoch 2 is the primary warm observation.

```bash
OUT="$HOME/tmp/fastlens-equal-image-rgb"
python -m fastlens.benchmarks.baselines.system_rgbnomore.training.equal_image_epoch_benchmark \
  --train-manifest "$TRAIN" --val-manifest "$VAL" --output-dir "$OUT" \
  --pipelines d2,d3,pytorch --device cuda:0 --audit-mode runtime-first-100 \
  --required-gpu-name-substring "RTX 4090"
```

Inspect the generated contract and execution plan, then repeat the command with
`--execute`. Completed checkpoint epochs can be resumed. For comparison with a
completed native B6 run, use
`python -m fastlens.benchmarks.training_pls.report_equal_image_performance --help`.
The report checks GPU, software, source identity and image/update counts.

## Outputs

Contracts, initial states, per-pipeline records, sample/augmentation schedules,
first-step probes, curves and checkpoints are written to the output directory.
Correctness, semantics, performance and convergence have separate statuses.
Native counters supplement the common throughput and sample statistics.
Independently validate a completed run with:

```bash
python -m fastlens.benchmarks.baselines.system_rgbnomore.training.validate \
  "$HOME/tmp/fastlens-training-runtime-smoke" --no-write
```
