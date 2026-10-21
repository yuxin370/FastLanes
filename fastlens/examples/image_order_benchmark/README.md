# Image-order benchmark

This supporting example compares random and rowgroup-aligned image batches in
FastLens's `component -> spatial block -> image` layout. It measures inference
throughput, prediction agreement and a short training-order sensitivity probe.
Final-paper experiments are in [paper/](../../benchmarks/paper/README.md).

## Protocol

Defaults are seed 11997733, batch 64, five warmup batches and 20 measured batches.
The conditions are interleaved and repeat 0 is excluded from the formal summary.

| Condition | Selection |
| --- | --- |
| `current_random` | Global shuffle of 48,615 eligible IDs; first 1,600 images |
| `paired_scattered` | Shuffle 25 aligned batches within their warmup/measured cohorts |
| `contiguous` | The same paired cohorts, retaining 64-image physical rowgroups |

The paired conditions have identical warmup and measured populations. The
current-random condition has a different population and is an operational
comparison. Timing includes read/decode, preprocessing, forward, accuracy metrics
and CUDA synchronization; model/reader creation and warmup are excluded.

Logits are aligned by image ID to compare Top-1/Top-5, prediction agreement,
absolute differences and cosine similarity. The training probe starts from the
same checkpoint and transformed samples, comparing global shuffle with shuffled
contiguous batches on the same held-out population. It is a short sensitivity
probe rather than full ImageNet retraining.

## Run

```bash
PYTHONPATH=build/fastlens/torch python \
  fastlens/examples/run_image_order_benchmark.py \
  --output-dir "$HOME/tmp/fastlens-image-order-e2e"
```

Add `--selection-only` to generate and check the selection without using a GPU,
or `--skip-training` to run only inference performance and semantics.

## Outputs

| File | Contents |
| --- | --- |
| `contract.json` | Input identities, environment and parameters |
| `selection.json` | Image IDs, paired invariants and class statistics |
| `performance.json/csv` | Per-condition, per-repeat measurements |
| `semantic_*.npz` | Image IDs, labels and logits |
| `training_probe.json` | Per-seed/epoch training and held-out evaluation |
| `training_order_stats.json` | Full-training-index class mixing |
| `summary.json`, `report.md` | Aggregates and interpretation |
