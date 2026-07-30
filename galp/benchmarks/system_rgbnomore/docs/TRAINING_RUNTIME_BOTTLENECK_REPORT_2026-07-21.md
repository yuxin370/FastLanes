# GALP training runtime bottleneck report — 2026-07-21

## Outcome

The benchmark now has an immutable `--execution-mode audit|runtime` contract. `audit` remains the default and preserves the exhaustive behavior. `runtime` keeps the same fresh-clone first-step semantic probe outside timing, then measures real `zero_grad -> forward -> loss -> backward -> optimizer.step -> scheduler.step` training without per-step parameter scans, `.item()`, or stage synchronizations.

This environment cannot expose a GPU to PyTorch (`torch.cuda.is_available() == false`, device count 0), so no after-change GPU number is claimed. The historical GALP result supplied with this task is 7.83 img/s. The required target-host commands are at the end of this report.

The supplied inputs were inspected in place: the train manifest declares 100,000 samples, the held-out validation manifest declares 10,000 samples, both carry logical IDs, labels, dimensions, and `galp_image_id`, and the GALP manifest plus payload-fingerprint cache are present. No ImageNet validation or final-accuracy claim is made.

## Root-cause diagnosis

The old result was not a clean production-loader measurement:

1. The measured audit step explicitly synchronized at step start and after forward, loss, backward, gradient processing, optimizer, and scheduler. CPU-backed inputs added an H2D-completion synchronization.
2. It materialized loss and finite checks on the host every step and scanned every gradient tensor with multiple device-to-host scalar reads. The first measured step also cloned all trainable parameters to CPU and computed a full update diff.
3. GALP requested `batch.execution_stats` in `next_batch`; that API waits for native GPU completion. It therefore forced a correctness/statistics boundary inside the loader call before the model could consume the batch.
4. GALP submitted several native handles, but the native reader intentionally executes an ordered single-producer chain. `--workers 4` was not explicitly mapped by the training adapter, so its meaning was misleading even though the native default happened to be four rowgroup workers.
5. Dropping pending handles relied on future destruction. Queue capacity, hit/miss, producer activity, cancellation/drain, and consumer wait were not available in the training artifact.

The runtime changes remove items 1--3 from the timed step, explicitly map `--workers` to native rowgroup-prefetch workers, and make item 5 bounded and observable. The batch producer remains deliberately single and ordered; the artifact states this rather than implying four batch producers.

## Timing and synchronization policy

| Property | audit | runtime |
|---|---|---|
| First-step semantic probe | Fresh clone, before warmup | Same audit probe, fresh clone, before warmup |
| Per-step gradient scan | Every step | None in measured region |
| Full parameter snapshot | First measured step | None in measured region |
| Loss host materialization | Per step | One deferred batch operation after region end |
| Explicit CUDA synchronization | Per stage | Once before and once after measured region |
| Stage timing | Synchronized host wall time | CUDA events resolved after region end |
| GALP execution stats | Inside loader step | Deferred until after region-end synchronization |
| Main throughput | Sum of synchronized step times | Measured-region wall clock |

Every runtime artifact records explicit synchronization count/reasons, GALP native internal sync count/reasons, deep-scan count, deferred materializations, and whether the first-step probe entered timing.

## GALP producer/consumer behavior

- Capacity is exactly `prefetch_depth + 1` batches: the current batch plus the configured number ahead.
- Submission fails at capacity until the consumer advances, providing hard backpressure.
- Consumption is FIFO, so canonical sample order and transform descriptors remain checked.
- Producer exceptions propagate from `pop`.
- `close` tries to cancel queued native work and then joins/drains every handle. The native handle reports `started`, `active`, `finished`, cancellation, and producer-active milliseconds.
- `--workers N` means native rowgroup-prefetch workers inside one asynchronous ordered batch producer. `N=0` is rejected. It never means N batch producers.
- Artifacts expose maximum/current queue depth, hit/miss and hit rate, submission time, native producer-active time, consumer wait and wait fraction, cancellation/drain/close errors, read/decode/projection time, native internal syncs, and allocator counters.

## Required four-pipeline result table

The table is intentionally incomplete because GPU execution is unavailable in this sandbox. Filling unavailable cells with estimates would make the comparison non-auditable.

| Pipeline | Input domain | First-step semantic | img/s | Median step | p95 step | Loader wait | Forward | Backward + optimizer | CPU/GPU memory | CV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GALP | DCT | Not run here | — | — | — | — | — | — | — | — | Environment-blocked |
| RGB-no-more | DCT | Not run here | — | — | — | — | — | — | — | — | Environment-blocked |
| DALI | RGB | Not run here | — | — | — | — | — | — | — | — | Environment-blocked |
| PyTorch | RGB | Not run here | — | — | — | — | — | — | — | — | Environment-blocked |

## Baseline versus after

| Metric | Historical audit baseline | Runtime after | Target |
|---|---:|---:|---:|
| GALP end-to-end train throughput | 7.83 img/s | Not measurable here | >=78.3 img/s or >=70% compute upper bound |
| GALP loader wait fraction | Not available in legacy artifact | Not measurable here | <20% |
| Hot-repeat throughput CV | Not available | Not measurable here | <=10% |
| Compute-only achieved fraction | Not available | Not measurable here | >=70% if 10x is not reached |
| Cold/hot ratio | Not available | Not measurable here | Diagnostic, no fabricated gate |

Runtime output computes an event-derived compute-only upper bound from forward, loss, backward, gradient-processing, and optimizer CUDA-event durations. It is marked diagnostic and is never treated as a fifth pipeline. Repeat 0 is reported as cold; repeats 1--4 form the hot aggregate and CV.

## Verification completed here

- Python training benchmark suite: 24 tests run; 23 pass and one CUDA-gated test is skipped when CUDA is unavailable.
- Native build: `_galp_direct_dct` and `galp_tests` compile successfully.
- Native suite: 113 tests pass; three existing fixture/opt-in tests skip.
- Covered behaviors include CLI default/runtime selection, runtime contract, immutable resume rejection across modes, zero measured-path audit scans, synchronization reason accounting, ordered queue behavior, hard backpressure, producer exception propagation, close cleanup, exact training-state reset, GALP worker failure semantics, canonical identity/descriptor checks, and the mode-independent audit probe policy.

## Exact target-host commands

Run these from `/home/tangyuxin/gfastlanes/FastLanes` on the target GPU host. Start with stability; do not use `--phase all`.

```bash
cmake --build build -j2 --target _galp_direct_dct galp_tests
./build/galp/tests/galp_tests

PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
TRAIN=/tmp/galp-training-manifests/train.json
VAL=/tmp/galp-training-manifests/val.json
GALP=/home/tangyuxin/gfastlanes/FastLanes/galp/data/imagedataset_dct/ImageNet-train/manifest.bin

PYTHONPATH=galp/benchmarks/system_rgbnomore "$PY" \
  -m unittest galp.tests.test_training_benchmark

CUDA_VISIBLE_DEVICES=0 "$PY" galp/benchmarks/system_rgbnomore/training/run.py \
  --enabled-pipelines galp,rgbnomore,dali,pytorch \
  --required-comparison-groups dct,rgb \
  --execution-mode runtime --phase smoke \
  --train-manifest "$TRAIN" --val-manifest "$VAL" --galp-manifest "$GALP" \
  --rgbnomore-root /home/tangyuxin/RGB-no-more \
  --device cuda:0 --batch-size 64 --workers 4 --prefetch-depth 2 \
  --warmup-steps 3 --measured-steps 10 \
  --output-dir /tmp/galp-training-runtime-stability-20260721

"$PY" galp/benchmarks/system_rgbnomore/training/validate.py \
  /tmp/galp-training-runtime-stability-20260721 --no-write
```

After stability passes, sweep GALP queue depth in independent directories:

```bash
for DEPTH in 0 2 4; do
  CUDA_VISIBLE_DEVICES=0 "$PY" galp/benchmarks/system_rgbnomore/training/run.py \
    --pipeline galp --execution-mode runtime --phase smoke \
    --train-manifest "$TRAIN" --val-manifest "$VAL" --galp-manifest "$GALP" \
    --rgbnomore-root /home/tangyuxin/RGB-no-more \
    --device cuda:0 --batch-size 64 --workers 4 --prefetch-depth "$DEPTH" \
    --warmup-steps 5 --measured-steps 30 \
    --output-dir "/tmp/galp-training-runtime-depth-${DEPTH}-20260721"
done
```

Then run the formal five-repeat comparison:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" galp/benchmarks/system_rgbnomore/training/run.py \
  --enabled-pipelines galp,rgbnomore,dali,pytorch \
  --required-comparison-groups dct,rgb \
  --execution-mode runtime --phase step \
  --train-manifest "$TRAIN" --val-manifest "$VAL" --galp-manifest "$GALP" \
  --rgbnomore-root /home/tangyuxin/RGB-no-more \
  --device cuda:0 --batch-size 64 --workers 4 --prefetch-depth 2 \
  --warmup-steps 10 --measured-steps 100 --repeats 5 \
  --output-dir /tmp/galp-training-runtime-step-20260721

"$PY" galp/benchmarks/system_rgbnomore/training/validate.py \
  /tmp/galp-training-runtime-step-20260721 --no-write
```

If neither 78.3 img/s nor 70% of the compute upper bound is reached, inspect `loader_measured_metrics`, `stage_latency_ms`, `native_allocator_metrics`, and `synchronization_accounting` in `pipeline_galp.json`. These fields distinguish storage/read, decode/projection, producer starvation, consumer wait, allocator churn, and remaining native synchronization without changing the measured workload.

No final ImageNet accuracy or convergence claim is made by these short runs.
