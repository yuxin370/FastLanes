"""Real classifier updates on native IC/FS compact mosaics; throughput, not accuracy."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time

import torch

from fastlens.benchmarks.dct_models.measure_process import ProcessWindow
from fastlens.benchmarks.baselines.system_rgbnomore.training.artifacts import tensor_state_sha256
from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import (
    capture_rng_state, restore_rng_state, seed_everything,
)
from .contracts import artifact, write_json
from .store import Counters
from .training import update_windows


def _ranges(count, size):
    """Retain a singleton tail by adding it to the preceding batch (BN-safe)."""
    start = 0
    while start < count:
        stop = min(start + size, count)
        if count - stop == 1:
            stop = count
        yield start, stop
        start = stop


def batch_plan(count, batch_size, pool_size, accumulation):
    if count < 2 or batch_size < 2 or pool_size < batch_size or accumulation < 1:
        raise ValueError("need >=2 images, batch_size>=2, pool_size>=batch_size and accumulation>=1")
    pools = []
    for pool_id, (start, stop) in enumerate(_ranges(count, pool_size)):
        rows = [dict(epoch=0, closed_pool_id=pool_id, microbatch_id=i, start=a, stop=b)
                for i, (a, b) in enumerate(_ranges(stop-start, batch_size))]
        pools.append((start, stop, update_windows(rows, accumulation)))
    return pools


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _train_pass(dataset, model, optimizer, plan, device):
    total_loss = torch.zeros((), device=device)
    counters = Counters()
    samples = microbatches = updates = h2d_bytes = 0
    stream = dataset.iter_training_batches(plan) if hasattr(dataset, "iter_training_batches") else None
    try:
        for start, stop, windows in plan:
            if stream is None:
                values, labels, read_counters = dataset.read(list(range(start, stop)))
                if read_counters is None:
                    counters = None
                else:
                    counters.add(Counters(**read_counters))
            for window in windows:
                optimizer.zero_grad(set_to_none=True)
                window_samples = sum(row["stop"]-row["start"] for row in window)
                for row in window:
                    left, right = row["start"], row["stop"]
                    if stream is None:
                        source_inputs, source_targets = values[left:right], labels[left:right]
                    else:
                        source_inputs, source_targets, read_counters = next(stream)
                        if len(source_inputs) != right-left or len(source_targets) != right-left:
                            raise AssertionError("streamed microbatch differs from the training plan")
                        if read_counters is None:
                            counters = None
                        else:
                            counters.add(Counters(**read_counters))
                    inputs = source_inputs.to(device)
                    targets = source_targets.to(device)
                    logits = model(inputs)
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                    (loss * ((right-left)/window_samples)).backward()
                    total_loss += loss.detach() * (right-left)
                    samples += right-left
                    microbatches += 1
                    if device.type == "cuda":
                        if source_inputs.device.type == "cpu":
                            h2d_bytes += inputs.numel()*inputs.element_size()
                        if source_targets.device.type == "cpu":
                            h2d_bytes += targets.numel()*targets.element_size()
                optimizer.step()
                updates += 1
        if stream is not None:
            try:
                next(stream)
            except StopIteration:
                pass
            else:
                raise AssertionError("streamed input contains extra training microbatches")
    finally:
        if stream is not None:
            stream.close()
    loss_sum = float(total_loss)
    if not math.isfinite(loss_sum):
        raise FloatingPointError("non-finite compact training loss")
    return dict(samples=samples, microbatches=microbatches, optimizer_updates=updates,
                loss_sum=loss_sum, h2d_bytes=h2d_bytes, reader_counters=asdict(counters) if counters is not None else None)


def _window(dataset, model, optimizer, plan, device, *, seconds, passes, phase="Measured"):
    """Start before read(), finish at a complete pass and synchronized update."""
    totals = dict(samples=0, microbatches=0, optimizer_updates=0, loss_sum=0., h2d_bytes=0)
    counters = Counters()
    pass_seconds = []
    _sync(device)
    started = previous = time.perf_counter()
    while True:
        row = _train_pass(dataset, model, optimizer, plan, device)
        _sync(device)
        now = time.perf_counter()
        pass_seconds.append(now-previous)
        previous = now
        for name in totals:
            totals[name] += row[name]
        if row["reader_counters"] is None:
            counters = None
        else:
            counters.add(Counters(**row["reader_counters"]))
        print(f"{phase} dataset passes: {len(pass_seconds)}; samples: {totals['samples']}; timed seconds: {now-started:.3f}", flush=True)
        if (passes is not None and len(pass_seconds) >= passes) or (passes is None and now-started >= seconds):
            break
    wall = now-started
    return dict(totals, wall_seconds=wall, first_pass_seconds=pass_seconds[0], pass_seconds=pass_seconds,
                complete_passes=len(pass_seconds), images_s=totals["samples"]/wall,
                updates_s=totals["optimizer_updates"]/wall,
                loss_mean=totals["loss_sum"]/totals["samples"], reader_counters=asdict(counters) if counters is not None else None)


def benchmark(dataset, *, device="cuda:0", batch_size=64, pool_size=256, accumulation=1,
              warmup_seconds=60., min_seconds=180., epochs=None, repeats=3, seed=20261014,
              learning_rate=.001, momentum=.9, weight_decay=4e-5, model_factory=None, model_name="resnet50"):
    """Repeat from one identical random state; each pass continues updating weights."""
    if not math.isfinite(warmup_seconds) or warmup_seconds < 0 or not math.isfinite(min_seconds) or min_seconds < 0:
        raise ValueError("timing durations must be finite and nonnegative")
    if (epochs is not None and epochs < 1) or repeats < 1:
        raise ValueError("epochs/repeats must be positive")
    plan = batch_plan(dataset.image_count, batch_size, pool_size, accumulation)
    device = torch.device(device)
    caller_rng = capture_rng_state()
    runtime_before = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                      torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    try:
        seed_everything(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        from .compact_models import build_model, input_policy, MODEL_LABELS
        model = (model_factory() if model_factory is not None else build_model(model_name)).to(device).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        initial_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        initial_optimizer = copy.deepcopy(optimizer.state_dict())
        initial_rng = capture_rng_state()
        initial_sha256 = tensor_state_sha256(initial_model)

        def restore():
            model.load_state_dict(initial_model)
            optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
            optimizer.zero_grad(set_to_none=True)
            restore_rng_state(initial_rng)

        results = []
        for repeat in range(repeats):
            restore()
            warmup = (_window(dataset, model, optimizer, plan, device, seconds=warmup_seconds, passes=None, phase="Warmup")
                      if warmup_seconds else None)
            restore()  # Model, optimizer, BN buffers and RNG; keep warmed OS/decoded caches.
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with ProcessWindow() as resources:
                row = _window(dataset, model, optimizer, plan, device, seconds=min_seconds, passes=epochs)
            row.update(repeat=repeat, warmup=warmup, resources=resources.result,
                       final_model_sha256=tensor_state_sha256(model.state_dict()))
            if device.type == "cuda":
                row["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                row["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            if getattr(dataset, "producer_h2d_unmeasured", False):
                for window in (row, warmup):
                    if window is not None:
                        window["model_input_h2d_bytes"] = window.pop("h2d_bytes")
                        window["h2d_bytes"] = None
                        window["h2d_scope"] = "model-input transfer measured; native producer internal transfer not measured"
            results.append(row)
        batch_hist = Counter(row["stop"]-row["start"] for _, _, windows in plan for window in windows for row in window)
        update_hist = Counter(sum(row["stop"]-row["start"] for row in window)
                              for _, _, windows in plan for window in windows)
        return dict(status="passed", experiment_type="native_compact_training_throughput", headline_eligible=False,
                    model=MODEL_LABELS[model_name] if model_factory is None else "injected_test_model",
                    model_name=model_name if model_factory is None else "injected_test_model",
                    model_input_policy=input_policy(model_name, dataset.input_shape) if model_factory is None else None,
                    initial_model_sha256=initial_sha256, checkpoint=None, seed=seed, precision="fp32",
                    runtime=dict(device=str(device), torch=torch.__version__, cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                                 torch_threads=torch.get_num_threads(), tf32=False, cudnn_deterministic=True),
                    execution_partition=dict(getattr(dataset, "execution_partition", dict(read_decode="cpu", normalization="cpu")),
                                             augmentation="disabled", model=str(device)),
                    input_shape=list(dataset.input_shape), dataset_images=dataset.image_count,
                    dataset={key: dataset.manifest[key] for key in ("split", "representation", "normalization",
                            "source_identity", "data_identity") if key in dataset.manifest},
                    configured_microbatch=batch_size, preparation_pool=pool_size,
                    gradient_accumulation=accumulation,
                    microbatch_sample_histogram_per_pass=dict(sorted(batch_hist.items())),
                    optimizer_sample_histogram_per_pass=dict(sorted(update_hist.items())),
                    tail_policy="retain singleton by merging into preceding pool/microbatch; actual microbatch may be configured+1",
                    optimizer=dict(name="SGD", learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay),
                    timing=dict(warmup_seconds=warmup_seconds, min_seconds=None if epochs else min_seconds,
                                requested_complete_passes=epochs, scope="request read through synchronized optimizer update",
                                warmup_state="restore initial weights, BN buffers, optimizer and RNG after warmup; retain cache state",
                                initialization_and_snapshot_restore="excluded", order="fixed manifest order for every pass",
                                cache_state="natural after actual complete warmup passes; no OS cache eviction"),
                    interpretation="Real parameter updates; dataset passes cover only this manifest. No convergence, accuracy or full-ImageNet-training claim.",
                    repeats=results, median_images_s=statistics.median(r["images_s"] for r in results),
                    min_images_s=min(r["images_s"] for r in results), max_images_s=max(r["images_s"] for r in results))
    finally:
        restore_rng_state(caller_rng)
        (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
         torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = runtime_before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", choices=("ic_shared", "ic_batched", "fs"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    from .compact_models import MODEL_NAMES
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet50")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--decode-budget-mib", type=int, default=1024)
    parser.add_argument("--reader-memory-mib", type=int, default=2048)
    parser.add_argument("--warmup-seconds", type=float, default=60.)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--min-seconds", type=float, default=180.)
    duration.add_argument("--epochs", type=int, help="complete passes over this training manifest; overrides time window")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--momentum", type=float, default=.9)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from .compact_data import CompactDataset
    torch.set_num_threads(args.torch_threads)
    with CompactDataset(args.manifest, args.method, pool_size=args.pool_size, workers=args.workers,
                        decode_budget_bytes=args.decode_budget_mib*1024**2,
                        memory_budget_bytes=args.reader_memory_mib*1024**2) as dataset:
        result = benchmark(dataset, device=args.device, batch_size=args.batch_size, pool_size=args.pool_size,
                           accumulation=args.accumulation, warmup_seconds=args.warmup_seconds, min_seconds=args.min_seconds,
                           epochs=args.epochs, repeats=args.repeats, seed=args.seed, learning_rate=args.learning_rate,
                           momentum=args.momentum, weight_decay=args.weight_decay, model_name=args.model)
    result["manifest"] = artifact(args.manifest)
    result["method"] = args.method
    result["workers"] = args.workers
    result["reader_budgets"] = dict(decode_bytes=args.decode_budget_mib*1024**2,
                                    memory_bytes=args.reader_memory_mib*1024**2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(status="passed", output=str(args.output), median_images_s=result["median_images_s"],
                          claim="training throughput only; no measured validation accuracy"), indent=2))


if __name__ == "__main__":
    main()
