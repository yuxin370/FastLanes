#!/usr/bin/env python3
"""Short RTX-4090 resource/performance A/B for Phase-4B lifetime ownership."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROFILE = "rgbnomore-validation-v1"
IMAGE_BATCHES = ([0, 1, 2, 3], [4, 5, 6, 7])


def _select_lifetime(native_lifetime: bool) -> None:
    # Both sides use the Phase-3 production scheduler.  Only the lifetime
    # construction-time choice changes.
    os.environ.pop("GALP_PHASE3_NATIVE_DELEGATE", None)
    os.environ["GALP_PHASE4_NATIVE_LIFETIME"] = "1" if native_lifetime else "0"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _counter_delta(after: dict[str, Any], before: dict[str, Any], key: str) -> int:
    return int(after[key]) - int(before[key])


def _run_once(
    DirectDctReader: Any,
    torch: Any,
    native_module: Any,
    manifest: Path,
    native_lifetime: bool | None,
    schedule_repeats: int,
) -> dict[str, Any]:
    if native_lifetime is None:
        os.environ.pop("GALP_PHASE3_NATIVE_DELEGATE", None)
        os.environ.pop("GALP_PHASE4_NATIVE_LIFETIME", None)
    else:
        _select_lifetime(native_lifetime)
    before = dict(native_module._lifetime_reclaim_stats_for_test())
    reader = DirectDctReader(manifest)
    pipeline = reader.pipeline(PROFILE)
    expected_backend = "native" if native_lifetime is not False else "legacy"
    if pipeline._native._lifetime_backend_for_test != expected_backend:
        raise AssertionError(
            f"lifetime selector mismatch: {pipeline._native._lifetime_backend_for_test} != {expected_backend}"
        )
    schedule = [list(IMAGE_BATCHES[index % 2]) for index in range(schedule_repeats)]
    torch.cuda.reset_peak_memory_stats()
    pipeline.start(schedule)
    batch_latencies_ms: list[float] = []
    peak_transient_bytes = 0
    cpu_begin = time.process_time()
    wall_begin = time.perf_counter()
    for expected_ids in schedule:
        batch_begin = time.perf_counter()
        batch = next(pipeline)
        if batch.global_image_ids != expected_ids:
            raise AssertionError(f"batch order mismatch: {batch.global_image_ids} != {expected_ids}")
        y = batch.y
        cbcr = batch.cbcr
        # A tiny real same-stream consumer.  It is deliberately asynchronous;
        # the final boundary synchronization is outside production code.
        sentinel = y.reshape(-1)[0] + cbcr.reshape(-1)[0]
        peak_transient_bytes = max(
            peak_transient_bytes,
            int(batch._native.execution_stats.get("actual_transient_total_allocated_high_water_bytes", 0)),
        )
        del sentinel, y, cbcr, batch
        batch_latencies_ms.append((time.perf_counter() - batch_begin) * 1000.0)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_begin
    cpu_seconds = time.process_time() - cpu_begin
    pipeline.close()
    del pipeline, reader
    gc.collect()
    native_module.manual_reclaim()
    after = dict(native_module._lifetime_reclaim_stats_for_test())
    queue_name = "native" if native_lifetime is not False else "legacy"
    queue_before = dict(before[queue_name])
    queue_after = dict(after[queue_name])
    device_before = dict(before["device_pool"])
    device_after = dict(after["device_pool"])
    pinned_before = dict(before["pinned_pool"])
    pinned_after = dict(after["pinned_pool"])
    return {
        "throughput_images_per_s": (4 * schedule_repeats) / wall_seconds,
        "wall_seconds": wall_seconds,
        "host_cpu_seconds": cpu_seconds,
        "batch_latencies_ms": batch_latencies_ms,
        "peak_torch_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_transient_bytes": peak_transient_bytes,
        "queue": {
            "consumer_events": _counter_delta(queue_after, queue_before, "consumer_event_count"),
            "producer_events": _counter_delta(queue_after, queue_before, "producer_event_count"),
            "enqueued_batches": _counter_delta(queue_after, queue_before, "enqueued_batch_count"),
            "reclaimed_batches": _counter_delta(queue_after, queue_before, "reclaimed_batch_count"),
            "pending_peak_observed": int(queue_after["pending_reclaim_peak"]),
            "live_peak_observed": int(queue_after["live_batch_peak"]),
        },
        "device_pool": {
            "allocation_requests": _counter_delta(device_after, device_before, "allocation_requests"),
            "cuda_allocations": _counter_delta(device_after, device_before, "cuda_allocation_count"),
            "cuda_allocation_bytes": _counter_delta(device_after, device_before, "cuda_allocation_bytes"),
            "in_use_bytes_after": int(device_after["in_use_bytes"]),
            "peak_in_use_bytes_observed": int(device_after["peak_in_use_bytes"]),
        },
        "pinned_pool": {
            "allocation_requests": _counter_delta(pinned_after, pinned_before, "allocation_requests"),
            "cuda_allocations": _counter_delta(pinned_after, pinned_before, "cuda_allocation_count"),
            "cuda_allocation_bytes": _counter_delta(pinned_after, pinned_before, "cuda_allocation_bytes"),
            "in_use_bytes_after": int(pinned_after["in_use_bytes"]),
            "peak_in_use_bytes_observed": int(pinned_after["peak_in_use_bytes"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--schedule-repeats", type=int, default=16)
    parser.add_argument("--default-smoke", action="store_true")
    args = parser.parse_args()
    if not args.default_smoke and (args.warmup < 2 or not 5 <= args.pairs <= 10):
        raise ValueError("Phase-4B protocol requires warmup >= 2 and 5 <= pairs <= 10")

    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(args.module_path.resolve()))
    import torch
    import _galp_direct_dct as native_module
    from galp.torch import DirectDctReader

    if args.default_smoke:
        run = _run_once(
            DirectDctReader,
            torch,
            native_module,
            args.manifest,
            None,
            args.schedule_repeats,
        )
        payload = {
            "schema": "galp-phase4b-default-performance-smoke-v1",
            "result": "PASS",
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "uuid": str(torch.cuda.get_device_properties(0).uuid),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            },
            "run": run,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print("PHASE 4B DEFAULT PERFORMANCE SMOKE PASS")
        print(
            json.dumps(
                {
                    "throughput_images_per_s": run["throughput_images_per_s"],
                    "p50_ms": _percentile(run["batch_latencies_ms"], 0.50),
                    "p95_ms": _percentile(run["batch_latencies_ms"], 0.95),
                    "host_cpu_seconds": run["host_cpu_seconds"],
                    "peak_transient_bytes": run["peak_transient_bytes"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    for _ in range(args.warmup):
        _run_once(DirectDctReader, torch, native_module, args.manifest, False, args.schedule_repeats)
        _run_once(DirectDctReader, torch, native_module, args.manifest, True, args.schedule_repeats)

    runs: dict[str, list[dict[str, Any]]] = {"legacy": [], "native": []}
    for index in range(args.pairs):
        order = (False, True) if index % 2 == 0 else (True, False)
        for native_lifetime in order:
            runs["native" if native_lifetime else "legacy"].append(
                _run_once(
                    DirectDctReader,
                    torch,
                    native_module,
                    args.manifest,
                    native_lifetime,
                    args.schedule_repeats,
                )
            )

    legacy = runs["legacy"]
    native = runs["native"]
    legacy_throughput = statistics.mean(run["throughput_images_per_s"] for run in legacy)
    native_throughput = statistics.mean(run["throughput_images_per_s"] for run in native)
    legacy_latencies = [value for run in legacy for value in run["batch_latencies_ms"]]
    native_latencies = [value for run in native for value in run["batch_latencies_ms"]]
    legacy_p50 = _percentile(legacy_latencies, 0.50)
    native_p50 = _percentile(native_latencies, 0.50)
    legacy_p95 = _percentile(legacy_latencies, 0.95)
    native_p95 = _percentile(native_latencies, 0.95)
    legacy_cpu = statistics.mean(run["host_cpu_seconds"] for run in legacy)
    native_cpu = statistics.mean(run["host_cpu_seconds"] for run in native)
    throughput_ratio = native_throughput / legacy_throughput
    p50_ratio = native_p50 / legacy_p50
    p95_ratio = native_p95 / legacy_p95
    peak_transient_legacy = max(run["peak_transient_bytes"] for run in legacy)
    peak_transient_native = max(run["peak_transient_bytes"] for run in native)
    peak_torch_legacy = max(run["peak_torch_bytes"] for run in legacy)
    peak_torch_native = max(run["peak_torch_bytes"] for run in native)

    gates = {
        "throughput": throughput_ratio >= 0.99,
        "p50": p50_ratio <= 1.02,
        "p95": p95_ratio <= 1.02,
        "peak_transient": peak_transient_native <= peak_transient_legacy,
        "peak_torch": peak_torch_native <= peak_torch_legacy,
        "producer_events": all(run["queue"]["producer_events"] == 0 for run in native),
        "no_new_cuda_allocations": sum(run["device_pool"]["cuda_allocations"] for run in native)
        <= sum(run["device_pool"]["cuda_allocations"] for run in legacy),
        "no_new_pinned_allocations": sum(run["pinned_pool"]["cuda_allocations"] for run in native)
        <= sum(run["pinned_pool"]["cuda_allocations"] for run in legacy),
    }
    payload = {
        "schema": "galp-phase4b-lifetime-ab-v1",
        "result": "PASS" if all(gates.values()) else "FAIL",
        "protocol": {
            "warmup": args.warmup,
            "pairs": args.pairs,
            "schedule_repeats": args.schedule_repeats,
            "ordering": "balanced alternating",
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "uuid": str(torch.cuda.get_device_properties(0).uuid),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "summary": {
            "throughput_images_per_s": {"legacy": legacy_throughput, "native": native_throughput, "ratio": throughput_ratio},
            "p50_ms": {"legacy": legacy_p50, "native": native_p50, "ratio": p50_ratio},
            "p95_ms": {"legacy": legacy_p95, "native": native_p95, "ratio": p95_ratio},
            "host_cpu_seconds": {"legacy": legacy_cpu, "native": native_cpu, "ratio": native_cpu / legacy_cpu},
            "peak_transient_bytes": {"legacy": peak_transient_legacy, "native": peak_transient_native},
            "peak_torch_bytes": {"legacy": peak_torch_legacy, "native": peak_torch_native},
        },
        "gates": gates,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE 4B LIFETIME A/B " + payload["result"])
    print(json.dumps({"summary": payload["summary"], "gates": gates}, indent=2, sort_keys=True))
    return 0 if payload["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
