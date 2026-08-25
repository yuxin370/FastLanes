#!/usr/bin/env python3
"""One-shot validation for the private Phase-3 Torch pipeline delegate switch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROFILE = "rgbnomore-validation-v1"
BATCHES = [[0, 1, 2, 3], [4, 5]]
TRANSFORMS = [
    [
        {
            "horizontal_flip": True,
            "logical_sample_id": f"phase3-a-{image_id}",
            "augmentation_key": "phase3-production-ab",
        }
        for image_id in BATCHES[0]
    ],
    [
        {
            "horizontal_flip": False,
            "logical_sample_id": f"phase3-b-{image_id}",
            "augmentation_key": "phase3-production-ab",
            "crop": {"x": 0, "y": 0, "width": 256, "height": 256},
        }
        for image_id in BATCHES[1]
    ],
]

EXACT_RESOURCE_KEYS = (
    "decode_kernel_launch_count",
    "materialize_kernel_launch_count",
    "planless_transform_kernel_launch_count",
    "planless_transform_dense_kernel_launch_count",
    "planless_transform_sparse_kernel_launch_count",
    "fixed_grid_finalize_kernel_launch_count",
    "internal_sync_count",
    "cached_gather_sync_count",
    "decoded_batch_sync_count",
    "copy_to_decode_event_handoff_count",
    "decode_to_transform_event_handoff_count",
    "fixed_grid_round_event_handoff_count",
    "workset_upload_dma_count",
    "workset_upload_dma_bytes",
    "compressed_payload_bytes_read",
    "selected_compressed_payload_bytes",
    "rowgroup_storage_bytes_read",
    "pread_count",
    "preadv_count",
)


def _select_backend(native: bool) -> None:
    if native:
        # Native is the production default. Absence of the private override is
        # part of this validation contract.
        os.environ.pop("GALP_PHASE3_NATIVE_DELEGATE", None)
    else:
        os.environ["GALP_PHASE3_NATIVE_DELEGATE"] = "0"


def _tensor_contract(tensor: Any) -> dict[str, Any]:
    payload = tensor.detach().contiguous().cpu().numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _resource_totals(stats: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int(batch.get(key, 0)) for batch in stats)
        for key in EXACT_RESOURCE_KEYS
    }


def _run_contract(DirectDctReader: Any, torch: Any, manifest: Path, native: bool) -> dict[str, Any]:
    _select_backend(native)
    reader = DirectDctReader(manifest)
    pipeline = reader.pipeline(PROFILE)
    torch.cuda.reset_peak_memory_stats()
    pipeline.start(BATCHES, transforms_by_batch=TRANSFORMS)
    prefetched = int(pipeline._native.prefetched_batch_count)
    batches: list[dict[str, Any]] = []
    raw_stats: list[dict[str, Any]] = []
    for expected_ids in BATCHES:
        batch = next(pipeline)
        if batch.global_image_ids != expected_ids:
            raise AssertionError(
                f"batch order mismatch: {batch.global_image_ids} != {expected_ids}"
            )
        batches.append(
            {
                "ids": batch.global_image_ids,
                "y": _tensor_contract(batch.y),
                "cbcr": _tensor_contract(batch.cbcr),
            }
        )
        raw_stats.append(dict(batch._native.execution_stats))

    try:
        next(pipeline)
    except StopIteration:
        pass
    else:
        raise AssertionError("exhausted pipeline did not raise StopIteration")

    torch.cuda.synchronize()
    metrics = dict(pipeline._native.metrics)
    peak_torch_bytes = int(torch.cuda.max_memory_allocated())
    pipeline.close()
    pipeline.close()
    return {
        "batches": batches,
        "prefetched_batch_count": prefetched,
        "resource": _resource_totals(raw_stats),
        "peak_transient_bytes": max(
            int(stats.get("actual_transient_total_allocated_high_water_bytes", 0))
            for stats in raw_stats
        ),
        "peak_torch_bytes": peak_torch_bytes,
        "metrics_schema": metrics.get("schema"),
        "metrics_complete": bool(metrics.get("complete")),
        "metrics_keys": sorted(metrics),
    }


def _run_lifecycle(DirectDctReader: Any, manifest: Path, native: bool) -> None:
    _select_backend(native)
    reader = DirectDctReader(manifest)
    pipeline = reader.pipeline(PROFILE)
    pipeline.start(BATCHES)
    next(pipeline)
    pipeline.close()  # early consumer stop with one pending batch
    pipeline.start([[0, 1, 2, 3]])  # reset/restart after close
    if next(pipeline).global_image_ids != [0, 1, 2, 3]:
        raise AssertionError("reset/restart changed output order")
    pipeline.close()


def _timed_run(DirectDctReader: Any, torch: Any, manifest: Path, native: bool) -> float:
    _select_backend(native)
    reader = DirectDctReader(manifest)
    pipeline = reader.pipeline(PROFILE)
    begin = time.perf_counter()
    pipeline.start(BATCHES)
    for _ in BATCHES:
        batch = next(pipeline)
        _ = batch.y
        _ = batch.cbcr
    torch.cuda.synchronize()
    pipeline.close()
    return (time.perf_counter() - begin) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(args.module_path.resolve()))
    import torch
    from galp.torch import DirectDctReader

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    legacy = _run_contract(DirectDctReader, torch, args.manifest, native=False)
    native = _run_contract(DirectDctReader, torch, args.manifest, native=True)
    if legacy["batches"] != native["batches"]:
        raise AssertionError("Legacy/Native tensor contract mismatch")
    if legacy["prefetched_batch_count"] != native["prefetched_batch_count"]:
        raise AssertionError("Legacy/Native pending depth mismatch")
    if legacy["resource"] != native["resource"]:
        raise AssertionError(
            f"Legacy/Native resource mismatch: legacy={legacy['resource']} native={native['resource']}"
        )
    if native["peak_transient_bytes"] > legacy["peak_transient_bytes"]:
        raise AssertionError("Native transient peak exceeds Legacy")
    if native["peak_torch_bytes"] > legacy["peak_torch_bytes"]:
        raise AssertionError("Native Torch CUDA peak exceeds Legacy")
    if legacy["metrics_schema"] != native["metrics_schema"]:
        raise AssertionError("Legacy/Native metrics schema mismatch")
    if legacy["metrics_keys"] != native["metrics_keys"]:
        raise AssertionError("Legacy/Native metrics key inventory mismatch")

    _run_lifecycle(DirectDctReader, args.manifest, native=False)
    _run_lifecycle(DirectDctReader, args.manifest, native=True)

    for _ in range(args.warmup):
        _timed_run(DirectDctReader, torch, args.manifest, native=False)
        _timed_run(DirectDctReader, torch, args.manifest, native=True)

    legacy_ms: list[float] = []
    native_ms: list[float] = []
    for index in range(args.repeats):
        order = (False, True) if index % 2 == 0 else (True, False)
        for use_native in order:
            elapsed = _timed_run(DirectDctReader, torch, args.manifest, native=use_native)
            (native_ms if use_native else legacy_ms).append(elapsed)

    time_ratio = statistics.mean(native_ms) / statistics.mean(legacy_ms)
    if time_ratio > 1.02:
        raise AssertionError(f"Native delegate performance smoke regressed: ratio={time_ratio}")

    device = torch.cuda.get_device_properties(0)
    payload = {
        "result": "PASS",
        "production_default": "native",
        "rollback_override": "GALP_PHASE3_NATIVE_DELEGATE=0",
        "gpu": {
            "name": device.name,
            "uuid": str(device.uuid),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "correctness": "PASS",
        "lifecycle": "PASS",
        "resource": {"legacy": legacy, "native": native, "result": "PASS"},
        "performance": {
            "legacy_ms": legacy_ms,
            "native_ms": native_ms,
            "native_over_legacy": time_ratio,
            "result": "PASS",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE 3 PRODUCTION DELEGATE VALIDATION PASS")
    print(json.dumps(payload["performance"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
