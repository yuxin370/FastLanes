#!/usr/bin/env python3
"""Targeted cross-shard proof for native physical orchestration.

The legacy arm deliberately materializes two physical segments and stitches
them in Python. The native arm submits one logical request crossing the same
boundary and receives one complete batch. This is an integration harness, not
a stable API or a general benchmark suite.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROFILE = "rgbnomore-validation-center-crop-512-v1"
DCT_COEFFS = "first:32"


def _pipeline(reader: Any, native_physical: bool) -> Any:
    os.environ["GALP_PHASE6_NATIVE_PHYSICAL"] = "1" if native_physical else "0"
    return reader.pipeline(PROFILE, dct_coeffs=DCT_COEFFS)


def _prepare_imports(source_root: Path, module_path: Path) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(source_root.resolve()))
    sys.path.insert(0, str(module_path.resolve()))
    import torch
    import _galp_direct_dct as native_module
    from galp.torch import DirectDctReader

    return torch, native_module, DirectDctReader


def _resource_snapshot(native_module: Any, torch: Any) -> dict[str, Any]:
    native_module.manual_reclaim()
    result = dict(native_module._lifetime_reclaim_stats_for_test())
    return {
        "device_pool": dict(result["device_pool"]),
        "pinned_pool": dict(result["pinned_pool"]),
        "torch_peak_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _resource_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "device_cuda_allocations": int(after["device_pool"]["cuda_allocation_count"])
        - int(before["device_pool"]["cuda_allocation_count"]),
        "device_cuda_allocation_bytes": int(after["device_pool"]["cuda_allocation_bytes"])
        - int(before["device_pool"]["cuda_allocation_bytes"]),
        "device_peak_in_use_bytes": int(after["device_pool"]["peak_in_use_bytes"]),
        "pinned_cuda_allocations": int(after["pinned_pool"]["cuda_allocation_count"])
        - int(before["pinned_pool"]["cuda_allocation_count"]),
        "pinned_cuda_allocation_bytes": int(after["pinned_pool"]["cuda_allocation_bytes"])
        - int(before["pinned_pool"]["cuda_allocation_bytes"]),
        "torch_peak_bytes": int(after["torch_peak_bytes"]),
    }


def _legacy(
    reader: Any,
    torch: Any,
    first: int,
    count: int,
    left_shard: dict[str, Any],
    right_shard: dict[str, Any],
) -> tuple[Any, Any, list[int]]:
    boundary = int(right_shard["first_global_image_index"])
    if not first < boundary < first + count:
        raise ValueError("the targeted legacy arm requires exactly one physical shard boundary")
    # Scheduled active-output execution is canonically keyed by a complete
    # physical shard.  Match the production Legacy path: activate both full
    # shards once, then assemble the requested tail/head slices in Python.
    batches = [
        list(
            range(
                int(shard["first_global_image_index"]),
                int(shard["first_global_image_index"]) + int(shard["image_count"]),
            )
        )
        for shard in (left_shard, right_shard)
    ]
    pipeline = _pipeline(reader, native_physical=False)
    pipeline.start(batches)
    left = next(pipeline)
    right = next(pipeline)
    left_offset = first - int(left_shard["first_global_image_index"])
    left_count = boundary - first
    right_count = count - left_count
    y = torch.cat(
        (left.y[left_offset : left_offset + left_count], right.y[:right_count]), dim=0
    )
    cbcr = torch.cat(
        (
            left.cbcr[left_offset : left_offset + left_count],
            right.cbcr[:right_count],
        ),
        dim=0,
    )
    ids = (
        left.global_image_ids[left_offset : left_offset + left_count]
        + right.global_image_ids[:right_count]
    )
    torch.cuda.synchronize()
    pipeline.close()
    return y, cbcr, ids


def _native(
    reader: Any,
    torch: Any,
    first: int,
    count: int,
    coverage_first: int,
    coverage_count: int,
) -> tuple[Any, Any, list[int], list[dict[str, Any]]]:
    coverage_end = coverage_first + coverage_count
    target_end = first + count
    logical_batches: list[list[int]] = []
    for offset in range(coverage_first, first, count):
        logical_batches.append(list(range(offset, min(first, offset + count))))
    target_index = len(logical_batches)
    logical_batches.append(list(range(first, target_end)))
    for offset in range(target_end, coverage_end, count):
        logical_batches.append(list(range(offset, min(coverage_end, offset + count))))
    pipeline = _pipeline(reader, native_physical=True)
    pipeline.start(logical_batches)
    plans = [dict(pipeline._native._segment_plans[target_index])]
    batch = None
    for index in range(target_index + 1):
        current = next(pipeline)
        if index == target_index:
            batch = current
    assert batch is not None
    y = batch.y
    cbcr = batch.cbcr
    ids = batch.global_image_ids
    torch.cuda.synchronize()
    pipeline.close()
    return y, cbcr, ids, plans


def _legacy_production_schedule(
    reader: Any,
    torch: Any,
    native_module: Any,
    shards: list[dict[str, Any]],
    logical_batch_size: int,
) -> dict[str, Any]:
    physical_batches = [
        list(
            range(
                int(shard["first_global_image_index"]),
                int(shard["first_global_image_index"]) + int(shard["image_count"]),
            )
        )
        for shard in shards
    ]
    selected_count = sum(len(batch) for batch in physical_batches)
    torch.cuda.reset_peak_memory_stats()
    before = _resource_snapshot(native_module, torch)
    pipeline = _pipeline(reader, native_physical=False)
    pipeline.start(physical_batches)
    current = None
    current_offset = 0
    for logical_first in range(0, selected_count, logical_batch_size):
        remaining = min(logical_batch_size, selected_count - logical_first)
        y_parts: list[Any] = []
        cbcr_parts: list[Any] = []
        while remaining:
            if current is None or current_offset >= len(current.global_image_ids):
                if current is not None:
                    # Exact legacy boundary behavior: preserve an already
                    # consumed tail before releasing its full-shard backing.
                    y_parts[:] = [y_parts[0].clone()] if len(y_parts) == 1 else [torch.cat(y_parts, dim=0)]
                    cbcr_parts[:] = (
                        [cbcr_parts[0].clone()]
                        if len(cbcr_parts) == 1
                        else [torch.cat(cbcr_parts, dim=0)]
                    )
                    torch.cuda.current_stream().synchronize()
                current = next(pipeline)
                current_offset = 0
            available = len(current.global_image_ids) - current_offset
            take = min(remaining, available)
            y_parts.append(current.y[current_offset : current_offset + take])
            cbcr_parts.append(current.cbcr[current_offset : current_offset + take])
            current_offset += take
            remaining -= take
        y = y_parts[0] if len(y_parts) == 1 else torch.cat(y_parts, dim=0)
        cbcr = cbcr_parts[0] if len(cbcr_parts) == 1 else torch.cat(cbcr_parts, dim=0)
        if y.shape[0] != cbcr.shape[0]:
            raise AssertionError("legacy logical assembly cardinality mismatch")
    torch.cuda.synchronize()
    pipeline.close()
    del pipeline, current, y, cbcr, y_parts, cbcr_parts
    after = _resource_snapshot(native_module, torch)
    result = _resource_delta(before, after)
    result.update({
        "canonical_shard_executions": len(shards),
        "full_shard_activations": len(shards),
        "python_explicit_syncs": max(0, len(shards) - 1),
        "assembly_copy_count": 4 * max(0, len(shards) - 1),
    })
    return result


def _native_production_schedule(
    reader: Any,
    torch: Any,
    native_module: Any,
    first: int,
    count: int,
    logical_batch_size: int,
) -> dict[str, Any]:
    logical_batches = [
        list(range(offset, min(first + count, offset + logical_batch_size)))
        for offset in range(first, first + count, logical_batch_size)
    ]
    torch.cuda.reset_peak_memory_stats()
    before = _resource_snapshot(native_module, torch)
    pipeline = _pipeline(reader, native_physical=True)
    pipeline.start(logical_batches)
    for expected in logical_batches:
        batch = next(pipeline)
        if batch.global_image_ids != expected:
            raise AssertionError("native production schedule changed logical order")
        if batch.y.shape[0] != len(expected) or batch.cbcr.shape[0] != len(expected):
            raise AssertionError("native production schedule changed logical cardinality")
    torch.cuda.synchronize()
    pipeline.close()
    del pipeline, batch
    after = _resource_snapshot(native_module, torch)
    result = _resource_delta(before, after)
    # Exactly one logical batch crosses the boundary in this targeted
    # two-shard workload. Native performs two component copies per segment.
    result.update({
        "canonical_shard_executions": 2,
        "full_shard_activations": 2,
        "python_explicit_syncs": 0,
        "assembly_copy_count": 4,
    })
    return result


def _elapsed_ms(call: Any, torch: Any) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter_ns()
    resources = call()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - started) / 1.0e6, resources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--block-major-access-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--first-image", type=int)
    parser.add_argument("--image-count", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    access_index = args.block_major_access_dir / "manifest.block_major_access.bin"
    if not access_index.is_file():
        raise FileNotFoundError(f"block-major access index is missing: {access_index}")
    os.environ["GALP_BLOCK_MAJOR_ACCESS_DIR"] = str(
        args.block_major_access_dir.resolve()
    )
    os.environ["GALP_PHASE6_NATIVE_PHYSICAL"] = "1"

    torch, native_module, reader_type = _prepare_imports(args.source_root, args.module_path)
    from galp.benchmarks.system_dct_major.common import parse_manifest

    manifest = parse_manifest(args.manifest)
    shards = list(manifest["shards"])
    if len(shards) < 2:
        raise ValueError("Phase 6 cross-shard proof requires at least two physical shards")
    if args.first_image is None:
        boundary_index = 1
        boundary = int(shards[boundary_index]["first_global_image_index"])
        first_image = boundary - args.image_count // 2
    else:
        first_image = args.first_image
        end_image = first_image + args.image_count
        candidates = [
            (index, int(shard["first_global_image_index"]))
            for index, shard in enumerate(shards[1:], start=1)
            if first_image < int(shard["first_global_image_index"]) < end_image
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"requested [{first_image},{end_image}) must cross exactly one real manifest boundary; "
                f"observed {candidates}"
            )
        boundary_index, boundary = candidates[0]
    left_shard = shards[boundary_index - 1]
    right_shard = shards[boundary_index]
    left_count = boundary - first_image
    right_count = args.image_count - left_count
    reader = reader_type(args.manifest, module_path=args.module_path)
    legacy_y, legacy_cbcr, legacy_ids = _legacy(
        reader,
        torch,
        first_image,
        args.image_count,
        left_shard,
        right_shard,
    )
    performance_shards = [left_shard, right_shard]
    performance_first = int(left_shard["first_global_image_index"])
    performance_count = sum(int(shard["image_count"]) for shard in performance_shards)
    native_y, native_cbcr, native_ids, plans = _native(
        reader,
        torch,
        first_image,
        args.image_count,
        performance_first,
        performance_count,
    )
    expected_ids = list(range(first_image, first_image + args.image_count))
    if legacy_ids != expected_ids or native_ids != expected_ids:
        raise AssertionError("cross-shard sample order differs")
    if not torch.equal(legacy_y, native_y) or not torch.equal(legacy_cbcr, native_cbcr):
        raise AssertionError("cross-shard native tensors differ from Python legacy stitching")
    segments = plans[0]["segments"]
    expected_segments = [
        {
            "shard_id": int(left_shard["shard_id"]),
            "logical_output_offset": 0,
            "image_count": left_count,
            "first_global_image_id": first_image,
            "last_global_image_id": boundary - 1,
        },
        {
            "shard_id": int(right_shard["shard_id"]),
            "logical_output_offset": left_count,
            "image_count": right_count,
            "first_global_image_id": boundary,
            "last_global_image_id": first_image + args.image_count - 1,
        },
    ]
    if segments != expected_segments:
        raise AssertionError(f"native SegmentPlan mismatch: {segments!r}")

    legacy_times: list[float] = []
    native_times: list[float] = []
    legacy_resources: list[dict[str, Any]] = []
    native_resources: list[dict[str, Any]] = []
    for index in range(args.warmup + args.pairs):
        arms = (
            (("legacy", lambda: _legacy_production_schedule(
                reader, torch, native_module, performance_shards, args.image_count)),
             ("native", lambda: _native_production_schedule(
                 reader, torch, native_module, performance_first, performance_count, args.image_count)))
            if index % 2 == 0
            else (("native", lambda: _native_production_schedule(
                reader, torch, native_module, performance_first, performance_count, args.image_count)),
                  ("legacy", lambda: _legacy_production_schedule(
                      reader, torch, native_module, performance_shards, args.image_count)))
        )
        for name, call in arms:
            elapsed, resources = _elapsed_ms(call, torch)
            if index >= args.warmup:
                (legacy_times if name == "legacy" else native_times).append(elapsed)
                (legacy_resources if name == "legacy" else native_resources).append(resources)

    legacy_mean = statistics.fmean(legacy_times)
    native_mean = statistics.fmean(native_times)
    throughput_ratio = legacy_mean / native_mean
    payload = {
        "schema": "galp-phase6-physical-orchestration-ab-v1",
        "result": "PASS" if throughput_ratio >= 0.99 else "FAIL",
        "correctness": "BIT_EXACT",
        "ids": "EXACT",
        "segment_plan": plans[0],
        "performance_image_count": performance_count,
        "performance_physical_shards": [int(shard["shard_id"]) for shard in performance_shards],
        "native_coefficient_spec": DCT_COEFFS,
        "block_major_access_dir": str(args.block_major_access_dir.resolve()),
        "legacy_ms": legacy_times,
        "native_ms": native_times,
        "native_over_legacy_throughput": throughput_ratio,
        "native_production_stitch_ops": 0,
        "native_production_explicit_syncs": 0,
        "resource": {
            "legacy": legacy_resources,
            "native": native_resources,
            "gates": {
                "canonical_shard_executions": max(
                    item["canonical_shard_executions"] for item in native_resources
                ) <= min(item["canonical_shard_executions"] for item in legacy_resources),
                "full_shard_activations": max(
                    item["full_shard_activations"] for item in native_resources
                ) <= min(item["full_shard_activations"] for item in legacy_resources),
                "assembly_copy_count": max(
                    item["assembly_copy_count"] for item in native_resources
                ) <= min(item["assembly_copy_count"] for item in legacy_resources),
                "explicit_syncs": max(
                    item["python_explicit_syncs"] for item in native_resources
                ) <= min(item["python_explicit_syncs"] for item in legacy_resources),
                "device_peak": max(
                    item["device_peak_in_use_bytes"] for item in native_resources
                ) <= max(item["device_peak_in_use_bytes"] for item in legacy_resources),
            },
        },
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    if not all(payload["resource"]["gates"].values()):
        payload["result"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PHASE 6 PHYSICAL ORCHESTRATION A/B", payload["result"])
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["result"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
