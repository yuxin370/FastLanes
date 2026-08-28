#!/usr/bin/env python3
"""Test-only Phase-2 Legacy Torch vs Native shadow GPU A/B driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROFILE = "rgbnomore-validation-v1"
BATCH_SIZE = 4
BASE_BATCH_COUNTS = (4, 4, 2)

EXECUTION_EXACT_KEYS = (
    "decode_kernel_launch_count",
    "gather_kernel_launch_count",
    "prefix_gather_kernel_launch_count",
    "cached_gather_kernel_launch_count",
    "materialize_kernel_launch_count",
    "planless_transform_kernel_launch_count",
    "planless_transform_dense_kernel_launch_count",
    "planless_transform_sparse_kernel_launch_count",
    "planless_transform_dense_output_block_count",
    "planless_transform_sparse_output_block_count",
    "planless_transform_selected_coefficient_count",
    "planless_transform_compact_binding_count",
    "planless_transform_dense_binding_equivalent_count",
    "fixed_grid_finalize_kernel_launch_count",
    "project_decoded_ycbcr_grid_launch_count",
    "internal_sync_count",
    "cached_gather_sync_count",
    "decoded_batch_sync_count",
    "cached_gather_event_handoff_count",
    "fixed_grid_round_event_handoff_count",
    "decode_to_transform_event_handoff_count",
    "copy_to_decode_event_handoff_count",
    "workset_upload_dma_count",
    "workset_upload_dma_bytes",
    "workset_upload_count",
    "scratch_upload_count",
    "compressed_payload_bytes_read",
    "selected_compressed_payload_bytes",
    "rowgroup_storage_bytes_read",
    "pread_count",
    "preadv_count",
    "actual_transient_total_used_high_water_bytes",
    "actual_transient_total_allocated_high_water_bytes",
    "rowgroup_count",
    "workset_count",
    "planless_transform_threads_per_cta",
    "planless_transform_max_blocks_per_launch",
    "planless_transform_max_output_blocks_per_launch",
)

ALLOCATION_KEYS = (
    "galp_native_device_peak_in_use_bytes",
    "galp_native_device_allocation_requests",
    "galp_native_device_cuda_allocation_count",
    "galp_native_device_cuda_allocation_bytes",
    "galp_native_pinned_peak_in_use_bytes",
    "galp_native_pinned_allocation_requests",
    "galp_native_pinned_cuda_allocation_count",
    "galp_native_pinned_cuda_allocation_bytes",
    "decode_workset_output_arena_growth_count",
    "decode_workset_chunk_arena_growth_count",
    "compact_batch_buffer_growth_count",
    "planless_axis_program_device_growth_count",
    "planless_axis_program_pinned_growth_count",
)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _tensor_capture(tensor: Any, *, capture_data: bool) -> dict[str, Any]:
    result = {
        "shape": list(tensor.shape),
        "strides": list(tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "bytes": b"",
    }
    if capture_data:
        result["bytes"] = tensor.detach().contiguous().cpu().numpy().tobytes()
    return result


def _capture_legacy_batch(batch: Any, *, capture_data: bool) -> dict[str, Any]:
    raw = batch._native
    y = batch.y
    cbcr = batch.cbcr
    result = {
        "y": _tensor_capture(y, capture_data=capture_data),
        "cbcr": _tensor_capture(cbcr, capture_data=capture_data),
        "global_image_ids": list(batch.global_image_ids),
        "transform_descriptors": _plain(batch.transform_descriptors),
        "image_layouts": _plain(list(raw.image_layouts)),
        "block_metadata": _plain(list(raw.block_metadata)),
        "rowgroups": _plain(list(raw.rowgroups)),
        "selected_coefficients": list(raw.selected_coefficients),
        "block_count": int(raw.block_count),
        "coefficients_per_block": int(raw.coefficients_per_block),
        "cuda_device": int(raw.cuda_device),
        "stats": _plain(dict(raw.execution_stats)),
        "cache": _plain(dict(raw.cache_stats)),
    }
    return result


def _select_requests(reader: Any, profile_id: str) -> tuple[list[list[int]], list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    profile = reader.profile_info(profile_id)
    reference_width_blocks, reference_height_blocks = profile["crop_reference_blocks"]
    crop_width = int(reference_width_blocks) * 8
    crop_height = int(reference_height_blocks) * 8
    selected: list[tuple[int, int, int]] = []
    for image_id in range(min(reader.image_count, 4096)):
        metadata = dict(reader._native.image_metadata(image_id))
        width = int(metadata["image_width"])
        height = int(metadata["image_height"])
        if width >= crop_width and height >= crop_height:
            selected.append((image_id, width, height))
        if len(selected) == sum(BASE_BATCH_COUNTS):
            break
    if len(selected) != sum(BASE_BATCH_COUNTS):
        raise RuntimeError(
            f"manifest has only {len(selected)} images meeting {crop_width}x{crop_height} profile input"
        )

    image_batches: list[list[int]] = []
    transform_batches: list[list[dict[str, Any]]] = []
    native_batches: list[list[dict[str, Any]]] = []
    cursor = 0
    for batch_ordinal, count in enumerate(BASE_BATCH_COUNTS):
        image_batch: list[int] = []
        transforms: list[dict[str, Any]] = []
        native_samples: list[dict[str, Any]] = []
        for sample_ordinal in range(count):
            image_id, width, height = selected[cursor]
            descriptor: dict[str, Any] = {
                "horizontal_flip": bool((cursor + batch_ordinal) % 2),
                "logical_sample_id": f"phase2-{batch_ordinal}-{sample_ordinal}-image-{image_id}",
                "augmentation_key": f"phase2-profile-{profile_id}-batch-{batch_ordinal}",
            }
            if batch_ordinal > 0:
                x = ((width - crop_width) // 2 // 16) * 16
                y = ((height - crop_height) // 2 // 16) * 16
                descriptor["crop"] = {
                    "x": x,
                    "y": y,
                    "width": crop_width,
                    "height": crop_height,
                }
            image_batch.append(image_id)
            transforms.append(descriptor)
            native_samples.append({"image_id": image_id, **descriptor})
            cursor += 1
        image_batches.append(image_batch)
        transform_batches.append(transforms)
        native_batches.append(native_samples)
    return image_batches, transform_batches, native_batches


def _repeat_schedule(
    image_batches: list[list[int]],
    transform_batches: list[list[dict[str, Any]]],
    native_batches: list[list[dict[str, Any]]],
    repeats: int,
) -> tuple[list[list[int]], list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    return (
        [list(batch) for _ in range(repeats) for batch in image_batches],
        [[dict(item) for item in batch] for _ in range(repeats) for batch in transform_batches],
        [[dict(item) for item in batch] for _ in range(repeats) for batch in native_batches],
    )


def _run_legacy(
    manifest: Path,
    module_path: Path,
    profile_id: str,
    dct_coeffs: str,
    image_batches: list[list[int]],
    transform_batches: list[list[dict[str, Any]]],
    *,
    capture_data: bool,
    capture_details: bool,
) -> dict[str, Any]:
    import torch
    from galp.torch import DirectDctReader

    reader = DirectDctReader(manifest, module_path=module_path)
    pipeline = reader.pipeline(profile_id, dct_coeffs=dct_coeffs)
    execution_begin = time.perf_counter()
    pipeline.start(image_batches, transforms_by_batch=transform_batches)
    prefetched_after_reset = int(pipeline._native.prefetched_batch_count)
    batches = []
    latencies = []
    output_ids = []
    final_stats: list[dict[str, Any]] = []
    for expected_ids in image_batches:
        batch_begin = time.perf_counter()
        batch = next(pipeline)
        if capture_details:
            captured = _capture_legacy_batch(batch, capture_data=capture_data)
            batches.append(captured)
            final_stats.append(captured["stats"])
            output_ids.append(captured["global_image_ids"])
        else:
            # execution_stats waits only for this batch's existing completion
            # event.  A process-wide torch.cuda.synchronize() would add a
            # test-harness sync absent from both pipeline implementations.
            stats = _plain(dict(batch._native.execution_stats))
            final_stats.append(stats)
            output_ids.append(list(batch.global_image_ids))
        latencies.append((time.perf_counter() - batch_begin) * 1000.0)
        if output_ids[-1] != expected_ids:
            raise RuntimeError(f"legacy output order mismatch: {output_ids[-1]} != {expected_ids}")
        del batch
    pipeline.close()
    wall_ms = (time.perf_counter() - execution_begin) * 1000.0
    return {
        "batches": batches,
        "stats": final_stats,
        "batch_latency_ms": latencies,
        "execution_wall_ms": wall_ms,
        "prefetched_after_reset": prefetched_after_reset,
        "output_ids": output_ids,
    }


def _run_native(
    native_module: Any,
    manifest: Path,
    profile_id: str,
    dct_coeffs: str,
    native_batches: list[list[dict[str, Any]]],
    *,
    capture_data: bool,
    capture_details: bool,
    trace_enabled: bool,
) -> dict[str, Any]:
    return _plain(
        native_module.run_native(
            str(manifest),
            profile_id,
            dct_coeffs,
            native_batches,
            BATCH_SIZE,
            trace_enabled=trace_enabled,
            capture_data=capture_data,
            capture_details=capture_details,
        )
    )


def _assert_equal(name: str, legacy: Any, native: Any) -> None:
    if legacy != native:
        raise AssertionError(f"{name} mismatch\nlegacy={legacy!r}\nnative={native!r}")


def _assert_plan_and_trace(native: dict[str, Any], batch_count: int) -> None:
    trace = native["trace"]
    identities = native["production_plan_identities"]
    if len(identities) != batch_count:
        raise AssertionError("production plan identity count mismatch")
    forbidden = {"failed", "cancelled"}
    if forbidden.intersection(event["stage"] for event in trace):
        raise AssertionError(f"native trace contains failure/cancellation: {trace}")
    expected_chain = (
        "request_accepted",
        "preparing",
        "plan_ready",
        "staged",
        "awaiting_predecessor",
        "read_started",
        "submitted",
        "completed",
    )
    for ordinal in range(batch_count):
        positions: dict[str, int] = {}
        for position, event in enumerate(trace):
            if event["request_ordinal"] == ordinal:
                positions.setdefault(event["stage"], position)
        missing = set(expected_chain).difference(positions)
        if missing:
            raise AssertionError(f"request {ordinal} missing trace stages {sorted(missing)}")
        ordered = [positions[stage] for stage in expected_chain]
        if ordered != sorted(ordered):
            raise AssertionError(f"request {ordinal} trace order is invalid: {positions}")
        gate = positions.get("gate_release_requested")
        if gate is None or gate >= positions["submitted"]:
            raise AssertionError(
                f"request {ordinal} gate release was not requested before submission"
            )
        plan_event = trace[positions["plan_ready"]]
        _assert_equal(
            f"request {ordinal} production/native plan hash",
            identities[ordinal]["plan_identity_hash"],
            plan_event["plan_identity_hash"],
        )
        _assert_equal(
            f"request {ordinal} production/native I/O hash",
            identities[ordinal]["io_identity_hash"],
            plan_event["io_identity_hash"],
        )
    submitted = [event for event in trace if event["stage"] == "submitted"]
    completed = [event for event in trace if event["stage"] == "completed"]
    _assert_equal("submission request order", list(range(batch_count)), [e["request_ordinal"] for e in submitted])
    _assert_equal("completion request order", list(range(batch_count)), [e["request_ordinal"] for e in completed])
    _assert_equal("submission ordinals", list(range(1, batch_count + 1)), [e["submission_ordinal"] for e in submitted])
    _assert_equal("completion ordinals", list(range(1, batch_count + 1)), [e["completion_ordinal"] for e in completed])
    if trace[-1]["stage"] != "closed":
        raise AssertionError("native trace does not end in close")


def _run_correctness(
    manifest: Path,
    module_path: Path,
    profile_id: str,
    dct_coeffs: str,
    native_module: Any,
    image_batches: list[list[int]],
    transform_batches: list[list[dict[str, Any]]],
    native_batches: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    from galp.torch import DirectDctReader

    plan_reader = DirectDctReader(manifest, module_path=module_path)
    previews = [
        _plain(
            plan_reader._native.plan(
                image_ids, profile_id, transforms, dct_coeffs=dct_coeffs
            )
        )
        for image_ids, transforms in zip(image_batches, transform_batches, strict=True)
    ]
    legacy = _run_legacy(
        manifest,
        module_path,
        profile_id,
        dct_coeffs,
        image_batches,
        transform_batches,
        capture_data=True,
        capture_details=True,
    )
    native = _run_native(
        native_module,
        manifest,
        profile_id,
        dct_coeffs,
        native_batches,
        capture_data=True,
        capture_details=True,
        trace_enabled=True,
    )
    _assert_equal("batch count", len(legacy["batches"]), len(native["batches"]))
    for index, (legacy_batch, native_batch, preview) in enumerate(
        zip(legacy["batches"], native["batches"], previews, strict=True)
    ):
        for key in (
            "global_image_ids",
            "transform_descriptors",
            "image_layouts",
            "block_metadata",
            "rowgroups",
            "selected_coefficients",
            "block_count",
            "coefficients_per_block",
            "cuda_device",
        ):
            _assert_equal(f"batch {index} {key}", legacy_batch[key], native_batch[key])
        for tensor_name in ("y", "cbcr"):
            for key in ("shape", "strides", "dtype", "bytes"):
                _assert_equal(
                    f"batch {index} {tensor_name}.{key}",
                    legacy_batch[tensor_name][key],
                    native_batch[tensor_name][key],
                )
        for key in EXECUTION_EXACT_KEYS:
            _assert_equal(
                f"batch {index} stats.{key}",
                legacy_batch["stats"][key],
                native_batch["stats"][key],
            )
        _assert_equal(f"batch {index} preview image layouts", preview["image_layouts"], native_batch["image_layouts"])
        _assert_equal(f"batch {index} preview block metadata", preview["block_metadata"], native_batch["block_metadata"])
        _assert_equal(f"batch {index} preview rowgroups", preview["rowgroups"], native_batch["rowgroups"])
        _assert_equal(
            f"batch {index} preview selected coefficients",
            preview["selected_coefficients"],
            native_batch["selected_coefficients"],
        )
        _assert_equal(f"batch {index} preview Y shape", preview["y_shape"], native_batch["y"]["shape"])
        _assert_equal(f"batch {index} preview CbCr shape", preview["cbcr_shape"], native_batch["cbcr"]["shape"])

    _assert_plan_and_trace(native, len(image_batches))
    _assert_equal("legacy initial bounded depth", min(2, len(image_batches)), legacy["prefetched_after_reset"])
    _assert_equal("native initial bounded depth", min(2, len(image_batches)), native["prefetched_after_reset"])
    _assert_equal("native max pending depth", 2, native["state_before_close"]["max_pending_count"])
    _assert_equal("native completed count", len(image_batches), native["state_before_close"]["completed_request_count"])
    _assert_equal("native drained state", "drained", native["state_before_close"]["lifecycle"])
    _assert_equal("native close cancellation", 0, native["close_cancelled"])
    _assert_equal("native closed state", "closed", native["state_after_close"]["lifecycle"])

    tensor_digests = []
    for batch in legacy["batches"]:
        tensor_digests.append(
            {
                "y": hashlib.sha256(batch["y"]["bytes"]).hexdigest(),
                "cbcr": hashlib.sha256(batch["cbcr"]["bytes"]).hexdigest(),
            }
        )
    return {
        "result": "PASS",
        "profile": profile_id,
        "dct_coeffs": dct_coeffs,
        "batch_sizes": [len(batch) for batch in image_batches],
        "sample_ids": image_batches,
        "tensor_digests": tensor_digests,
        "tensor_comparison": "bit-exact",
        "plan_hashes": native["production_plan_identities"],
        "trace_event_count": len(native["trace"]),
        "legacy_prefetched_after_reset": legacy["prefetched_after_reset"],
        "native_state_before_close": native["state_before_close"],
        "execution_exact_keys": list(EXECUTION_EXACT_KEYS),
    }


def _aggregate_stats(stats: list[dict[str, Any]]) -> dict[str, int]:
    peak_keys = {
        "galp_native_device_peak_in_use_bytes",
        "galp_native_pinned_peak_in_use_bytes",
        "actual_transient_total_used_high_water_bytes",
        "actual_transient_total_allocated_high_water_bytes",
    }
    keys = set(EXECUTION_EXACT_KEYS).union(ALLOCATION_KEYS)
    return {
        key: int(max((batch[key] for batch in stats), default=0) if key in peak_keys else sum(batch[key] for batch in stats))
        for key in keys
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _summarize_run(result: dict[str, Any], image_count: int) -> dict[str, Any]:
    wall_ms = float(result["execution_wall_ms"])
    latencies = [float(value) for value in result["batch_latency_ms"]]
    stats = result.get("stats") or [batch["stats"] for batch in result["batches"]]
    return {
        "throughput_images_per_s": image_count / (wall_ms / 1000.0),
        "wall_ms": wall_ms,
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "p99_ms": _percentile(latencies, 0.99),
        "resource": _aggregate_stats(stats),
    }


def _ratio_ci(native_values: list[float], legacy_values: list[float]) -> dict[str, float]:
    logs = [math.log(native / legacy) for native, legacy in zip(native_values, legacy_values, strict=True)]
    mean = statistics.mean(logs)
    if len(logs) == 1:
        half = math.inf
    else:
        t_critical = {5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(len(logs), 2.228)
        half = t_critical * statistics.stdev(logs) / math.sqrt(len(logs))
    return {"ratio": math.exp(mean), "lower_95": math.exp(mean - half), "upper_95": math.exp(mean + half)}


def _run_benchmark(
    manifest: Path,
    module_path: Path,
    profile_id: str,
    dct_coeffs: str,
    native_module: Any,
    base_image_batches: list[list[int]],
    base_transform_batches: list[list[dict[str, Any]]],
    base_native_batches: list[list[dict[str, Any]]],
    warmup: int,
    repeats: int,
    schedule_repeats: int,
) -> dict[str, Any]:
    image_batches, transform_batches, native_batches = _repeat_schedule(
        base_image_batches, base_transform_batches, base_native_batches, schedule_repeats
    )
    image_count = sum(len(batch) for batch in image_batches)

    def one(kind: str) -> dict[str, Any]:
        cpu_begin = time.process_time()
        if kind == "legacy":
            raw = _run_legacy(
                manifest,
                module_path,
                profile_id,
                dct_coeffs,
                image_batches,
                transform_batches,
                capture_data=False,
                capture_details=False,
            )
        else:
            raw = _run_native(
                native_module,
                manifest,
                profile_id,
                dct_coeffs,
                native_batches,
                capture_data=False,
                capture_details=False,
                trace_enabled=False,
            )
        summary = _summarize_run(raw, image_count)
        summary["host_cpu_ms"] = (time.process_time() - cpu_begin) * 1000.0
        return summary

    for _ in range(warmup):
        one("legacy")
        one("native")

    measured = {"legacy": [], "native": []}
    for repeat in range(repeats):
        order = ("legacy", "native") if repeat % 2 == 0 else ("native", "legacy")
        pair: dict[str, Any] = {}
        for kind in order:
            pair[kind] = one(kind)
        measured["legacy"].append(pair["legacy"])
        measured["native"].append(pair["native"])

    result: dict[str, Any] = {
        "warmup_per_path": warmup,
        "measured_repeats": repeats,
        "schedule_batch_count": len(image_batches),
        "schedule_image_count": image_count,
        "runs": measured,
    }
    for metric in ("throughput_images_per_s", "p50_ms", "p95_ms", "p99_ms", "host_cpu_ms"):
        result[f"{metric}_ci"] = _ratio_ci(
            [run[metric] for run in measured["native"]],
            [run[metric] for run in measured["legacy"]],
        )
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--dct-coeffs", default="all")
    parser.add_argument("--mode", choices=("correctness", "legacy", "native", "benchmark"), default="correctness")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--schedule-repeats", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(args.module_path.resolve()))
    import torch
    import _galp_phase2_native_ab as native_module
    from galp.torch import DirectDctReader

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    selector = DirectDctReader(args.manifest, module_path=args.module_path)
    image_batches, transform_batches, native_batches = _select_requests(selector, args.profile)
    del selector

    if args.mode == "correctness":
        payload = _run_correctness(
            args.manifest,
            args.module_path,
            args.profile,
            args.dct_coeffs,
            native_module,
            image_batches,
            transform_batches,
            native_batches,
        )
    elif args.mode == "legacy":
        raw = _run_legacy(
            args.manifest,
            args.module_path,
            args.profile,
            args.dct_coeffs,
            image_batches,
            transform_batches,
            capture_data=False,
            capture_details=False,
        )
        payload = {"mode": "legacy", **_summarize_run(raw, sum(map(len, image_batches)))}
    elif args.mode == "native":
        raw = _run_native(
            native_module,
            args.manifest,
            args.profile,
            args.dct_coeffs,
            native_batches,
            capture_data=False,
            capture_details=False,
            trace_enabled=False,
        )
        payload = {"mode": "native", **_summarize_run(raw, sum(map(len, image_batches)))}
    else:
        payload = _run_benchmark(
            args.manifest,
            args.module_path,
            args.profile,
            args.dct_coeffs,
            native_module,
            image_batches,
            transform_batches,
            native_batches,
            args.warmup,
            args.repeats,
            args.schedule_repeats,
        )

    device = torch.cuda.get_device_properties(0)
    payload = {
        "environment": {
            "cuda_device_name": device.name,
            "cuda_device_uuid": str(device.uuid),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
        "profile": args.profile,
        "dct_coeffs": args.dct_coeffs,
        **payload,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
