#!/usr/bin/env python3
"""Validate DCT-major correctness, physical pushdown evidence, and performance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common import (
    PIPELINE_RESULT_SCHEMA,
    SUMMARY_SCHEMA,
    distribution,
    load_contract,
    load_sample_manifest,
    read_json,
    sample_trace,
    selected_batches,
    sha256_file,
    sha256_json,
    write_json,
)


def _require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def _compact_plan_memory_is_consistent(native: dict[str, Any]) -> bool:
    """Validate cumulative compact-plan bytes against the per-segment peak."""

    segment_count = int(native.get("segment_count", 0))
    compact_bytes = int(native.get("compact_plan_bytes", 0))
    compact_peak = int(native.get("compact_plan_peak_bytes", 0))
    return (
        segment_count > 0
        and compact_bytes > 0
        and compact_peak > 0
        and compact_bytes <= compact_peak * segment_count
    )


def _load_result(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"pipeline result is not an object: {path}")
    return payload


def _hot_repeats(contract: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    repeats = list(result["repeats"])
    if contract["execution"].get("aggregate_exclude_first_repeat") and len(repeats) > 1:
        return repeats[1:]
    return repeats


def _sum_hot_native(contract: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    records = _hot_repeats(contract, result)
    keys = {
        key
        for repeat in records
        for key, value in repeat.get("native_totals", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return {
        key: sum(float(repeat.get("native_totals", {}).get(key, 0.0)) for repeat in records) / len(records)
        for key in keys
    }


def _aggregate(contract: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    cold = result["repeats"][0]
    hot = _hot_repeats(contract, result)
    throughput = [float(item["throughput_images_per_s"]) for item in hot]
    steady = [
        float(item["steady_throughput_images_per_s"])
        for item in hot
        if item.get("steady_throughput_images_per_s") is not None
    ]

    def native_distribution(key: str) -> dict[str, float | int] | None:
        values = [
            float(item["native_totals"][key])
            for item in hot
            if isinstance(item.get("native_totals", {}).get(key), (int, float))
        ]
        return distribution(values) if values else None

    def process_io_distribution(key: str) -> dict[str, float | int] | None:
        values = [
            float(item["process_io"][key])
            for item in hot
            if isinstance(item.get("process_io"), dict)
            and isinstance(item["process_io"].get(key), (int, float))
        ]
        return distribution(values) if len(values) == len(hot) else None

    controlled_io = contract["execution"].get("cold_protocol") == "controlled-io"
    cold_throughput = (
        float(cold["repeat_scope_throughput_images_per_s"])
        if controlled_io
        else float(cold.get("process_scope_throughput_images_per_s") or cold["throughput_images_per_s"])
    )
    cold_ttfb = (
        float(cold["repeat_scope_time_to_first_batch_ms"])
        if controlled_io
        else float(cold.get("process_scope_time_to_first_batch_ms") or cold["time_to_first_batch_ms"])
    )
    cold_seconds = float(cold["images"]) / cold_throughput
    cold_input_ready_stall_ms = float(
        cold.get("native_totals", {}).get("prefetch_consumer_wait_ms", 0.0)
    )

    def input_ready_stall_percent(item: dict[str, Any]) -> float:
        seconds = float(item.get("seconds", 0.0))
        wait_ms = float(item.get("native_totals", {}).get("prefetch_consumer_wait_ms", 0.0))
        return 100.0 * wait_ms / (seconds * 1000.0) if seconds > 0.0 else 0.0

    return {
        "pipeline": result["pipeline"],
        "domain": result["domain"],
        "cold_start": {
            "repeat": int(cold["repeat"]),
            "scope": "controlled-repeat-scope" if controlled_io else "process-scope",
            "seconds": cold_seconds,
            "throughput_images_per_s": cold_throughput,
            "time_to_first_batch_ms": cold_ttfb,
            "batch_loop_throughput_images_per_s": float(cold["throughput_images_per_s"]),
            "batch_loop_time_to_first_batch_ms": float(cold["time_to_first_batch_ms"]),
            "first_shard_ready_ms": (
                float(
                    cold.get("first_shard_ready_ms")
                    if controlled_io
                    else cold.get("process_scope_first_shard_ready_ms")
                    or cold.get("first_shard_ready_ms")
                )
                if cold.get("first_shard_ready_ms") is not None
                or (not controlled_io and cold.get("process_scope_first_shard_ready_ms") is not None)
                else None
            ),
            "steady_throughput_images_per_s": (
                float(cold["steady_throughput_images_per_s"])
                if cold.get("steady_throughput_images_per_s") is not None
                else None
            ),
            "loader_mean_ms": float(cold["loader_submit_ms"]["mean"]),
            "h2d_mean_ms": float(cold["top_level_h2d_ms"]["mean"]),
            "model_mean_ms": float(cold["model_ms"]["mean"]),
            "batch_latency_p50_ms": float(cold["latency_ms"]["p50"]),
            "batch_latency_p95_ms": float(cold["latency_ms"]["p95"]),
            "gpu_utilization_mean_percent": (
                float(cold["gpu_utilization_percent"]["mean"])
                if int(cold.get("gpu_utilization_percent", {}).get("count", 0)) > 0
                else None
            ),
            "native_totals": cold.get("native_totals", {}),
            "process_io": cold.get("process_io"),
            "input_ready_stall_ms": cold_input_ready_stall_ms,
            "input_ready_stall_percent": (
                100.0 * cold_input_ready_stall_ms / (cold_seconds * 1000.0)
                if cold_seconds > 0.0
                else 0.0
            ),
        },
        "throughput_images_per_s": distribution(throughput),
        "throughput_endpoint_drift": (
            abs(throughput[-1] - throughput[0]) / throughput[0]
            if len(throughput) > 1 and throughput[0]
            else 0.0
        ),
        "time_to_first_batch_ms": distribution([float(item["time_to_first_batch_ms"]) for item in hot]),
        "steady_throughput_images_per_s": distribution(steady) if steady else None,
        "latency_mean_ms": distribution([float(item["latency_ms"]["mean"]) for item in hot]),
        "batch_latency_p50_ms": distribution([float(item["latency_ms"]["p50"]) for item in hot]),
        "batch_latency_p95_ms": distribution([float(item["latency_ms"]["p95"]) for item in hot]),
        "loader_mean_ms": distribution([float(item["loader_submit_ms"]["mean"]) for item in hot]),
        "process_io_logical_read_bytes": process_io_distribution("logical_read_bytes"),
        "process_io_storage_read_bytes": process_io_distribution("storage_read_bytes"),
        "process_io_read_syscalls": process_io_distribution("read_syscalls"),
        "input_ready_stall_percent": distribution(
            [input_ready_stall_percent(item) for item in hot]
        ),
        "h2d_mean_ms": distribution([float(item["top_level_h2d_ms"]["mean"]) for item in hot]),
        "model_mean_ms": distribution([float(item["model_ms"]["mean"]) for item in hot]),
        "host_peak_rss_bytes": distribution([float(item["host_peak_rss_bytes"]) for item in hot]),
        "peak_torch_gpu_allocated_bytes": distribution(
            [float(item["peak_torch_gpu_allocated_bytes"]) for item in hot]
        ),
        "peak_torch_gpu_reserved_bytes": distribution(
            [float(item["peak_torch_gpu_reserved_bytes"]) for item in hot]
        ),
        "gpu_utilization_percent": distribution(
            [
                float(item["gpu_utilization_percent"]["mean"])
                for item in hot
                if int(item.get("gpu_utilization_percent", {}).get("count", 0)) > 0
            ]
        )
        if any(int(item.get("gpu_utilization_percent", {}).get("count", 0)) > 0 for item in hot)
        else None,
        "galp_native_device_peak_in_use_bytes": native_distribution(
            "galp_native_device_peak_in_use_bytes"
        ),
        "galp_native_pinned_peak_in_use_bytes": native_distribution(
            "galp_native_pinned_peak_in_use_bytes"
        ),
        "compact_plan_peak_bytes": native_distribution("compact_plan_peak_bytes"),
        "native_hot_mean": _sum_hot_native(contract, result),
    }


def _artifact(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files if name != "metadata_json"}


def _array_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_match": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    if left.size == 0:
        return {"shape_match": True, "max_abs": 0.0, "mean_abs": 0.0, "cosine_mean": 1.0}
    lhs = left.astype(np.float64)
    rhs = right.astype(np.float64)
    delta = np.abs(lhs - rhs)
    flat_lhs = lhs.reshape(lhs.shape[0], -1)
    flat_rhs = rhs.reshape(rhs.shape[0], -1)
    denominator = np.linalg.norm(flat_lhs, axis=1) * np.linalg.norm(flat_rhs, axis=1)
    cosine = np.divide(
        np.sum(flat_lhs * flat_rhs, axis=1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator != 0,
    )
    return {
        "shape_match": True,
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "cosine_mean": float(cosine.mean()),
    }


def _semantic_compare(
    left_name: str,
    right_name: str,
    left: dict[str, Any],
    right: dict[str, Any],
    contract: dict[str, Any],
    *,
    strict: bool,
    failures: list[str],
) -> dict[str, Any]:
    lhs = _artifact(left["semantic_artifact"])
    rhs = _artifact(right["semantic_artifact"])
    thresholds = contract["semantic_validation"]
    result: dict[str, Any] = {
        "pipelines": [left_name, right_name],
        "enforcement": "strict" if strict else "diagnostic",
        "inputs": {},
    }
    identity_ok = np.array_equal(lhs.get("ordinals"), rhs.get("ordinals")) and np.array_equal(
        lhs.get("labels"), rhs.get("labels")
    )
    result["sample_identity_match"] = bool(identity_ok)
    if strict and not identity_ok:
        failures.append(f"semantic {left_name}/{right_name}: sample identity differs")
    common_inputs = sorted(set(name for name in lhs if name.startswith("input_")).intersection(rhs))
    inputs_ok = True
    for name in common_inputs:
        metrics = _array_metrics(lhs[name], rhs[name])
        max_abs_limit = float(thresholds["input_max_abs"])
        mean_abs_limit = float(thresholds.get("input_mean_abs", math.inf))
        ok = bool(
            metrics.get("shape_match")
            and metrics.get("max_abs", math.inf) <= max_abs_limit
            and metrics.get("mean_abs", math.inf) <= mean_abs_limit
        )
        metrics["max_abs_limit"] = max_abs_limit
        metrics["mean_abs_limit"] = mean_abs_limit
        metrics["ok"] = ok
        result["inputs"][name] = metrics
        inputs_ok = inputs_ok and ok
        if strict and not ok:
            failures.append(f"semantic {left_name}/{right_name}: {name} differs")
    output_metrics = _array_metrics(lhs["output"], rhs["output"])
    output_kind = "feature" if contract["workload"]["kind"] == "feature-extraction" else "logit"
    cosine_min = float(thresholds.get(f"{output_kind}_cosine_min", thresholds.get("cosine_min", 1.0)))
    max_abs_limit = thresholds.get(f"{output_kind}_max_abs")
    mean_abs_limit = thresholds.get(f"{output_kind}_mean_abs")
    output_ok = bool(
        output_metrics.get("shape_match")
        and output_metrics.get("cosine_mean", -1.0) >= cosine_min
        and (max_abs_limit is None or output_metrics.get("max_abs", math.inf) <= float(max_abs_limit))
        and (mean_abs_limit is None or output_metrics.get("mean_abs", math.inf) <= float(mean_abs_limit))
    )
    output_metrics["cosine_min"] = cosine_min
    if max_abs_limit is not None:
        output_metrics["max_abs_limit"] = float(max_abs_limit)
    if mean_abs_limit is not None:
        output_metrics["mean_abs_limit"] = float(mean_abs_limit)
    output_metrics["ok"] = output_ok
    result["output"] = output_metrics
    prediction_ok = True
    if contract["workload"]["kind"] == "evaluation":
        semantic_top1_agreement = float(np.mean(np.argmax(lhs["output"], axis=1) == np.argmax(rhs["output"], axis=1)))
        semantic_top1_min = float(thresholds.get("semantic_top1_agreement_min", 1.0))
        semantic_prediction_ok = semantic_top1_agreement >= semantic_top1_min
        prediction_ok = prediction_ok and semantic_prediction_ok
        result["semantic_top1_agreement"] = semantic_top1_agreement
        result["semantic_top1_agreement_min"] = semantic_top1_min
        if strict and not semantic_prediction_ok:
            failures.append(f"semantic {left_name}/{right_name}: semantic Top-1 agreement is too low")
    if contract["workload"]["kind"] == "evaluation" and "top1_predictions" in lhs and "top1_predictions" in rhs:
        full_top1_agreement = float(np.mean(lhs["top1_predictions"] == rhs["top1_predictions"]))
        full_top1_min = float(thresholds.get("full_prediction_top1_agreement_min", 1.0))
        full_prediction_ok = full_top1_agreement >= full_top1_min
        prediction_ok = prediction_ok and full_prediction_ok
        result["full_top1_prediction_agreement"] = full_top1_agreement
        result["full_top1_prediction_agreement_min"] = full_top1_min
        if strict and not full_prediction_ok:
            failures.append(f"semantic {left_name}/{right_name}: full Top-1 agreement is too low")
    result["ok"] = bool(identity_ok and inputs_ok and output_ok and prediction_ok)
    if strict and not output_ok:
        failures.append(f"semantic {left_name}/{right_name}: model output differs")
    return result


def _validate_result(
    name: str,
    result: dict[str, Any],
    contract: dict[str, Any],
    expected_trace: list[dict[str, int]],
    failures: list[str],
) -> None:
    _require(result.get("schema_version") == PIPELINE_RESULT_SCHEMA, failures, f"{name}: bad result schema")
    _require(result.get("pipeline") == name, failures, f"{name}: pipeline identity mismatch")
    _require(result.get("contract_sha256") == sha256_json(contract), failures, f"{name}: contract hash mismatch")
    _require(result.get("sample_manifest_sha256") == contract["dataset"]["sample_manifest_sha256"], failures, f"{name}: sample manifest hash mismatch")
    repeats = result.get("repeats")
    _require(isinstance(repeats, list) and len(repeats) == contract["execution"]["repeats"], failures, f"{name}: repeat count mismatch")
    for repeat in repeats if isinstance(repeats, list) else []:
        _require(repeat.get("sample_trace") == expected_trace, failures, f"{name}: no-shuffle sample trace mismatch")
        _require(int(repeat.get("images", -1)) == len(expected_trace), failures, f"{name}: measured image count mismatch")
        first_batch_ms = repeat.get("time_to_first_batch_ms")
        _require(
            isinstance(first_batch_ms, (int, float)) and math.isfinite(float(first_batch_ms)) and first_batch_ms > 0.0,
            failures,
            f"{name}: invalid time-to-first-batch measurement",
        )
        steady = repeat.get("steady_throughput_images_per_s")
        if int(repeat.get("batches", 0)) > 1:
            _require(
                isinstance(steady, (int, float)) and math.isfinite(float(steady)) and steady > 0.0,
                failures,
                f"{name}: missing steady-state throughput",
            )
        pipeline_config = contract["pipelines"].get(name, {})
        if pipeline_config.get("segment_mode") == "manifest-shard":
            native = repeat.get("native_totals", {})
            native_segments = repeat.get("native_segments", [])
            _require(
                isinstance(native_segments, list) and bool(native_segments),
                failures,
                f"{name}: manifest-shard execution reported no shard segments",
            )
            shard_ids = [int(item.get("segment_shard_id", -1)) for item in native_segments]
            _require(
                all(shard_id >= 0 for shard_id in shard_ids) and len(shard_ids) == len(set(shard_ids)),
                failures,
                f"{name}: manifest shards were missing or reactivated: {shard_ids}",
            )
            _require(
                int(native.get("segment_count", -1)) == len(native_segments),
                failures,
                f"{name}: one native plan/result is required for every manifest shard",
            )
            for counter in (
                "segment_cross_shard_count",
                "shard_reactivation_count",
                "duplicate_physical_read_count",
                "rowgroup_revisit_count",
                "vector_run_revisit_count",
                "physical_read_order_inversions",
            ):
                _require(
                    int(native.get(counter, -1)) == 0,
                    failures,
                    f"{name}: {counter} must be zero in manifest-shard mode, got {native.get(counter)!r}",
                )
            first_shard_ready = repeat.get("first_shard_ready_ms")
            _require(
                isinstance(first_shard_ready, (int, float))
                and math.isfinite(float(first_shard_ready))
                and float(first_shard_ready) > 0.0,
                failures,
                f"{name}: missing first-shard ready time",
            )
        if name == "dct_major_pushdown":
            native = repeat.get("native_totals", {})
            _require(
                isinstance(contract["dataset"].get("block_major_access"), dict),
                failures,
                "dct_major_pushdown: missing immutable block-major descriptor contract",
            )
            _require(
                contract["dataset"].get("block_major_access", {}).get("passes_one_percent") is True,
                failures,
                "dct_major_pushdown: block-major descriptor did not pass the 1% storage gate",
            )
            for counter in (
                "host_expanded_transform_items_created",
                "host_output_block_source_lists_created",
                "host_global_transform_sort_items",
            ):
                _require(
                    int(native.get(counter, -1)) == 0,
                    failures,
                    f"dct_major_pushdown: {counter} must be zero, got {native.get(counter)!r}",
                )
            _require(
                int(native.get("planless_transform_kernel_launch_count", 0)) > 0,
                failures,
                "dct_major_pushdown: no planless transform kernel was launched",
            )
            _require(
                int(native.get("planless_image_descriptor_count", -1)) == int(repeat.get("images", -2)),
                failures,
                "dct_major_pushdown: logical planless image descriptor count must equal measured images, "
                f"got {native.get('planless_image_descriptor_count')!r} for {repeat.get('images')!r}",
            )
            full_scan_outputs = int(
                native.get("planless_transform_full_scan_output_block_count", 0)
            )
            active_outputs = int(native.get("planless_transform_output_block_count", 0))
            skipped_outputs = int(
                native.get("planless_transform_skipped_output_block_count", 0)
            )
            if full_scan_outputs > 0:
                _require(
                    0 < active_outputs <= full_scan_outputs,
                    failures,
                    "dct_major_pushdown: resident-output schedule is empty or exceeds its full scan",
                )
                _require(
                    skipped_outputs == full_scan_outputs - active_outputs,
                    failures,
                    "dct_major_pushdown: resident-output skip accounting is inconsistent",
                )
                _require(
                    int(native.get("planless_transform_active_output_index_bytes", -1))
                    == active_outputs * 4,
                    failures,
                    "dct_major_pushdown: active-output index byte accounting is inconsistent",
                )
                schedule_builds = int(
                    native.get("planless_transform_active_output_schedule_build_count", 0)
                )
                schedule_hits = int(
                    native.get("active_output_schedule_sidecar_hit_count", 0)
                )
                schedule_executions = schedule_builds + schedule_hits
                schedule_worksets = int(
                    native.get("planless_transform_active_output_workset_count", -1)
                )
                _require(
                    schedule_executions == int(native.get("segment_count", -1)),
                    failures,
                    "dct_major_pushdown: active-output schedule must be built or loaded exactly once per segment",
                )
                _require(
                    schedule_worksets == int(native.get("workset_count", -2)),
                    failures,
                    "dct_major_pushdown: active-output workset coverage disagrees with execution",
                )
                _require(
                    int(native.get("planless_transform_active_output_offset_bytes", -1))
                    == (schedule_worksets + schedule_executions) * 8,
                    failures,
                    "dct_major_pushdown: active-output offset byte accounting is inconsistent",
                )
                _require(
                    int(native.get("planless_transform_active_output_offsets_valid", 0))
                    == schedule_executions,
                    failures,
                    "dct_major_pushdown: active-output offsets were not monotonic/bounded for every segment",
                )
                # Native peak is a per-segment maximum while index/offset bytes
                # above are totals across segments. A maximum must cover at
                # least the average persistent schedule, without pretending
                # that all segment schedules coexist.
                persistent_schedule_bytes = (
                    (
                        active_outputs * 4
                        + int(native.get("planless_transform_active_output_offset_bytes", 0))
                    )
                    / schedule_executions
                    if schedule_executions > 0
                    else float("inf")
                )
                _require(
                    int(native.get("planless_transform_active_output_schedule_peak_bytes", -1))
                    >= persistent_schedule_bytes,
                    failures,
                    "dct_major_pushdown: active-output schedule peak does not cover its persistent arrays",
                )
                _require(
                    int(native.get("planless_transform_source_contribution_count", 0)) > 0,
                    failures,
                    "dct_major_pushdown: active-output schedule reported no source contributions",
                )
                source_contributions = int(
                    native.get("planless_transform_source_contribution_count", 0)
                )
                _require(
                    int(native.get("planless_transform_source_contribution_visit_count", -1))
                    == 2 * source_contributions,
                    failures,
                    "dct_major_pushdown: deterministic count/fill contribution visits are inconsistent",
                )
                _require(
                    int(native.get("planless_transform_output_workset_ownership_count", -1))
                    == active_outputs,
                    failures,
                    "dct_major_pushdown: output/workset ownership count disagrees with the active index",
                )
                coordinate_entries = int(native.get("coordinate_group_index_entries", -1))
                coordinate_populated = int(native.get("coordinate_group_index_populated", -1))
                coordinate_holes = int(native.get("coordinate_group_index_holes", -1))
                _require(
                    int(native.get("coordinate_group_lookup_count", 0)) > 0
                    and coordinate_populated > 0
                    and coordinate_entries == coordinate_populated + coordinate_holes,
                    failures,
                    "dct_major_pushdown: coordinate-to-group index accounting is inconsistent",
                )
                _require(
                    int(native.get("coordinate_group_index_bytes", -1)) >= coordinate_entries * 4,
                    failures,
                    "dct_major_pushdown: coordinate-to-group index byte accounting is incomplete",
                )
                expected_density = (
                    coordinate_populated / coordinate_entries if coordinate_entries > 0 else 0.0
                )
                _require(
                    abs(float(native.get("coordinate_group_index_density", -1.0)) - expected_density)
                    <= 1.0e-12,
                    failures,
                    "dct_major_pushdown: coordinate-to-group index density is inconsistent",
                )
                split_schedule_ms = sum(
                    float(native.get(key, -1.0))
                    for key in (
                        "planless_transform_group_workset_build_ms",
                        "planless_transform_active_output_count_ms",
                        "planless_transform_active_output_prefix_ms",
                        "planless_transform_active_output_fill_ms",
                    )
                )
                _require(
                    split_schedule_ms >= 0.0
                    and split_schedule_ms
                    <= float(native.get("planless_transform_active_output_planning_ms", 0.0)),
                    failures,
                    "dct_major_pushdown: split active-output timings are missing or exceed total planning",
                )
                _require(
                    float(native.get("planless_transform_active_output_planning_ms", 0.0)) > 0.0,
                    failures,
                    "dct_major_pushdown: active-output CPU schedule timing is missing",
                )
                _require(
                    float(native.get("planless_transform_gpu_kernel_ms", 0.0)) > 0.0,
                    failures,
                    "dct_major_pushdown: separately timed planless CUDA kernel elapsed is missing",
                )
            strategy_count = sum(
                int(native.get(key, 0))
                for key in (
                    "run_interval_exact_rowgroup_count",
                    "run_interval_bounded_rowgroup_count",
                    "bitmap_exact_rowgroup_count",
                    "full_rowgroup_strategy_count",
                )
            )
            _require(
                strategy_count == int(native.get("rowgroup_count", -1)),
                failures,
                "dct_major_pushdown: adaptive strategy counts do not cover every rowgroup",
            )
            crop_execution_mode = contract["pipelines"][name].get("crop_execution_mode")
            if crop_execution_mode in (
                "vector-range-read-selected-decode",
                "bounded-range-read-selected-decode",
                "bounded-io-uring-range-read-selected-decode",
                "bounded-io-uring-scheduled-range-read-selected-decode",
            ):
                planned_vectors = int(native.get("planned_vector_count", -1))
                actual_vectors = int(native.get("actual_vector_count", -2))
                _require(
                    planned_vectors > 0 and planned_vectors == actual_vectors,
                    failures,
                    "dct_major_pushdown: explicit selected decode requires positive planned=actual vectors, "
                    f"got {planned_vectors}/{actual_vectors}",
                )
                _require(
                    int(native.get("sparse_read_fallback_rowgroup_count", -1)) == 0,
                    failures,
                    "dct_major_pushdown: explicit selected-range mode used sparse fallback",
                )
            if crop_execution_mode in (
                "bounded-range-read-selected-decode",
                "bounded-io-uring-range-read-selected-decode",
                "bounded-io-uring-scheduled-range-read-selected-decode",
            ):
                scheduled_mode = (
                    crop_execution_mode
                    == "bounded-io-uring-scheduled-range-read-selected-decode"
                )
                io_uring_mode = crop_execution_mode in (
                    "bounded-io-uring-range-read-selected-decode",
                    "bounded-io-uring-scheduled-range-read-selected-decode",
                )
                cap_ppm = int(math.floor(float(contract["pipelines"][name].get(
                    "bounded_read_amplification_cap", 1.0
                )) * 1_000_000 + 0.5))
                local_cap = float(contract["pipelines"][name].get(
                    "bounded_read_local_amplification_cap", 0.0
                ))
                local_cap_ppm = (
                    0 if local_cap == 0.0 else int(math.floor(local_cap * 1_000_000 + 0.5))
                )
                max_run_bytes = int(contract["pipelines"][name].get(
                    "bounded_read_max_run_bytes", 0
                ))
                _require(
                    1_000_000 <= cap_ppm <= 1_100_000,
                    failures,
                    f"dct_major_pushdown: bounded amplification cap is outside [1.0, 1.10]: {cap_ppm}",
                )
                _require(
                    int(native.get("bounded_read_amplification_ppm", -1)) == cap_ppm
                    and int(native.get("bounded_read_local_amplification_ppm", -1)) == local_cap_ppm
                    and int(native.get("bounded_read_max_run_bytes", -1)) == max_run_bytes,
                    failures,
                    "dct_major_pushdown: runtime bounded-read configuration differs from the contract",
                )
                exact_bytes = int(native.get("bounded_exact_storage_bytes", -1))
                physical_bytes = int(native.get("bounded_physical_storage_bytes", -1))
                gap_bytes = int(native.get("bounded_merged_gap_bytes", -1))
                full_bytes = int(native.get("full_compressed_payload_bytes", -1))
                selected_bytes = int(native.get("selected_compressed_payload_bytes", -1))
                actual_read_bytes = int(native.get("compressed_payload_bytes_read", -1))
                _require(
                    exact_bytes > 0
                    and exact_bytes == selected_bytes
                    and physical_bytes == actual_read_bytes
                    and gap_bytes == physical_bytes - exact_bytes,
                    failures,
                    "dct_major_pushdown: bounded exact/physical/gap byte accounting is inconsistent",
                )
                _require(
                    physical_bytes <= (exact_bytes * cap_ppm) // 1_000_000
                    and physical_bytes <= full_bytes,
                    failures,
                    "dct_major_pushdown: bounded physical bytes exceed the whole-run cap or full payload",
                )
                expected_read_amplification = (
                    physical_bytes / exact_bytes if exact_bytes > 0 else 0.0
                )
                _require(
                    math.isclose(
                        float(native.get("read_amplification", math.nan)),
                        expected_read_amplification,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    ),
                    failures,
                    "dct_major_pushdown: read amplification is not the whole-run physical/exact ratio",
                )
                _require(
                    int(native.get("merged_gap_bytes", -1)) == gap_bytes
                    and int(native.get("hole_clear_bytes", -1)) == gap_bytes
                    and int(native.get("static_prefix_restore_bytes", 0)) > 0,
                    failures,
                    "dct_major_pushdown: bounded hole clear/static-prefix accounting is inconsistent",
                )
                physical_runs = int(native.get("bounded_physical_run_count", -1))
                exact_extents = int(native.get("bounded_exact_extent_count", -1))
                _require(
                    0 < physical_runs <= exact_extents
                    and (
                        physical_runs <= int(native.get("io_uring_read_request_count", -2))
                        if io_uring_mode
                        else physical_runs == int(native.get("pread_count", -2))
                    ),
                    failures,
                    "dct_major_pushdown: bounded physical run/submission accounting is inconsistent",
                )
                if io_uring_mode:
                    io_requests = int(native.get("io_uring_read_request_count", -1))
                    io_completions = int(native.get("io_uring_completion_count", -2))
                    io_submits = int(native.get("io_uring_submit_syscall_count", -1))
                    _require(
                        native.get("bounded_io_backend") == "io-uring"
                        and int(native.get("bounded_io_uring_queue_depth", -1)) == 256
                        and int(native.get("pread_count", -1)) == 0
                        and io_requests == io_completions
                        and io_requests >= physical_runs
                        and 0 < io_submits < io_requests
                        and int(native.get("io_uring_setup_count", 0)) > 0
                        and int(native.get("io_uring_ring_mapped_bytes", 0)) > 0
                        and int(native.get("io_uring_fallback_count", -1)) == 0,
                        failures,
                        "dct_major_pushdown: explicit io_uring mode did not use valid batched submissions",
                    )
                else:
                    _require(
                        native.get("bounded_io_backend", "sync-pread") == "sync-pread"
                        and int(native.get("bounded_io_uring_queue_depth", 0)) == 0
                        and int(native.get("io_uring_read_request_count", 0)) == 0
                        and int(native.get("io_uring_fallback_count", 0)) == 0,
                        failures,
                        "dct_major_pushdown: synchronous bounded mode reported io_uring activity",
                    )
                if scheduled_mode:
                    segment_count = int(native.get("segment_count", 0))
                    sidecar_hits = int(native.get("active_output_schedule_sidecar_hit_count", 0))
                    sidecar_misses = int(native.get("active_output_schedule_sidecar_miss_count", 0))
                    schedule_builds = int(
                        native.get("planless_transform_active_output_schedule_build_count", -1)
                    )
                    _require(
                        segment_count > 0
                        and sidecar_hits == segment_count
                        and sidecar_misses == 0
                        and schedule_builds == 0
                        and int(native.get("active_output_schedule_sidecar_reject_count", -1)) == 0
                        and int(native.get("active_output_schedule_sidecar_persist_count", -1)) == 0
                        and int(native.get("active_output_schedule_sidecar_bytes", 0)) > 0
                        and int(native.get("active_output_schedule_interval_count", 0)) > 0
                        and int(native.get("active_output_schedule_mmap_capacity_bytes", 0))
                        == 16 * 1024 * 1024
                        and int(native.get("active_output_schedule_mmap_window_count", 0)) == 2
                        and int(native.get("active_output_schedule_mapped_bytes_peak", 0)) > 0,
                        failures,
                        "dct_major_pushdown: frozen schedule sidecar evidence is inconsistent",
                    )
                _require(
                    int(native.get("run_interval_bounded_rowgroup_count", 0))
                    == int(native.get("rowgroup_count", -1))
                    and native.get("storage_read_granularity") == "bounded-selected-vector-range"
                    and native.get("decode_granularity") == "selected-vector",
                    failures,
                    "dct_major_pushdown: bounded mode did not remain bounded-read + selected-decode",
                )
            if contract["pipelines"][name].get("crop_execution_mode") == "auto":
                adaptive_candidates = int(
                    native.get("automatic_sparse_storage_candidate_rowgroup_count", 0)
                )
                _require(
                    adaptive_candidates > 0,
                    failures,
                    "dct_major_pushdown: auto mode did not evaluate any three-strategy candidates",
                )
                for counter in (
                    "adaptive_run_interval_estimated_ns",
                    "adaptive_bitmap_estimated_ns",
                    "adaptive_full_rowgroup_estimated_ns",
                ):
                    _require(
                        float(native.get(counter, 0.0)) > 0.0,
                        failures,
                        f"dct_major_pushdown: auto mode did not report {counter}",
                    )
            capacity = int(native.get("decode_workset_capacity_bytes", 0))
            expected_capacity = (
                int(contract["pipelines"][name]["decode_workset_capacity_mib"])
                * 1024
                * 1024
            )
            estimated_peak = max(
                int(native.get("max_estimated_decode_workset_bytes", 0)),
                int(native.get("bounded_double_buffer_peak_estimated_bytes", 0)),
            )
            _require(
                capacity == expected_capacity,
                failures,
                f"dct_major_pushdown: runtime workset capacity {capacity} differs from contract {expected_capacity}",
            )
            _require(
                0 < estimated_peak <= capacity,
                failures,
                f"dct_major_pushdown: estimated workset peak {estimated_peak} exceeds capacity {capacity}",
            )
            _require(
                int(native.get("oversized_decode_rowgroup_count", 0)) == 0,
                failures,
                "dct_major_pushdown: at least one rowgroup exceeded the byte budget",
            )
            _require(
                int(native.get("host_io_staged_rowgroups", 0)) == 0,
                failures,
                "dct_major_pushdown: full-segment host I/O staging was used",
            )
            actual_used = int(native.get("actual_transient_total_used_high_water_bytes", 0))
            actual_allocated = int(
                native.get("actual_transient_total_allocated_high_water_bytes", 0)
            )
            _require(
                0 < actual_used <= actual_allocated <= 512 * 1024 * 1024
                and native.get("actual_transient_memory_gate_passed") is True,
                failures,
                "dct_major_pushdown: actual transient high-water is missing or exceeds 512 MiB",
            )
            segment_count = int(native.get("segment_count", 0))
            _require(
                _compact_plan_memory_is_consistent(native),
                failures,
                "dct_major_pushdown: cumulative compact plan bytes/per-segment peak are missing "
                f"or inconsistent across {segment_count} segments",
            )

            pipeline_contract = contract["pipelines"][name]
            _require(
                int(pipeline_contract.get("plan_cache_capacity", -1)) == 0,
                failures,
                "dct_major_pushdown: count-bounded exact plan cache must be disabled",
            )
            _require(
                int(native.get("exact_batch_plan_cache_enabled", 0)) == 0,
                failures,
                "dct_major_pushdown: runtime enabled the exact expanded-plan cache",
            )
            for counter in ("plan_cache_hits", "plan_cache_misses", "plan_cache_evictions"):
                _require(
                    int(native.get(counter, 0)) == 0,
                    failures,
                    f"dct_major_pushdown: {counter} must remain zero on the planless path",
                )

            # The block-major implementation keeps axis programs in the
            # compact batch and owns FLS readers/sparse plans only for that
            # batch.  It must not quietly fall through to the historical
            # thread-local transform or sparse-vector caches, whose retention
            # is controlled by entry count rather than the formal byte budget.
            for counter in (
                "sparse_vector_cache_hits",
                "sparse_vector_cache_misses",
                "dct_resize_weight_cache_hits",
                "dct_resize_weight_cache_misses",
                "dct_conversion_matrix_cache_hits",
                "dct_conversion_matrix_cache_misses",
            ):
                _require(
                    int(native.get(counter, 0)) == 0,
                    failures,
                    f"dct_major_pushdown: persistent helper cache counter {counter} must remain zero",
                )

            decoded_capacity = int(native.get("decoded_rowgroup_cache_capacity_bytes", 0))
            expected_decoded_capacity = int(pipeline_contract.get("cache_capacity_mib", 0)) * 1024 * 1024
            decoded_current = int(native.get("decoded_rowgroup_cache_current_bytes", 0))
            decoded_peak = int(native.get("decoded_rowgroup_cache_peak_bytes", 0))
            decoded_entries = int(native.get("decoded_rowgroup_cache_current_rowgroups", 0))
            decoded_peak_entries = int(native.get("decoded_rowgroup_cache_peak_rowgroups", 0))
            _require(
                decoded_capacity == expected_decoded_capacity,
                failures,
                "dct_major_pushdown: decoded-rowgroup cache byte capacity differs from the contract",
            )
            _require(
                0 <= decoded_current <= decoded_peak <= decoded_capacity,
                failures,
                "dct_major_pushdown: decoded-rowgroup cache current/peak exceeds its byte capacity",
            )
            if decoded_capacity == 0:
                _require(
                    decoded_entries == 0 and decoded_peak_entries == 0,
                    failures,
                    "dct_major_pushdown: disabled decoded-rowgroup cache retained entries",
                )
                for counter in (
                    "decoded_rowgroup_cache_hits",
                    "decoded_rowgroup_cache_misses",
                    "decoded_rowgroup_cache_inserts",
                    "decoded_rowgroup_cache_evictions",
                ):
                    _require(
                        int(native.get(counter, 0)) == 0,
                        failures,
                        f"dct_major_pushdown: disabled decoded-rowgroup cache reported {counter}",
                    )


def _physical_evidence(
    contract: dict[str, Any],
    aggregates: dict[str, dict[str, Any]],
    failures: list[str],
) -> dict[str, Any] | None:
    if "dct_major_full" not in aggregates or "dct_major_pushdown" not in aggregates:
        return None
    full = aggregates["dct_major_full"]["native_hot_mean"]
    push = aggregates["dct_major_pushdown"]["native_hot_mean"]

    def metric(payload: dict[str, float], *names: str) -> float:
        for name in names:
            if name in payload:
                value = float(payload[name])
                if value > 0.0:
                    return value
        return 0.0

    full_bytes = metric(full, "compressed_payload_bytes_read", "rowgroup_storage_bytes_read")
    push_bytes = metric(push, "compressed_payload_bytes_read", "rowgroup_storage_bytes_read")
    full_selected_bytes = metric(full, "selected_compressed_payload_bytes")
    push_selected_bytes = metric(push, "selected_compressed_payload_bytes")
    full_available_bytes = metric(full, "full_compressed_payload_bytes")
    push_available_bytes = metric(push, "full_compressed_payload_bytes")
    full_vectors = metric(full, "actual_vector_count", "selected_vector_count")
    push_vectors = metric(push, "actual_vector_count", "selected_vector_count")
    full_blocks = metric(
        full,
        "requested_source_block_count",
        "source_blocks_transformed",
        "fixed_transform_source_block_count",
        "block_count",
        "planned_selected_vector_count",
        "planned_vector_count",
    )
    push_blocks = metric(
        push,
        "requested_source_block_count",
        "source_blocks_transformed",
        "fixed_transform_source_block_count",
        "block_count",
        "planned_selected_vector_count",
        "planned_vector_count",
    )
    bytes_reduced = 0.0 < push_bytes < full_bytes
    vectors_reduced = 0.0 < push_vectors < full_vectors
    blocks_reduced = 0.0 < push_blocks < full_blocks
    _require(bytes_reduced, failures, f"DCT-major pushdown did not reduce physical bytes: {push_bytes} vs {full_bytes}")
    _require(vectors_reduced, failures, f"DCT-major pushdown did not reduce decoded vectors: {push_vectors} vs {full_vectors}")
    _require(blocks_reduced, failures, f"DCT-major pushdown did not reduce transformed source blocks: {push_blocks} vs {full_blocks}")
    return {
        "full_compressed_payload_bytes_read": full_bytes,
        "pushdown_compressed_payload_bytes_read": push_bytes,
        "full_selected_compressed_payload_bytes": full_selected_bytes,
        "pushdown_selected_compressed_payload_bytes": push_selected_bytes,
        "full_available_compressed_payload_bytes": full_available_bytes,
        "pushdown_available_compressed_payload_bytes": push_available_bytes,
        "full_read_amplification": (
            full_bytes / full_selected_bytes if full_selected_bytes else 0.0
        ),
        "pushdown_read_amplification": (
            push_bytes / push_selected_bytes if push_selected_bytes else 0.0
        ),
        "physical_bytes_saved_percent": 100.0 * (1.0 - push_bytes / full_bytes) if full_bytes else 0.0,
        "full_actual_vector_count": full_vectors,
        "pushdown_actual_vector_count": push_vectors,
        "decoded_vectors_saved_percent": 100.0 * (1.0 - push_vectors / full_vectors) if full_vectors else 0.0,
        "full_source_blocks": full_blocks,
        "pushdown_source_blocks": push_blocks,
        "source_blocks_saved_percent": 100.0 * (1.0 - push_blocks / full_blocks) if full_blocks else 0.0,
        "full_pread_count": metric(full, "pread_count"),
        "pushdown_pread_count": metric(push, "pread_count"),
        "full_repeat_counters": {
            name: float(full.get(name, 0.0))
            for name in (
                "segment_cross_shard_count",
                "shard_reactivation_count",
                "duplicate_physical_read_count",
                "rowgroup_revisit_count",
                "vector_run_revisit_count",
                "physical_read_order_inversions",
            )
        },
        "pushdown_repeat_counters": {
            name: float(push.get(name, 0.0))
            for name in (
                "segment_cross_shard_count",
                "shard_reactivation_count",
                "duplicate_physical_read_count",
                "rowgroup_revisit_count",
                "vector_run_revisit_count",
                "physical_read_order_inversions",
            )
        },
        "ok": bytes_reduced and vectors_reduced and blocks_reduced,
    }


def _planless_resource_evidence(
    aggregates: dict[str, dict[str, Any]],
    failures: list[str],
) -> dict[str, Any] | None:
    legacy = aggregates.get("dct_major_legacy_pushdown")
    planless = aggregates.get("dct_major_pushdown")
    if legacy is None or planless is None:
        return None

    metric_names = (
        "host_peak_rss_bytes",
        "galp_native_pinned_peak_in_use_bytes",
        "galp_native_device_peak_in_use_bytes",
        "peak_torch_gpu_allocated_bytes",
        "peak_torch_gpu_reserved_bytes",
    )
    comparisons: dict[str, dict[str, float | bool]] = {}
    for name in metric_names:
        legacy_distribution = legacy.get(name)
        planless_distribution = planless.get(name)
        available = isinstance(legacy_distribution, dict) and isinstance(planless_distribution, dict)
        legacy_peak = float(legacy_distribution.get("max", 0.0)) if available else 0.0
        planless_peak = float(planless_distribution.get("max", 0.0)) if available else 0.0
        ok = available and planless_peak <= legacy_peak
        _require(available, failures, f"planless/legacy memory comparison is missing {name}")
        if available:
            _require(
                ok,
                failures,
                f"planless {name} exceeds legacy pushdown: {planless_peak:.0f} vs {legacy_peak:.0f}",
            )
        comparisons[name] = {
            "legacy_peak_bytes": legacy_peak,
            "planless_peak_bytes": planless_peak,
            "planless_over_legacy": planless_peak / legacy_peak if legacy_peak else 0.0,
            "ok": ok,
        }
    return {"ok": all(bool(item["ok"]) for item in comparisons.values()), "metrics": comparisons}


def _write_csv(
    path: Path,
    results: Sequence[dict[str, Any]],
    block_major_access: dict[str, Any] | None,
) -> None:
    columns = (
        "pipeline",
        "domain",
        "repeat",
        "images",
        "seconds",
        "throughput_images_per_s",
        "time_to_first_batch_ms",
        "first_shard_ready_ms",
        "steady_throughput_images_per_s",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "loader_mean_ms",
        "h2d_mean_ms",
        "model_mean_ms",
        "gpu_utilization_mean_percent",
        "accuracy_top1",
        "accuracy_top5",
        "planning_ms",
        "planning_item_count",
        "sort_item_count",
        "compact_plan_bytes",
        "compact_plan_peak_bytes",
        "selected_compressed_payload_bytes",
        "compressed_payload_bytes_read",
        "full_compressed_payload_bytes",
        "physical_io_saving",
        "read_amplification",
        "process_io_logical_read_bytes",
        "process_io_storage_read_bytes",
        "process_io_read_syscalls",
        "input_ready_stall_ms",
        "input_ready_stall_percent",
        "pread_count",
        "rowgroup_count",
        "planned_vector_count",
        "actual_vector_count",
        "full_vector_count",
        "duplicate_physical_read_count",
        "rowgroup_revisit_count",
        "vector_run_revisit_count",
        "physical_read_order_inversions",
        "decoded_coefficient_bytes",
        "requested_source_block_count",
        "host_peak_rss_bytes",
        "native_pinned_peak_bytes",
        "native_gpu_peak_bytes",
        "torch_gpu_allocated_peak_bytes",
        "torch_gpu_reserved_peak_bytes",
        "decode_workset_capacity_bytes",
        "decode_workset_peak_estimated_bytes",
        "oversized_decode_rowgroups",
        "run_interval_exact_rowgroups",
        "bitmap_exact_rowgroups",
        "full_rowgroup_strategy_count",
        "run_interval_bounded_rowgroups",
        "bounded_exact_storage_bytes",
        "bounded_physical_storage_bytes",
        "bounded_merged_gap_bytes",
        "bounded_exact_extent_count",
        "bounded_physical_run_count",
        "bounded_selected_gap_count",
        "bounded_read_amplification_ppm",
        "bounded_read_local_amplification_ppm",
        "bounded_read_max_run_bytes",
        "hole_clear_bytes",
        "static_prefix_restore_bytes",
        "actual_transient_total_used_high_water_bytes",
        "actual_transient_total_allocated_high_water_bytes",
        "actual_transient_memory_gate_passed",
        "adaptive_run_interval_estimated_ns",
        "adaptive_bitmap_estimated_ns",
        "adaptive_full_rowgroup_estimated_ns",
        "adaptive_selected_memory_fit_rowgroups",
        "adaptive_full_memory_fit_rowgroups",
        "decoded_rowgroup_cache_capacity_bytes",
        "decoded_rowgroup_cache_current_bytes",
        "decoded_rowgroup_cache_peak_bytes",
        "decoded_rowgroup_cache_current_rowgroups",
        "decoded_rowgroup_cache_peak_rowgroups",
        "decoded_rowgroup_cache_hits",
        "decoded_rowgroup_cache_misses",
        "decoded_rowgroup_cache_hit_rate",
        "decoded_rowgroup_cache_inserts",
        "decoded_rowgroup_cache_evictions",
        "plan_cache_hits",
        "plan_cache_misses",
        "plan_cache_evictions",
        "sparse_vector_cache_hits",
        "sparse_vector_cache_misses",
        "dct_resize_weight_cache_hits",
        "dct_resize_weight_cache_misses",
        "dct_conversion_matrix_cache_hits",
        "dct_conversion_matrix_cache_misses",
        "descriptor_bytes",
        "base_storage_bytes",
        "total_storage_bytes_with_descriptor",
        "descriptor_storage_increase_percent",
    )

    def native(native_totals: dict[str, Any], *names: str) -> float | int:
        for name in names:
            value = native_totals.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
        return 0

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            for repeat in result["repeats"]:
                native_totals = repeat.get("native_totals", {})
                compressed = float(native(native_totals, "compressed_payload_bytes_read"))
                selected_compressed = float(
                    native(native_totals, "selected_compressed_payload_bytes")
                )
                full_compressed = float(native(native_totals, "full_compressed_payload_bytes"))
                read_denominator = selected_compressed or full_compressed
                process_io = repeat.get("process_io") or {}
                cache_hits = float(native(native_totals, "decoded_rowgroup_cache_hits"))
                cache_misses = float(native(native_totals, "decoded_rowgroup_cache_misses"))
                repeat_seconds = float(repeat.get("seconds", 0.0))
                input_ready_stall_ms = float(
                    native(native_totals, "prefetch_consumer_wait_ms")
                )
                writer.writerow(
                    {
                        "pipeline": result["pipeline"],
                        "domain": result["domain"],
                        "repeat": repeat["repeat"],
                        "images": repeat["images"],
                        "seconds": repeat["seconds"],
                        "throughput_images_per_s": repeat["throughput_images_per_s"],
                        "time_to_first_batch_ms": repeat["time_to_first_batch_ms"],
                        "first_shard_ready_ms": repeat.get("process_scope_first_shard_ready_ms")
                        or repeat.get("first_shard_ready_ms"),
                        "steady_throughput_images_per_s": repeat.get("steady_throughput_images_per_s"),
                        "latency_mean_ms": repeat["latency_ms"]["mean"],
                        "latency_p50_ms": repeat["latency_ms"].get(
                            "p50", repeat["latency_ms"]["mean"]
                        ),
                        "latency_p95_ms": repeat["latency_ms"].get(
                            "p95", repeat["latency_ms"]["mean"]
                        ),
                        "loader_mean_ms": repeat["loader_submit_ms"]["mean"],
                        "h2d_mean_ms": repeat.get("top_level_h2d_ms", {}).get("mean", 0.0),
                        "model_mean_ms": repeat["model_ms"]["mean"],
                        "gpu_utilization_mean_percent": (
                            repeat["gpu_utilization_percent"].get("mean")
                            if int(repeat.get("gpu_utilization_percent", {}).get("count", 0)) > 0
                            else None
                        ),
                        "accuracy_top1": repeat.get("accuracy_top1"),
                        "accuracy_top5": repeat.get("accuracy_top5"),
                        "planning_ms": native(native_totals, "planning_ms", "prefetch_planning_ms"),
                        "planning_item_count": native(
                            native_totals, "host_expanded_transform_items_created"
                        ),
                        "sort_item_count": native(native_totals, "host_global_transform_sort_items"),
                        "compact_plan_bytes": native(native_totals, "compact_plan_bytes"),
                        "compact_plan_peak_bytes": native(native_totals, "compact_plan_peak_bytes"),
                        "selected_compressed_payload_bytes": selected_compressed,
                        "compressed_payload_bytes_read": compressed,
                        "full_compressed_payload_bytes": full_compressed,
                        "physical_io_saving": (
                            1.0 - compressed / full_compressed if full_compressed else 0.0
                        ),
                        "read_amplification": (
                            compressed / read_denominator if read_denominator else 0.0
                        ),
                        "process_io_logical_read_bytes": process_io.get(
                            "logical_read_bytes"
                        ),
                        "process_io_storage_read_bytes": process_io.get(
                            "storage_read_bytes"
                        ),
                        "process_io_read_syscalls": process_io.get("read_syscalls"),
                        "input_ready_stall_ms": input_ready_stall_ms,
                        "input_ready_stall_percent": (
                            100.0 * input_ready_stall_ms / (repeat_seconds * 1000.0)
                            if repeat_seconds > 0.0
                            else 0.0
                        ),
                        "pread_count": native(native_totals, "pread_count"),
                        "rowgroup_count": native(native_totals, "rowgroup_count"),
                        "planned_vector_count": native(native_totals, "planned_vector_count"),
                        "actual_vector_count": native(native_totals, "actual_vector_count"),
                        "full_vector_count": native(native_totals, "full_vector_count"),
                        "duplicate_physical_read_count": native(
                            native_totals, "duplicate_physical_read_count"
                        ),
                        "rowgroup_revisit_count": native(native_totals, "rowgroup_revisit_count"),
                        "vector_run_revisit_count": native(
                            native_totals, "vector_run_revisit_count"
                        ),
                        "physical_read_order_inversions": native(
                            native_totals, "physical_read_order_inversions"
                        ),
                        "decoded_coefficient_bytes": native(native_totals, "decoded_coefficient_bytes"),
                        "requested_source_block_count": native(
                            native_totals, "requested_source_block_count"
                        ),
                        "host_peak_rss_bytes": repeat["host_peak_rss_bytes"],
                        "native_pinned_peak_bytes": native(
                            native_totals, "galp_native_pinned_peak_in_use_bytes"
                        ),
                        "native_gpu_peak_bytes": native(
                            native_totals, "galp_native_device_peak_in_use_bytes"
                        ),
                        "torch_gpu_allocated_peak_bytes": repeat["peak_torch_gpu_allocated_bytes"],
                        "torch_gpu_reserved_peak_bytes": repeat["peak_torch_gpu_reserved_bytes"],
                        "decode_workset_capacity_bytes": native(
                            native_totals, "decode_workset_capacity_bytes"
                        ),
                        "decode_workset_peak_estimated_bytes": max(
                            int(native(native_totals, "max_estimated_decode_workset_bytes")),
                            int(native(native_totals, "bounded_double_buffer_peak_estimated_bytes")),
                        ),
                        "oversized_decode_rowgroups": native(
                            native_totals, "oversized_decode_rowgroup_count"
                        ),
                        "run_interval_exact_rowgroups": native(
                            native_totals, "run_interval_exact_rowgroup_count"
                        ),
                        "bitmap_exact_rowgroups": native(
                            native_totals, "bitmap_exact_rowgroup_count"
                        ),
                        "full_rowgroup_strategy_count": native(
                            native_totals, "full_rowgroup_strategy_count"
                        ),
                        "run_interval_bounded_rowgroups": native(
                            native_totals, "run_interval_bounded_rowgroup_count"
                        ),
                        "bounded_exact_storage_bytes": native(
                            native_totals, "bounded_exact_storage_bytes"
                        ),
                        "bounded_physical_storage_bytes": native(
                            native_totals, "bounded_physical_storage_bytes"
                        ),
                        "bounded_merged_gap_bytes": native(
                            native_totals, "bounded_merged_gap_bytes"
                        ),
                        "bounded_exact_extent_count": native(
                            native_totals, "bounded_exact_extent_count"
                        ),
                        "bounded_physical_run_count": native(
                            native_totals, "bounded_physical_run_count"
                        ),
                        "bounded_selected_gap_count": native(
                            native_totals, "bounded_selected_gap_count"
                        ),
                        "bounded_read_amplification_ppm": native(
                            native_totals, "bounded_read_amplification_ppm"
                        ),
                        "bounded_read_local_amplification_ppm": native(
                            native_totals, "bounded_read_local_amplification_ppm"
                        ),
                        "bounded_read_max_run_bytes": native(
                            native_totals, "bounded_read_max_run_bytes"
                        ),
                        "hole_clear_bytes": native(native_totals, "hole_clear_bytes"),
                        "static_prefix_restore_bytes": native(
                            native_totals, "static_prefix_restore_bytes"
                        ),
                        "actual_transient_total_used_high_water_bytes": native(
                            native_totals, "actual_transient_total_used_high_water_bytes"
                        ),
                        "actual_transient_total_allocated_high_water_bytes": native(
                            native_totals, "actual_transient_total_allocated_high_water_bytes"
                        ),
                        "actual_transient_memory_gate_passed": bool(
                            native_totals.get("actual_transient_memory_gate_passed", False)
                        ),
                        "adaptive_run_interval_estimated_ns": native(
                            native_totals, "adaptive_run_interval_estimated_ns"
                        ),
                        "adaptive_bitmap_estimated_ns": native(
                            native_totals, "adaptive_bitmap_estimated_ns"
                        ),
                        "adaptive_full_rowgroup_estimated_ns": native(
                            native_totals, "adaptive_full_rowgroup_estimated_ns"
                        ),
                        "adaptive_selected_memory_fit_rowgroups": native(
                            native_totals, "adaptive_selected_memory_fit_rowgroup_count"
                        ),
                        "adaptive_full_memory_fit_rowgroups": native(
                            native_totals, "adaptive_full_memory_fit_rowgroup_count"
                        ),
                        "decoded_rowgroup_cache_capacity_bytes": native(
                            native_totals, "decoded_rowgroup_cache_capacity_bytes"
                        ),
                        "decoded_rowgroup_cache_current_bytes": native(
                            native_totals, "decoded_rowgroup_cache_current_bytes"
                        ),
                        "decoded_rowgroup_cache_peak_bytes": native(
                            native_totals, "decoded_rowgroup_cache_peak_bytes"
                        ),
                        "decoded_rowgroup_cache_current_rowgroups": native(
                            native_totals, "decoded_rowgroup_cache_current_rowgroups"
                        ),
                        "decoded_rowgroup_cache_peak_rowgroups": native(
                            native_totals, "decoded_rowgroup_cache_peak_rowgroups"
                        ),
                        "decoded_rowgroup_cache_hits": cache_hits,
                        "decoded_rowgroup_cache_misses": cache_misses,
                        "decoded_rowgroup_cache_hit_rate": (
                            cache_hits / (cache_hits + cache_misses)
                            if cache_hits + cache_misses
                            else 0.0
                        ),
                        "decoded_rowgroup_cache_inserts": native(
                            native_totals, "decoded_rowgroup_cache_inserts"
                        ),
                        "decoded_rowgroup_cache_evictions": native(
                            native_totals, "decoded_rowgroup_cache_evictions"
                        ),
                        "plan_cache_hits": native(native_totals, "plan_cache_hits"),
                        "plan_cache_misses": native(native_totals, "plan_cache_misses"),
                        "plan_cache_evictions": native(native_totals, "plan_cache_evictions"),
                        "sparse_vector_cache_hits": native(native_totals, "sparse_vector_cache_hits"),
                        "sparse_vector_cache_misses": native(native_totals, "sparse_vector_cache_misses"),
                        "dct_resize_weight_cache_hits": native(
                            native_totals, "dct_resize_weight_cache_hits"
                        ),
                        "dct_resize_weight_cache_misses": native(
                            native_totals, "dct_resize_weight_cache_misses"
                        ),
                        "dct_conversion_matrix_cache_hits": native(
                            native_totals, "dct_conversion_matrix_cache_hits"
                        ),
                        "dct_conversion_matrix_cache_misses": native(
                            native_totals, "dct_conversion_matrix_cache_misses"
                        ),
                        "descriptor_bytes": (
                            block_major_access["descriptor_bytes"] if block_major_access else 0
                        ),
                        "base_storage_bytes": (
                            block_major_access["base_storage_bytes"] if block_major_access else 0
                        ),
                        "total_storage_bytes_with_descriptor": (
                            block_major_access["total_storage_bytes_with_descriptor"]
                            if block_major_access
                            else 0
                        ),
                        "descriptor_storage_increase_percent": (
                            block_major_access["storage_increase_percent"]
                            if block_major_access
                            else 0.0
                        ),
                    }
                )


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# DCT-major no-shuffle benchmark",
        "",
        f"Validation: **{'PASS' if summary['ok'] else 'FAIL'}**",
        "",
        f"Workload: `{summary['workload']}`; sample order: `galp_image_id_ascending`; shuffle: `false`.",
        f"Preprocess: `{summary.get('preprocess_profile', 'unspecified')}`.",
        "",
        "| Pipeline | Domain | Cold E2E (img/s) | Cold first batch (ms) | Hot E2E p50 (img/s) | Steady p50 (img/s) | Hot first batch p50 (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for aggregate in summary["aggregates"]:
        steady = aggregate["steady_throughput_images_per_s"]
        steady_text = f"{steady['p50']:.3f}" if steady is not None else "n/a"
        lines.append(
            f"| {aggregate['pipeline']} | {aggregate['domain']} | "
            f"{aggregate['cold_start']['throughput_images_per_s']:.3f} | "
            f"{aggregate['cold_start']['time_to_first_batch_ms']:.3f} | "
            f"{aggregate['throughput_images_per_s']['p50']:.3f} | "
            f"{steady_text} | "
            f"{aggregate['time_to_first_batch_ms']['p50']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Timing decomposition",
            "",
            "| Pipeline | Cold first shard ready (ms) | Batch p50 (ms) | Batch p95 (ms) | Load (ms) | Input-ready stall (%) | H2D (ms) | Model (ms) | GPU util mean (%) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for aggregate in summary["aggregates"]:
        cold = aggregate["cold_start"]
        first_shard = cold.get("first_shard_ready_ms")
        gpu_util = aggregate.get("gpu_utilization_percent")
        first_shard_text = f"{first_shard:.3f}" if first_shard is not None else "n/a"
        gpu_util_text = f"{gpu_util['p50']:.3f}" if gpu_util is not None else "n/a"
        lines.append(
            f"| {aggregate['pipeline']} | "
            f"{first_shard_text} | "
            f"{aggregate['batch_latency_p50_ms']['p50']:.3f} | "
            f"{aggregate['batch_latency_p95_ms']['p50']:.3f} | "
            f"{aggregate['loader_mean_ms']['p50']:.3f} | "
            f"{aggregate['input_ready_stall_percent']['p50']:.3f} | "
            f"{aggregate['h2d_mean_ms']['p50']:.3f} | "
            f"{aggregate['model_mean_ms']['p50']:.3f} | "
            f"{gpu_util_text} |"
        )
    lines.extend(
        [
            "",
            "## Peak resources (hot-repeat p50)",
            "",
            "| Pipeline | Host RSS (MiB) | Torch GPU allocated (MiB) | Torch GPU reserved (MiB) | Native pinned (MiB) | Native GPU (MiB) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    mib = float(1024**2)
    for aggregate in summary["aggregates"]:
        native_pinned = aggregate.get("galp_native_pinned_peak_in_use_bytes")
        native_gpu = aggregate.get("galp_native_device_peak_in_use_bytes")
        lines.append(
            f"| {aggregate['pipeline']} | "
            f"{aggregate['host_peak_rss_bytes']['p50'] / mib:.3f} | "
            f"{aggregate['peak_torch_gpu_allocated_bytes']['p50'] / mib:.3f} | "
            f"{aggregate['peak_torch_gpu_reserved_bytes']['p50'] / mib:.3f} | "
            f"{native_pinned['p50'] / mib if native_pinned is not None else 0.0:.3f} | "
            f"{native_gpu['p50'] / mib if native_gpu is not None else 0.0:.3f} |"
        )
    lines.extend(["", "## Cold-start speedups (primary)", ""])
    for name, value in summary["cold_speedups"].items():
        lines.append(f"- `{name}`: `{value:.6f}x`.")
    lines.extend(["", "## Hot-repeat speedups (diagnostic)", ""])
    for name, value in summary["speedups"].items():
        lines.append(f"- `{name}`: `{value:.6f}x`.")
    lines.extend(["", "## Stability", ""])
    for aggregate in summary["aggregates"]:
        lines.append(
            f"- `{aggregate['pipeline']}`: throughput CV="
            f"`{aggregate['throughput_images_per_s']['cv_population']:.6f}`, endpoint drift="
            f"`{aggregate['throughput_endpoint_drift']:.6f}`."
        )
    lines.extend(["", "## Physical crop evidence", ""])
    if summary["physical_evidence"] is None:
        lines.append("- DCT-major full/pushdown pair was not run.")
    else:
        evidence = summary["physical_evidence"]
        lines.extend(
            [
                f"- Compressed bytes: `{evidence['full_compressed_payload_bytes_read']:.0f}` -> `{evidence['pushdown_compressed_payload_bytes_read']:.0f}` (`-{evidence['physical_bytes_saved_percent']:.3f}%`).",
                f"- Full/selected/actual compressed bytes (full): `{evidence['full_available_compressed_payload_bytes']:.0f}` / `{evidence['full_selected_compressed_payload_bytes']:.0f}` / `{evidence['full_compressed_payload_bytes_read']:.0f}`; read amplification=`{evidence['full_read_amplification']:.6f}x`.",
                f"- Full/selected/actual compressed bytes (pushdown): `{evidence['pushdown_available_compressed_payload_bytes']:.0f}` / `{evidence['pushdown_selected_compressed_payload_bytes']:.0f}` / `{evidence['pushdown_compressed_payload_bytes_read']:.0f}`; read amplification=`{evidence['pushdown_read_amplification']:.6f}x`.",
                f"- Decoded vectors: `{evidence['full_actual_vector_count']:.0f}` -> `{evidence['pushdown_actual_vector_count']:.0f}` (`-{evidence['decoded_vectors_saved_percent']:.3f}%`).",
                f"- Source blocks: `{evidence['full_source_blocks']:.0f}` -> `{evidence['pushdown_source_blocks']:.0f}` (`-{evidence['source_blocks_saved_percent']:.3f}%`).",
                f"- Preads: `{evidence['full_pread_count']:.0f}` -> `{evidence['pushdown_pread_count']:.0f}`.",
                "- Full repeat counters: "
                + ", ".join(
                    f"`{name}={value:.0f}`"
                    for name, value in evidence["full_repeat_counters"].items()
                )
                + ".",
                "- Pushdown repeat counters: "
                + ", ".join(
                    f"`{name}={value:.0f}`"
                    for name, value in evidence["pushdown_repeat_counters"].items()
                )
                + ".",
            ]
        )
    lines.extend(["", "## Semantic comparisons", ""])
    for comparison in summary["semantic_comparisons"]:
        if comparison["ok"]:
            comparison_status = "PASS"
        elif comparison["enforcement"] == "diagnostic":
            comparison_status = "DIAGNOSTIC-DRIFT"
        else:
            comparison_status = "FAIL"
        lines.append(
            f"- `{comparison['pipelines'][0]}` vs `{comparison['pipelines'][1]}`: "
            f"{comparison_status} ({comparison['enforcement']}); "
            f"output max_abs=`{comparison['output'].get('max_abs', float('nan')):.6g}`."
        )
    lines.extend(["", "## Block-major sidecar storage", ""])
    storage = summary["block_major_access_storage"]
    if storage is None:
        lines.append("- Block-major access descriptor was not part of this contract.")
    else:
        lines.extend(
            [
                f"- Descriptor files: `{storage['shard_count']}` shards plus companion index.",
                f"- Descriptor bytes: `{storage['descriptor_bytes']}`.",
                f"- Active-output schedule bytes: "
                f"`{storage.get('active_output_schedule_bytes', 0)}`.",
                f"- Base manifest/FLS/metadata bytes: `{storage['base_storage_bytes']}`.",
                f"- Total with all sidecars: "
                f"`{storage.get('total_storage_bytes_with_sidecars', storage['total_storage_bytes_with_descriptor'])}`.",
                f"- All-sidecar storage increase: "
                f"`{storage.get('all_sidecar_storage_increase_percent', storage['storage_increase_percent']):.6f}%` "
                f"(1% gate: PASS; 0.5% target: {'PASS' if storage['passes_half_percent'] else 'MISS'}).",
            ]
        )
    total_storage = summary.get("storage_evidence", {})
    if total_storage:
        lines.extend(
            [
                f"- JPEG bytes: `{total_storage['jpeg_bytes']}`.",
                f"- FLS payload bytes: `{total_storage['fls_payload_bytes']}`.",
                f"- Metadata bytes: `{total_storage['metadata_bytes']}`.",
                f"- Block-major access sidecar bytes: `{total_storage['sidecar_bytes']}`.",
                f"- Total block-major bytes: `{total_storage['total_block_major_bytes']}` "
                f"(`{total_storage['relative_to_jpeg']:.6f}x` JPEG, "
                f"`{total_storage['relative_to_raw_rgb']:.6f}x` raw RGB).",
            ]
        )
    lines.extend(["", "## Planless versus legacy memory", ""])
    resource = summary["planless_resource_evidence"]
    if resource is None:
        lines.append("- Planless and legacy crop-pushdown pair was not run.")
    else:
        for name, comparison in resource["metrics"].items():
            lines.append(
                f"- `{name}`: `{comparison['legacy_peak_bytes']:.0f}` -> "
                f"`{comparison['planless_peak_bytes']:.0f}` bytes "
                f"(`{comparison['planless_over_legacy']:.6f}x`)."
            )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in summary["failures"])
    lines.extend(
        [
            "",
            "## Comparability boundary",
            "",
            "- DCT-major full/pushdown, image-major v2/v3 pushdown, and RGB-no-more use the DCT checkpoint and are strict same-domain comparisons.",
            "- DALI and PyTorch use the RGB checkpoint. Their comparison is same-domain; ratios against GALP are deployment-level context only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(contract_path: Path, output_dir: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    samples = load_sample_manifest(
        Path(contract["dataset"]["sample_manifest"]),
        contract["dataset"]["sample_manifest_sha256"],
    )
    _, measured = selected_batches(contract, samples)
    expected_trace = sample_trace(measured)
    failures: list[str] = []
    if contract["preprocess"].get("profile") == "fixed-center-224-from-512":
        geometry = contract["dataset"].get("source_geometry_validation") or {}
        _require(
            int(geometry.get("validated_image_count", -1)) == len(samples)
            and [geometry.get("source_width"), geometry.get("source_height")] == [512, 512]
            and [geometry.get("crop_origin_x"), geometry.get("crop_origin_y")] == [144, 144]
            and [geometry.get("crop_width"), geometry.get("crop_height")] == [224, 224],
            failures,
            "fixed 512 source / center crop (144,144,224,224) geometry was not fully validated",
        )
        _require(
            contract["preprocess"]["rgb"].get("resize_shorter") is None
            and contract["preprocess"]["dct"].get("crop_reference_size_blocks") == [64, 64],
            failures,
            "fixed crop profile unexpectedly contains RGB resize or non-64-block DCT reference",
        )
    results: dict[str, dict[str, Any]] = {}
    for name in contract["pipelines"]["enabled"]:
        path = output_dir / f"pipeline_{name}.json"
        if not path.is_file():
            failures.append(f"missing pipeline result: {path}")
            continue
        result = _load_result(path)
        _validate_result(name, result, contract, expected_trace, failures)
        results[name] = result

    source_changes: list[str] = []
    for label, fingerprint in contract.get("source_fingerprints", {}).items():
        path = Path(fingerprint["path"])
        if not path.is_file() or sha256_file(path) != fingerprint.get("sha256"):
            source_changes.append(label)
    _require(not source_changes, failures, f"runtime source files changed: {source_changes}")

    access = contract.get("dataset", {}).get("block_major_access") or {}
    changed_schedules: list[str] = []
    for fingerprint in access.get("active_output_schedules", []):
        path = Path(fingerprint["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(fingerprint.get("size_bytes", -1))
            or sha256_file(path) != fingerprint.get("sha256")
        ):
            changed_schedules.append(str(path))
    _require(
        not changed_schedules,
        failures,
        f"active-output schedule sidecars changed after contract snapshot: {changed_schedules}",
    )

    semantic_pairs = [
        ("dct_major_full", "dct_major_pushdown", True),
        ("dct_major_legacy_pushdown", "dct_major_pushdown", True),
        ("dct_major_pushdown", "image_major_pushdown", True),
        ("dct_major_pushdown", "image_major_v2_pushdown", True),
        ("dct_major_pushdown", "image_major_v3_pushdown", True),
        ("image_major_v2_pushdown", "image_major_v3_pushdown", True),
        ("dct_major_pushdown", "rgbnomore", True),
        ("dali", "pytorch", False),
    ]
    semantic_comparisons = [
        _semantic_compare(left, right, results[left], results[right], contract, strict=strict, failures=failures)
        for left, right, strict in semantic_pairs
        if left in results and right in results
    ]
    aggregate_list = [_aggregate(contract, results[name]) for name in contract["pipelines"]["enabled"] if name in results]
    aggregates = {item["pipeline"]: item for item in aggregate_list}
    maximum_cv = float(contract["stability_gates"]["maximum_hot_cv"])
    maximum_endpoint_drift = float(contract["stability_gates"]["maximum_endpoint_drift"])
    for name, aggregate in aggregates.items():
        cv = float(aggregate["throughput_images_per_s"]["cv_population"])
        _require(cv <= maximum_cv, failures, f"{name}: hot throughput CV {cv:.6f} exceeds {maximum_cv:.6f}")
        endpoint_drift = float(aggregate["throughput_endpoint_drift"])
        _require(
            endpoint_drift <= maximum_endpoint_drift,
            failures,
            f"{name}: throughput endpoint drift {endpoint_drift:.6f} exceeds {maximum_endpoint_drift:.6f}",
        )
    physical_evidence = _physical_evidence(contract, aggregates, failures)
    planless_resource_evidence = _planless_resource_evidence(aggregates, failures)

    speedups: dict[str, float] = {}
    throughput = {name: float(item["throughput_images_per_s"]["p50"]) for name, item in aggregates.items()}
    cold_throughput = {
        name: float(item["cold_start"]["throughput_images_per_s"])
        for name, item in aggregates.items()
    }

    def ratio(label: str, numerator: str, denominator: str) -> None:
        if numerator in throughput and denominator in throughput and throughput[denominator] > 0.0:
            speedups[label] = throughput[numerator] / throughput[denominator]

    ratio("crop_pushdown_over_full", "dct_major_pushdown", "dct_major_full")
    ratio("planless_over_legacy_pushdown", "dct_major_pushdown", "dct_major_legacy_pushdown")
    ratio("dct_major_over_image_major", "dct_major_pushdown", "image_major_pushdown")
    ratio("dct_major_over_image_major_v2", "dct_major_pushdown", "image_major_v2_pushdown")
    ratio("dct_major_over_image_major_v3", "dct_major_pushdown", "image_major_v3_pushdown")
    ratio("image_major_v3_over_v2", "image_major_v3_pushdown", "image_major_v2_pushdown")
    ratio("dct_major_over_dali", "dct_major_pushdown", "dali")
    ratio("dct_major_over_pytorch", "dct_major_pushdown", "pytorch")
    ratio("dali_over_pytorch", "dali", "pytorch")

    cold_speedups: dict[str, float] = {}

    def cold_ratio(label: str, numerator: str, denominator: str) -> None:
        if numerator in cold_throughput and denominator in cold_throughput and cold_throughput[denominator] > 0.0:
            cold_speedups[label] = cold_throughput[numerator] / cold_throughput[denominator]

    cold_ratio("crop_pushdown_over_full", "dct_major_pushdown", "dct_major_full")
    cold_ratio("planless_over_legacy_pushdown", "dct_major_pushdown", "dct_major_legacy_pushdown")
    cold_ratio("dct_major_over_image_major", "dct_major_pushdown", "image_major_pushdown")
    cold_ratio("dct_major_over_image_major_v2", "dct_major_pushdown", "image_major_v2_pushdown")
    cold_ratio("dct_major_over_image_major_v3", "dct_major_pushdown", "image_major_v3_pushdown")
    cold_ratio("image_major_v3_over_v2", "image_major_v3_pushdown", "image_major_v2_pushdown")
    cold_ratio("dct_major_over_dali", "dct_major_pushdown", "dali")
    cold_ratio("dct_major_over_pytorch", "dct_major_pushdown", "pytorch")
    cold_ratio("dali_over_pytorch", "dali", "pytorch")

    dct_storage = contract["dataset"]["dct_major_storage"]
    access_storage = contract["dataset"].get("block_major_access") or {}
    fls_payload_bytes = sum(
        int(payload["size_bytes"])
        for payload in dct_storage["payloads"]
        if payload.get("kind") == "fls"
    )
    metadata_bytes = sum(
        int(payload["size_bytes"])
        for payload in dct_storage["payloads"]
        if payload.get("kind") == "metadata"
    )
    sidecar_bytes = int(
        access_storage.get(
            "all_sidecar_bytes",
            access_storage.get("descriptor_bytes", 0),
        )
    )
    manifest_bytes = int(dct_storage["manifest"]["size_bytes"])
    jpeg_bytes = int(
        contract["dataset"].get("jpeg_full_dataset_bytes")
        or contract["dataset"]["jpeg_selected_bytes"]
    )
    raw_rgb_bytes = int(contract["dataset"]["raw_rgb_selected_bytes"])
    total_block_major_bytes = fls_payload_bytes + metadata_bytes + manifest_bytes + sidecar_bytes
    storage_evidence = {
        "jpeg_bytes": jpeg_bytes,
        "fls_payload_bytes": fls_payload_bytes,
        "metadata_bytes": metadata_bytes + manifest_bytes,
        "sidecar_bytes": sidecar_bytes,
        "total_block_major_bytes": total_block_major_bytes,
        "relative_to_jpeg": total_block_major_bytes / jpeg_bytes if jpeg_bytes else 0.0,
        "relative_to_raw_rgb": total_block_major_bytes / raw_rgb_bytes if raw_rgb_bytes else 0.0,
    }

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "ok": not failures,
        "failures": failures,
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256_json(contract),
        "workload": contract["workload"]["kind"],
        "preprocess_profile": contract["preprocess"].get("profile"),
        "expected_sample_trace": expected_trace,
        "aggregates": aggregate_list,
        "speedups": speedups,
        "cold_speedups": cold_speedups,
        "physical_evidence": physical_evidence,
        "planless_resource_evidence": planless_resource_evidence,
        "block_major_access_storage": contract["dataset"].get("block_major_access"),
        "storage_evidence": storage_evidence,
        "semantic_comparisons": semantic_comparisons,
        "source_changes": source_changes,
        "pipeline_results": [results[name] for name in contract["pipelines"]["enabled"] if name in results],
    }
    write_json(output_dir / "results.json", summary)
    write_json(
        output_dir / "validation.json",
        {
            "ok": summary["ok"],
            "failures": failures,
            "physical_evidence": physical_evidence,
            "planless_resource_evidence": planless_resource_evidence,
            "block_major_access_storage": contract["dataset"].get("block_major_access"),
            "semantic_comparisons": semantic_comparisons,
        },
    )
    _write_csv(
        output_dir / "results.csv",
        summary["pipeline_results"],
        contract["dataset"].get("block_major_access"),
    )
    _write_report(output_dir / "report.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate(args.contract, args.output_dir)
    print("RESULT_JSON " + json.dumps({"ok": summary["ok"], "failures": len(summary["failures"])}, sort_keys=True))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
