#!/usr/bin/env python3
"""Validate contract compliance, semantic alignment, and summarize all pipelines."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common import (
    PIPELINES,
    RESULT_SCHEMA,
    distribution,
    finite_number,
    load_contract,
    load_sample_manifest,
    measured_samples,
    sample_trace,
    sha256_file,
    sha256_json,
    source_tree_metadata,
    write_json,
)


def _require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def _load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"pipeline result must be an object: {path}")
    return payload


def _validate_pipeline_result(
    pipeline: str,
    payload: dict[str, Any],
    contract: dict[str, Any],
    expected_trace: dict[str, Any],
    failures: list[str],
) -> None:
    label = f"pipeline {pipeline}"
    _require(payload.get("schema_version") == RESULT_SCHEMA, failures, f"{label}: bad result schema")
    _require(payload.get("pipeline") == pipeline, failures, f"{label}: pipeline name mismatch")
    _require(payload.get("contract_sha256") == sha256_json(contract), failures, f"{label}: contract hash mismatch")
    _require(
        payload.get("sample_manifest_sha256") == contract["dataset"]["manifest_sha256"],
        failures,
        f"{label}: sample manifest hash mismatch",
    )
    expected_domain = "dct" if pipeline in ("galp", "galp_legacy", "rgbnomore") else "rgb"
    _require(payload.get("domain") == expected_domain, failures, f"{label}: expected domain {expected_domain}")
    _require(payload.get("execution") == contract["execution"], failures, f"{label}: execution contract mismatch")
    model = payload.get("model", {})
    expected_model = contract["models"][expected_domain]
    for key in ("architecture", "input_domain", "recipe_id", "checkpoint", "checkpoint_sha256"):
        _require(model.get(key) == expected_model.get(key), failures, f"{label}: model.{key} mismatch")
    semantic = Path(str(payload.get("semantic_artifact", "")))
    _require(semantic.is_file(), failures, f"{label}: missing semantic artifact {semantic}")
    if semantic.is_file():
        _require(sha256_file(semantic) == payload.get("semantic_artifact_sha256"), failures, f"{label}: semantic artifact hash mismatch")

    repeats = payload.get("repeats")
    expected_repeats = int(contract["execution"]["repeats"])
    _require(isinstance(repeats, list) and len(repeats) == expected_repeats, failures, f"{label}: repeat count mismatch")
    if not isinstance(repeats, list):
        return
    expected_images = int(contract["execution"]["batch_size"]) * int(contract["execution"]["measurement_batches"])
    for index, record in enumerate(repeats):
        record_label = f"{label} repeat {index}"
        _require(record.get("repeat") == index, failures, f"{record_label}: repeat index mismatch")
        _require(record.get("images") == expected_images, failures, f"{record_label}: image count mismatch")
        for key in ("seconds", "throughput_images_per_s", "accuracy_top1", "accuracy_top5", "cpu_process_seconds"):
            _require(finite_number(record.get(key)), failures, f"{record_label}: {key} is not finite")
        _require(float(record.get("seconds", 0.0)) > 0.0, failures, f"{record_label}: seconds must be positive")
        _require(float(record.get("throughput_images_per_s", 0.0)) > 0.0, failures, f"{record_label}: throughput must be positive")
        top1 = float(record.get("accuracy_top1", -1.0))
        top5 = float(record.get("accuracy_top5", -1.0))
        _require(0.0 <= top1 <= top5 <= 1.0, failures, f"{record_label}: invalid top1/top5")
        latency = record.get("end_to_end_latency_ms")
        _require(isinstance(latency, dict), failures, f"{record_label}: latency distribution missing")
        if isinstance(latency, dict):
            _require(latency.get("count") == contract["execution"]["measurement_batches"], failures, f"{record_label}: latency count mismatch")
            _require(float(latency.get("mean", 0.0)) > 0.0, failures, f"{record_label}: latency mean must be positive")
        _require(record.get("sample_trace", {}).get("sha256") == expected_trace["sha256"], failures, f"{record_label}: sample trace mismatch")
        _require(isinstance(record.get("peak_gpu_memory_allocated_bytes"), int), failures, f"{record_label}: peak memory missing")
        host_rss_before = record.get("host_process_rss_before_measurement_bytes")
        host_rss_after = record.get("host_process_rss_after_measurement_bytes")
        host_peak_rss = record.get("host_process_peak_rss_bytes")
        _require(
            isinstance(host_rss_before, int) and host_rss_before > 0,
            failures,
            f"{record_label}: host RSS before measurement is missing",
        )
        _require(
            isinstance(host_rss_after, int) and host_rss_after > 0,
            failures,
            f"{record_label}: host RSS after measurement is missing",
        )
        _require(
            isinstance(host_peak_rss, int)
            and host_peak_rss > 0
            and isinstance(host_rss_after, int)
            and host_peak_rss >= host_rss_after,
            failures,
            f"{record_label}: host peak RSS is missing or smaller than final RSS",
        )
        _require(
            record.get("host_process_memory_scope")
            == (
                "linux_main_pipeline_process_VmRSS_and_lifetime_VmHWM_includes_python_torch_"
                "and_native_pipeline_allocations_excludes_loader_worker_processes"
            ),
            failures,
            f"{record_label}: host process memory scope is missing",
        )
        _require(isinstance(record.get("stage_breakdown_ms"), dict), failures, f"{record_label}: stage breakdown missing")
        native_counters = record.get("native_counters")
        _require(isinstance(native_counters, dict), failures, f"{record_label}: native counters missing")
        if pipeline == "galp" and contract["pipelines"]["galp"]["preprocess"] == "rgbnomore-val-pushdown":
            if isinstance(native_counters, dict):
                _require(
                    int(native_counters.get("fixed_transform_items", -1)) == 0
                    and int(native_counters.get("planless_image_descriptors", 0)) == expected_images
                    and int(native_counters.get("host_expanded_transform_items_created", -1)) == 0
                    and int(native_counters.get("host_output_block_source_lists_created", -1)) == 0
                    and int(native_counters.get("host_global_transform_sort_items", -1)) == 0,
                    failures,
                    f"{record_label}: planless transformed-grid structural counters failed",
                )
                _require(
                    int(native_counters.get("device_mapping_fused_batches", 0))
                    == int(contract["execution"]["measurement_batches"]),
                    failures,
                    f"{record_label}: device mapping was not fused for every batch",
                )
                _require(
                    int(native_counters.get("plan_cache_hits", -1)) == 0,
                    failures,
                    f"{record_label}: exact-batch plan cache was exercised",
                )
                _require(
                    int(native_counters.get("exact_batch_plan_cache_enabled_batches", -1)) == 0,
                    failures,
                    f"{record_label}: exact-batch plan cache was enabled",
                )
                _require(
                    int(native_counters.get("decoded_rowgroup_cache_enabled_batches", -1)) == 0,
                    failures,
                    f"{record_label}: decoded-rowgroup cache was enabled",
                )
                _require(
                    int(native_counters.get("rowgroup_storage_bytes_read", 0)) > 0,
                    failures,
                    f"{record_label}: compressed rowgroup read-byte counter is missing",
                )
                _require(
                    int(native_counters.get("galp_native_device_peak_in_use_bytes", 0)) > 0
                    and int(native_counters.get("galp_native_device_cuda_allocation_count", 0)) > 0,
                    failures,
                    f"{record_label}: GALP native device allocation counters are missing",
                )
                _require(
                    int(native_counters.get("projection_items", 0)) == 0
                    and int(native_counters.get("decoded_projection_items", 0)) == 0
                    and int(native_counters.get("project_decoded_ycbcr_grid_launches", 0)) == 0,
                    failures,
                    f"{record_label}: generic projection fallback was used",
                )
                gate = contract.get("performance_gates", {}).get("galp", {})
                manifest_version = int(contract["pipelines"]["galp"].get("manifest_version", 1))
                if manifest_version >= int(gate.get("image_major_manifest_minimum_version", 2)):
                    expected_batches = int(contract["execution"]["measurement_batches"])
                    structural_expectations = {
                        "rowgroups": expected_images * int(gate.get("rowgroups_per_image", 1)),
                        "worksets": expected_batches * int(gate.get("worksets_per_batch", 1)),
                        "internal_syncs": expected_batches * int(gate.get("internal_syncs_per_batch", 1)),
                        "decode_kernels": expected_batches * int(gate.get("decode_kernels_per_batch", 1)),
                    }
                    for counter, expected in structural_expectations.items():
                        _require(
                            int(native_counters.get(counter, -1)) == expected,
                            failures,
                            f"{record_label}: {counter}={native_counters.get(counter)}; expected {expected}",
                        )
                native_per_batch = record.get("stage_breakdown_ms", {}).get("native_per_batch_ms", {})
                stage_gates = (
                    ("planning_seconds", "p50", gate.get("planning_median_ms_max")),
                    ("planning_seconds", "p95", gate.get("planning_p95_ms_max")),
                    ("device_mapping_seconds", "p50", gate.get("device_mapping_median_ms_max")),
                    (
                        "device_mapping_plus_fixed_transform_seconds",
                        "p50",
                        gate.get("device_mapping_plus_fixed_transform_median_ms_max"),
                    ),
                )
                for stage, statistic, maximum in stage_gates:
                    if maximum is None:
                        continue
                    summary = native_per_batch.get(stage, {}) if isinstance(native_per_batch, dict) else {}
                    actual = summary.get(statistic) if isinstance(summary, dict) else None
                    _require(
                        finite_number(actual) and float(actual) <= float(maximum),
                        failures,
                        f"{record_label}: {stage}.{statistic}={actual} ms exceeds {maximum} ms",
                    )
                    _require(
                        summary.get("count") == contract["execution"]["measurement_batches"],
                        failures,
                        f"{record_label}: {stage} per-batch distribution is incomplete",
                    )

        if (
            pipeline == "galp_legacy"
            and contract["pipelines"]["galp_legacy"]["preprocess"] == "rgbnomore-val-pushdown"
            and isinstance(native_counters, dict)
        ):
            _require(
                int(native_counters.get("planless_image_descriptors", -1)) == 0
                and int(native_counters.get("fixed_transform_items", 0)) > 0
                and int(native_counters.get("host_expanded_transform_items_created", 0)) > 0
                and int(native_counters.get("host_global_transform_sort_items", 0)) > 0,
                failures,
                f"{record_label}: legacy A/B expanded-graph structural counters failed",
            )
            _require(
                int(native_counters.get("exact_batch_plan_cache_enabled_batches", -1)) == 0,
                failures,
                f"{record_label}: legacy A/B exact-batch cache must remain disabled",
            )
            _require(
                int(native_counters.get("decoded_rowgroup_cache_enabled_batches", -1)) == 0,
                failures,
                f"{record_label}: legacy A/B decoded-rowgroup cache must remain disabled",
            )


def _array_diff(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if expected.shape != actual.shape:
        return {"shape_match": False, "expected_shape": list(expected.shape), "actual_shape": list(actual.shape)}
    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    return {
        "shape_match": True,
        "shape": list(expected.shape),
        "max_abs": float(delta.max()) if delta.size else 0.0,
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.square(delta).mean())) if delta.size else 0.0,
    }


def _logit_diff(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    result = _array_diff(expected, actual)
    if not result.get("shape_match"):
        return result
    expected64 = expected.astype(np.float64)
    actual64 = actual.astype(np.float64)
    numerator = np.sum(expected64 * actual64, axis=1)
    denominator = np.linalg.norm(expected64, axis=1) * np.linalg.norm(actual64, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-30)
    result["cosine_mean"] = float(cosine.mean())
    result["top1_agreement"] = float(np.mean(np.argmax(expected, axis=1) == np.argmax(actual, axis=1)))
    return result


def _semantic_compare(
    left_name: str,
    right_name: str,
    left_path: Path,
    right_path: Path,
    thresholds: dict[str, Any],
    enforcement: str,
    failures: list[str],
) -> dict[str, Any]:
    left = np.load(left_path, allow_pickle=False)
    right = np.load(right_path, allow_pickle=False)
    result: dict[str, Any] = {
        "pipelines": [left_name, right_name],
        "identity": {},
        "inputs": [],
        "logits": {},
        "thresholds": thresholds,
        "enforcement": enforcement,
        "ok": True,
    }
    for key in ("ordinals", "labels"):
        equal = key in left and key in right and np.array_equal(left[key], right[key])
        result["identity"][key + "_equal"] = bool(equal)
        if not equal:
            failures.append(f"semantic {left_name}/{right_name}: {key} differ")
            result["ok"] = False

    prediction_identity_ok = True
    for key in ("prediction_ordinals", "prediction_labels"):
        equal = key in left and key in right and np.array_equal(left[key], right[key])
        result["identity"][key + "_equal"] = bool(equal)
        prediction_identity_ok = prediction_identity_ok and bool(equal)
        if not equal:
            if enforcement == "strict":
                failures.append(f"semantic {left_name}/{right_name}: full {key} differ")
            result["ok"] = False
    left_metadata = json.loads(str(left["metadata_json"].item()))
    right_metadata = json.loads(str(right["metadata_json"].item()))
    expected_predictions = int(
        thresholds.get("full_prediction_sample_count", left_metadata.get("prediction_agreement_sample_count", 0))
    )
    left_top1 = left["top1_predictions"] if "top1_predictions" in left else np.asarray([])
    right_top1 = right["top1_predictions"] if "top1_predictions" in right else np.asarray([])
    left_top5 = left["top5_predictions"] if "top5_predictions" in left else np.asarray([])
    right_top5 = right["top5_predictions"] if "top5_predictions" in right else np.asarray([])
    labels = left["prediction_labels"] if "prediction_labels" in left else np.asarray([])
    prediction_shape_ok = (
        len(left_top1) == expected_predictions
        and left_metadata.get("prediction_agreement_sample_count") == expected_predictions
        and right_metadata.get("prediction_agreement_sample_count") == expected_predictions
        and left_top1.shape == right_top1.shape
        and left_top5.shape == right_top5.shape
        and len(labels) == expected_predictions
    )
    top1_agreement = (
        float(np.mean(left_top1 == right_top1)) if prediction_shape_ok and expected_predictions else 0.0
    )
    left_top1_correct = int(np.sum(left_top1 == labels)) if prediction_shape_ok else -1
    right_top1_correct = int(np.sum(right_top1 == labels)) if prediction_shape_ok else -1
    left_top5_correct = (
        int(np.sum(np.any(left_top5 == labels[:, None], axis=1))) if prediction_shape_ok else -1
    )
    right_top5_correct = (
        int(np.sum(np.any(right_top5 == labels[:, None], axis=1))) if prediction_shape_ok else -1
    )
    prediction_ok = bool(
        prediction_identity_ok
        and prediction_shape_ok
        and top1_agreement >= float(thresholds.get("full_prediction_top1_agreement_min", 0.0))
        and left_top1_correct == right_top1_correct
        and left_top5_correct == right_top5_correct
    )
    result["full_prediction"] = {
        "count": expected_predictions,
        "shape_match": prediction_shape_ok,
        "top1_agreement": top1_agreement,
        "left_top1_correct": left_top1_correct,
        "right_top1_correct": right_top1_correct,
        "left_top5_correct": left_top5_correct,
        "right_top5_correct": right_top5_correct,
        "within_tolerance": prediction_ok,
    }
    if not prediction_ok:
        if enforcement == "strict":
            failures.append(f"semantic {left_name}/{right_name}: full prediction audit failed")
        result["ok"] = False

    input_keys = sorted(key for key in left.files if key.startswith("input_"))
    if input_keys != sorted(key for key in right.files if key.startswith("input_")):
        failures.append(f"semantic {left_name}/{right_name}: input tensor sets differ")
        result["ok"] = False
    for key in input_keys:
        comparison = {"name": key, **_array_diff(left[key], right[key])}
        comparison["within_tolerance"] = bool(
            comparison.get("shape_match")
            and comparison.get("max_abs", math.inf) <= float(thresholds["input_max_abs"])
            and comparison.get("mean_abs", math.inf) <= float(thresholds["input_mean_abs"])
        )
        if not comparison["within_tolerance"]:
            if enforcement == "strict":
                failures.append(f"semantic {left_name}/{right_name}: {key} exceeds tolerance")
            result["ok"] = False
        result["inputs"].append(comparison)

    logits = _logit_diff(left["logits"], right["logits"])
    logits["within_tolerance"] = bool(
        logits.get("shape_match")
        and logits.get("max_abs", math.inf) <= float(thresholds.get("logit_max_abs", math.inf))
        and logits.get("cosine_mean", -1.0) >= float(thresholds["logit_cosine_min"])
        and logits.get("top1_agreement", -1.0)
        >= float(thresholds.get("logit_top1_agreement_min", 0.0))
    )
    if not logits["within_tolerance"]:
        if enforcement == "strict":
            failures.append(f"semantic {left_name}/{right_name}: logits exceed tolerance")
        result["ok"] = False
    result["logits"] = logits
    left.close()
    right.close()
    return result


def _aggregate_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    repeats = payload["repeats"]
    aggregate_repeats = repeats[1:] if payload["execution"].get("aggregate_exclude_first_repeat") and len(repeats) > 1 else repeats
    return {
        "pipeline": payload["pipeline"],
        "domain": payload["domain"],
        "aggregate_repeat_indices": [record["repeat"] for record in aggregate_repeats],
        "throughput_images_per_s": distribution([record["throughput_images_per_s"] for record in aggregate_repeats]),
        "mean_end_to_end_latency_ms": distribution([record["end_to_end_latency_ms"]["mean"] for record in aggregate_repeats]),
        "p95_end_to_end_latency_ms": distribution([record["end_to_end_latency_ms"]["p95"] for record in aggregate_repeats]),
        "accuracy_top1": distribution([record["accuracy_top1"] for record in aggregate_repeats]),
        "accuracy_top5": distribution([record["accuracy_top5"] for record in aggregate_repeats]),
        "peak_gpu_memory_allocated_bytes": distribution([record["peak_gpu_memory_allocated_bytes"] for record in aggregate_repeats]),
        "peak_gpu_memory_reserved_bytes": distribution([record["peak_gpu_memory_reserved_bytes"] for record in aggregate_repeats]),
        "peak_gpu_memory_scope": repeats[0]["peak_gpu_memory_scope"],
        "host_process_rss_after_measurement_bytes": distribution(
            [record["host_process_rss_after_measurement_bytes"] for record in aggregate_repeats]
        ),
        "host_process_peak_rss_bytes": distribution(
            [record["host_process_peak_rss_bytes"] for record in aggregate_repeats]
        ),
        "host_process_memory_scope": repeats[0]["host_process_memory_scope"],
    }


def _evaluate_performance_gates(
    contract: dict[str, Any], aggregates: Sequence[dict[str, Any]], failures: list[str]
) -> list[dict[str, Any]]:
    configured = contract.get("performance_gates", {}).get("galp", {})
    if "galp" not in contract["pipelines"]["enabled"]:
        return []
    galp = next((item for item in aggregates if item["pipeline"] == "galp"), None)
    if galp is None:
        return []
    dali = next((item for item in aggregates if item["pipeline"] == "dali"), None)
    gates: list[dict[str, Any]] = []

    def add_gate(metric: str, actual: float, target: float, passed: bool, message: str, comparison: str) -> None:
        if not passed:
            failures.append(message)
        gates.append(
            {
                "pipeline": "galp",
                "metric": metric,
                "target": target,
                "actual": actual,
                "comparison": comparison,
                "ok": passed,
            }
        )

    minimum = configured.get("minimum_median_throughput_images_per_s")
    if minimum is not None:
        actual = float(galp["throughput_images_per_s"]["p50"])
        target = float(minimum)
        add_gate(
            "median_throughput_images_per_s",
            actual,
            target,
            actual >= target,
            f"pipeline galp: median throughput {actual:.3f} img/s is below required {target:.3f} img/s",
            ">=",
        )

    ratio_target = configured.get("minimum_hot_median_to_dali_hot_median_ratio")
    if ratio_target is not None:
        target = float(ratio_target)
        actual = (
            float(galp["throughput_images_per_s"]["p50"])
            / float(dali["throughput_images_per_s"]["p50"])
            if dali is not None and float(dali["throughput_images_per_s"]["p50"]) > 0.0
            else 0.0
        )
        add_gate(
            "hot_median_to_dali_hot_median_ratio",
            actual,
            target,
            actual >= target,
            f"pipeline galp: hot median ratio to same-round DALI {actual:.4f} is below {target:.4f}",
            ">=",
        )

    if configured.get("require_hot_min_above_dali_hot_median"):
        actual = float(galp["throughput_images_per_s"]["min"])
        target = float(dali["throughput_images_per_s"]["p50"]) if dali is not None else math.inf
        add_gate(
            "hot_min_throughput_above_dali_hot_median",
            actual,
            target,
            actual > target,
            f"pipeline galp: hot minimum {actual:.3f} img/s does not exceed DALI hot median {target:.3f} img/s",
            ">",
        )

    cv_maximum = configured.get("maximum_hot_throughput_cv")
    if cv_maximum is not None:
        target = float(cv_maximum)
        for pipeline_name, aggregate in (("galp", galp), ("dali", dali)):
            actual = (
                float(aggregate["throughput_images_per_s"].get("cv_population", math.inf))
                if aggregate is not None
                else math.inf
            )
            add_gate(
                f"{pipeline_name}_hot_throughput_cv",
                actual,
                target,
                actual <= target,
                f"pipeline {pipeline_name}: hot throughput CV {actual:.4f} exceeds {target:.4f}",
                "<=",
            )
    return gates


CSV_COLUMNS = [
    "pipeline",
    "domain",
    "repeat",
    "images",
    "seconds",
    "throughput_images_per_s",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "accuracy_top1",
    "accuracy_top5",
    "cpu_process_seconds",
    "host_process_rss_after_measurement_bytes",
    "host_process_peak_rss_bytes",
    "host_process_memory_scope",
    "peak_gpu_memory_allocated_bytes",
    "peak_gpu_memory_reserved_bytes",
    "batch_size",
    "workers",
    "warmup_batches",
    "measurement_batches",
    "precision",
    "device",
    "checkpoint",
    "checkpoint_sha256",
    "sample_manifest_sha256",
    "contract_sha256",
    "pipeline_preprocess",
]


def _write_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for payload in results:
            for record in payload["repeats"]:
                latency = record["end_to_end_latency_ms"]
                row = {
                    "pipeline": payload["pipeline"],
                    "domain": payload["domain"],
                    "repeat": record["repeat"],
                    "images": record["images"],
                    "seconds": record["seconds"],
                    "throughput_images_per_s": record["throughput_images_per_s"],
                    "latency_mean_ms": latency["mean"],
                    "latency_p50_ms": latency["p50"],
                    "latency_p95_ms": latency["p95"],
                    "latency_p99_ms": latency["p99"],
                    "accuracy_top1": record["accuracy_top1"],
                    "accuracy_top5": record["accuracy_top5"],
                    "cpu_process_seconds": record["cpu_process_seconds"],
                    "host_process_rss_after_measurement_bytes": record[
                        "host_process_rss_after_measurement_bytes"
                    ],
                    "host_process_peak_rss_bytes": record["host_process_peak_rss_bytes"],
                    "host_process_memory_scope": record["host_process_memory_scope"],
                    "peak_gpu_memory_allocated_bytes": record["peak_gpu_memory_allocated_bytes"],
                    "peak_gpu_memory_reserved_bytes": record["peak_gpu_memory_reserved_bytes"],
                    "checkpoint": payload["model"]["checkpoint"],
                    "checkpoint_sha256": payload["model"]["checkpoint_sha256"],
                    "sample_manifest_sha256": payload["sample_manifest_sha256"],
                    "contract_sha256": payload["contract_sha256"],
                    "pipeline_preprocess": payload.get("pipeline_config", {}).get("preprocess", ""),
                    **{key: payload["execution"][key] for key in ("batch_size", "workers", "warmup_batches", "measurement_batches", "precision", "device")},
                }
                writer.writerow(row)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    aggregates = summary["aggregates"]
    lines = [
        "# Canonical end-to-end benchmark report",
        "",
        f"Validation: **{'PASS' if summary['ok'] else 'FAIL'}**",
        "",
        "| Pipeline | Domain | Throughput median (img/s) | Mean latency median (ms/batch) | Peak host RSS (MiB) | Top-1 median | Top-5 median |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['pipeline']} | {item['domain']} | {item['throughput_images_per_s']['p50']:.3f} | "
            f"{item['mean_end_to_end_latency_ms']['p50']:.3f} | "
            f"{item['host_process_peak_rss_bytes']['max'] / (1024 * 1024):.1f} | "
            f"{item['accuracy_top1']['p50']:.4f} | "
            f"{item['accuracy_top5']['p50']:.4f} |"
        )
    lines.extend(["", "## Performance gates", ""])
    if summary["performance_gates"]:
        for gate in summary["performance_gates"]:
            lines.append(
                f"- {gate['pipeline']} {gate['metric']}: "
                f"{'PASS' if gate['ok'] else 'FAIL'}; {gate['actual']:.3f} "
                f"{gate.get('comparison', '>=')} {gate['target']:.3f}."
            )
    else:
        lines.append("- No throughput gate applies to this preset.")
    lines.extend(["", "## Comparability", ""])
    for comparison in summary["comparability"]:
        lines.append(f"- **{comparison['pair']}**: `{comparison['classification']}` — {comparison['reason']}")
    lines.extend(["", "## Semantic validation", ""])
    for comparison in summary["semantic_validation"]["comparisons"]:
        semantic_status = (
            "PASS"
            if comparison["ok"]
            else "DRIFT (diagnostic)"
            if comparison.get("enforcement") == "diagnostic"
            else "FAIL"
        )
        lines.append(
            f"- {comparison['pipelines'][0]} vs {comparison['pipelines'][1]}: "
            f"{semantic_status}; logits top-1 agreement "
            f"{comparison['logits'].get('top1_agreement', float('nan')):.4f}."
        )
    lines.extend(["", "## Caveats", ""])
    for caveat in summary["caveats"]:
        lines.append(f"- {caveat}")
    if summary["failures"]:
        lines.extend(["", "## Validation failures", ""])
        lines.extend(f"- {failure}" for failure in summary["failures"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_and_summarize(contract_path: Path, output_dir: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    _, samples = load_sample_manifest(Path(contract["dataset"]["sample_manifest"]), contract["dataset"]["manifest_sha256"])
    measured = measured_samples(
        samples,
        int(contract["execution"]["batch_size"]),
        int(contract["execution"]["warmup_batches"]),
        int(contract["execution"]["measurement_batches"]),
    )
    expected_trace = sample_trace(measured)
    failures: list[str] = []
    observed_source_revisions: dict[str, Any] = {}
    for source_name, expected_revision in contract.get("source_revisions", {}).items():
        expected_files = expected_revision.get("runtime_file_sha256", {})
        if not isinstance(expected_files, dict):
            failures.append(f"source revision {source_name}: runtime file hashes are missing")
            continue
        observed_revision = source_tree_metadata(
            Path(str(expected_revision.get("root", ""))),
            list(expected_files),
        )
        observed_source_revisions[source_name] = observed_revision
        _require(
            expected_revision.get("benchmark_source_clean") is True,
            failures,
            f"source revision {source_name}: contract was not created from committed runtime sources",
        )
        _require(
            observed_revision.get("benchmark_source_clean") is True,
            failures,
            f"source revision {source_name}: runtime sources became dirty during the benchmark",
        )
        _require(
            observed_revision.get("git_commit") == expected_revision.get("git_commit"),
            failures,
            f"source revision {source_name}: git commit changed during the benchmark",
        )
        _require(
            observed_revision.get("runtime_file_sha256") == expected_files,
            failures,
            f"source revision {source_name}: runtime file hashes changed during the benchmark",
        )
    results: dict[str, dict[str, Any]] = {}
    for pipeline in contract["pipelines"]["enabled"]:
        path = output_dir / f"pipeline_{pipeline}.json"
        if not path.is_file():
            failures.append(f"pipeline result is missing: {path}")
            continue
        payload = _load_result(path)
        results[pipeline] = payload
        _validate_pipeline_result(pipeline, payload, contract, expected_trace, failures)

    semantic_comparisons: list[dict[str, Any]] = []
    groups = contract["semantic_validation"]["comparison_groups"]
    for group in groups:
        left, right = group["pipelines"]
        if left not in results or right not in results:
            continue
        semantic_comparisons.append(
            _semantic_compare(
                left,
                right,
                Path(results[left]["semantic_artifact"]),
                Path(results[right]["semantic_artifact"]),
                group["thresholds"],
                group.get("enforcement", "strict" if group.get("domain") == "dct" else "diagnostic"),
                failures,
            )
        )

    semantic_by_pair = {tuple(item["pipelines"]): item for item in semantic_comparisons}
    ordered_results = [results[name] for name in contract["pipelines"]["enabled"] if name in results]
    aggregates = [_aggregate_pipeline(payload) for payload in ordered_results]
    performance_gates = _evaluate_performance_gates(contract, aggregates, failures)

    architecture_ab: dict[str, Any] | None = None
    if "galp" in results and "galp_legacy" in results:
        planless_aggregate = next(item for item in aggregates if item["pipeline"] == "galp")
        legacy_aggregate = next(item for item in aggregates if item["pipeline"] == "galp_legacy")

        def hot_stage_distribution(pipeline: str, stage: str, statistic: str) -> dict[str, Any]:
            indices = set(
                next(item for item in aggregates if item["pipeline"] == pipeline)["aggregate_repeat_indices"]
            )
            values = []
            for repeat in results[pipeline]["repeats"]:
                if int(repeat["repeat"]) not in indices:
                    continue
                summary = repeat.get("stage_breakdown_ms", {}).get("native_per_batch_ms", {}).get(stage, {})
                if finite_number(summary.get(statistic)):
                    values.append(float(summary[statistic]))
            if not values:
                return {"count": 0}
            return distribution(values)

        planless_throughput = float(planless_aggregate["throughput_images_per_s"]["p50"])
        legacy_throughput = float(legacy_aggregate["throughput_images_per_s"]["p50"])
        architecture_ab = {
            "pipelines": ["galp", "galp_legacy"],
            "same_contract_except_execution_representation": True,
            "shared_native_binary_sha256": contract["pipelines"]["galp"]["native_binary_fingerprint"]["sha256"],
            "semantic_classification": (
                "strict"
                if semantic_by_pair.get(("galp", "galp_legacy"), {}).get("ok", False)
                else "unproven"
            ),
            "planless_hot_median_throughput_images_per_s": planless_throughput,
            "legacy_hot_median_throughput_images_per_s": legacy_throughput,
            "planless_to_legacy_hot_median_speedup": (
                planless_throughput / legacy_throughput if legacy_throughput > 0.0 else math.inf
            ),
            "planless_planning_p50_ms_across_hot_repeats": hot_stage_distribution(
                "galp", "planning_seconds", "p50"
            ),
            "legacy_planning_p50_ms_across_hot_repeats": hot_stage_distribution(
                "galp_legacy", "planning_seconds", "p50"
            ),
        }

    def classification(left: str, right: str) -> str:
        if left not in results or right not in results:
            return "not_run"
        comparison = semantic_by_pair.get((left, right))
        if comparison is None:
            return "not_yet_validated"
        if comparison.get("enforcement") == "diagnostic":
            return "system_level_reference_only"
        if not comparison["ok"] or failures:
            return "not_yet_validated"
        return "strict_system_comparison"

    comparability = [
        {
            "pair": "GALP vs RGB-no-more",
            "classification": classification("galp", "rgbnomore"),
            "reason": "Same canonical samples/order/labels, DCT preprocessing contract, architecture, DCT checkpoint and precision; tensors/logits are validated numerically.",
        },
        {
            "pair": "GALP planless vs GALP legacy",
            "classification": classification("galp", "galp_legacy"),
            "reason": "Same compressed data, sample order, model, checkpoint, precision, cache/prefetch contract, and native binary; only the execution-representation switch differs.",
        },
        {
            "pair": "DALI vs PyTorch",
            "classification": classification("dali", "pytorch"),
            "reason": "Samples/order/labels, architecture, checkpoint and high-level resize/crop/range recipe match, but DALI/nvJPEG and PIL/torchvision preprocessing show implementation-level numerical drift; tensor/logit differences are reported diagnostically.",
        },
        {
            "pair": "DCT-domain pair vs RGB-domain pair",
            "classification": "system_level_reference_only",
            "reason": "DCT and RGB models require different input-domain checkpoints; both checkpoints declare the same RGB-no-more ImageNet ViT-Ti recipe family, but tensors and weights are not interchangeable.",
        },
    ]
    summary = {
        "schema_version": "galp_system_benchmark_summary_v2",
        "ok": not failures,
        "failures": failures,
        "contract": str(contract_path.resolve()),
        "contract_snapshot": contract,
        "source_revisions_observed_at_validation": observed_source_revisions,
        "contract_sha256": sha256_json(contract),
        "sample_manifest_sha256": contract["dataset"]["manifest_sha256"],
        "expected_measured_sample_trace": expected_trace,
        "aggregates": aggregates,
        "performance_gates": performance_gates,
        "architecture_ab": architecture_ab,
        "semantic_validation": {
            "sample_count": contract["semantic_validation"]["sample_count"],
            "comparisons": semantic_comparisons,
            "cross_domain_boundary": "RGB and DCT inputs/logits are not compared elementwise across domain-specific models.",
        },
        "comparability": comparability,
        "caveats": [
            "GALP, DALI, and PyTorch expose different worker abstractions; the configured count is identical where the API permits, and each result records worker_semantics.",
            "Torch peak-memory counters exclude GALP/DALI native allocator memory; peak_gpu_memory_scope makes this limitation explicit.",
            "Top-1/top-5 are computed on the contract's measured subset, not silently presented as full ImageNet validation accuracy.",
			"Adapters may submit bounded prefetch before the current forward; GALP records its ordered batch lookahead depth, and synchronization waits only for the current model stream without draining independent preprocessing streams.",
        ],
        "pipeline_results": ordered_results,
    }
    dct_spec = importlib.util.find_spec("dct_manip")
    if dct_spec is not None and dct_spec.origin and Path(dct_spec.origin).is_file():
        summary["source_revisions_observed_at_validation"]["dct_manip_extension"] = {
            "path": dct_spec.origin,
            "sha256": sha256_file(Path(dct_spec.origin)),
        }
    write_json(output_dir / "results.json", summary)
    _write_csv(output_dir / "results.csv", ordered_results)
    _write_report(output_dir / "report.md", summary)
    write_json(output_dir / "validation.json", {"ok": not failures, "failures": failures, "semantic_validation": summary["semantic_validation"]})
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = validate_and_summarize(args.contract, args.output_dir)
    print(json.dumps({"ok": summary["ok"], "failures": len(summary["failures"])}, sort_keys=True))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
