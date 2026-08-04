#!/usr/bin/env python3
"""Build a machine-readable acceptance report for image-major manifest v3 training."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

MIN_PERFORMANCE_REPEATS = 3
COMPUTE_CONDITION_RELATIVE_TOLERANCE = 0.10
COMPUTE_CONDITION_CV_LIMIT = 0.10


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _artifact(directory: Path | None, name: str) -> dict[str, Any] | None:
    return _read_json(None if directory is None else directory / name)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _coefficient_of_variation(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    if mean == 0.0:
        return 0.0 if all(value == 0.0 for value in values) else None
    return statistics.pstdev(values) / abs(mean)


def _phase_repeats(pipeline: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not pipeline:
        return []
    phase = pipeline.get("phase_results", {}).get("smoke", {})
    repeats = phase.get("repeats", []) if isinstance(phase, Mapping) else []
    return [dict(value) for value in repeats if isinstance(value, Mapping)]


def _nested_number(value: Mapping[str, Any], *path: str) -> float | None:
    current: Any = value
    for name in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(name)
    return _number(current)


def _native_mean(repeats: Sequence[Mapping[str, Any]], name: str) -> float | None:
    total = 0.0
    count = 0.0
    for repeat in repeats:
        aggregate = (
            repeat.get("native_execution_stats_by_phase", {})
            .get("measured", {})
            .get("numeric_aggregates", {})
            .get(name)
        )
        if not isinstance(aggregate, Mapping):
            aggregate = (
                repeat.get("native_execution_stats", {})
                .get("numeric_aggregates", {})
                .get(name)
            )
        if not isinstance(aggregate, Mapping):
            continue
        item_sum = _number(aggregate.get("sum"))
        item_count = _number(aggregate.get("count"))
        if item_sum is not None and item_count:
            total += item_sum
            count += item_count
    return total / count if count else None


def _native_latest(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not repeats:
        return {}
    repeat = repeats[-1]
    phase = repeat.get("native_execution_stats_by_phase", {})
    measured = phase.get("measured", {}) if isinstance(phase, Mapping) else {}
    latest = measured.get("latest", {}) if isinstance(measured, Mapping) else {}
    if not isinstance(latest, Mapping) or not latest:
        latest = repeat.get("native_execution_stats", {}).get("latest", {})
    return dict(latest) if isinstance(latest, Mapping) else {}


def _stage_mean(repeats: Sequence[Mapping[str, Any]], name: str) -> float | None:
    values = [
        value
        for repeat in repeats
        if (value := _nested_number(repeat, "stage_latency_ms", name, "mean")) is not None
    ]
    return _mean(values)


def _training_metrics(directory: Path | None) -> dict[str, Any] | None:
    pipeline = _artifact(directory, "pipeline_galp.json")
    if pipeline is None:
        return None
    repeats = _phase_repeats(pipeline)
    throughputs = [
        value
        for repeat in repeats
        if (value := _number(repeat.get("throughput_images_per_s"))) is not None
    ]
    step_ms = [
        elapsed / steps * 1000.0
        for repeat in repeats
        if (elapsed := _number(repeat.get("measured_region_wall_clock_seconds"))) is not None
        and (steps := _number(repeat.get("measured_steps")))
    ]
    wait_fractions = [
        value
        for repeat in repeats
        if (
            value := _nested_number(
                repeat,
                "loader_measured_metrics",
                "consumer_wait_fraction_of_measured_wall",
            )
        )
        is not None
    ]
    compute_upper_bounds = [
        value
        for repeat in repeats
        if (
            value := _nested_number(
                repeat, "compute_only_diagnostic", "upper_bound_images_per_s"
            )
        )
        is not None
    ]
    stability = [
        repeat.get("native_allocation_stability")
        for repeat in repeats
        if isinstance(repeat.get("native_allocation_stability"), Mapping)
    ]
    status = pipeline.get("status", {})
    probe = pipeline.get("first_step_semantic_probe", {})
    if not isinstance(probe, Mapping):
        probe = {}
    probe_inputs = probe.get("inputs", [])
    input_fingerprints = [
        {
            "shape": value.get("shape"),
            "dtype": value.get("dtype"),
            "sha256": value.get("sha256"),
            "finite": value.get("finite"),
        }
        for value in probe_inputs
        if isinstance(value, Mapping)
    ]
    has_phase_native_stats = bool(repeats) and all(
        isinstance(repeat.get("native_execution_stats_by_phase"), Mapping)
        for repeat in repeats
    )
    return {
        "directory": str(directory),
        "repeat_count": len(repeats),
        "pipeline_status": status,
        "first_step_semantic_probe": {
            "failures": probe.get("failures"),
            "before_warmup": probe.get("before_warmup"),
            "fresh_clone": probe.get("fresh_clone"),
            "formal_repeat_polluted": probe.get("formal_repeat_polluted"),
            "sample_ids": probe.get("sample_ids"),
            "labels": probe.get("labels"),
            "augmentation_decisions": probe.get("augmentation_decisions"),
            "input_fingerprints": input_fingerprints,
            "initial_logits": probe.get("initial_logits"),
            "first_step_loss": probe.get("first_step_loss"),
            "reproducibility": probe.get("reproducibility"),
        },
        "throughput_images_per_s_mean": _mean(throughputs),
        "throughput_images_per_s_cv": _coefficient_of_variation(throughputs),
        "end_to_end_step_ms_mean": _mean(step_ms),
        "loader_wait_fraction_mean": _mean(wait_fractions),
        "compute_upper_bound_images_per_s_mean": _mean(compute_upper_bounds),
        "compute_upper_bound_images_per_s_cv": _coefficient_of_variation(compute_upper_bounds),
        "compute_upper_bound_sample_count": len(compute_upper_bounds),
        "native_phase_scope": (
            "measured batches"
            if has_phase_native_stats
            else "all consumed batches including warmup (legacy artifact)"
        ),
        "planning_ms_per_batch_mean": _native_mean(repeats, "planning_ms"),
        "physical_read_ms_per_batch_mean": (
            _stage_mean(repeats, "read")
            if _stage_mean(repeats, "read") is not None
            else _native_mean(repeats, "sync_rowgroup_read_ms")
        ),
        "decode_ms_per_batch_mean": (
            _stage_mean(repeats, "decode")
            if _stage_mean(repeats, "decode") is not None
            else _native_mean(repeats, "decode_ms")
        ),
        "transform_ms_per_batch_mean": _native_mean(repeats, "fixed_transform_ms"),
        "native_internal_syncs_per_batch_mean": _native_mean(repeats, "internal_sync_count"),
        "coalesced_read_runs_per_batch_mean": _native_mean(repeats, "coalesced_read_run_count"),
        "preadv_calls_per_batch_mean": _native_mean(repeats, "preadv_count"),
        "native_latest": _native_latest(repeats),
        "allocation_stability": stability,
        "all_repeats_ok": bool(repeats) and all(repeat.get("ok") is True for repeat in repeats),
    }


def _baseline_metrics(directory: Path | None) -> dict[str, Any] | None:
    galp = _training_metrics(directory)
    dali_pipeline = _artifact(directory, "pipeline_dali.json")
    dali_repeats = _phase_repeats(dali_pipeline)
    dali_throughputs = [
        value
        for repeat in dali_repeats
        if (value := _number(repeat.get("throughput_images_per_s"))) is not None
    ]
    dali_step_ms = [
        elapsed / steps * 1000.0
        for repeat in dali_repeats
        if (elapsed := _number(repeat.get("measured_region_wall_clock_seconds"))) is not None
        and (steps := _number(repeat.get("measured_steps")))
    ]
    dali_compute = [
        value
        for repeat in dali_repeats
        if (
            value := _nested_number(
                repeat, "compute_only_diagnostic", "upper_bound_images_per_s"
            )
        )
        is not None
    ]
    if galp is None and not dali_repeats:
        return None
    return {
        "directory": str(directory),
        "galp": galp,
        "dali": {
            "throughput_images_per_s_mean": _mean(dali_throughputs),
            "end_to_end_step_ms_mean": _mean(dali_step_ms),
            "compute_upper_bound_images_per_s_mean": _mean(dali_compute),
            "model_path_note": "DALI and GALP use different model/domain paths; this is not a codec-only comparison",
        },
    }


def _normalize_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(contract))
    for name in ("contract_sha256", "provenance"):
        value.pop(name, None)
    pipelines = value.get("pipelines")
    if isinstance(pipelines, dict):
        for name in (
            "galp_manifest",
            "galp_manifest_expectations",
            "galp_manifest_preflight",
            "galp_manifest_sha256",
            "galp_payload_fingerprint_cache",
            "galp_payload_fingerprints",
            "galp_validation_manifest",
            "galp_validation_manifest_expectations",
            "galp_validation_manifest_preflight",
            "galp_validation_manifest_sha256",
            "galp_validation_payload_fingerprint_cache",
            "galp_validation_payload_fingerprints",
        ):
            pipelines.pop(name, None)
    # Each independent run writes the same audited initial state into its own
    # output directory.  The absolute artifact path is therefore layout-run
    # provenance, not a training-condition difference.  Likewise, the embedded
    # pickle is not a canonical byte representation (equal RNG states can have
    # different pickle bytes); rng_state.sha256 is the canonical semantic
    # identity emitted by the training harness.
    initial_states = value.get("initial_states")
    if isinstance(initial_states, list):
        for state in initial_states:
            if not isinstance(state, dict):
                continue
            artifact = state.get("artifact")
            if isinstance(artifact, dict):
                artifact.pop("path", None)
            rng_state = state.get("rng_state")
            if isinstance(rng_state, dict) and isinstance(rng_state.get("sha256"), str):
                rng_state.pop("data", None)
    return value


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool | None,
    *,
    actual: Any,
    expected: Any,
    blocking: bool = True,
    evidence: str,
) -> None:
    checks.append(
        {
            "name": name,
            "status": "unverified" if passed is None else "pass" if passed else "fail",
            "blocking": blocking,
            "actual": actual,
            "expected": expected,
            "evidence": evidence,
        }
    )


def _gpu_semantic_checks(
    checks: list[dict[str, Any]], xml_path: Path | None
) -> dict[str, Any]:
    required = {
        "JpegDct.ManifestV3PlanlessMatchesLegacyAcrossRaggedShardsAndSampling":
            "semantics.gpu_manifest_v3_planless_matches_legacy_ragged",
        "JpegDct.PlanlessDeviceMatchesLegacyAcrossGeneralityMatrix":
            "semantics.gpu_planless_matches_legacy_generality_matrix",
    }
    if xml_path is None or not xml_path.is_file():
        observed: dict[str, str] = {}
    else:
        root = ET.parse(xml_path).getroot()
        observed = {}
        for case in root.iter("testcase"):
            full_name = f"{case.get('classname', '')}.{case.get('name', '')}"
            if case.find("failure") is not None or case.find("error") is not None:
                observed[full_name] = "failed"
            elif case.find("skipped") is not None or case.get("status") == "notrun":
                observed[full_name] = "skipped"
            else:
                observed[full_name] = "passed"
    states = {name: observed.get(name, "missing") for name in required}
    for case_name, check_name in required.items():
        state = states[case_name]
        _add_check(
            checks,
            check_name,
            None if not observed else state == "passed",
            actual=state,
            expected="passed (not skipped)",
            evidence=f"GoogleTest XML case {case_name}",
        )
    return {
        "xml": None if xml_path is None else str(xml_path),
        "required_cases": states,
    }


def _planner_checks(
    checks: list[dict[str, Any]],
    v2: Mapping[str, Any] | None,
    v3: Mapping[str, Any] | None,
) -> dict[str, Any]:
    same_fields = (
        "batch_size",
        "crop",
        "horizontal_flip",
        "iterations",
        "warmup_iterations",
        "fixed_transform_source_block_count",
        "fixed_transform_output_block_count",
        "planned_selected_vector_count",
        "full_vector_count",
    )
    same = None if v2 is None or v3 is None else all(v2.get(name) == v3.get(name) for name in same_fields)
    _add_check(
        checks,
        "planner.same_workload",
        same,
        actual=None if v2 is None or v3 is None else {name: [v2.get(name), v3.get(name)] for name in same_fields},
        expected="all pairs equal",
        evidence="standalone planner probes",
    )
    expansion = None if v3 is None else [
        v3.get("host_expanded_transform_items_created"),
        v3.get("host_output_block_source_lists_created"),
        v3.get("host_global_transform_sort_items"),
    ]
    _add_check(
        checks,
        "planner.zero_host_per_block_expansion_and_sort",
        None if expansion is None else expansion == [0, 0, 0],
        actual=expansion,
        expected=[0, 0, 0],
        evidence="v3 standalone planner probe",
    )
    planless = None if v3 is None else v3.get("uses_planless_fixed_transform") is True
    _add_check(
        checks,
        "planner.compact_planless_enabled",
        planless,
        actual=None if v3 is None else v3.get("uses_planless_fixed_transform"),
        expected=True,
        evidence="v3 standalone planner probe",
    )
    v2_ms = None if v2 is None else _number(v2.get("planning_ms_mean"))
    v3_ms = None if v3 is None else _number(v3.get("planning_ms_mean"))
    ratio = v3_ms / v2_ms if v2_ms and v3_ms is not None else None
    _add_check(
        checks,
        "planner.v3_at_most_1_2x_v2",
        None if ratio is None or same is not True else ratio <= 1.2,
        actual=ratio,
        expected="<= 1.2",
        evidence="same-workload standalone planner probes",
    )
    return {
        "v2_ms_per_batch": v2_ms,
        "v3_ms_per_batch": v3_ms,
        "v3_to_v2_ratio": ratio,
        "v3_ms_per_image": None if v3 is None else v3.get("planning_ms_per_image_mean"),
        "v3_compact_image_descriptors": None if v3 is None else v3.get("compact_image_descriptor_count"),
        "v3_compact_plan_bytes": None if v3 is None else v3.get("compact_plan_bytes"),
    }


def _reader_checks(checks: list[dict[str, Any]], reader: Mapping[str, Any] | None) -> dict[str, Any]:
    saved = None if reader is None else _number(reader.get("actual_saved_rowgroups"))
    avoided = None if reader is None else _number(reader.get("compressed_bytes_avoided"))
    planned = None if reader is None else _number(reader.get("planned_decode_vectors"))
    full = None if reader is None else _number(reader.get("full_rowgroups"))
    amplification = None if reader is None else _number(reader.get("source_output_block_amplification"))
    _add_check(
        checks,
        "pushdown.saved_vectors_or_rowgroups",
        None if saved is None else saved > 0,
        actual=saved,
        expected="> 0",
        evidence="controlled crop reader probe",
    )
    _add_check(
        checks,
        "pushdown.compressed_bytes_reduced",
        None if avoided is None else avoided > 0,
        actual=avoided,
        expected="> 0",
        evidence="controlled crop reader probe",
    )
    decode_reduced = None if planned is None or full is None else planned < full
    _add_check(
        checks,
        "pushdown.decode_vectors_reduced",
        decode_reduced,
        actual=None if planned is None or full is None else {"selected": planned, "full": full},
        expected="selected < full",
        evidence="controlled crop reader probe",
    )
    _add_check(
        checks,
        "pushdown.source_output_amplification_target",
        None if amplification is None else amplification <= 1.3,
        actual=amplification,
        expected="<= 1.3",
        blocking=False,
        evidence="layout-efficiency target; a miss requires a layout recommendation, not a planless rollback",
    )
    return dict(reader) if reader is not None else {}


def _training_checks(
    checks: list[dict[str, Any]],
    v2_dir: Path | None,
    v3_dir: Path | None,
    v2: Mapping[str, Any] | None,
    v3: Mapping[str, Any] | None,
) -> dict[str, Any]:
    v2_contract = _artifact(v2_dir, "contract.json")
    v3_contract = _artifact(v3_dir, "contract.json")
    same_contract = (
        None
        if v2_contract is None or v3_contract is None
        else _normalize_contract(v2_contract) == _normalize_contract(v3_contract)
    )
    _add_check(
        checks,
        "training.same_conditions_except_manifest_layout",
        same_contract,
        actual=same_contract,
        expected=True,
        evidence="normalized v2/v3 training contracts",
    )
    for label, metrics in (("v2", v2), ("v3", v3)):
        ok = None if metrics is None else metrics.get("all_repeats_ok") is True
        probe = None if metrics is None else metrics.get("first_step_semantic_probe")
        probe_ok = None
        if isinstance(probe, Mapping) and probe:
            inputs = probe.get("input_fingerprints")
            probe_ok = (
                probe.get("failures") == []
                and probe.get("before_warmup") is True
                and probe.get("fresh_clone") is True
                and probe.get("formal_repeat_polluted") is False
                and isinstance(inputs, list)
                and bool(inputs)
                and all(
                    isinstance(value, Mapping)
                    and value.get("finite") is True
                    and isinstance(value.get("sha256"), str)
                    for value in inputs
                )
            )
        _add_check(
            checks,
            f"training.{label}_runtime_ok",
            ok,
            actual=ok,
            expected=True,
            evidence=f"{label} pipeline_galp.json repeat results",
        )
        _add_check(
            checks,
            f"training.{label}_first_step_probe_valid",
            probe_ok,
            actual=probe,
            expected="fresh pre-warmup finite probe with no failures",
            evidence=f"{label} pipeline first-step semantic probe",
        )
    v2_probe = None if v2 is None else v2.get("first_step_semantic_probe")
    v3_probe = None if v3 is None else v3.get("first_step_semantic_probe")
    semantic_fields = (
        "sample_ids",
        "labels",
        "augmentation_decisions",
        "input_fingerprints",
        "initial_logits",
        "first_step_loss",
        "reproducibility",
    )
    comparable = (
        isinstance(v2_probe, Mapping)
        and bool(v2_probe)
        and isinstance(v3_probe, Mapping)
        and bool(v3_probe)
    )
    semantics_match = (
        None
        if not comparable
        else all(v2_probe.get(name) == v3_probe.get(name) for name in semantic_fields)
    )
    _add_check(
        checks,
        "training.v2_v3_first_step_semantics_match",
        semantics_match,
        actual=(
            None
            if not comparable
            else {name: v2_probe.get(name) == v3_probe.get(name) for name in semantic_fields}
        ),
        expected="all first-step sample/augmentation/input/logit/loss fields equal",
        evidence="same-condition v2/v3 fresh-clone first-step probes",
    )
    v3_throughput = None if v3 is None else _number(v3.get("throughput_images_per_s_mean"))
    v2_throughput = None if v2 is None else _number(v2.get("throughput_images_per_s_mean"))
    v2_repeat_count = None if v2 is None else int(v2.get("repeat_count", 0))
    v3_repeat_count = None if v3 is None else int(v3.get("repeat_count", 0))
    repeat_count_ok = (
        None
        if v2_repeat_count is None or v3_repeat_count is None
        else v2_repeat_count >= MIN_PERFORMANCE_REPEATS
        and v3_repeat_count >= MIN_PERFORMANCE_REPEATS
    )
    _add_check(
        checks,
        "training.performance_repeat_count_at_least_3",
        repeat_count_ok,
        actual={"v2": v2_repeat_count, "v3": v3_repeat_count},
        expected=f"both >= {MIN_PERFORMANCE_REPEATS}",
        evidence="true repeated smoke measurements",
    )
    v2_compute = None if v2 is None else _number(v2.get("compute_upper_bound_images_per_s_mean"))
    v3_compute = None if v3 is None else _number(v3.get("compute_upper_bound_images_per_s_mean"))
    v2_compute_cv = None if v2 is None else _number(v2.get("compute_upper_bound_images_per_s_cv"))
    v3_compute_cv = None if v3 is None else _number(v3.get("compute_upper_bound_images_per_s_cv"))
    compute_relative_difference = (
        None
        if v2_compute is None or v3_compute is None or max(abs(v2_compute), abs(v3_compute)) == 0.0
        else abs(v3_compute - v2_compute) / max(abs(v2_compute), abs(v3_compute))
    )
    compute_conditions_comparable = (
        None
        if repeat_count_ok is not True
        or compute_relative_difference is None
        or v2_compute_cv is None
        or v3_compute_cv is None
        else compute_relative_difference <= COMPUTE_CONDITION_RELATIVE_TOLERANCE
        and v2_compute_cv <= COMPUTE_CONDITION_CV_LIMIT
        and v3_compute_cv <= COMPUTE_CONDITION_CV_LIMIT
    )
    _add_check(
        checks,
        "training.v2_v3_compute_conditions_comparable",
        compute_conditions_comparable,
        actual={
            "v2_compute_upper_bound_images_per_s": v2_compute,
            "v3_compute_upper_bound_images_per_s": v3_compute,
            "relative_difference": compute_relative_difference,
            "v2_cv": v2_compute_cv,
            "v3_cv": v3_compute_cv,
        },
        expected={
            "relative_difference_at_most": COMPUTE_CONDITION_RELATIVE_TOLERANCE,
            "per_side_cv_at_most": COMPUTE_CONDITION_CV_LIMIT,
        },
        evidence="three-repeat compute-only diagnostics under the same model recipe",
    )
    _add_check(
        checks,
        "training.v3_throughput_at_least_900",
        None if repeat_count_ok is not True or v3_throughput is None else v3_throughput >= 900.0,
        actual=v3_throughput,
        expected=">= 900 img/s",
        evidence="v3 end-to-end measured train region",
    )
    _add_check(
        checks,
        "training.v3_not_slower_than_v2",
        (
            None
            if compute_conditions_comparable is not True
            or v2_throughput is None
            or v3_throughput is None
            else v3_throughput >= v2_throughput
        ),
        actual={
            "v2": v2_throughput,
            "v3": v3_throughput,
            "compute_conditions_comparable": compute_conditions_comparable,
        },
        expected="v3 >= v2",
        evidence="three-repeat end-to-end regions with comparable compute-only conditions",
    )
    wait = None if v3 is None else _number(v3.get("loader_wait_fraction_mean"))
    _add_check(
        checks,
        "training.loader_wait_below_20_percent",
        None if wait is None else wait < 0.2,
        actual=wait,
        expected="< 0.2",
        evidence="v3 measured loader wait / measured wall",
    )
    planning = None if v3 is None else _number(v3.get("planning_ms_per_batch_mean"))
    _add_check(
        checks,
        "training.planning_at_most_20_ms_per_batch",
        None if planning is None else planning <= 20.0,
        actual=planning,
        expected="<= 20 ms/batch",
        evidence="v3 measured native execution statistics",
    )
    detailed_timings = (
        None
        if v3 is None
        else {
            name: _number(v3.get(name))
            for name in (
                "planning_ms_per_batch_mean",
                "physical_read_ms_per_batch_mean",
                "decode_ms_per_batch_mean",
                "transform_ms_per_batch_mean",
                "loader_wait_fraction_mean",
                "end_to_end_step_ms_mean",
                "throughput_images_per_s_mean",
                "compute_upper_bound_images_per_s_mean",
            )
        }
    )
    _add_check(
        checks,
        "training.detailed_timing_breakdown_observed",
        (
            None
            if detailed_timings is None
            else all(value is not None for value in detailed_timings.values())
        ),
        actual=detailed_timings,
        expected="planner/read/decode/transform/loader/model/step/throughput metrics all present",
        evidence="v3 measured repeat and native execution statistics",
    )
    latest = {} if v3 is None else v3.get("native_latest", {})
    runtime_planless = None
    if isinstance(latest, Mapping) and latest:
        runtime_planless = latest.get("uses_planless_fixed_transform") is True and all(
            latest.get(name) == 0
            for name in (
                "host_expanded_transform_items_created",
                "host_output_block_source_lists_created",
                "host_global_transform_sort_items",
            )
        )
    _add_check(
        checks,
        "training.v3_runtime_compact_planless_evidence",
        runtime_planless,
        actual={
            name: latest.get(name) if isinstance(latest, Mapping) else None
            for name in (
                "uses_planless_fixed_transform",
                "host_expanded_transform_items_created",
                "host_output_block_source_lists_created",
                "host_global_transform_sort_items",
                "compact_plan_bytes",
            )
        },
        expected="planless=true and all host expansion/sort counters zero",
        evidence="v3 measured native latest snapshot",
    )
    stability_items = [] if v3 is None else v3.get("allocation_stability", [])
    stability = (
        None
        if not stability_items
        else all(
            item.get("verifiable") is True and item.get("stable_after_warmup") is True
            for item in stability_items
        )
    )
    _add_check(
        checks,
        "training.native_allocations_stable_after_warmup",
        stability,
        actual=stability_items,
        expected="all repeats verifiable and stable_after_warmup=true",
        evidence="warmup-boundary native allocation deltas",
    )
    read_runs = None if v3 is None else _number(v3.get("coalesced_read_runs_per_batch_mean"))
    preadv = None if v3 is None else _number(v3.get("preadv_calls_per_batch_mean"))
    _add_check(
        checks,
        "training.merged_range_read_stats_observed",
        None if read_runs is None or preadv is None else read_runs > 0 and preadv > 0,
        actual={"coalesced_runs_per_batch": read_runs, "preadv_per_batch": preadv},
        expected="both > 0",
        evidence="v3 measured native execution statistics",
    )
    v2_sync = None if v2 is None else _number(v2.get("native_internal_syncs_per_batch_mean"))
    v3_sync = None if v3 is None else _number(v3.get("native_internal_syncs_per_batch_mean"))
    _add_check(
        checks,
        "training.v3_internal_syncs_not_above_v2",
        None if v2_sync is None or v3_sync is None else v3_sync <= v2_sync,
        actual={"v2": v2_sync, "v3": v3_sync},
        expected="v3 <= v2",
        evidence="same-condition measured native synchronization counters",
    )
    return {"v2": v2, "v3": v3}


def build_report(
    *,
    baseline_dir: Path | None,
    v2_dir: Path | None,
    v3_dir: Path | None,
    planner_v2_path: Path | None,
    planner_v3_path: Path | None,
    reader_v3_path: Path | None,
    gpu_gtest_xml_path: Path | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    gpu_semantics = _gpu_semantic_checks(checks, gpu_gtest_xml_path)
    planner = _planner_checks(checks, _read_json(planner_v2_path), _read_json(planner_v3_path))
    reader = _reader_checks(checks, _read_json(reader_v3_path))
    v2 = _training_metrics(v2_dir)
    v3 = _training_metrics(v3_dir)
    training = _training_checks(checks, v2_dir, v3_dir, v2, v3)
    blocking_failures = [item["name"] for item in checks if item["blocking"] and item["status"] == "fail"]
    blocking_unverified = [
        item["name"] for item in checks if item["blocking"] and item["status"] == "unverified"
    ]
    nonblocking_failures = [
        item["name"] for item in checks if not item["blocking"] and item["status"] == "fail"
    ]
    if blocking_failures:
        overall = "fail"
    elif blocking_unverified:
        overall = "incomplete"
    elif nonblocking_failures:
        overall = "pass_with_layout_followup"
    else:
        overall = "pass"
    amplification = _number(reader.get("source_output_block_amplification"))
    layout = {
        "current_image_major_pushdown_target_met": amplification is not None and amplification <= 1.3,
        "recommendation": (
            "keep image-major on the training mainline; no layout migration is justified by this probe"
            if amplification is not None and amplification <= 1.3
            else "retain image-major for the current fix and continue isolated block-major evaluation for spatial locality"
        ),
        "block_major_migration": "not approved by this report; requires stable reader API, semantic tests, and an independent benchmark",
    }
    return {
        "schema": "galp-training-v3-acceptance-v1",
        "overall_status": overall,
        "blocking_failures": blocking_failures,
        "blocking_unverified": blocking_unverified,
        "nonblocking_failures": nonblocking_failures,
        "timing_boundary_note": "native stages may overlap and must not be summed into end-to-end latency",
        "codec_claim_boundary": "GALP and DALI use different model/domain paths; end-to-end differences are not pure codec differences",
        "baseline": _baseline_metrics(baseline_dir),
        "planner": planner,
        "physical_pushdown": reader,
        "gpu_semantics": gpu_semantics,
        "training": training,
        "layout_conclusion": layout,
        "checks": checks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--v2-dir", type=Path)
    parser.add_argument("--v3-dir", type=Path)
    parser.add_argument("--planner-v2", type=Path)
    parser.add_argument("--planner-v3", type=Path)
    parser.add_argument("--reader-v3", type=Path)
    parser.add_argument("--gpu-gtest-xml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="return success when required GPU artifacts are absent; failed verified gates still return failure",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(
        baseline_dir=args.baseline_dir,
        v2_dir=args.v2_dir,
        v3_dir=args.v3_dir,
        planner_v2_path=args.planner_v2,
        planner_v3_path=args.planner_v3,
        reader_v3_path=args.reader_v3,
        gpu_gtest_xml_path=args.gpu_gtest_xml,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "overall_status": report["overall_status"],
        "blocking_failures": report["blocking_failures"],
        "blocking_unverified": report["blocking_unverified"],
        "nonblocking_failures": report["nonblocking_failures"],
    }, indent=2, sort_keys=True))
    if report["blocking_failures"]:
        return 1
    if report["blocking_unverified"] and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
