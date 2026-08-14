#!/usr/bin/env python3
"""Evaluate the official image-major-v3 full-training Gate 2 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "galp-image-major-v3-full-training-gate2-v1"
DEFAULT_BASELINE_THROUGHPUT = 33.1568
DEFAULT_MAX_PLANNING_MS_PER_BATCH = 20.0
DEFAULT_MAX_READER_WAIT_FRACTION = 0.20
DEFAULT_MIN_QUEUE_HIT_RATE = 0.95

_ZERO_MEASURED_GROWTH_FIELDS = (
    "compact_batch_buffer_growth_count",
    "compact_batch_buffer_pageable_fallback_count",
    "decode_workset_output_arena_growth_count",
    "decode_workset_chunk_arena_growth_count",
    "planless_axis_program_device_growth_count",
    "planless_axis_program_pinned_growth_count",
)

_ZERO_GLOBAL_ALLOCATION_DELTA_FIELDS = (
    "galp_native_device_cuda_allocation_count",
    "galp_native_pinned_cuda_allocation_count",
)

_ZERO_REBUILD_FIELDS = (
    "descriptor_map_count",
    "descriptor_open_ms",
    "schema_plan_build_ms",
    "static_metadata_cache_miss_count",
)

_ZERO_HOST_EXPANSION_FIELDS = (
    "host_expanded_transform_items_created",
    "host_output_block_source_lists_created",
    "host_global_transform_sort_items",
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _measured_aggregate(repeat: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    phase_stats = _mapping(repeat.get("native_execution_stats_by_phase"))
    measured = _mapping(phase_stats.get("measured"))
    aggregate = _mapping(measured.get("numeric_aggregates")).get(name)
    return _mapping(aggregate)


def _measured_sum(repeat: Mapping[str, Any], name: str) -> float | None:
    return _number(_measured_aggregate(repeat, name).get("sum"))


def _measured_latest(repeat: Mapping[str, Any]) -> Mapping[str, Any]:
    phase_stats = _mapping(repeat.get("native_execution_stats_by_phase"))
    return _mapping(_mapping(phase_stats.get("measured")).get("latest"))


def _phase_repeats(pipeline: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    phases = _mapping(pipeline.get("phase_results"))
    repeats = _mapping(phases.get("smoke")).get("repeats")
    if not isinstance(repeats, Sequence) or isinstance(repeats, (str, bytes)):
        return []
    return [value for value in repeats if isinstance(value, Mapping)]


def evaluate_gate2(
    run_dir: Path,
    *,
    baseline_throughput: float = DEFAULT_BASELINE_THROUGHPUT,
    max_planning_ms_per_batch: float = DEFAULT_MAX_PLANNING_MS_PER_BATCH,
    max_reader_wait_fraction: float = DEFAULT_MAX_READER_WAIT_FRACTION,
    min_queue_hit_rate: float = DEFAULT_MIN_QUEUE_HIT_RATE,
    allow_semantic_not_applicable: bool = False,
    require_compact_pool_contract: bool = True,
) -> dict[str, Any]:
    pipeline_path = run_dir / "pipeline_galp.json"
    results_path = run_dir / "results.json"
    required_paths = (pipeline_path, results_path, run_dir / "contract.json", run_dir / "sample_order.json")
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_dir": str(run_dir.resolve()),
            "ok": False,
            "complete": False,
            "failures": [f"missing required artifact: {path}" for path in missing_paths],
            "checks": {},
            "metrics": {},
            "artifacts": {},
        }

    pipeline = _read_object(pipeline_path)
    results = _read_object(results_path)
    repeats = _phase_repeats(pipeline)
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    metrics: dict[str, Any] = {"repeat_count": len(repeats)}

    def check(name: str, ok: bool, actual: Any, expected: str) -> None:
        checks[name] = {"ok": bool(ok), "actual": actual, "expected": expected}
        if not ok:
            failures.append(f"{name}: expected {expected}, got {actual!r}")

    pipeline_status = _mapping(pipeline.get("status"))
    results_status = _mapping(_mapping(results.get("pipeline_status")).get("galp"))
    semantic_probe = _mapping(pipeline.get("first_step_semantic_probe"))
    status_basics_pass = all(
        _mapping(status).get(name) == "passed"
        for status in (pipeline_status, results_status)
        for name in ("artifact", "correctness")
    )
    semantic_pass = all(
        _mapping(status).get("semantic") == "passed"
        for status in (pipeline_status, results_status)
    )
    semantic_not_applicable_with_probe = (
        allow_semantic_not_applicable
        and all(
            _mapping(status).get("semantic") == "not_applicable"
            for status in (pipeline_status, results_status)
        )
        and semantic_probe.get("before_warmup") is True
        and semantic_probe.get("fresh_clone") is True
        and semantic_probe.get("formal_repeat_polluted") is False
        and semantic_probe.get("failures") == []
    )
    check(
        "artifact_and_semantic_status",
        status_basics_pass and (semantic_pass or semantic_not_applicable_with_probe),
        {
            "pipeline": dict(pipeline_status),
            "results": dict(results_status),
            "first_step_probe_clean": semantic_not_applicable_with_probe,
        },
        (
            "artifact/correctness passed and semantic passed, or explicitly allowed "
            "semantic=not_applicable with a clean pre-warmup fresh-clone probe"
        ),
    )
    check("smoke_repeat_present", bool(repeats), len(repeats), ">=1")

    repeat_metrics: list[dict[str, Any]] = []
    for repeat_index, repeat in enumerate(repeats):
        loader = _mapping(repeat.get("loader_measured_metrics"))
        submitted = _number(loader.get("submitted_batches"))
        planning_seconds = _number(loader.get("producer_planning_seconds"))
        planning_ms = (
            planning_seconds * 1000.0 / submitted
            if planning_seconds is not None and submitted is not None and submitted > 0.0
            else None
        )
        wait_fraction = _number(loader.get("consumer_wait_fraction_of_measured_wall"))
        queue_hit_rate = _number(loader.get("queue_hit_rate"))
        throughput = _number(repeat.get("throughput_images_per_s"))
        throughput_gain = throughput / baseline_throughput if throughput is not None and baseline_throughput > 0 else None
        stability = _mapping(repeat.get("native_allocation_stability"))
        allocation_deltas = _mapping(stability.get("global_counter_deltas"))
        growth_totals = _mapping(stability.get("measured_per_batch_totals"))
        latest = _measured_latest(repeat)

        prefix = f"repeat_{repeat_index}"
        check(
            f"{prefix}.planning_ms_per_batch",
            planning_ms is not None and planning_ms < max_planning_ms_per_batch,
            planning_ms,
            f"<{max_planning_ms_per_batch}",
        )
        check(
            f"{prefix}.reader_wait_fraction",
            wait_fraction is not None and wait_fraction < max_reader_wait_fraction,
            wait_fraction,
            f"<{max_reader_wait_fraction}",
        )
        check(
            f"{prefix}.queue_hit_rate",
            queue_hit_rate is not None and queue_hit_rate >= min_queue_hit_rate,
            queue_hit_rate,
            f">={min_queue_hit_rate}",
        )
        check(
            f"{prefix}.allocation_stability",
            stability.get("verifiable") is True
            and stability.get("stable_after_warmup") is True
            and stability.get("capacity_contract_complete") is True,
            {
                "verifiable": stability.get("verifiable"),
                "stable_after_warmup": stability.get("stable_after_warmup"),
                "capacity_contract_complete": stability.get("capacity_contract_complete"),
            },
            "verifiable=true, stable_after_warmup=true, capacity_contract_complete=true",
        )
        for name in _ZERO_GLOBAL_ALLOCATION_DELTA_FIELDS:
            value = _number(allocation_deltas.get(name))
            check(f"{prefix}.{name}_delta", value == 0.0, value, "0")
        for name in _ZERO_MEASURED_GROWTH_FIELDS:
            value = _number(growth_totals.get(name))
            check(f"{prefix}.{name}", value == 0.0, value, "0")
        for name in _ZERO_REBUILD_FIELDS + _ZERO_HOST_EXPANSION_FIELDS:
            value = _measured_sum(repeat, name)
            check(f"{prefix}.{name}", value == 0.0, value, "0")

        contract_bytes = _number(latest.get("planless_axis_program_capacity_contract_bytes"))
        device_capacity = _number(latest.get("planless_axis_program_device_capacity_bytes"))
        pinned_capacity = _number(latest.get("planless_axis_program_pinned_capacity_bytes"))
        check(
            f"{prefix}.axis_program_capacity_contract",
            latest.get("planless_axis_program_capacity_contract_complete") is True
            and contract_bytes is not None
            and contract_bytes > 0.0
            and device_capacity is not None
            and device_capacity >= contract_bytes
            and pinned_capacity is not None
            and pinned_capacity >= contract_bytes,
            {
                "complete": latest.get("planless_axis_program_capacity_contract_complete"),
                "contract_bytes": contract_bytes,
                "device_capacity_bytes": device_capacity,
                "pinned_capacity_bytes": pinned_capacity,
            },
            "complete contract with device/pinned capacity >= contract bytes",
        )
        if require_compact_pool_contract:
            compact_contract_bytes = _number(
                latest.get("compact_batch_pool_capacity_contract_bytes")
            )
            compact_prewarmed_bytes = _number(latest.get("compact_batch_pool_prewarmed_bytes"))
            compact_images = _number(latest.get("compact_batch_pool_capacity_contract_images"))
            compact_groups = _number(latest.get("compact_batch_pool_capacity_contract_groups"))
            compact_batches = _number(latest.get("compact_batch_pool_capacity_contract_batches"))
            compact_slots = _number(latest.get("compact_batch_pool_prewarmed_slots"))
            check(
                f"{prefix}.compact_pool_capacity_contract",
                latest.get("compact_batch_pool_capacity_contract_complete") is True
                and compact_contract_bytes is not None
                and compact_contract_bytes > 0.0
                and compact_prewarmed_bytes is not None
                and compact_prewarmed_bytes >= compact_contract_bytes
                and compact_images is not None
                and compact_images > 0.0
                and compact_groups is not None
                and compact_groups > 0.0
                and compact_batches is not None
                and compact_batches > 0.0
                and compact_slots is not None
                and compact_slots >= compact_groups * compact_batches,
                {
                    "complete": latest.get("compact_batch_pool_capacity_contract_complete"),
                    "contract_bytes": compact_contract_bytes,
                    "prewarmed_bytes": compact_prewarmed_bytes,
                    "images": compact_images,
                    "groups": compact_groups,
                    "concurrent_batches": compact_batches,
                    "prewarmed_slots": compact_slots,
                },
                "complete dataset-derived profile with bytes and slots covering its contract",
            )
        sample_validation = _mapping(_mapping(repeat.get("sample_order")).get("validation"))
        check(
            f"{prefix}.sample_order",
            sample_validation.get("ok") is True,
            dict(sample_validation),
            "sample_order.validation.ok=true",
        )
        check(
            f"{prefix}.throughput_above_original_baseline",
            throughput_gain is not None and throughput_gain > 1.0,
            {"images_per_s": throughput, "gain": throughput_gain},
            f">{baseline_throughput} img/s",
        )
        repeat_metrics.append(
            {
                "repeat": repeat_index,
                "throughput_images_per_s": throughput,
                "throughput_gain_vs_original": throughput_gain,
                "planning_ms_per_batch": planning_ms,
                "reader_wait_fraction": wait_fraction,
                "queue_hit_rate": queue_hit_rate,
                "capacity_contract_bytes": contract_bytes,
                "device_capacity_bytes": device_capacity,
                "pinned_capacity_bytes": pinned_capacity,
            }
        )

    metrics["repeats"] = repeat_metrics
    artifacts = {
        path.name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in required_paths
    }
    evaluator_path = Path(__file__).resolve()
    artifacts["gate2_evaluator"] = {
        "path": str(evaluator_path),
        "sha256": _sha256(evaluator_path),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "ok": not failures,
        "complete": True,
        "failures": failures,
        "checks": checks,
        "metrics": metrics,
        "thresholds": {
            "baseline_throughput_images_per_s": baseline_throughput,
            "max_planning_ms_per_batch": max_planning_ms_per_batch,
            "max_reader_wait_fraction": max_reader_wait_fraction,
            "min_queue_hit_rate": min_queue_hit_rate,
        },
        "artifacts": artifacts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-throughput", type=float, default=DEFAULT_BASELINE_THROUGHPUT)
    parser.add_argument("--max-planning-ms-per-batch", type=float, default=DEFAULT_MAX_PLANNING_MS_PER_BATCH)
    parser.add_argument("--max-reader-wait-fraction", type=float, default=DEFAULT_MAX_READER_WAIT_FRACTION)
    parser.add_argument("--min-queue-hit-rate", type=float, default=DEFAULT_MIN_QUEUE_HIT_RATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_gate2(
        args.run_dir,
        baseline_throughput=args.baseline_throughput,
        max_planning_ms_per_batch=args.max_planning_ms_per_batch,
        max_reader_wait_fraction=args.max_reader_wait_fraction,
        min_queue_hit_rate=args.min_queue_hit_rate,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if not report.get("complete"):
        return 2
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
