#!/usr/bin/env python3
"""Evaluate long, repeated image-major-v3 GALP step runs for Gate 3."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from training.gate2_acceptance import (
        DEFAULT_MAX_PLANNING_MS_PER_BATCH,
        DEFAULT_MAX_READER_WAIT_FRACTION,
        DEFAULT_MIN_QUEUE_HIT_RATE,
        _ZERO_GLOBAL_ALLOCATION_DELTA_FIELDS,
        _ZERO_HOST_EXPANSION_FIELDS,
        _ZERO_MEASURED_GROWTH_FIELDS,
        _ZERO_REBUILD_FIELDS,
        _mapping,
        _measured_latest,
        _measured_sum,
        _number,
        _read_object,
        _sha256,
    )
except ModuleNotFoundError:  # Direct script execution from this directory.
    from gate2_acceptance import (  # type: ignore[no-redef]
        DEFAULT_MAX_PLANNING_MS_PER_BATCH,
        DEFAULT_MAX_READER_WAIT_FRACTION,
        DEFAULT_MIN_QUEUE_HIT_RATE,
        _ZERO_GLOBAL_ALLOCATION_DELTA_FIELDS,
        _ZERO_HOST_EXPANSION_FIELDS,
        _ZERO_MEASURED_GROWTH_FIELDS,
        _ZERO_REBUILD_FIELDS,
        _mapping,
        _measured_latest,
        _measured_sum,
        _number,
        _read_object,
        _sha256,
    )


SCHEMA_VERSION = "galp-image-major-v3-full-training-gate3-v1"
DEFAULT_MIN_REPEATS = 3
DEFAULT_MAX_THROUGHPUT_CV = 0.10
DEFAULT_MAX_RSS_RANGE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_RSS_RANGE_FRACTION = 0.05
DEFAULT_MAX_FD_RANGE = 8
DEFAULT_MAX_MAPPING_RANGE = 64

_EXACT_REPEAT_RESOURCE_FIELDS = (
    "payload_fd_current_count",
    "descriptor_mapping_current_count",
    "descriptor_mapped_current_bytes",
    "static_metadata_count",
    "static_metadata_bytes",
    "galp_native_device_cuda_allocation_count",
    "galp_native_device_cuda_allocation_bytes",
    "galp_native_pinned_cuda_allocation_count",
    "galp_native_pinned_cuda_allocation_bytes",
    "galp_native_device_cached_bytes",
    "galp_native_pinned_cached_bytes",
    "compact_batch_buffer_capacity_bytes",
    "compact_batch_buffer_high_water_bytes",
    "decode_workset_output_arena_capacity_bytes",
    "decode_workset_chunk_arena_capacity_bytes",
    "planless_axis_program_device_capacity_bytes",
    "planless_axis_program_pinned_capacity_bytes",
)


def _step_repeats(pipeline: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    repeats = _mapping(_mapping(pipeline.get("phase_results")).get("step")).get("repeats")
    if not isinstance(repeats, Sequence) or isinstance(repeats, (str, bytes)):
        return []
    return [value for value in repeats if isinstance(value, Mapping)]


def _sample_cv(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean == 0.0:
        return None
    return statistics.stdev(values) / mean


def _range(values: Sequence[float]) -> float | None:
    return max(values) - min(values) if values else None


def evaluate_gate3(
    run_dir: Path,
    *,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    max_throughput_cv: float = DEFAULT_MAX_THROUGHPUT_CV,
    max_planning_ms_per_batch: float = DEFAULT_MAX_PLANNING_MS_PER_BATCH,
    max_reader_wait_fraction: float = DEFAULT_MAX_READER_WAIT_FRACTION,
    min_queue_hit_rate: float = DEFAULT_MIN_QUEUE_HIT_RATE,
    max_rss_range_bytes: int = DEFAULT_MAX_RSS_RANGE_BYTES,
    max_rss_range_fraction: float = DEFAULT_MAX_RSS_RANGE_FRACTION,
    max_fd_range: int = DEFAULT_MAX_FD_RANGE,
    max_mapping_range: int = DEFAULT_MAX_MAPPING_RANGE,
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
    repeats = _step_repeats(pipeline)
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def check(name: str, ok: bool, actual: Any, expected: str) -> None:
        checks[name] = {"ok": bool(ok), "actual": actual, "expected": expected}
        if not ok:
            failures.append(f"{name}: expected {expected}, got {actual!r}")

    pipeline_status = _mapping(pipeline.get("status"))
    results_status = _mapping(_mapping(results.get("pipeline_status")).get("galp"))
    check(
        "artifact_semantic_correctness_performance_status",
        all(
            _mapping(status).get(name) == "passed"
            for status in (pipeline_status, results_status)
            for name in ("artifact", "correctness", "semantic", "performance")
        ),
        {"pipeline": dict(pipeline_status), "results": dict(results_status)},
        "artifact/correctness/semantic/performance passed in pipeline and results",
    )
    check("minimum_repeat_count", len(repeats) >= min_repeats, len(repeats), f">={min_repeats}")

    throughputs: list[float] = []
    rss_values: list[float] = []
    fd_values: list[float] = []
    mapping_values: list[float] = []
    resource_series: dict[str, list[float]] = {name: [] for name in _EXACT_REPEAT_RESOURCE_FIELDS}
    repeat_metrics: list[dict[str, Any]] = []

    for repeat_index, repeat in enumerate(repeats):
        prefix = f"repeat_{repeat_index}"
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
        stability = _mapping(repeat.get("native_allocation_stability"))
        allocation_deltas = _mapping(stability.get("global_counter_deltas"))
        growth_totals = _mapping(stability.get("measured_per_batch_totals"))
        latest = _measured_latest(repeat)
        host = _mapping(repeat.get("host_memory"))

        check(
            f"{prefix}.repeat_correctness",
            repeat.get("ok") is True and not repeat.get("failures"),
            {"ok": repeat.get("ok"), "failures": repeat.get("failures")},
            "ok=true and no failures",
        )
        sample_validation = _mapping(_mapping(repeat.get("sample_order")).get("validation"))
        check(
            f"{prefix}.sample_order",
            sample_validation.get("ok") is True,
            dict(sample_validation),
            "sample_order.validation.ok=true",
        )
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

        if throughput is not None:
            throughputs.append(throughput)
        rss = _number(host.get("rss_bytes"))
        fds = _number(host.get("open_fd_count"))
        mappings = _number(host.get("memory_mapping_count"))
        if rss is not None:
            rss_values.append(rss)
        if fds is not None:
            fd_values.append(fds)
        if mappings is not None:
            mapping_values.append(mappings)
        for name in _EXACT_REPEAT_RESOURCE_FIELDS:
            value = _number(latest.get(name))
            if value is not None:
                resource_series[name].append(value)
        repeat_metrics.append(
            {
                "repeat": repeat_index,
                "throughput_images_per_s": throughput,
                "planning_ms_per_batch": planning_ms,
                "reader_wait_fraction": wait_fraction,
                "queue_hit_rate": queue_hit_rate,
                "rss_bytes": rss,
                "open_fd_count": fds,
                "memory_mapping_count": mappings,
            }
        )

    hot_throughputs = throughputs[1:] if len(throughputs) > 1 else throughputs
    throughput_cv = _sample_cv(hot_throughputs)
    check(
        "hot_repeat_throughput_cv",
        throughput_cv is not None and throughput_cv <= max_throughput_cv,
        throughput_cv,
        f"<={max_throughput_cv}",
    )

    rss_range = _range(rss_values)
    rss_limit = (
        max(float(max_rss_range_bytes), min(rss_values) * max_rss_range_fraction)
        if rss_values
        else None
    )
    check(
        "rss_repeat_stability",
        len(rss_values) == len(repeats)
        and rss_range is not None
        and rss_limit is not None
        and rss_range <= rss_limit,
        {"values": rss_values, "range": rss_range, "limit": rss_limit},
        "all repeats present and range within absolute/fractional limit",
    )
    fd_range = _range(fd_values)
    check(
        "fd_repeat_stability",
        len(fd_values) == len(repeats) and fd_range is not None and fd_range <= max_fd_range,
        {"values": fd_values, "range": fd_range},
        f"all repeats present and range <={max_fd_range}",
    )
    mapping_range = _range(mapping_values)
    check(
        "mapping_repeat_stability",
        len(mapping_values) == len(repeats)
        and mapping_range is not None
        and mapping_range <= max_mapping_range,
        {"values": mapping_values, "range": mapping_range},
        f"all repeats present and range <={max_mapping_range}",
    )
    for name, values in resource_series.items():
        check(
            f"repeat_resource_stability.{name}",
            len(values) == len(repeats) and len(set(values)) == 1,
            values,
            "present and identical after every repeat warmup",
        )

    aggregate = _mapping(_mapping(_mapping(pipeline.get("phase_results")).get("step")).get("aggregate"))
    check(
        "runner_step_aggregate",
        aggregate.get("performance_status") == "passed",
        dict(aggregate),
        "performance_status=passed",
    )

    artifacts = {
        path.name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in required_paths
    }
    evaluator_path = Path(__file__).resolve()
    artifacts["gate3_evaluator"] = {
        "path": str(evaluator_path),
        "sha256": _sha256(evaluator_path),
    }
    metrics = {
        "repeat_count": len(repeats),
        "repeats": repeat_metrics,
        "hot_repeat_throughput": {
            "values": hot_throughputs,
            "median": statistics.median(hot_throughputs) if hot_throughputs else None,
            "min": min(hot_throughputs) if hot_throughputs else None,
            "max": max(hot_throughputs) if hot_throughputs else None,
            "cv": throughput_cv,
        },
        "resource_series": resource_series,
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
            "min_repeats": min_repeats,
            "max_throughput_cv": max_throughput_cv,
            "max_planning_ms_per_batch": max_planning_ms_per_batch,
            "max_reader_wait_fraction": max_reader_wait_fraction,
            "min_queue_hit_rate": min_queue_hit_rate,
            "max_rss_range_bytes": max_rss_range_bytes,
            "max_rss_range_fraction": max_rss_range_fraction,
            "max_fd_range": max_fd_range,
            "max_mapping_range": max_mapping_range,
        },
        "artifacts": artifacts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-repeats", type=int, default=DEFAULT_MIN_REPEATS)
    parser.add_argument("--max-throughput-cv", type=float, default=DEFAULT_MAX_THROUGHPUT_CV)
    parser.add_argument("--max-planning-ms-per-batch", type=float, default=DEFAULT_MAX_PLANNING_MS_PER_BATCH)
    parser.add_argument("--max-reader-wait-fraction", type=float, default=DEFAULT_MAX_READER_WAIT_FRACTION)
    parser.add_argument("--min-queue-hit-rate", type=float, default=DEFAULT_MIN_QUEUE_HIT_RATE)
    parser.add_argument("--max-rss-range-bytes", type=int, default=DEFAULT_MAX_RSS_RANGE_BYTES)
    parser.add_argument("--max-rss-range-fraction", type=float, default=DEFAULT_MAX_RSS_RANGE_FRACTION)
    parser.add_argument("--max-fd-range", type=int, default=DEFAULT_MAX_FD_RANGE)
    parser.add_argument("--max-mapping-range", type=int, default=DEFAULT_MAX_MAPPING_RANGE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_gate3(
        args.run_dir,
        min_repeats=args.min_repeats,
        max_throughput_cv=args.max_throughput_cv,
        max_planning_ms_per_batch=args.max_planning_ms_per_batch,
        max_reader_wait_fraction=args.max_reader_wait_fraction,
        min_queue_hit_rate=args.min_queue_hit_rate,
        max_rss_range_bytes=args.max_rss_range_bytes,
        max_rss_range_fraction=args.max_rss_range_fraction,
        max_fd_range=args.max_fd_range,
        max_mapping_range=args.max_mapping_range,
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
