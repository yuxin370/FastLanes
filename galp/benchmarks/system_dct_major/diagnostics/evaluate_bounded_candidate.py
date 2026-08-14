#!/usr/bin/env python3
"""Evaluate one 50K bounded-read cap without silently widening the search."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "galp_dct_major_bounded_candidate_v2"
PIPELINE = "dct_major_pushdown"
EXACT_BYTES_50K = 824_678_664
FULL_BYTES_50K = 1_241_259_700
SELECTED_VECTORS_50K = 57_534
ROWGROUPS_50K = 788
TRANSIENT_LIMIT_BYTES = 512 * 1024 * 1024
CAP_SEQUENCE = (1.00, 1.02, 1.05, 1.10)
FROZEN_REPLAY_50K = {
    1_000_000: (824_678_664, 810_711),
    1_020_000: (841_123_964, 247_761),
    1_050_000: (865_744_171, 154_294),
    1_100_000: (906_760_327, 98_078),
}
ZERO_COUNTERS = (
    "segment_cross_shard_count",
    "shard_reactivation_count",
    "duplicate_physical_read_count",
    "rowgroup_revisit_count",
    "vector_run_revisit_count",
    "physical_read_order_inversions",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _cap_ppm(cap: float) -> int:
    if not math.isfinite(cap) or not 1.0 <= cap <= 1.10:
        raise ValueError("cap must be in [1.0, 1.10]")
    return int(math.floor(cap * 1_000_000.0 + 0.5))


def _next_cap(cap: float) -> float | None:
    ppm = _cap_ppm(cap)
    sequence_ppm = [_cap_ppm(value) for value in CAP_SEQUENCE]
    if ppm not in sequence_ppm:
        raise ValueError(f"cap {cap} is not in the frozen sequence {CAP_SEQUENCE}")
    index = sequence_ppm.index(ppm)
    return CAP_SEQUENCE[index + 1] if index + 1 < len(CAP_SEQUENCE) else None


def _resolve_contract_path(summary_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = summary_path.parent / path
    return path


def _hot_evidence(path: Path | None, cap_ppm: int) -> tuple[float | None, list[str]]:
    if path is None:
        return None, []
    payload = _read_json(path)
    failures: list[str] = []
    if payload.get("ok") is not True:
        failures.append("hot summary validation did not pass")
    failures.extend(f"hot: {failure}" for failure in payload.get("failures", []))
    if payload.get("source_changes"):
        failures.append("hot: runtime source files changed")

    contract_path = _resolve_contract_path(path, payload.get("contract"))
    if contract_path is None or not contract_path.is_file():
        failures.append("hot: contract is missing")
    else:
        contract = _read_json(contract_path)
        config = contract.get("pipelines", {}).get(PIPELINE)
        if not isinstance(config, dict):
            failures.append(f"hot: contract is missing {PIPELINE}")
        else:
            if config.get("crop_execution_mode") not in (
                "bounded-range-read-selected-decode",
                "bounded-io-uring-range-read-selected-decode",
                "bounded-io-uring-scheduled-range-read-selected-decode",
            ):
                failures.append("hot: execution mode is not bounded selected decode")
            try:
                hot_cap_ppm = _cap_ppm(float(config.get("bounded_read_amplification_cap")))
            except (TypeError, ValueError):
                failures.append("hot: bounded amplification cap is missing or invalid")
            else:
                if hot_cap_ppm != cap_ppm:
                    failures.append("hot: bounded amplification cap differs from cold candidate")

    median: float | None = None
    if isinstance(payload.get("pipelines"), dict):
        pipeline = payload["pipelines"].get(PIPELINE, {})
        distribution = pipeline.get("throughput_images_per_s", {})
        if "median" in distribution:
            median = float(distribution["median"])
    for aggregate in payload.get("aggregates", []):
        if aggregate.get("pipeline") == PIPELINE:
            distribution = aggregate.get("throughput_images_per_s", {})
            if "median" in distribution:
                median = float(distribution["median"])
                break
    if median is None:
        failures.append(f"hot: cannot find {PIPELINE} throughput median")
    return median, failures


def evaluate(
    controlled_cold_path: Path,
    cap: float,
    *,
    throughput_target: float = 4_717.0,
    stall_target_percent: float = 2.0,
    hot_target: float = 4_871.0,
    hot_summary_path: Path | None = None,
) -> dict[str, Any]:
    payload = _read_json(controlled_cold_path)
    pipeline = payload.get("pipelines", {}).get(PIPELINE)
    if not isinstance(pipeline, dict):
        raise ValueError(f"missing {PIPELINE} in {controlled_cold_path}")
    cap_ppm = _cap_ppm(cap)
    expected_physical_bytes, expected_physical_runs = FROZEN_REPLAY_50K[cap_ppm]
    failures = list(payload.get("failures", []))
    if not payload.get("ok", False):
        failures.append("controlled-cold wrapper did not pass")
    process_repeats = int(payload.get("process_repeats", -1))
    round_validation = payload.get("round_validation", [])
    if len(round_validation) != process_repeats or not round_validation:
        failures.append("round validation count does not match controlled-cold process count")
    for index, validation in enumerate(round_validation, start=1):
        if not validation.get("ok", False):
            failures.append(f"round {index} validation failed")

    native_repeats = pipeline.get("native_totals", [])
    if len(native_repeats) != process_repeats or not native_repeats:
        failures.append("native repeat count does not match controlled-cold process count")
    physical_values: list[int] = []
    run_values: list[int] = []
    memory_used_values: list[int] = []
    memory_values: list[int] = []
    for index, native in enumerate(native_repeats, start=1):
        label = f"round {index}"
        exact = int(native.get("bounded_exact_storage_bytes", -1))
        physical = int(native.get("bounded_physical_storage_bytes", -1))
        gap = int(native.get("bounded_merged_gap_bytes", -1))
        runs = int(native.get("bounded_physical_run_count", -1))
        actual_memory_used = int(native.get("actual_transient_total_used_high_water_bytes", -1))
        actual_memory = int(native.get("actual_transient_total_allocated_high_water_bytes", -1))
        physical_values.append(physical)
        run_values.append(runs)
        memory_used_values.append(actual_memory_used)
        memory_values.append(actual_memory)
        if int(native.get("bounded_read_amplification_ppm", -1)) != cap_ppm:
            failures.append(f"{label}: runtime cap differs from requested cap")
        if int(native.get("bounded_read_local_amplification_ppm", -1)) != 0:
            failures.append(f"{label}: local cap is not the frozen inherit-global setting")
        if int(native.get("bounded_read_max_run_bytes", -1)) != 0:
            failures.append(f"{label}: max-run limit is not the frozen unlimited setting")
        if exact != EXACT_BYTES_50K:
            failures.append(f"{label}: exact bytes {exact} != {EXACT_BYTES_50K}")
        if physical != int(native.get("compressed_payload_bytes_read", -2)):
            failures.append(f"{label}: planned physical bytes differ from actual reads")
        if exact != int(native.get("selected_compressed_payload_bytes", -2)):
            failures.append(f"{label}: exact bytes differ from selected-storage accounting")
        if gap != physical - exact:
            failures.append(f"{label}: merged gap bytes are inconsistent")
        expected_amplification = physical / exact if exact > 0 else math.inf
        try:
            reported_amplification = float(native.get("read_amplification"))
        except (TypeError, ValueError):
            reported_amplification = math.nan
        if not math.isclose(
            reported_amplification,
            expected_amplification,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            failures.append(f"{label}: read amplification is not the whole-run physical/exact ratio")
        if physical > (exact * cap_ppm) // 1_000_000 or physical > FULL_BYTES_50K:
            failures.append(f"{label}: physical bytes exceed cap or full payload")
        if physical != expected_physical_bytes or runs != expected_physical_runs:
            failures.append(
                f"{label}: physical bytes/runs differ from frozen 50K gap replay "
                f"({physical}/{runs} != {expected_physical_bytes}/{expected_physical_runs})"
            )
        if int(native.get("planned_vector_count", -1)) != SELECTED_VECTORS_50K or int(
            native.get("actual_vector_count", -2)
        ) != SELECTED_VECTORS_50K:
            failures.append(f"{label}: planned/actual selected vectors are not {SELECTED_VECTORS_50K}")
        if int(native.get("rowgroup_count", -1)) != ROWGROUPS_50K or int(
            native.get("run_interval_bounded_rowgroup_count", -2)
        ) != ROWGROUPS_50K:
            failures.append(f"{label}: bounded strategy does not cover all {ROWGROUPS_50K} rowgroups")
        if int(native.get("sparse_read_fallback_rowgroup_count", -1)) != 0:
            failures.append(f"{label}: sparse fallback is nonzero")
        for counter in ZERO_COUNTERS:
            if int(native.get(counter, -1)) != 0:
                failures.append(f"{label}: {counter} is nonzero or missing")
        if native.get("actual_transient_memory_gate_passed") is not True or not (
            0 < actual_memory_used <= actual_memory <= TRANSIENT_LIMIT_BYTES
        ):
            failures.append(f"{label}: actual transient memory gate failed")

    throughput_distribution = pipeline.get("throughput_images_per_s")
    if not isinstance(throughput_distribution, dict) or "median" not in throughput_distribution:
        failures.append("controlled-cold throughput evidence is missing")
        throughput = 0.0
    else:
        throughput = float(throughput_distribution["median"])
        if not math.isfinite(throughput) or throughput <= 0.0:
            failures.append("controlled-cold throughput evidence is invalid")
    stall_distribution = pipeline.get("input_ready_stall_percent")
    if not isinstance(stall_distribution, dict) or "median" not in stall_distribution:
        failures.append("controlled-cold input-ready stall evidence is missing")
        stall = math.inf
    else:
        stall = float(stall_distribution["median"])
        if not math.isfinite(stall) or stall < 0.0:
            failures.append("controlled-cold input-ready stall evidence is invalid")
    storage_read_distribution = pipeline.get("process_io_storage_read_bytes")
    if not isinstance(storage_read_distribution, dict) or "median" not in storage_read_distribution:
        failures.append("controlled-cold storage-layer read-byte evidence is missing")
        storage_read_bytes = 0.0
    else:
        storage_read_bytes = float(storage_read_distribution["median"])
        if not math.isfinite(storage_read_bytes) or storage_read_bytes <= 0.0:
            failures.append("controlled-cold storage-layer read-byte evidence is invalid")
        elif storage_read_bytes < expected_physical_bytes:
            failures.append(
                "controlled-cold storage-layer read bytes are smaller than the "
                "application physical payload"
            )
    cold_performance_passed = throughput >= throughput_target and stall <= stall_target_percent
    cold_correctness_passed = not failures
    hot_median, hot_failures = _hot_evidence(hot_summary_path, cap_ppm)
    failures.extend(hot_failures)
    hot_evidence_passed = hot_summary_path is not None and not hot_failures
    hot_passed = hot_evidence_passed and hot_median is not None and hot_median >= hot_target
    correctness_passed = cold_correctness_passed and (hot_summary_path is None or hot_evidence_passed)
    next_cap = _next_cap(cap)
    if not cold_correctness_passed:
        decision = "reject-correctness-or-hard-gate"
    elif not cold_performance_passed:
        decision = "try-next-cap" if next_cap is not None else "stop-hard-cap-missed"
    elif hot_median is None:
        decision = "run-hot-at-same-cap"
    elif not hot_evidence_passed:
        decision = "reject-hot-hard-gate"
    elif not hot_passed:
        decision = "stop-hot-regression"
    else:
        decision = "stop-success"

    return {
        "schema_version": SCHEMA,
        "controlled_cold": str(controlled_cold_path.resolve()),
        "hot_summary": str(hot_summary_path.resolve()) if hot_summary_path is not None else None,
        "cap": cap,
        "cap_ppm": cap_ppm,
        "correctness_passed": correctness_passed,
        "cold_performance_passed": cold_performance_passed,
        "hot_passed": hot_passed if hot_median is not None else None,
        "hot_evidence_passed": hot_evidence_passed if hot_summary_path is not None else None,
        "throughput_images_per_s_median": throughput,
        "throughput_target": throughput_target,
        "input_ready_stall_percent_median": stall,
        "input_ready_stall_target_percent": stall_target_percent,
        "storage_layer_read_bytes_median": storage_read_bytes,
        "hot_throughput_images_per_s_median": hot_median,
        "hot_target": hot_target,
        "physical_bytes": sorted(set(physical_values)),
        "physical_run_count": sorted(set(run_values)),
        "actual_transient_used_high_water_bytes": max(memory_used_values, default=0),
        "actual_transient_allocated_high_water_bytes": max(memory_values, default=0),
        "failures": failures,
        "decision": decision,
        "next_cap": next_cap if decision == "try-next-cap" else None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controlled_cold", type=Path)
    parser.add_argument("--cap", type=float, required=True)
    parser.add_argument("--throughput-target", type=float, default=4_717.0)
    parser.add_argument("--stall-target-percent", type=float, default=2.0)
    parser.add_argument("--hot-target", type=float, default=4_871.0)
    parser.add_argument("--hot-summary", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate(
        args.controlled_cold,
        args.cap,
        throughput_target=args.throughput_target,
        stall_target_percent=args.stall_target_percent,
        hot_target=args.hot_target,
        hot_summary_path=args.hot_summary,
    )
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["decision"] == "stop-success" else 3


if __name__ == "__main__":
    raise SystemExit(main())
