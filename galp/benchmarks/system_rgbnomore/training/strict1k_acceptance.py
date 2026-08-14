#!/usr/bin/env python3
"""Evaluate current-code strict-1K image-major-v3 planless regression."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from training.gate2_acceptance import (
        _mapping,
        _measured_latest,
        _number,
        _phase_repeats,
        _read_object,
        _sha256,
        evaluate_gate2,
    )
except ModuleNotFoundError:  # Direct script execution from this directory.
    from gate2_acceptance import (  # type: ignore[no-redef]
        _mapping,
        _measured_latest,
        _number,
        _phase_repeats,
        _read_object,
        _sha256,
        evaluate_gate2,
    )


SCHEMA_VERSION = "galp-image-major-v3-strict1k-regression-v1"
HISTORICAL_V3_THROUGHPUT_IMAGES_PER_S = 1497.070
DEFAULT_MAX_THROUGHPUT_REGRESSION_FRACTION = 0.10
DEFAULT_MAX_THROUGHPUT_CV = 0.10
DEFAULT_MIN_REPEATS = 3


def evaluate_strict1k(
    run_dir: Path,
    *,
    historical_throughput: float = HISTORICAL_V3_THROUGHPUT_IMAGES_PER_S,
    max_regression_fraction: float = DEFAULT_MAX_THROUGHPUT_REGRESSION_FRACTION,
    max_throughput_cv: float = DEFAULT_MAX_THROUGHPUT_CV,
    min_repeats: int = DEFAULT_MIN_REPEATS,
) -> dict[str, Any]:
    # Historical strict-1K artifacts predate the full-dataset compact-pool
    # capacity fields; they remain useful as a no-growth regression gate. The
    # official Gate 2 evaluator keeps the new contract mandatory.
    base = evaluate_gate2(
        run_dir,
        allow_semantic_not_applicable=True,
        require_compact_pool_contract=False,
    )
    if not base.get("complete"):
        return {
            **base,
            "schema_version": SCHEMA_VERSION,
            "strict1k": {"historical_throughput_images_per_s": historical_throughput},
        }

    pipeline = _read_object(run_dir / "pipeline_galp.json")
    repeats = _phase_repeats(pipeline)
    checks = dict(_mapping(base.get("checks")))
    failures = list(base.get("failures", []))

    def check(name: str, ok: bool, actual: Any, expected: str) -> None:
        checks[name] = {"ok": bool(ok), "actual": actual, "expected": expected}
        if not ok:
            failures.append(f"{name}: expected {expected}, got {actual!r}")

    check("strict1k.minimum_repeat_count", len(repeats) >= min_repeats, len(repeats), f">={min_repeats}")
    throughputs = [
        value
        for repeat in repeats
        if (value := _number(repeat.get("throughput_images_per_s"))) is not None
    ]
    median = statistics.median(throughputs) if throughputs else None
    cv = None
    if len(throughputs) >= 2 and statistics.fmean(throughputs) != 0.0:
        cv = statistics.stdev(throughputs) / statistics.fmean(throughputs)
    minimum_accepted = historical_throughput * (1.0 - max_regression_fraction)
    check(
        "strict1k.throughput_regression",
        len(throughputs) == len(repeats) and median is not None and median >= minimum_accepted,
        {"values": throughputs, "median": median},
        f"median >={minimum_accepted} img/s",
    )
    check(
        "strict1k.throughput_cv",
        cv is not None and cv <= max_throughput_cv,
        cv,
        f"<={max_throughput_cv}",
    )

    for index, repeat in enumerate(repeats):
        latest = _measured_latest(repeat)
        host_counters = {
            name: _number(latest.get(name))
            for name in (
                "host_expanded_transform_items_created",
                "host_output_block_source_lists_created",
                "host_global_transform_sort_items",
            )
        }
        check(
            f"strict1k.repeat_{index}.compact_planless_path",
            latest.get("uses_planless_fixed_transform") is True
            and all(value == 0.0 for value in host_counters.values()),
            {
                "uses_planless_fixed_transform": latest.get("uses_planless_fixed_transform"),
                **host_counters,
            },
            "planless=true and all host expansion/list/sort counters=0",
        )

    artifacts = dict(_mapping(base.get("artifacts")))
    evaluator_path = Path(__file__).resolve()
    artifacts["strict1k_evaluator"] = {
        "path": str(evaluator_path),
        "sha256": _sha256(evaluator_path),
    }
    return {
        **base,
        "schema_version": SCHEMA_VERSION,
        "ok": not failures,
        "failures": failures,
        "checks": checks,
        "artifacts": artifacts,
        "strict1k": {
            "historical_throughput_images_per_s": historical_throughput,
            "max_regression_fraction": max_regression_fraction,
            "minimum_accepted_throughput_images_per_s": minimum_accepted,
            "throughput_images_per_s": throughputs,
            "throughput_median_images_per_s": median,
            "throughput_cv": cv,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--historical-throughput", type=float, default=HISTORICAL_V3_THROUGHPUT_IMAGES_PER_S)
    parser.add_argument(
        "--max-regression-fraction",
        type=float,
        default=DEFAULT_MAX_THROUGHPUT_REGRESSION_FRACTION,
    )
    parser.add_argument("--max-throughput-cv", type=float, default=DEFAULT_MAX_THROUGHPUT_CV)
    parser.add_argument("--min-repeats", type=int, default=DEFAULT_MIN_REPEATS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_strict1k(
        args.run_dir,
        historical_throughput=args.historical_throughput,
        max_regression_fraction=args.max_regression_fraction,
        max_throughput_cv=args.max_throughput_cv,
        min_repeats=args.min_repeats,
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
