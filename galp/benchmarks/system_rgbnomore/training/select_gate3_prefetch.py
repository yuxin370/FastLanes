#!/usr/bin/env python3
"""Select K=2/4/8 only from image-major-v3 Gate-3-passing runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from training.gate3_acceptance import _sha256, evaluate_gate3
except ModuleNotFoundError:  # Direct script execution from this directory.
    from gate3_acceptance import _sha256, evaluate_gate3  # type: ignore[no-redef]


SCHEMA_VERSION = "galp-image-major-v3-gate3-prefetch-selection-v1"
REQUIRED_DEPTHS = (2, 4, 8)
DEFAULT_MIN_IMPROVEMENT_FRACTION = 0.02


def select_gate3_prefetch(
    reports: Mapping[int, Mapping[str, Any]],
    *,
    min_improvement_fraction: float = DEFAULT_MIN_IMPROVEMENT_FRACTION,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    passed: dict[int, float] = {}
    failures: list[str] = []
    for depth in REQUIRED_DEPTHS:
        report = reports.get(depth)
        if report is None:
            failures.append(f"missing K={depth} Gate 3 report")
            candidates[str(depth)] = {"present": False, "ok": False, "median_images_per_s": None}
            continue
        hot = report.get("metrics", {}).get("hot_repeat_throughput", {})
        median = hot.get("median") if isinstance(hot, Mapping) else None
        numeric_median = float(median) if isinstance(median, (int, float)) and not isinstance(median, bool) else None
        ok = report.get("complete") is True and report.get("ok") is True and numeric_median is not None
        candidates[str(depth)] = {
            "present": True,
            "ok": ok,
            "median_images_per_s": numeric_median,
            "failure_count": len(report.get("failures", [])) if isinstance(report.get("failures"), list) else None,
        }
        if ok and numeric_median is not None:
            passed[depth] = numeric_median

    if 2 not in passed:
        failures.append("K=2 reference did not pass Gate 3")
    if len(reports) == len(REQUIRED_DEPTHS) and not passed:
        failures.append("no K candidate passed Gate 3")
    if failures:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "complete": len(reports) == len(REQUIRED_DEPTHS),
            "failures": failures,
            "candidates": candidates,
            "selection": None,
            "min_improvement_fraction": min_improvement_fraction,
        }

    best_depth = max(passed, key=lambda depth: (passed[depth], -depth))
    reference = passed[2]
    best = passed[best_depth]
    improvement = best / reference - 1.0 if reference else None
    if best_depth != 2 and improvement is not None and improvement >= min_improvement_fraction:
        selected_depth = best_depth
        reason = "best passing candidate exceeds K=2 by the required margin"
    else:
        selected_depth = 2
        reason = "no passing candidate exceeds K=2 by the required margin"
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "complete": True,
        "failures": [],
        "candidates": candidates,
        "selection": {
            "prefetch_depth_batches": selected_depth,
            "median_images_per_s": passed[selected_depth],
            "best_raw_prefetch_depth_batches": best_depth,
            "best_raw_median_images_per_s": best,
            "best_raw_improvement_vs_k2_fraction": improvement,
            "reason": reason,
        },
        "min_improvement_fraction": min_improvement_fraction,
    }


def _candidate(value: str) -> tuple[int, Path]:
    depth_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("candidate must be K=RUN_DIR")
    try:
        depth = int(depth_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("candidate K must be an integer") from error
    if depth not in REQUIRED_DEPTHS:
        raise argparse.ArgumentTypeError(f"candidate K must be one of {REQUIRED_DEPTHS}")
    return depth, Path(path_text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=_candidate, required=True)
    parser.add_argument("--min-improvement-fraction", type=float, default=DEFAULT_MIN_IMPROVEMENT_FRACTION)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dirs = dict(args.candidate)
    duplicate_count = len(args.candidate) - len(run_dirs)
    reports = {depth: evaluate_gate3(path) for depth, path in run_dirs.items()}
    report = select_gate3_prefetch(
        reports,
        min_improvement_fraction=args.min_improvement_fraction,
    )
    report["run_dirs"] = {str(depth): str(path.resolve()) for depth, path in run_dirs.items()}
    selector_path = Path(__file__).resolve()
    report["selector"] = {"path": str(selector_path), "sha256": _sha256(selector_path)}
    if duplicate_count:
        report["ok"] = False
        report["failures"].append("duplicate --candidate depth")
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
