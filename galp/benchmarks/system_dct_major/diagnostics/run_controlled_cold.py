#!/usr/bin/env python3
"""Run repeated process-cold benchmark rounds with per-pipeline file eviction."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
REPO_ROOT = BENCHMARK_ROOT.parents[2]
DEFAULT_PYTHON = Path("/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python")
SCHEMA = "galp_dct_major_process_cold_v2"


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of no values")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    ordered = [float(value) for value in values]
    mean = statistics.fmean(ordered)
    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p95": _percentile(ordered, 0.95),
        "mean": mean,
        "cv_population": statistics.pstdev(ordered) / mean if len(ordered) > 1 and mean else 0.0,
        "min": min(ordered),
        "max": max(ordered),
    }


def _load_round(round_dir: Path) -> dict[str, Any]:
    result = json.loads((round_dir / "results.json").read_text(encoding="utf-8"))
    pipelines: dict[str, Any] = {}
    for aggregate in result["aggregates"]:
        name = str(aggregate["pipeline"])
        raw = json.loads((round_dir / f"pipeline_{name}.json").read_text(encoding="utf-8"))
        if len(raw["repeats"]) != 1:
            raise RuntimeError(f"controlled-cold pipeline must contain exactly one repeat: {name}")
        pipelines[name] = {
            "aggregate": aggregate,
            "repeat": raw["repeats"][0],
            "semantic_artifact": raw["semantic_artifact"],
        }
    return {"summary": result, "pipelines": pipelines}


def _aggregate(rounds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = list(rounds[0]["pipelines"])
    if any(list(item["pipelines"]) != names for item in rounds[1:]):
        raise RuntimeError("controlled-cold rounds used different pipeline orders")
    result: dict[str, Any] = {}
    for name in names:
        cold = [item["pipelines"][name]["aggregate"]["cold_start"] for item in rounds]
        repeats = [item["pipelines"][name]["repeat"] for item in rounds]
        first_shard = [
            float(item["first_shard_ready_ms"])
            for item in cold
            if item.get("first_shard_ready_ms") is not None
        ]
        input_ready_stall = [
            float(item["input_ready_stall_percent"])
            for item in cold
            if item.get("input_ready_stall_percent") is not None
        ]
        process_io_storage_read = [
            float(item["process_io"]["storage_read_bytes"])
            for item in cold
            if isinstance(item.get("process_io"), dict)
            and isinstance(item["process_io"].get("storage_read_bytes"), (int, float))
        ]
        gpu_util = [
            float(item["gpu_utilization_percent"]["mean"])
            for item in repeats
            if int(item.get("gpu_utilization_percent", {}).get("count", 0)) > 0
        ]
        result[name] = {
            "throughput_images_per_s": _distribution(
                [float(item["throughput_images_per_s"]) for item in cold]
            ),
            "cold_time_to_first_batch_ms": _distribution(
                [float(item["time_to_first_batch_ms"]) for item in cold]
            ),
            "first_shard_ready_ms": _distribution(first_shard) if first_shard else None,
            "input_ready_stall_percent": (
                _distribution(input_ready_stall) if len(input_ready_stall) == len(cold) else None
            ),
            "process_io_storage_read_bytes": (
                _distribution(process_io_storage_read)
                if len(process_io_storage_read) == len(cold)
                else None
            ),
            "batch_latency_p50_ms": _distribution(
                [float(item["latency_ms"]["p50"]) for item in repeats]
            ),
            "batch_latency_p95_ms": _distribution(
                [float(item["latency_ms"]["p95"]) for item in repeats]
            ),
            "loader_mean_ms": _distribution(
                [float(item["loader_submit_ms"]["mean"]) for item in repeats]
            ),
            "h2d_mean_ms": _distribution(
                [float(item["top_level_h2d_ms"]["mean"]) for item in repeats]
            ),
            "model_mean_ms": _distribution(
                [float(item["model_ms"]["mean"]) for item in repeats]
            ),
            "gpu_utilization_percent": _distribution(gpu_util) if gpu_util else None,
            "host_peak_rss_bytes": _distribution(
                [float(item["host_peak_rss_bytes"]) for item in repeats]
            ),
            "peak_torch_gpu_allocated_bytes": _distribution(
                [float(item["peak_torch_gpu_allocated_bytes"]) for item in repeats]
            ),
            "peak_torch_gpu_reserved_bytes": _distribution(
                [float(item["peak_torch_gpu_reserved_bytes"]) for item in repeats]
            ),
            "native_totals": [item.get("native_totals", {}) for item in repeats],
        }
    return result


def _write_report(path: Path, result: dict[str, Any]) -> None:
    protocol_name = str(result["cold_protocol"]).replace("-", " ")
    lines = [
        f"# {protocol_name.title()} process-cold results",
        "",
        f"Validation: **{'PASS' if result['ok'] else 'FAIL'}**",
        "",
        "Each observation used a new process and per-file `POSIX_FADV_DONTNEED`; no global cache drop was used.",
        "",
        "| Pipeline | Throughput median (img/s) | Throughput p95 | CV | Process TTFB median (ms) | First shard median (ms) | Input-ready stall median (%) | Storage read median (bytes) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in result["pipelines"].items():
        first_shard = item["first_shard_ready_ms"]
        first_shard_text = f"{first_shard['median']:.3f}" if first_shard is not None else "n/a"
        input_ready_stall = item["input_ready_stall_percent"]
        input_ready_stall_text = (
            f"{input_ready_stall['median']:.3f}" if input_ready_stall is not None else "n/a"
        )
        storage_read = item["process_io_storage_read_bytes"]
        storage_read_text = f"{storage_read['median']:.0f}" if storage_read is not None else "n/a"
        lines.append(
            f"| {name} | {item['throughput_images_per_s']['median']:.3f} | "
            f"{item['throughput_images_per_s']['p95']:.3f} | "
            f"{item['throughput_images_per_s']['cv_population']:.6f} | "
            f"{item['cold_time_to_first_batch_ms']['median']:.3f} | "
            f"{first_shard_text} | "
            f"{input_ready_stall_text} | "
            f"{storage_read_text} |"
        )
    if result.get("cold_speedups"):
        lines.extend(["", "## Process-cold speedups", ""])
        lines.extend(
            f"- `{name}`: `{value:.6f}x`."
            for name, value in result["cold_speedups"].items()
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_arguments = list(args.run_args)
    if run_arguments and run_arguments[0] == "--":
        run_arguments.pop(0)
    forbidden = {
        "--output-dir",
        "--repeats",
        "--dry-run",
        "--evict-pipeline-file-cache",
        "--no-evict-pipeline-file-cache",
        "--python",
        "--cold-protocol",
    }
    overlap = sorted(forbidden.intersection(run_arguments))
    if overlap:
        raise ValueError(f"wrapper owns these run.py options: {overlap}")
    rounds: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for index in range(args.process_repeats):
        round_dir = output_dir / f"round_{index + 1:02d}"
        command = [
            str(args.python.resolve()),
            "-B",
            str(BENCHMARK_ROOT / "run.py"),
            *run_arguments,
            "--python",
            str(args.python.resolve()),
            "--output-dir",
            str(round_dir),
            "--repeats",
            "1",
            "--evict-pipeline-file-cache",
            "--cold-protocol",
            args.protocol,
        ]
        commands.append(command)
        print("COMMAND " + shlex.join(command), flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0 and not (round_dir / "results.json").is_file():
            raise RuntimeError(f"process-cold round {index + 1} failed: {completed.returncode}")
        rounds.append(_load_round(round_dir))
    pipelines = _aggregate(rounds)
    failures = [
        f"round {index + 1}: {failure}"
        for index, item in enumerate(rounds)
        for failure in item["summary"].get("failures", [])
    ]
    failures.extend(
        [
            f"{name}: throughput CV {item['throughput_images_per_s']['cv_population']:.6f} exceeds 0.05"
            for name, item in pipelines.items()
            if float(item["throughput_images_per_s"]["cv_population"]) > 0.05
        ]
    )
    cold_speedups: dict[str, float] = {}
    if "dct_major_pushdown" in pipelines and "dct_major_full" in pipelines:
        full_throughput = float(pipelines["dct_major_full"]["throughput_images_per_s"]["median"])
        pushdown_throughput = float(
            pipelines["dct_major_pushdown"]["throughput_images_per_s"]["median"]
        )
        speedup = pushdown_throughput / full_throughput if full_throughput else 0.0
        cold_speedups["crop_pushdown_over_full"] = speedup
        if speedup <= 1.0:
            failures.append(
                f"dct_major_pushdown process-cold median did not exceed full: {speedup:.6f}x"
            )
    result = {
        "schema_version": SCHEMA,
        "ok": not failures,
        "failures": failures,
        "process_repeats": args.process_repeats,
        "cold_protocol": args.protocol,
        "cache_protocol": "per-pipeline posix_fadvise(DONTNEED); no global drop_caches",
        "model_prime": (
            "disabled" if "--no-cold-start-model-prime" in run_arguments else "enabled"
        ),
        "pipelines": pipelines,
        "cold_speedups": cold_speedups,
        "round_directories": [str((output_dir / f"round_{index + 1:02d}").resolve()) for index in range(args.process_repeats)],
        "round_validation": [
            {
                "ok": bool(item["summary"].get("ok")),
                "failures": list(item["summary"].get("failures", [])),
            }
            for item in rounds
        ],
        "commands": commands,
    }
    (output_dir / "controlled_cold_results.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output_dir / "REPORT.md", result)
    return 0 if result["ok"] else 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--process-repeats", type=int, default=5)
    parser.add_argument(
        "--protocol",
        choices=("controlled-io", "application-overlapped"),
        default="controlled-io",
        help="run.py cold-start accounting protocol",
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("run_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.process_repeats <= 0:
        parser.error("--process-repeats must be positive")
    if not args.run_args:
        parser.error("run.py arguments must follow --")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
