#!/usr/bin/env python3
"""Reproducible baseline/candidate FLS metadata benchmark orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument(
        "--benchmark-tool",
        type=Path,
        default=Path("build/galp/tools/metadata/galp_fls_metadata_benchmark"),
    )
    parser.add_argument(
        "--metadata-tool",
        type=Path,
        default=Path("build/galp/tools/metadata/galp_fls_metadata_tool"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--metadata-iterations", type=int, default=100_000)
    parser.add_argument("--decode-iterations", type=int, default=200)
    parser.add_argument("--metadata-warmup", type=int, default=1_000)
    parser.add_argument("--decode-warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    # Keep benchmark processes on one NUMA node when requested. The value is
    # passed directly to taskset as a CPU-list (for example, 0-3).
    parser.add_argument("--cpu-list")
    parser.add_argument("--seed", type=int, default=20_260_720)
    parser.add_argument(
        "--cache-state", choices=("cold", "warm", "uncontrolled"), default="uncontrolled"
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.metadata_iterations < 1 or args.decode_iterations < 1:
        parser.error("repeat and iteration counts must be positive")
    if args.metadata_warmup < 0 or args.decode_warmup < 0 or args.threads < 1:
        parser.error("warmup counts must be non-negative and threads must be positive")
    return args


def run(command: list[str], commands: list[dict[str, Any]]) -> None:
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    commands.append(
        {
            "argv": command,
            "elapsed_seconds": elapsed,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}"
        )


def lookup(value: dict[str, Any], path: str) -> float:
    current: Any = value
    for component in path.split("."):
        current = current[component]
    return float(current)


METADATA_METRICS = {
    "open_ms": "descriptor_open_ms",
    "loaded_bytes": "descriptor_loaded_bytes",
    "rss_after_open_bytes": "rss_after_open_bytes",
    "peak_rss_bytes": "peak_rss_bytes",
    "random_p50_ms": "random.latency.p50_ms",
    "random_p95_ms": "random.latency.p95_ms",
    "random_p99_ms": "random.latency.p99_ms",
    "random_operations_per_second": "random.operations_per_second",
    "sequential_operations_per_second": "sequential.operations_per_second",
    "multithread_operations_per_second": "multithread_random.operations_per_second",
}

DECODE_METRICS = {
    "open_ms": "table_reader_open_ms",
    "rss_after_open_bytes": "rss_after_reader_open_bytes",
    "peak_rss_bytes": "peak_rss_bytes",
    "random_p50_ms": "random.latency.p50_ms",
    "random_p95_ms": "random.latency.p95_ms",
    "random_p99_ms": "random.latency.p99_ms",
    "random_stored_gib_per_second": "random.stored_gib_per_second",
    "sequential_stored_gib_per_second": "sequential_batch.stored_gib_per_second",
    "multithread_stored_gib_per_second": (
        "multithread_random.measurement.stored_gib_per_second"
    ),
}


def summarize(samples: list[dict[str, Any]], metrics: dict[str, str]) -> dict[str, Any]:
    output: dict[str, Any] = {"repeat_count": len(samples), "metrics": {}}
    for name, path in metrics.items():
        values = [lookup(sample, path) for sample in samples]
        output["metrics"][name] = {
            "mean": statistics.fmean(values),
            "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "values": values,
        }
    return output


def comparison(
    baseline: dict[str, Any], candidate: dict[str, Any], lower_is_better: set[str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, baseline_value in baseline["metrics"].items():
        candidate_value = candidate["metrics"][name]
        base = baseline_value["mean"]
        cand = candidate_value["mean"]
        ratio = cand / base if base else None
        improvement = None
        if ratio is not None:
            improvement = (1.0 - ratio) if name in lower_is_better else (ratio - 1.0)
        output[name] = {
            "baseline_mean": base,
            "candidate_mean": cand,
            "candidate_over_baseline": ratio,
            "improvement_fraction": improvement,
            "lower_is_better": name in lower_is_better,
        }
    return output


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    variants = {"baseline": args.baseline.resolve()}
    if args.candidate:
        variants["candidate"] = args.candidate.resolve()

    for name, input_path in variants.items():
        breakdown = args.output_dir / f"breakdown_{name}.json"
        run(
            [
                str(args.metadata_tool),
                "analyze",
                "--input",
                str(input_path),
                "--output",
                str(breakdown),
                "--pretty",
            ],
            commands,
        )

    if args.candidate:
        run(
            [
                str(args.metadata_tool),
                "compare",
                "--input",
                str(args.baseline.resolve()),
                "--candidate",
                str(args.candidate.resolve()),
                "--output",
                str(args.output_dir / "equivalence.json"),
                "--pretty",
            ],
            commands,
        )

    raw: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"metadata": [], "decode": []} for name in variants
    }
    for repeat in range(args.repeats):
        for name, input_path in variants.items():
            for phase, iterations, warmup in (
                ("metadata", args.metadata_iterations, args.metadata_warmup),
                ("decode", args.decode_iterations, args.decode_warmup),
            ):
                output = args.output_dir / f"{name}_{phase}_repeat_{repeat:02d}.json"
                prefix = ["taskset", "--cpu-list", args.cpu_list] if args.cpu_list else []
                run(
                    prefix
                    + [
                        str(args.benchmark_tool),
                        "--input",
                        str(input_path),
                        "--phase",
                        phase,
                        "--iterations",
                        str(iterations),
                        "--warmup",
                        str(warmup),
                        "--threads",
                        str(args.threads),
                        "--seed",
                        str(args.seed),
                        "--cache-state",
                        args.cache_state,
                        "--output",
                        str(output),
                        "--pretty",
                    ],
                    commands,
                )
                raw[name][phase].append(json.loads(output.read_text())["result"])

    summaries: dict[str, Any] = {}
    for name in variants:
        summaries[name] = {
            "metadata": summarize(raw[name]["metadata"], METADATA_METRICS),
            "decode": summarize(raw[name]["decode"], DECODE_METRICS),
        }

    result: dict[str, Any] = {
        "schema_version": "galp_fls_metadata_benchmark_summary_v1",
        "parameters": {
            "repeats": args.repeats,
            "metadata_iterations": args.metadata_iterations,
            "decode_iterations": args.decode_iterations,
            "metadata_warmup": args.metadata_warmup,
            "decode_warmup": args.decode_warmup,
            "threads": args.threads,
            "cpu_list": args.cpu_list,
            "seed": args.seed,
            "cache_state": args.cache_state,
        },
        "summaries": summaries,
    }
    if args.candidate:
        lower = {
            "open_ms",
            "loaded_bytes",
            "rss_after_open_bytes",
            "peak_rss_bytes",
            "random_p50_ms",
            "random_p95_ms",
            "random_p99_ms",
        }
        result["comparisons"] = {
            phase: comparison(summaries["baseline"][phase], summaries["candidate"][phase], lower)
            for phase in ("metadata", "decode")
        }

    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    metadata = {
        "schema_version": "galp_fls_metadata_run_metadata_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "load_average_at_summary": os.getloadavg(),
        "argv": vars(args) | {"baseline": str(args.baseline), "candidate": str(args.candidate)},
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
