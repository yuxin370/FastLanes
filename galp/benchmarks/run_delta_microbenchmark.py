#!/usr/bin/env python3

"""Run and compare the controlled GALP DELTA micro-benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable


CSV_COLUMNS = [
    "encoding",
    "data_type",
    "bit_width",
    "mode",
    "unpack_n_vectors",
    "n_vectors",
    "selected_chunks",
    "sample",
    "kernel_us",
    "gvalues_per_s",
    "ns_per_value",
]


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+", shlex.join(command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_generated_bindings(repo: Path, build_dir: Path) -> dict[str, Any]:
    script = repo / "galp/scripts/codegen/generate_kernel_bindings.py"
    generated_root = build_dir / "generated/galp/benchmarks"
    checked_files = [
        "bindings/delta-int8_t-decompress_column-bindings.cu",
        "bindings/delta-int16_t-decompress_column-bindings.cu",
        "bindings/ffor-int8_t-decompress_column-bindings.cu",
        "bindings/ffor-int16_t-decompress_column-bindings.cu",
        "include/galp_bench/generated/kernel_bindings.cuh",
    ]
    with tempfile.TemporaryDirectory(prefix="galp-delta-codegen-") as temp:
        temp_root = Path(temp)
        run(
            [
                sys.executable,
                str(script),
                "--out-dir",
                str(temp_root / "bindings"),
                "--header-out-dir",
                str(temp_root / "include/galp_bench/generated"),
            ],
            cwd=repo,
        )
        files: list[dict[str, Any]] = []
        for relative in checked_files:
            expected = generated_root / relative
            regenerated = temp_root / relative
            matches = expected.is_file() and regenerated.is_file() and expected.read_bytes() == regenerated.read_bytes()
            files.append(
                {
                    "path": relative,
                    "matches": matches,
                    "build_sha256": sha256(expected) if expected.is_file() else None,
                    "regenerated_sha256": sha256(regenerated) if regenerated.is_file() else None,
                }
            )
    return {"success": all(item["matches"] for item in files), "files": files}


def occupancy_model(architecture: str, registers: int, shared_bytes: int) -> dict[str, Any]:
    # All generated DELTA bindings launch the FastLanes mapping with eight
    # warps (256 threads) per block. Resource limits are CUDA's architectural
    # limits for the cubin target, not measurements inferred from throughput.
    limits = {
        "sm_89": {"threads": 1536, "warps": 48, "blocks": 24, "registers": 65536, "shared": 102400},
        "sm_90": {"threads": 2048, "warps": 64, "blocks": 32, "registers": 65536, "shared": 233472},
    }.get(architecture)
    if limits is None:
        return {"architecture": architecture, "available": False}

    threads_per_block = 256
    warps_per_block = threads_per_block // 32
    # Registers are allocated to each warp in 256-register units.
    registers_per_warp = math.ceil(registers * 32 / 256) * 256
    registers_per_block = registers_per_warp * warps_per_block
    by_registers = limits["registers"] // registers_per_block if registers_per_block else limits["blocks"]
    by_threads = limits["threads"] // threads_per_block
    by_warps = limits["warps"] // warps_per_block
    by_shared = limits["shared"] // shared_bytes if shared_bytes else limits["blocks"]
    active_blocks = min(limits["blocks"], by_registers, by_threads, by_warps, by_shared)
    return {
        "architecture": architecture,
        "available": True,
        "threads_per_block": threads_per_block,
        "active_blocks_per_sm": active_blocks,
        "active_warps_per_sm": active_blocks * warps_per_block,
        "max_warps_per_sm": limits["warps"],
        "theoretical_occupancy_percent": 100.0 * active_blocks * warps_per_block / limits["warps"],
        "limiting_active_blocks": {
            "registers": by_registers,
            "threads": by_threads,
            "warps": by_warps,
            "shared_memory": by_shared,
            "architectural_blocks": limits["blocks"],
        },
    }


def parse_ptxas_output(output: str, data_type: str, architecture: str) -> list[dict[str, Any]]:
    function_pattern = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
    stack_pattern = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
    register_pattern = re.compile(r"Used (\d+) registers")
    unpack_pattern = re.compile(r"decompress_columnI[as]Li(\d+)E")

    current: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    for line in output.splitlines():
        function_match = function_pattern.search(line)
        if function_match:
            symbol = function_match.group(1)
            if "DELTADecompressor" not in symbol:
                current = None
                continue
            unpack_match = unpack_pattern.search(symbol)
            if unpack_match is None:
                raise RuntimeError(f"could not decode DELTA unpack width from {symbol}")
            current = {
                "data_type": data_type,
                "unpack_n_vectors": int(unpack_match.group(1)),
                "architecture": function_match.group(2),
                "symbol": symbol,
            }
            continue
        if current is None:
            continue
        stack_match = stack_pattern.search(line)
        if stack_match:
            current.update(
                {
                    "stack_bytes_per_thread": int(stack_match.group(1)),
                    "spill_store_bytes": int(stack_match.group(2)),
                    "spill_load_bytes": int(stack_match.group(3)),
                }
            )
            continue
        register_match = register_pattern.search(line)
        if register_match:
            current["registers_per_thread"] = int(register_match.group(1))
            current["shared_bytes_per_block"] = 0
            current["occupancy"] = occupancy_model(
                architecture, current["registers_per_thread"], current["shared_bytes_per_block"]
            )
            results.append(current)
            current = None
    return sorted(results, key=lambda item: item["unpack_n_vectors"])


def collect_resource_metrics(build_dir: Path, output_dir: Path) -> dict[str, Any]:
    commands_path = build_dir / "compile_commands.json"
    if not commands_path.is_file():
        raise RuntimeError(f"missing {commands_path}; configure CMake with CMAKE_EXPORT_COMPILE_COMMANDS=ON")
    compile_commands = json.loads(commands_path.read_text())
    selected = [
        entry
        for entry in compile_commands
        if re.search(r"delta-int(8|16)_t-decompress_column-bindings\.cu$", entry.get("file", ""))
    ]
    if len(selected) != 2:
        raise RuntimeError(f"expected two generated DELTA compile commands, found {len(selected)}")

    resource_dir = output_dir / "resource_probe"
    resource_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []
    raw_logs: list[str] = []
    architecture = ""
    for entry in sorted(selected, key=lambda item: item["file"]):
        source = Path(entry["file"])
        data_type = "i8" if "int8_t" in source.name else "i16"
        command = shlex.split(entry["command"])
        output_index = command.index("-o") + 1
        command[output_index] = str(resource_dir / f"delta_{data_type}.o")
        command.insert(command.index("-x"), "-Xptxas=-v")
        completed = run(command, cwd=Path(entry["directory"]))
        log_path = resource_dir / f"delta_{data_type}.ptxas.txt"
        log_path.write_text(completed.stdout)
        raw_logs.append(str(log_path))
        architecture_match = re.search(r"arch=compute_(\d+)", entry["command"])
        architecture = f"sm_{architecture_match.group(1)}" if architecture_match else "unknown"
        all_metrics.extend(parse_ptxas_output(completed.stdout, data_type, architecture))

    if {(row["data_type"], row["unpack_n_vectors"]) for row in all_metrics} != {
        (data_type, unpack) for data_type in ("i8", "i16") for unpack in (1, 2, 4)
    }:
        raise RuntimeError("PTXAS resource probe did not report all DELTA I8/I16 unpack=1/2/4 kernels")
    return {"architecture": architecture, "kernels": all_metrics, "raw_logs": raw_logs}


def benchmark_matrix(full_vectors: int, tail_vectors: int) -> Iterable[tuple[str, str, int, int, str, int, int]]:
    widths = {"i8": (1, 8), "i16": (1, 16)}
    for encoding in ("delta", "ffor"):
        for data_type in ("i8", "i16"):
            start_width, end_width = widths[data_type]
            for mode in ("full", "selected"):
                for unpack in (1, 2, 4):
                    yield encoding, data_type, start_width, end_width, mode, unpack, full_vectors
            for unpack in (2, 4):
                yield encoding, data_type, start_width, end_width, "tail", unpack, tail_vectors


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_COLUMNS:
            raise RuntimeError(f"unexpected CSV header in {path}: {reader.fieldnames}")
        return list(reader)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    key_columns = [
        "encoding",
        "data_type",
        "bit_width",
        "mode",
        "unpack_n_vectors",
        "n_vectors",
        "selected_chunks",
    ]
    for row in rows:
        groups.setdefault(tuple(row[column] for column in key_columns), []).append(row)

    output: list[dict[str, Any]] = []
    for key, samples in sorted(groups.items()):
        kernel_us = [float(sample["kernel_us"]) for sample in samples]
        ns_per_value = [float(sample["ns_per_value"]) for sample in samples]
        summary: dict[str, Any] = dict(zip(key_columns, key))
        summary.update(
            {
                "samples": len(samples),
                "median_kernel_us": statistics.median(kernel_us),
                "min_kernel_us": min(kernel_us),
                "max_kernel_us": max(kernel_us),
                "median_ns_per_value": statistics.median(ns_per_value),
                "median_gvalues_per_s": 1.0 / statistics.median(ns_per_value),
            }
        )
        output.append(summary)
    return output


def compare_summaries(
    baseline: list[dict[str, Any]], current: list[dict[str, Any]], max_regression_percent: float
) -> dict[str, Any]:
    key_columns = [
        "encoding",
        "data_type",
        "bit_width",
        "mode",
        "unpack_n_vectors",
        "n_vectors",
        "selected_chunks",
    ]
    baseline_by_key = {tuple(str(row[column]) for column in key_columns): row for row in baseline}
    rows: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    for current_row in current:
        key = tuple(str(current_row[column]) for column in key_columns)
        baseline_row = baseline_by_key.get(key)
        if baseline_row is None:
            continue
        before = float(baseline_row["median_ns_per_value"])
        after = float(current_row["median_ns_per_value"])
        regression = 100.0 * (after / before - 1.0) if before else 0.0
        row = dict(zip(key_columns, key))
        row.update(
            {
                "baseline_median_ns_per_value": before,
                "current_median_ns_per_value": after,
                "regression_percent": regression,
            }
        )
        rows.append(row)
        if current_row["encoding"] == "delta" and regression > max_regression_percent:
            regressions.append(row)
    missing = sorted(set(baseline_by_key) - {tuple(str(row[column]) for column in key_columns) for row in current})
    return {
        "max_allowed_regression_percent": max_regression_percent,
        "success": not regressions and not missing,
        "regressions": regressions,
        "missing_current_keys": missing,
        "rows": rows,
    }


def collect_machine_metadata(repo: Path, env: dict[str, str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", "")}
    for name, command in (
        ("gpu", ["nvidia-smi", "--query-gpu=index,name,driver_version,compute_cap", "--format=csv,noheader"]),
        ("git_commit", ["git", "rev-parse", "HEAD"]),
        ("git_status", ["git", "status", "--short"]),
    ):
        completed = run(command, cwd=repo, env=env, check=False)
        metadata[name] = completed.stdout.strip()
        metadata[f"{name}_exit_code"] = completed.returncode
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--micro-bench", type=Path, default=Path("build/galp/benchmarks/micro_bench"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--device", default="0")
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--full-vectors", type=int, default=4096)
    parser.add_argument("--tail-vectors", type=int)
    parser.add_argument("--compare-to", type=Path, help="Baseline summary JSON created by this script")
    parser.add_argument("--max-regression-percent", type=float, default=5.0)
    parser.add_argument("--skip-codegen-check", action="store_true")
    parser.add_argument("--skip-resource-probe", action="store_true")
    args = parser.parse_args()

    if args.samples < 5:
        parser.error("--samples must be at least 5")
    repo = Path(__file__).resolve().parents[2]
    build_dir = (repo / args.build_dir).resolve() if not args.build_dir.is_absolute() else args.build_dir.resolve()
    micro_bench = (repo / args.micro_bench).resolve() if not args.micro_bench.is_absolute() else args.micro_bench.resolve()
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tail_vectors = args.tail_vectors if args.tail_vectors is not None else args.full_vectors - 1
    if args.full_vectors <= 0 or args.full_vectors % 4 != 0 or tail_vectors <= 0 or tail_vectors % 2 == 0:
        parser.error(
            "full-vectors must be a positive multiple of 4 and tail-vectors must be positive and odd"
        )
    if not micro_bench.is_file():
        parser.error(f"micro_bench not found: {micro_bench}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device
    rows: list[dict[str, str]] = []
    commands: list[list[str]] = []
    for encoding, data_type, start_width, end_width, mode, unpack, n_vectors in benchmark_matrix(
        args.full_vectors, tail_vectors
    ):
        csv_path = raw_dir / f"{encoding}_{data_type}_{mode}_u{unpack}.csv"
        command = [
            str(micro_bench),
            data_type,
            encoding,
            "decompress",
            str(unpack),
            "1",
            "stateful-branchless",
            "none",
            "none",
            str(start_width),
            str(end_width),
            "0",
            "0",
            str(n_vectors),
            str(args.samples),
            "0",
            mode,
            str(csv_path),
        ]
        completed = run(command, cwd=repo, env=env)
        if completed.stdout:
            print(completed.stdout, end="")
        commands.append(command)
        rows.extend(read_csv_rows(csv_path))

    aggregate_csv = output_dir / f"{args.label}.csv"
    with aggregate_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    codegen = {"success": True, "skipped": True}
    if not args.skip_codegen_check:
        codegen = check_generated_bindings(repo, build_dir)
    resources: dict[str, Any] = {"skipped": True}
    if not args.skip_resource_probe:
        resources = collect_resource_metrics(build_dir, output_dir)

    summary_rows = summarize(rows)
    report: dict[str, Any] = {
        "label": args.label,
        "aggregate_csv": str(aggregate_csv),
        "configuration": {
            "samples": args.samples,
            "full_vectors": args.full_vectors,
            "tail_vectors": tail_vectors,
            "delta_bit_widths": {"i8": [1, 8], "i16": [1, 16]},
            "unpack_n_vectors": [1, 2, 4],
            "modes": ["full", "tail", "selected"],
            "timing_scope": "CUDA event time around production workset kernels only",
        },
        "machine": collect_machine_metadata(repo, env),
        "generated_binding_reproducibility": codegen,
        "resources": resources,
        "commands": commands,
        "summary": summary_rows,
    }
    exit_code = 0
    if not codegen.get("success", False):
        exit_code = 1
    if args.compare_to:
        baseline_report = json.loads(args.compare_to.read_text())
        comparison = compare_summaries(
            baseline_report["summary"], summary_rows, args.max_regression_percent
        )
        report["comparison"] = comparison
        if not comparison["success"]:
            exit_code = 1

    report_path = output_dir / f"{args.label}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {aggregate_csv}")
    print(f"wrote {report_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
