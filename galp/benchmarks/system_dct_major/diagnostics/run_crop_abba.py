#!/usr/bin/env python3
"""Run a strict legacy-eager/planless DCT-major crop-pushdown ABBA sequence."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from common import (  # noqa: E402
    PIPELINE_RESULT_SCHEMA,
    distribution,
    load_contract,
    read_json,
    sha256_json,
    write_json,
)
from run import _prepare_output_dir  # noqa: E402
from validate import _aggregate, _planless_resource_evidence, _semantic_compare  # noqa: E402


SEQUENCE = (
    "dct_major_legacy_pushdown",
    "dct_major_pushdown",
    "dct_major_pushdown",
    "dct_major_legacy_pushdown",
)


def _stream(command: list[str], *, environment: dict[str, str], log: Path) -> int:
    print("COMMAND " + shlex.join(command), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=BENCHMARK_ROOT.parents[2],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        return int(process.wait())


def _summarize(contract: dict[str, Any], legs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    expected_hash = sha256_json(contract)
    expected_trace: list[dict[str, int]] | None = None
    for index, (expected_pipeline, result) in enumerate(zip(SEQUENCE, legs, strict=True)):
        if result.get("schema_version") != PIPELINE_RESULT_SCHEMA:
            failures.append(f"leg {index}: bad result schema")
        if result.get("pipeline") != expected_pipeline:
            failures.append(f"leg {index}: pipeline mismatch")
        if result.get("contract_sha256") != expected_hash:
            failures.append(f"leg {index}: contract hash mismatch")
        repeats = result.get("repeats", [])
        if len(repeats) != 1:
            failures.append(f"leg {index}: expected exactly one repeat")
            continue
        trace = repeats[0].get("sample_trace")
        if expected_trace is None:
            expected_trace = trace
        elif trace != expected_trace:
            failures.append(f"leg {index}: sample trace differs")

    comparisons = []
    for left_index, right_index, label in (
        (0, 3, "legacy-repeatability"),
        (1, 2, "planless-repeatability"),
        (0, 1, "forward-cross-path"),
        (3, 2, "reverse-cross-path"),
    ):
        comparison = _semantic_compare(
            f"leg-{left_index}",
            f"leg-{right_index}",
            legs[left_index],
            legs[right_index],
            contract,
            strict=True,
            failures=failures,
        )
        comparison["label"] = label
        comparisons.append(comparison)

    legacy = [legs[0], legs[3]]
    planless = [legs[1], legs[2]]
    combined_legacy = dict(legacy[0])
    combined_legacy["repeats"] = [item["repeats"][0] for item in legacy]
    combined_planless = dict(planless[0])
    combined_planless["repeats"] = [item["repeats"][0] for item in planless]
    aggregates = {
        "dct_major_legacy_pushdown": _aggregate(contract, combined_legacy),
        "dct_major_pushdown": _aggregate(contract, combined_planless),
    }
    resources = _planless_resource_evidence(aggregates, failures)
    legacy_throughput = distribution(
        [float(item["repeats"][0]["throughput_images_per_s"]) for item in legacy]
    )
    planless_throughput = distribution(
        [float(item["repeats"][0]["throughput_images_per_s"]) for item in planless]
    )
    return {
        "schema_version": "galp_dct_major_planless_abba_v1",
        "ok": not failures,
        "failures": failures,
        "sequence": list(SEQUENCE),
        "legacy_throughput_images_per_s": legacy_throughput,
        "planless_throughput_images_per_s": planless_throughput,
        "planless_over_legacy": float(planless_throughput["p50"]) / float(legacy_throughput["p50"]),
        "planless_resource_evidence": resources,
        "semantic_comparisons": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_contract(args.contract)
    enabled = set(contract["pipelines"]["enabled"])
    required = {"dct_major_legacy_pushdown", "dct_major_pushdown"}
    if not required.issubset(enabled):
        raise ValueError(f"contract must enable {sorted(required)}")
    if int(contract["execution"]["repeats"]) != 1:
        raise ValueError("ABBA contract must use --repeats 1")
    if not args.python.is_file():
        raise FileNotFoundError(args.python)

    output_dir = args.output_dir.resolve()
    _prepare_output_dir(output_dir)
    binding_dir = Path(contract["pipelines"]["dct_major_pushdown"]["torch_binding_dir"])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(binding_dir) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    commands: list[dict[str, Any]] = []
    result_paths: list[Path] = []
    for index, pipeline in enumerate(SEQUENCE):
        leg_dir = output_dir / f"leg_{index:02d}_{pipeline}"
        result_path = leg_dir / f"pipeline_{pipeline}.json"
        command = [
            str(args.python.resolve()),
            str(BENCHMARK_ROOT / "pipeline.py"),
            "--pipeline",
            pipeline,
            "--contract",
            str(args.contract.resolve()),
            "--output",
            str(result_path),
        ]
        commands.append({"leg": index, "pipeline": pipeline, "command": command})
        result_paths.append(result_path)
        if args.dry_run:
            print("COMMAND " + shlex.join(command), flush=True)
            continue
        leg_dir.mkdir(parents=True, exist_ok=False)
        code = _stream(command, environment=environment, log=leg_dir / "pipeline.log")
        if code != 0:
            write_json(output_dir / "commands.json", commands)
            write_json(output_dir / "failed.json", {"leg": index, "pipeline": pipeline, "exit_code": code})
            raise SystemExit(code)

    write_json(output_dir / "commands.json", commands)
    if args.dry_run:
        write_json(output_dir / "run_metadata.json", {"dry_run": True, "sequence": list(SEQUENCE)})
        return
    summary = _summarize(contract, [read_json(path) for path in result_paths])
    write_json(output_dir / "abba_results.json", summary)
    print(
        "RESULT_JSON "
        + json.dumps(
            {"ok": summary["ok"], "planless_over_legacy": summary["planless_over_legacy"]},
            sort_keys=True,
        )
    )
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
