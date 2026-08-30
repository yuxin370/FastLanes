#!/usr/bin/env python3
"""Combine equal-image D2/D3/PyTorch results with native physical B6 metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_EPOCHS = (1, 2)
EXPECTED_SAMPLES = 1_281_167
EXPECTED_MICROBATCHES = 20_019
EXPECTED_UPDATES = 1_252
STANDARD_PIPELINES = ("d2", "d3", "pytorch")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _epoch_map(records: Sequence[Mapping[str, Any]], source: str) -> dict[int, dict[str, Any]]:
    selected = [
        dict(row)
        for row in records
        if row.get("record_type") == "train"
        and row.get("scope") == "epoch"
        and int(row.get("epoch", -1)) in EXPECTED_EPOCHS
    ]
    result: dict[int, dict[str, Any]] = {}
    for row in selected:
        epoch = int(row["epoch"])
        if epoch in result:
            raise ValueError(f"{source} duplicates epoch {epoch}")
        result[epoch] = row
    if tuple(sorted(result)) != EXPECTED_EPOCHS:
        raise ValueError(f"{source} must contain exact epochs {EXPECTED_EPOCHS}")
    for epoch, row in result.items():
        observed = {
            "samples": int(row["epoch_samples"]),
            "microbatches": int(row["epoch_microbatches"]),
            "updates": int(row["epoch_optimizer_updates"]),
        }
        expected = {
            "samples": EXPECTED_SAMPLES,
            "microbatches": EXPECTED_MICROBATCHES,
            "updates": EXPECTED_UPDATES,
        }
        if observed != expected:
            raise ValueError(
                f"{source} epoch {epoch} workload differs: {observed} != {expected}"
            )
    return result


def _gpu_identity(environment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": environment.get("gpu_name"),
        "uuid": environment.get("gpu_uuid"),
        "hostname": environment.get("hostname"),
        "torch": environment.get("torch"),
        "torch_cuda_build": environment.get("torch_cuda_build"),
    }


def build_report(
    *,
    standard_root: Path,
    native_run: Path,
    standard_pipelines: Sequence[str] = STANDARD_PIPELINES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    standard_pipelines = tuple(standard_pipelines)
    if not standard_pipelines or len(set(standard_pipelines)) != len(standard_pipelines):
        raise ValueError("standard_pipelines must be non-empty and unique")
    unsupported = set(standard_pipelines) - set(STANDARD_PIPELINES)
    if unsupported:
        raise ValueError(f"unsupported standard pipelines: {sorted(unsupported)}")
    standard_root = standard_root.resolve()
    native_run = native_run.resolve()
    contract = _read_json(standard_root / "contract.json")
    if contract.get("benchmark") != "equal-image-epoch-aware-rgb-training-v2":
        raise ValueError("standard root is not an equal-image RGB benchmark")
    if bool(contract.get("profiling", {}).get("enabled", False)):
        raise ValueError(
            "profiling-only RGB runs cannot be used as formal performance evidence"
        )
    schedule = contract.get("prefix_schedule", {})
    expected_schedule = {
        "processed_images": 2 * EXPECTED_SAMPLES,
        "total_microbatches": 2 * EXPECTED_MICROBATCHES,
        "total_optimizer_updates": 2 * EXPECTED_UPDATES,
    }
    for key, expected in expected_schedule.items():
        if int(schedule.get(key, -1)) != expected:
            raise ValueError(f"standard contract {key} differs")

    standard_environment = _read_json(standard_root / "environment.json")
    native_environment = _read_json(native_run / "environment.json")
    if (
        native_environment.get("execution_mode") != "native_physical_pls"
        or native_environment.get("physical_fls_observed") is not True
        or native_environment.get("physical_gpu_pool") is not True
    ):
        raise ValueError("native run is not the physical PLS/GPU-pool backend")
    native_condition = _read_json(native_run / "condition_contract.json")
    if native_condition.get("condition_id") != "B6":
        raise ValueError("native run condition is not B6")
    standard_gpu = _gpu_identity(standard_environment)
    native_gpu = _gpu_identity(native_environment)
    if standard_gpu != native_gpu:
        raise ValueError(
            f"standard/native hardware-software identity differs: "
            f"{standard_gpu} != {native_gpu}"
        )
    if "H100" not in str(standard_gpu["name"]):
        raise ValueError(f"comparison GPU is not H100: {standard_gpu['name']}")

    epoch_maps: dict[str, dict[int, dict[str, Any]]] = {}
    for pipeline in standard_pipelines:
        result = _read_json(standard_root / "runs" / pipeline / "final_result.json")
        if result.get("state") != "completed" or int(result.get("completed_epoch", -1)) != 2:
            raise ValueError(f"{pipeline} equal-image run is incomplete")
        epoch_maps[pipeline] = _epoch_map(
            result.get("epoch_records", []), f"standard {pipeline}"
        )

    native_status = _read_json(native_run / "run_status.json")
    if int(native_status.get("completed_epoch", -1)) < 2:
        raise ValueError("native B6 run has not completed epoch 2")
    if int(native_status.get("seed", -1)) != int(contract.get("seed", -2)):
        raise ValueError("standard/native training seed differs")
    native_records = _read_jsonl(native_run / "metrics.jsonl")
    epoch_maps["native_b6"] = _epoch_map(native_records, "native B6")

    rows: list[dict[str, Any]] = []
    native_warm_ips = float(epoch_maps["native_b6"][2]["images_per_second"])
    domains = {
        "native_b6": "dct",
        "d2": "rgb",
        "d3": "rgb",
        "pytorch": "rgb",
    }
    for pipeline in ("native_b6", *standard_pipelines):
        for epoch in EXPECTED_EPOCHS:
            source = epoch_maps[pipeline][epoch]
            ips = float(source["images_per_second"])
            rows.append(
                {
                    "pipeline": pipeline,
                    "input_domain": domains[pipeline],
                    "epoch": epoch,
                    "observation": "cold" if epoch == 1 else "warm-primary",
                    "epoch_samples": int(source["epoch_samples"]),
                    "epoch_microbatches": int(source["epoch_microbatches"]),
                    "epoch_optimizer_updates": int(source["epoch_optimizer_updates"]),
                    "epoch_seconds": float(source["epoch_seconds"]),
                    "images_per_second": ips,
                    "warm_throughput_relative_to_native_b6": (
                        None if epoch == 1 else ips / native_warm_ips
                    ),
                    "data_preparation_seconds": float(
                        source.get(
                            "data_preparation_seconds",
                            source.get("native_pool_load_seconds", 0.0),
                        )
                    ),
                    "loader_wait_seconds": float(
                        source.get(
                            "loader_wait_seconds",
                            source.get("native_pool_boundary_wait_seconds", 0.0),
                        )
                    ),
                }
            )

    report = {
        "schema_version": "galp-equal-image-performance-report-v2",
        "hardware": standard_gpu,
        "workload": {
            "epochs": [1, 2],
            "epoch_1_role": "cold observation",
            "epoch_2_role": "primary warm observation",
            "samples_per_epoch": EXPECTED_SAMPLES,
            "microbatch_images": 64,
            "gradient_accumulation": 16,
            "microbatches_per_epoch": EXPECTED_MICROBATCHES,
            "optimizer_updates_per_epoch": EXPECTED_UPDATES,
            "validation_excluded_from_training_throughput": True,
        },
        "rows": rows,
        "included_pipelines": ["native_b6", *standard_pipelines],
        "claims": {
            "d2_vs_pytorch": (
                "direct RGB comparison with canonical order and planned crop/flip"
                if {"d2", "pytorch"}.issubset(standard_pipelines)
                else "pending; D2 and PyTorch are not both included"
            ),
            "d3_performance_ceiling": (
                "DALI-native shuffle/crop/flip performance ceiling; order and "
                "augmentation decisions differ from D2/PyTorch"
                if "d3" in standard_pipelines
                else "pending; D3 is not included"
            ),
            "rgb_vs_native_b6": (
                "equal-image full-application comparison, not loader-only and not "
                "model-compute-isolated because RGB and DCT input domains differ"
            ),
            "epoch_1_cold_comparison": False,
            "epoch_2_warm_comparison": True,
            "accuracy_or_convergence_claim": False,
            "long_run_throughput_qualification": False,
        },
        "source_artifacts": {
            "standard_root": str(standard_root),
            "native_run": str(native_run),
            "standard_contract_hash": contract["contract_hash"],
        },
    }
    return report, rows


def _write_outputs(
    output_dir: Path, report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "equal_image_performance.json", report)
    fields = [
        "pipeline",
        "input_domain",
        "epoch",
        "observation",
        "epoch_samples",
        "epoch_microbatches",
        "epoch_optimizer_updates",
        "epoch_seconds",
        "images_per_second",
        "warm_throughput_relative_to_native_b6",
        "data_preparation_seconds",
        "loader_wait_seconds",
    ]
    temporary = output_dir / ".equal_image_performance.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    os.replace(temporary, output_dir / "equal_image_performance.csv")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-root", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--standard-pipelines",
        default=",".join(STANDARD_PIPELINES),
        help="comma-separated completed RGB pipelines to include",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    standard_pipelines = tuple(
        item.strip() for item in args.standard_pipelines.split(",") if item.strip()
    )
    report, rows = build_report(
        standard_root=args.standard_root,
        native_run=args.native_run,
        standard_pipelines=standard_pipelines,
    )
    _write_outputs(args.output_dir.resolve(), report, rows)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "gpu": report["hardware"],
                "warm_epoch": [row for row in rows if int(row["epoch"]) == 2],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
