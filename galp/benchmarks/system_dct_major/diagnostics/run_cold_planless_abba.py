#!/usr/bin/env python3
"""Run four independent process-cold planless legs in an A/B/B/A order.

A and B intentionally use the same immutable binding and configuration.  The
labels balance natural page-cache/order effects while measuring repeatability;
this is not a comparison between different implementations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any, Sequence

from run_cold_double_buffer_abba import (  # type: ignore
    BENCHMARK_ROOT,
    DEFAULT_PYTHON,
    REPO_ROOT,
    _record,
    _semantic_difference,
    write_json,
)


def _median(records: Sequence[dict[str, Any]], path: Sequence[str]) -> float:
    values: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        values.append(float(value))
    return statistics.median(values)


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sequence = ("A", "B", "B", "A")
    records: list[dict[str, Any]] = []
    for order, label in enumerate(sequence, start=1):
        run_dir = output_dir / f"{order:02d}_{label.lower()}"
        command = [
            str(args.python.resolve()),
            str(BENCHMARK_ROOT / "run.py"),
            "--preset",
            "smoke",
            "--workload",
            "feature-extraction",
            "--pipelines",
            "dct_major_pushdown",
            "--block-major-access-dir",
            str(args.block_major_access_dir.resolve()),
            "--dct-major-crop-execution-mode",
            "auto",
            "--dct-major-segment-size",
            str(args.sample_count),
            "--decode-workset-capacity-mib",
            str(args.decode_workset_capacity_mib),
            "--block-major-double-buffer",
            "on",
            "--batch-size",
            str(args.batch_size),
            "--warmup-batches",
            "0",
            "--sample-count",
            str(args.sample_count),
            "--repeats",
            "1",
            "--output-dir",
            str(run_dir),
        ]
        print("COMMAND " + " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"cold planless run {order} ({label}) failed: {completed.returncode}")
        records.append(_record(run_dir, label, order))

    reference = Path(records[0]["semantic_artifact"])
    semantic = [
        {
            "reference_order": 1,
            "candidate_order": record["order"],
            **_semantic_difference(reference, Path(record["semantic_artifact"])),
        }
        for record in records[1:]
    ]
    spawn_values = [
        float(record["spawn_to_first_output_seconds"])
        for record in records
        if record["spawn_to_first_output_seconds"] is not None
    ]
    result = {
        "schema_version": "galp_dct_major_cold_planless_abba_v1",
        "ok": all(item["equal"] for item in semantic),
        "primary_scope": "four independent process-cold runs; warmup=0; double-buffer=on",
        "sequence": list(sequence),
        "labels_are_same_configuration": True,
        "records": records,
        "semantic_comparisons": semantic,
        "medians": {
            "application_cold_ttft_ms": _median(records, ("cold_start", "time_to_first_batch_ms")),
            "application_cold_throughput_images_per_s": _median(
                records, ("cold_start", "throughput_images_per_s")
            ),
            "spawn_to_first_output_seconds": statistics.median(spawn_values) if spawn_values else None,
            "subprocess_full_wall_seconds": _median(records, ("subprocess_wall_seconds",)),
            "active_output_planning_ms": _median(
                records, ("native", "planless_transform_active_output_planning_ms")
            ),
            "group_workset_build_ms": _median(
                records, ("native", "planless_transform_group_workset_build_ms")
            ),
            "active_output_count_ms": _median(
                records, ("native", "planless_transform_active_output_count_ms")
            ),
            "active_output_prefix_ms": _median(
                records, ("native", "planless_transform_active_output_prefix_ms")
            ),
            "active_output_fill_ms": _median(
                records, ("native", "planless_transform_active_output_fill_ms")
            ),
            "planless_gpu_kernel_ms": _median(
                records, ("native", "planless_transform_gpu_kernel_ms")
            ),
            "producer_active_ms": _median(records, ("native", "prefetch_producer_active_ms")),
            "model_mean_ms": _median(records, ("cold_start", "model_mean_ms")),
            "loader_mean_ms": _median(records, ("cold_start", "loader_mean_ms")),
            "host_peak_rss_bytes": _median(records, ("host_peak_rss_bytes",)),
            "torch_gpu_peak_allocated_bytes": _median(
                records, ("peak_torch_gpu_allocated_bytes",)
            ),
            "torch_gpu_peak_reserved_bytes": _median(
                records, ("peak_torch_gpu_reserved_bytes",)
            ),
            "galp_native_gpu_peak_bytes": _median(
                records, ("native", "galp_native_device_peak_in_use_bytes")
            ),
            "galp_native_pinned_peak_bytes": _median(
                records, ("native", "galp_native_pinned_peak_in_use_bytes")
            ),
            "adapter_construction_ms": _median(
                records, ("cold_start_scope", "adapter_construction_ms")
            ),
            "cuda_availability_probe_ms": _median(
                records, ("cold_start_scope", "cuda_availability_probe_ms")
            ),
            "cuda_set_device_ms": _median(
                records, ("cold_start_scope", "cuda_set_device_ms")
            ),
            "binding_extension_import_ms": _median(
                records,
                ("cold_start_scope", "adapter_subphases", "binding_extension_import_ms"),
            ),
            "direct_dct_profile_import_ms": _median(
                records,
                ("cold_start_scope", "adapter_subphases", "direct_dct_profile_import_ms"),
            ),
            "postdecode_diagnostics_import_ms": _median(
                records,
                ("cold_start_scope", "adapter_subphases", "postdecode_diagnostics_import_ms"),
            ),
            "reader_python_constructor_ms": _median(
                records,
                (
                    "cold_start_scope",
                    "adapter_subphases",
                    "direct_dct_reader_python_constructor_ms",
                ),
            ),
            "reader_native_total_ms": _median(
                records,
                (
                    "cold_start_scope",
                    "adapter_subphases",
                    "native_reader_initialization",
                    "total_ms",
                ),
            ),
            "reader_metadata_load_ms": _median(
                records,
                (
                    "cold_start_scope",
                    "adapter_subphases",
                    "native_reader_initialization",
                    "shard_metadata_load_ms",
                ),
            ),
            "reader_metadata_index_ms": _median(
                records,
                (
                    "cold_start_scope",
                    "adapter_subphases",
                    "native_reader_initialization",
                    "shard_metadata_index_ms",
                ),
            ),
            "descriptor_open_ms": _median(
                records,
                (
                    "cold_start_scope",
                    "adapter_subphases",
                    "native_reader_initialization",
                    "block_major_descriptor_open_ms",
                ),
            ),
            "descriptor_validation_ms": _median(
                records,
                (
                    "cold_start_scope",
                    "adapter_subphases",
                    "native_reader_initialization",
                    "block_major_descriptor_validation_ms",
                ),
            ),
        },
        "hot_batch_loop_is_diagnostic_only": True,
    }
    write_json(output_dir / "cold_planless_abba.json", result)
    print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["ok"] else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-major-access-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--decode-workset-capacity-mib", type=int, default=512)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
