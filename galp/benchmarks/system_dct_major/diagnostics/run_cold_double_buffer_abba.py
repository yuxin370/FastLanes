#!/usr/bin/env python3
"""Run process-cold ON/OFF/OFF/ON block-major double-buffer validation."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
REPO_ROOT = BENCHMARK_ROOT.parents[2]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from common import write_json  # noqa: E402


DEFAULT_PYTHON = Path("/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python")


def _semantic_difference(reference: Path, candidate: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"equal": True, "arrays": {}}
    with np.load(reference, allow_pickle=False) as lhs, np.load(candidate, allow_pickle=False) as rhs:
        if set(lhs.files) != set(rhs.files):
            return {"equal": False, "reason": "array-name mismatch"}
        for name in lhs.files:
            if name == "metadata_json":
                continue
            left = lhs[name]
            right = rhs[name]
            if left.shape != right.shape or left.dtype != right.dtype:
                result["equal"] = False
                result["arrays"][name] = {
                    "equal": False,
                    "left_shape": list(left.shape),
                    "right_shape": list(right.shape),
                    "left_dtype": str(left.dtype),
                    "right_dtype": str(right.dtype),
                }
                continue
            if np.issubdtype(left.dtype, np.number):
                difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
                maximum = float(difference.max()) if difference.size else 0.0
            else:
                maximum = 0.0
            cosine = None
            if name.startswith("input_") and np.issubdtype(left.dtype, np.floating):
                equal = maximum <= 1.0e-3
            elif name == "output" and np.issubdtype(left.dtype, np.floating):
                flat_left = left.astype(np.float64).reshape(-1)
                flat_right = right.astype(np.float64).reshape(-1)
                denominator = float(np.linalg.norm(flat_left) * np.linalg.norm(flat_right))
                cosine = float(np.dot(flat_left, flat_right) / denominator) if denominator else 1.0
                equal = maximum <= 2.1e-2 and cosine >= 0.99999
            else:
                equal = bool(np.array_equal(left, right))
            result["equal"] = result["equal"] and equal
            result["arrays"][name] = {"equal": equal, "max_abs_difference": maximum}
            if cosine is not None:
                result["arrays"][name]["cosine_similarity"] = cosine
    return result


def _record(run_dir: Path, policy: str, order: int) -> dict[str, Any]:
    summary = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    aggregate = next(item for item in summary["aggregates"] if item["pipeline"] == "dct_major_pushdown")
    pipeline = json.loads((run_dir / "pipeline_dct_major_pushdown.json").read_text(encoding="utf-8"))
    cold_repeat = pipeline["repeats"][0]
    commands = json.loads((run_dir / "commands.json").read_text(encoding="utf-8"))
    pipeline_command = next(item for item in commands if item["name"] == "dct_major_pushdown")
    native = cold_repeat.get("native_totals", {})
    return {
        "order": order,
        "policy": policy,
        "run_dir": str(run_dir.resolve()),
        "subprocess_wall_seconds": float(pipeline_command.get("subprocess_wall_seconds", 0.0)),
        "spawn_to_first_output_seconds": pipeline_command.get("spawn_to_first_output_seconds"),
        "cold_start": aggregate["cold_start"],
        "cold_start_scope": pipeline["cold_start_scope"],
        "batch_loop_throughput_images_per_s": float(cold_repeat["throughput_images_per_s"]),
        "batch_loop_time_to_first_batch_ms": float(cold_repeat["time_to_first_batch_ms"]),
        "host_peak_rss_bytes": int(cold_repeat.get("host_peak_rss_bytes", 0)),
        "peak_torch_gpu_allocated_bytes": int(cold_repeat.get("peak_torch_gpu_allocated_bytes", 0)),
        "peak_torch_gpu_reserved_bytes": int(cold_repeat.get("peak_torch_gpu_reserved_bytes", 0)),
        "native_segments": cold_repeat.get("native_segments", []),
        "native": {
            key: native.get(key)
            for key in (
                "planning_ms",
                "prefetch_producer_active_ms",
                "prefetch_ordered_submission_ms",
                "sync_rowgroup_read_ms",
                "workset_count",
                "workset_build_ms",
                "workset_upload_ms",
                "decode_ms",
                "fixed_transform_ms",
                "planless_transform_active_output_planning_ms",
                "planless_transform_group_workset_build_ms",
                "planless_transform_active_output_count_ms",
                "planless_transform_active_output_prefix_ms",
                "planless_transform_active_output_fill_ms",
                "planless_transform_gpu_kernel_ms",
                "planless_transform_full_scan_output_block_count",
                "planless_transform_output_block_count",
                "planless_transform_skipped_output_block_count",
                "planless_transform_active_output_index_bytes",
                "planless_transform_active_output_offset_bytes",
                "planless_transform_active_output_schedule_peak_bytes",
                "planless_transform_source_contribution_count",
                "planless_transform_source_contribution_visit_count",
                "planless_transform_output_workset_ownership_count",
                "planless_transform_active_output_workset_count",
                "planless_transform_active_output_schedule_build_count",
                "planless_transform_active_output_offsets_valid",
                "coordinate_group_lookup_count",
                "coordinate_group_index_entries",
                "coordinate_group_index_populated",
                "coordinate_group_index_holes",
                "coordinate_group_index_bytes",
                "coordinate_group_index_density",
                "bounded_double_buffer_policy",
                "bounded_double_buffer_candidate",
                "bounded_double_buffer_enabled",
                "bounded_double_buffer_workset_count",
                "bounded_double_buffer_peak_estimated_bytes",
                "automatic_sparse_storage_candidate_rowgroup_count",
                "automatic_sparse_storage_early_rejected_rowgroup_count",
                "galp_native_device_peak_in_use_bytes",
                "galp_native_pinned_peak_in_use_bytes",
            )
        },
        "semantic_artifact": pipeline["semantic_artifact"],
    }


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sequence = ("on", "off", "off", "on")
    records: list[dict[str, Any]] = []
    for order, policy in enumerate(sequence):
        run_dir = output_dir / f"{order + 1:02d}_{policy}"
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
            policy,
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
            raise RuntimeError(f"cold double-buffer run {order + 1} ({policy}) failed: {completed.returncode}")
        records.append(_record(run_dir, policy, order + 1))

    reference = Path(records[0]["semantic_artifact"])
    semantic = [
        {
            "reference_order": 1,
            "candidate_order": record["order"],
            **_semantic_difference(reference, Path(record["semantic_artifact"])),
        }
        for record in records[1:]
    ]
    policies: dict[str, Any] = {}
    for policy in ("on", "off"):
        selected = [item for item in records if item["policy"] == policy]
        spawn_to_first = [
            float(item["spawn_to_first_output_seconds"])
            for item in selected
            if item["spawn_to_first_output_seconds"] is not None
        ]
        policies[policy] = {
            "cold_throughput_median_images_per_s": statistics.median(
                float(item["cold_start"]["throughput_images_per_s"]) for item in selected
            ),
            "cold_ttft_median_ms": statistics.median(
                float(item["cold_start"]["time_to_first_batch_ms"]) for item in selected
            ),
            "subprocess_wall_median_seconds": statistics.median(
                float(item["subprocess_wall_seconds"]) for item in selected
            ),
            "spawn_to_first_output_median_seconds": (
                statistics.median(spawn_to_first) if spawn_to_first else None
            ),
        }
    on_throughput = policies["on"]["cold_throughput_median_images_per_s"]
    off_throughput = policies["off"]["cold_throughput_median_images_per_s"]
    result = {
        "schema_version": "galp_dct_major_cold_double_buffer_abba_v1",
        "ok": all(item["equal"] for item in semantic),
        "primary_scope": "independent-process cold start; no internal warm repeat",
        "sequence": list(sequence),
        "records": records,
        "semantic_comparisons": semantic,
        "policies": policies,
        "on_over_off_cold_throughput": on_throughput / off_throughput if off_throughput else 0.0,
        "recommended_policy": "on" if on_throughput >= off_throughput else "off",
    }
    write_json(output_dir / "cold_double_buffer_abba.json", result)
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
