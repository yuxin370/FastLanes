#!/usr/bin/env python3
"""Run and summarize Direct-DCT/model CUDA scheduling policies on one contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


POLICIES = ("fully-overlapped", "limited-overlap", "serial")
HERE = Path(__file__).resolve().parent


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": stat.st_size,
        "file_identity": {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        },
    }


def _selected_repeats(contract: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    repeats = list(result["repeats"])
    if contract["execution"].get("aggregate_exclude_first_repeat", False) and len(repeats) > 1:
        return repeats[1:]
    return repeats


def _median(values: list[float]) -> float:
    if not values:
        raise RuntimeError("scheduler matrix found no repeat values")
    return float(statistics.median(values))


def _summarize_policy(contract: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    repeats = _selected_repeats(contract, result)
    return {
        "repeat_count": len(repeats),
        "throughput_images_per_s_median": _median(
            [float(repeat["throughput_images_per_s"]) for repeat in repeats]
        ),
        "model_forward_gpu_p50_ms_median": _median(
            [float(repeat["stage_breakdown_ms"]["model_forward_gpu"]["p50"]) for repeat in repeats]
        ),
        "model_forward_gpu_mean_ms_median": _median(
            [float(repeat["stage_breakdown_ms"]["model_forward_gpu"]["mean"]) for repeat in repeats]
        ),
        "end_to_end_p50_ms_median": _median(
            [float(repeat["end_to_end_latency_ms"]["p50"]) for repeat in repeats]
        ),
        "end_to_end_mean_ms_median": _median(
            [float(repeat["end_to_end_latency_ms"]["mean"]) for repeat in repeats]
        ),
        "accuracy_top1": _median([float(repeat["accuracy_top1"]) for repeat in repeats]),
        "accuracy_top5": _median([float(repeat["accuracy_top5"]) for repeat in repeats]),
        "planless_transform_kernel_launches": _median(
            [float(repeat["native_counters"].get("planless_transform_kernel_launches", 0)) for repeat in repeats]
        ),
        "planless_transform_max_blocks_per_launch": max(
            int(repeat["native_counters"].get("planless_transform_max_blocks_per_launch", 0))
            for repeat in repeats
        ),
        "decode_to_transform_event_handoffs": _median(
            [float(repeat["native_counters"].get("decode_to_transform_event_handoffs", 0)) for repeat in repeats]
        ),
        "copy_to_decode_event_handoffs": _median(
            [float(repeat["native_counters"].get("copy_to_decode_event_handoffs", 0)) for repeat in repeats]
        ),
    }


def _policy_contract(
    base: dict[str, Any], policy: str, transform_blocks: int, binding_dir: Path, binding: Path
) -> dict[str, Any]:
    contract = copy.deepcopy(base)
    contract["benchmark_id"] = f'{base.get("benchmark_id", "benchmark")}-scheduler-{policy}'
    galp = contract["pipelines"]["galp"]
    galp["scheduling_policy"] = policy
    galp["transform_blocks_per_launch"] = transform_blocks if policy == "limited-overlap" else 0
    galp["use_low_priority_streams"] = True
    galp["torch_binding_dir"] = str(binding_dir.resolve())
    galp["native_binary_fingerprint"] = _binding_fingerprint(binding)
    contract["execution"]["model_stream_priority"] = -1
    contract["timing"]["next_batch_prefetch_overlap"] = policy != "serial"
    return contract


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    base_contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if "galp" not in base_contract.get("pipelines", {}).get("enabled", []):
        raise ValueError("base contract must enable the galp pipeline")
    if args.transform_blocks <= 0:
        raise ValueError("--transform-blocks must be positive")
    binding_candidates = sorted(args.binding_dir.glob("_galp_direct_dct*.so"))
    if len(binding_candidates) != 1:
        raise RuntimeError(f"expected one Direct-DCT binding in {args.binding_dir}, got {binding_candidates}")

    policy_summaries: dict[str, dict[str, Any]] = {}
    result_paths: dict[str, str] = {}
    semantic_paths: dict[str, Path] = {}
    result_payloads: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        policy_dir = args.output_dir / policy
        contract = _policy_contract(
            base_contract, policy, args.transform_blocks, args.binding_dir, binding_candidates[0]
        )
        contract_path = policy_dir / "contract.json"
        output_path = policy_dir / "pipeline_galp.json"
        _write_json(contract_path, contract)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(args.binding_dir.resolve()), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        subprocess.run(
            [
                str(args.python),
                str(HERE / "pipeline.py"),
                "--pipeline",
                "galp",
                "--contract",
                str(contract_path),
                "--output",
                str(output_path),
            ],
            check=True,
            env=env,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        result_payloads[policy] = result
        policy_summaries[policy] = _summarize_policy(contract, result)
        result_paths[policy] = str(output_path.resolve())
        semantic_paths[policy] = policy_dir / "semantic_galp.npz"

    reference_traces = [repeat["sample_trace"]["sha256"] for repeat in result_payloads["serial"]["repeats"]]
    for policy, payload in result_payloads.items():
        traces = [repeat["sample_trace"]["sha256"] for repeat in payload["repeats"]]
        if traces != reference_traces:
            raise RuntimeError(f"sample trace mismatch for {policy}")

    semantic_keys = (
        "ordinals",
        "labels",
        "logits",
        "input_0",
        "input_1",
        "prediction_ordinals",
        "prediction_labels",
        "top1_predictions",
        "top5_predictions",
    )
    with np.load(semantic_paths["serial"]) as reference:
        for policy in ("fully-overlapped", "limited-overlap"):
            with np.load(semantic_paths[policy]) as candidate:
                for key in semantic_keys:
                    if not np.array_equal(reference[key], candidate[key]):
                        raise RuntimeError(f"semantic artifact mismatch for {policy}: {key}")

    model_only_p50 = policy_summaries["serial"]["model_forward_gpu_p50_ms_median"]
    model_only_mean = policy_summaries["serial"]["model_forward_gpu_mean_ms_median"]
    for summary in policy_summaries.values():
        summary["model_extra_p50_ms_vs_serial"] = (
            summary["model_forward_gpu_p50_ms_median"] - model_only_p50
        )
        summary["model_extra_mean_ms_vs_serial"] = (
            summary["model_forward_gpu_mean_ms_median"] - model_only_mean
        )

    batch_size = int(base_contract["execution"]["batch_size"])
    transform = base_contract["preprocess"]["dct"].get("grid_transform", {})
    # The canonical RGB-no-more pushdown shape is fixed at Y=28x28 and
    # Cb/Cr=14x14 blocks if the contract does not repeat the native transform.
    y_width = int(transform.get("y_output_width_blocks", 28))
    y_height = int(transform.get("y_output_height_blocks", 28))
    c_width = int(transform.get("cbcr_output_width_blocks", 14))
    c_height = int(transform.get("cbcr_output_height_blocks", 14))
    blocks_per_image = y_width * y_height + 2 * c_width * c_height
    output_blocks = batch_size * blocks_per_image
    limited_launches = math.ceil(output_blocks / args.transform_blocks)
    device = json.loads(Path(result_paths["limited-overlap"]).read_text(encoding="utf-8"))["device_metadata"]
    sm_count = int(device.get("multi_processor_count", 0))

    result = {
        "schema_version": "galp_scheduler_matrix_v1",
        "base_contract": str(args.contract.resolve()),
        "binding": str(binding_candidates[0].resolve()),
        "binding_sha256": _sha256(binding_candidates[0]),
        "core_metric": "T_model_with_transform - T_model_only",
        "model_only_reference": "serial policy under the same contract",
        "semantic_bit_exact_across_policies": True,
        "policies": policy_summaries,
        "policy_results": result_paths,
        "theoretical_work": {
            "batch_size": batch_size,
            "transform_output_blocks_per_image": blocks_per_image,
            "transform_output_blocks_per_batch": output_blocks,
            "limited_blocks_per_launch": args.transform_blocks,
            "limited_launches_per_batch_theoretical": limited_launches,
            "device_sm_count": sm_count,
            "limited_max_resident_block_fraction_upper_bound": (
                min(1.0, args.transform_blocks / sm_count) if sm_count > 0 else None
            ),
            "interpretation": (
                "Each transform CUDA block produces one 8x8 DCT output block with 64 threads. "
                "The chunk limit bounds simultaneously eligible low-priority transform blocks; "
                "launch count grows as ceil(output_blocks/chunk), trading launch overhead for model isolation."
            ),
        },
    }
    _write_json(args.output_dir / "scheduler_matrix.json", result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binding-dir", type=Path, default=Path("build/galp/torch"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--transform-blocks", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run_matrix(_parse_args()), indent=2, sort_keys=True))
