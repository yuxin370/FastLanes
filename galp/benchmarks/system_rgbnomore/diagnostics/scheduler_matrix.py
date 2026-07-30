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


HERE = Path(__file__).resolve().parent
INFERENCE_DIR = HERE.parent / "inference"
PLANLESS_RESOURCE_COUNTERS = (
    "planless_transform_registers_per_thread",
    "planless_transform_static_shared_bytes_per_cta",
    "planless_transform_local_bytes_per_thread",
    "planless_transform_threads_per_cta",
    "planless_transform_max_active_ctas_per_sm",
    "cuda_max_threads_per_sm",
    "cuda_warp_size",
)


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


def _invariant_counter(repeats: list[dict[str, Any]], key: str) -> int:
    values = {int(repeat["native_counters"][key]) for repeat in repeats}
    if len(values) != 1:
        raise RuntimeError(f"native counter is not invariant across repeats: {key}={sorted(values)}")
    return values.pop()


def _normalize_transform_blocks(values: int | list[int] | None) -> list[int]:
    if values is None:
        values = [64]
    elif isinstance(values, int):
        values = [values]
    normalized = sorted(set(int(value) for value in values))
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("--transform-blocks must contain positive integers")
    return normalized


def _normalize_limited_candidates(
    transform_blocks: int | list[int] | None,
    transform_ctas: int | list[int] | None,
) -> list[tuple[int, int]]:
    if transform_blocks is None:
        blocks = [64]
    elif isinstance(transform_blocks, int):
        blocks = [transform_blocks]
    else:
        blocks = [int(value) for value in transform_blocks]
    if not blocks or any(value <= 0 for value in blocks):
        raise ValueError("--transform-blocks must contain positive integers")
    if transform_ctas is None:
        ctas = [64]
    elif isinstance(transform_ctas, int):
        ctas = [transform_ctas]
    else:
        ctas = [int(value) for value in transform_ctas]
    if not ctas or any(value <= 0 for value in ctas):
        raise ValueError("--transform-ctas must contain positive integers")
    if len(ctas) == 1:
        ctas *= len(blocks)
    if len(ctas) != len(blocks):
        raise ValueError("--transform-blocks and --transform-ctas must have equal counts")
    return sorted(set(zip(blocks, ctas)))


def _policy_specs(limited_candidates: list[tuple[int, int]]) -> list[tuple[str, str, int, int]]:
    single_limited = len(limited_candidates) == 1
    specs = [("fully-overlapped", "fully-overlapped", 0, 0)]
    specs.extend(
        (
            "limited-overlap" if single_limited else f"limited-overlap-o{blocks}-c{ctas}",
            "limited-overlap",
            blocks,
            ctas,
        )
        for blocks, ctas in limited_candidates
    )
    specs.append(("serial", "serial", 0, 0))
    return specs


def _residency_bounds(
    *,
    submitted_ctas: int,
    sm_count: int,
    max_active_ctas_per_sm: int,
    threads_per_cta: int,
    max_threads_per_sm: int,
) -> dict[str, float | int]:
    values = (
        submitted_ctas,
        sm_count,
        max_active_ctas_per_sm,
        threads_per_cta,
        max_threads_per_sm,
    )
    if any(value <= 0 for value in values):
        raise ValueError("kernel residency inputs must be positive")
    device_cta_capacity = sm_count * max_active_ctas_per_sm
    resident_ctas = min(submitted_ctas, device_cta_capacity)
    resident_threads = resident_ctas * threads_per_cta
    return {
        "submitted_ctas_per_launch": submitted_ctas,
        "device_resident_cta_capacity": device_cta_capacity,
        "max_resident_ctas_per_launch": resident_ctas,
        "average_resident_ctas_per_sm_upper_bound": resident_ctas / sm_count,
        "sm_coverage_fraction_upper_bound": min(1.0, resident_ctas / sm_count),
        "resident_cta_capacity_fraction_upper_bound": resident_ctas / device_cta_capacity,
        "thread_occupancy_fraction_upper_bound": resident_threads
        / (sm_count * max_threads_per_sm),
    }


def _pareto_frontier(
    policy_summaries: dict[str, dict[str, Any]], policy_labels: list[str]
) -> list[str]:
    frontier: list[str] = []
    for candidate_label in policy_labels:
        candidate = policy_summaries[candidate_label]
        candidate_throughput = float(candidate["throughput_images_per_s_median"])
        candidate_extra = float(candidate["model_extra_p50_ms_vs_serial"])
        dominated = False
        for other_label in policy_labels:
            if other_label == candidate_label:
                continue
            other = policy_summaries[other_label]
            other_throughput = float(other["throughput_images_per_s_median"])
            other_extra = float(other["model_extra_p50_ms_vs_serial"])
            if (
                other_throughput >= candidate_throughput
                and other_extra <= candidate_extra
                and (other_throughput > candidate_throughput or other_extra < candidate_extra)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate_label)
    return frontier


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
        "loader_submit_p50_ms_median": _median(
            [
                float(repeat["stage_breakdown_ms"]["loader_and_preprocess_submit"]["p50"])
                for repeat in repeats
            ]
        ),
        "fixed_transform_gpu_p50_ms_median": _median(
            [
                float(
                    repeat["stage_breakdown_ms"]["native_per_batch_ms"]
                    ["fixed_transform_kernel_seconds"]["p50"]
                )
                for repeat in repeats
            ]
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
        "planless_transform_max_output_blocks_per_launch": max(
            int(
                repeat["native_counters"].get(
                    "planless_transform_max_output_blocks_per_launch", 0
                )
            )
            for repeat in repeats
        ),
        "decode_to_transform_event_handoffs": _median(
            [float(repeat["native_counters"].get("decode_to_transform_event_handoffs", 0)) for repeat in repeats]
        ),
        "copy_to_decode_event_handoffs": _median(
            [float(repeat["native_counters"].get("copy_to_decode_event_handoffs", 0)) for repeat in repeats]
        ),
        "direct_dct_low_priority_batches": _median(
            [
                float(repeat["native_counters"].get("direct_dct_low_priority_batches", 0))
                for repeat in repeats
            ]
        ),
        "direct_dct_stream_priority": _invariant_counter(repeats, "direct_dct_stream_priority"),
        "direct_dct_h2d_stream_priority": _invariant_counter(repeats, "direct_dct_h2d_stream_priority"),
        "direct_dct_decode_stream_priority": _invariant_counter(repeats, "direct_dct_decode_stream_priority"),
        "direct_dct_transform_stream_priority": _invariant_counter(
            repeats, "direct_dct_transform_stream_priority"
        ),
        "direct_dct_round_stream_priority": _invariant_counter(repeats, "direct_dct_round_stream_priority"),
        "cuda_least_stream_priority": _invariant_counter(repeats, "cuda_least_stream_priority"),
        "cuda_greatest_stream_priority": _invariant_counter(repeats, "cuda_greatest_stream_priority"),
        **{
            key: _invariant_counter(repeats, key)
            for key in PLANLESS_RESOURCE_COUNTERS
        },
    }


def _policy_contract(
    base: dict[str, Any],
    policy_label: str,
    policy: str,
    transform_blocks: int,
    transform_ctas: int,
    binding_dir: Path,
    binding: Path,
) -> dict[str, Any]:
    contract = copy.deepcopy(base)
    contract["benchmark_id"] = f'{base.get("benchmark_id", "benchmark")}-scheduler-{policy_label}'
    galp = contract["pipelines"]["galp"]
    galp["scheduling_policy"] = policy
    galp["transform_blocks_per_launch"] = transform_blocks if policy == "limited-overlap" else 0
    galp["transform_ctas_per_launch"] = transform_ctas if policy == "limited-overlap" else 0
    galp["use_low_priority_streams"] = True
    galp["torch_binding_dir"] = str(binding_dir.resolve())
    galp["native_binary_fingerprint"] = _binding_fingerprint(binding)
    contract["execution"]["model_stream_priority"] = "greatest"
    contract["timing"]["next_batch_prefetch_overlap"] = policy != "serial"
    return contract


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    base_contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if "galp" not in base_contract.get("pipelines", {}).get("enabled", []):
        raise ValueError("base contract must enable the galp pipeline")
    limited_candidates = _normalize_limited_candidates(args.transform_blocks, args.transform_ctas)
    policy_specs = _policy_specs(limited_candidates)
    limited_labels = [label for label, policy, _, _ in policy_specs if policy == "limited-overlap"]
    binding_candidates = sorted(args.binding_dir.glob("_galp_direct_dct*.so"))
    if len(binding_candidates) != 1:
        raise RuntimeError(f"expected one Direct-DCT binding in {args.binding_dir}, got {binding_candidates}")

    policy_summaries: dict[str, dict[str, Any]] = {}
    result_paths: dict[str, str] = {}
    semantic_paths: dict[str, Path] = {}
    result_payloads: dict[str, dict[str, Any]] = {}
    for policy_label, policy, transform_blocks, transform_ctas in policy_specs:
        policy_dir = args.output_dir / policy_label
        contract = _policy_contract(
            base_contract,
            policy_label,
            policy,
            transform_blocks,
            transform_ctas,
            args.binding_dir,
            binding_candidates[0],
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
                str(INFERENCE_DIR / "pipeline.py"),
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
        result_payloads[policy_label] = result
        policy_summaries[policy_label] = _summarize_policy(contract, result)
        policy_summaries[policy_label]["scheduling_policy"] = policy
        policy_summaries[policy_label]["transform_blocks_per_launch"] = transform_blocks
        policy_summaries[policy_label]["transform_ctas_per_launch"] = transform_ctas
        result_paths[policy_label] = str(output_path.resolve())
        semantic_paths[policy_label] = policy_dir / "semantic_galp.npz"

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
        for policy_label in (label for label in semantic_paths if label != "serial"):
            with np.load(semantic_paths[policy_label]) as candidate:
                for key in semantic_keys:
                    if not np.array_equal(reference[key], candidate[key]):
                        raise RuntimeError(f"semantic artifact mismatch for {policy_label}: {key}")

    for policy, payload in result_payloads.items():
        summary = policy_summaries[policy]
        cuda_scheduling = payload["cuda_scheduling"]
        model_priority = int(cuda_scheduling["model_stream_priority_actual"])
        torch_least_priority = int(cuda_scheduling["torch_least_stream_priority"])
        torch_greatest_priority = int(cuda_scheduling["torch_greatest_stream_priority"])
        least_priority = int(summary["cuda_least_stream_priority"])
        greatest_priority = int(summary["cuda_greatest_stream_priority"])
        direct_dct_priorities = {
            stage: int(summary[f"direct_dct_{stage}_stream_priority"])
            for stage in ("h2d", "decode", "transform", "round")
        }
        if greatest_priority >= least_priority:
            raise RuntimeError(
                f"device does not expose distinct CUDA stream priorities: "
                f"greatest={greatest_priority} least={least_priority}"
            )
        if torch_greatest_priority >= torch_least_priority:
            raise RuntimeError(
                f"PyTorch does not expose distinct CUDA stream priorities: "
                f"greatest={torch_greatest_priority} least={torch_least_priority}"
            )
        if model_priority != torch_greatest_priority:
            raise RuntimeError(
                f"model stream is not at PyTorch greatest priority for {policy}: "
                f"actual={model_priority} greatest={torch_greatest_priority}"
            )
        for stage, actual_priority in direct_dct_priorities.items():
            if actual_priority != least_priority:
                raise RuntimeError(
                    f"Direct-DCT {stage} stream is not at least priority for {policy}: "
                    f"actual={actual_priority} least={least_priority}"
                )
            if model_priority >= actual_priority:
                raise RuntimeError(
                    f"model stream does not outrank Direct-DCT {stage} for {policy}: "
                    f"model={model_priority} Direct-DCT={actual_priority}"
                )
        summary["model_stream_priority_actual"] = model_priority
        summary["torch_least_stream_priority"] = torch_least_priority
        summary["torch_greatest_stream_priority"] = torch_greatest_priority
        summary["native_cuda_greatest_priority_unavailable_to_torch"] = (
            torch_greatest_priority != greatest_priority
        )
        summary["priority_isolation_verified"] = True

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
    limited_launches = {
        blocks: math.ceil(output_blocks / blocks) for blocks, _ in limited_candidates
    }
    device = result_payloads[limited_labels[0]]["device_metadata"]
    sm_count = int(device.get("multi_processor_count", 0))
    if sm_count <= 0:
        raise RuntimeError(f"invalid CUDA multiprocessor count: {sm_count}")
    resource_reference = {
        key: int(policy_summaries["fully-overlapped"][key])
        for key in PLANLESS_RESOURCE_COUNTERS
    }
    required_positive_resources = (
        "planless_transform_registers_per_thread",
        "planless_transform_threads_per_cta",
        "planless_transform_max_active_ctas_per_sm",
        "cuda_max_threads_per_sm",
        "cuda_warp_size",
    )
    if any(resource_reference[key] <= 0 for key in required_positive_resources):
        raise RuntimeError(f"invalid planless transform resource counters: {resource_reference}")
    for policy_label, summary in policy_summaries.items():
        actual_resources = {
            key: int(summary[key]) for key in PLANLESS_RESOURCE_COUNTERS
        }
        if actual_resources != resource_reference:
            raise RuntimeError(
                f"planless transform resources changed for {policy_label}: "
                f"actual={actual_resources} expected={resource_reference}"
            )
    measurement_batches = int(base_contract["execution"]["measurement_batches"])
    for policy_label, summary in policy_summaries.items():
        policy = str(summary["scheduling_policy"])
        blocks = int(summary["transform_blocks_per_launch"])
        ctas = int(summary["transform_ctas_per_launch"])
        expected_launches_per_batch = limited_launches[blocks] if policy == "limited-overlap" else 1
        expected_max_blocks = min(ctas, blocks) if policy == "limited-overlap" else output_blocks
        expected_max_output_blocks = blocks if policy == "limited-overlap" else output_blocks
        expected_total_launches = measurement_batches * expected_launches_per_batch
        if int(summary["planless_transform_kernel_launches"]) != expected_total_launches:
            raise RuntimeError(
                f"unexpected transform launch count for {policy_label}: "
                f"actual={summary['planless_transform_kernel_launches']} "
                f"expected={expected_total_launches}"
            )
        if int(summary["planless_transform_max_blocks_per_launch"]) != expected_max_blocks:
            raise RuntimeError(
                f"unexpected max transform grid for {policy_label}: "
                f"actual={summary['planless_transform_max_blocks_per_launch']} "
                f"expected={expected_max_blocks}"
            )
        if (
            int(summary["planless_transform_max_output_blocks_per_launch"])
            != expected_max_output_blocks
        ):
            raise RuntimeError(
                f"unexpected max transform output work for {policy_label}: "
                f"actual={summary['planless_transform_max_output_blocks_per_launch']} "
                f"expected={expected_max_output_blocks}"
            )
        for handoff in ("copy_to_decode_event_handoffs", "decode_to_transform_event_handoffs"):
            if int(summary[handoff]) != measurement_batches:
                raise RuntimeError(
                    f"unexpected {handoff} for {policy_label}: "
                    f"actual={summary[handoff]} expected={measurement_batches}"
                )
        if int(summary["direct_dct_low_priority_batches"]) != measurement_batches:
            raise RuntimeError(
                f"low-priority Direct-DCT was not active for every batch in {policy_label}"
            )

    fully = policy_summaries["fully-overlapped"]
    serial = policy_summaries["serial"]
    dali_throughput: float | None = None
    dali_result_path = args.contract.parent / "pipeline_dali.json"
    if dali_result_path.exists():
        dali_payload = json.loads(dali_result_path.read_text(encoding="utf-8"))
        dali_repeats = _selected_repeats(base_contract, dali_payload)
        dali_throughput = _median(
            [float(repeat["throughput_images_per_s"]) for repeat in dali_repeats]
        )
    model_only_ceiling = batch_size * 1000.0 / serial["model_forward_gpu_p50_ms_median"]
    limited_comparisons: dict[str, dict[str, Any]] = {}
    for policy_label in limited_labels:
        limited = policy_summaries[policy_label]
        throughput_ratio = (
            limited["throughput_images_per_s_median"]
            / fully["throughput_images_per_s_median"]
        )
        limited_comparisons[policy_label] = {
            "transform_blocks_per_launch": limited["transform_blocks_per_launch"],
            "transform_ctas_per_launch": limited["transform_ctas_per_launch"],
            "throughput_ratio_to_fully_overlapped": throughput_ratio,
            "model_extra_reduction_ms_vs_fully_overlapped": (
                fully["model_extra_p50_ms_vs_serial"]
                - limited["model_extra_p50_ms_vs_serial"]
            ),
            "throughput_ratio_to_dali": (
                limited["throughput_images_per_s_median"] / dali_throughput
                if dali_throughput is not None
                else None
            ),
            "preserves_at_least_98pct_fully_overlapped_throughput": (
                throughput_ratio >= 0.98
            ),
            "reduces_model_extra_p50_vs_fully_overlapped": (
                limited["model_extra_p50_ms_vs_serial"]
                <= fully["model_extra_p50_ms_vs_serial"]
            ),
        }
    eligible_limited = [
        policy_label
        for policy_label, comparison in limited_comparisons.items()
        if comparison["preserves_at_least_98pct_fully_overlapped_throughput"]
        and comparison["reduces_model_extra_p50_vs_fully_overlapped"]
    ]
    recommended_limited_policy = (
        min(
            eligible_limited,
            key=lambda policy_label: policy_summaries[policy_label][
                "model_extra_p50_ms_vs_serial"
            ],
        )
        if eligible_limited
        else None
    )
    pareto_frontier_limited_policies = _pareto_frontier(policy_summaries, limited_labels)

    result = {
        "schema_version": "galp_scheduler_matrix_v3",
        "base_contract": str(args.contract.resolve()),
        "binding": str(binding_candidates[0].resolve()),
        "binding_sha256": _sha256(binding_candidates[0]),
        "core_metric": "T_model_with_transform - T_model_only",
        "model_only_reference": "serial policy under the same contract",
        "semantic_bit_exact_across_policies": True,
        "priority_isolation_verified_across_policies": True,
        "structural_scheduler_counters_verified_across_policies": True,
        "kernel_resource_limits_verified_across_policies": True,
        "comparison": {
            "model_only_ceiling_images_per_s_from_serial_p50": model_only_ceiling,
            "dali_reference_throughput_images_per_s": dali_throughput,
            "limited_candidates": limited_comparisons,
            "pareto_frontier_limited_policies": pareto_frontier_limited_policies,
            "recommended_limited_policy": recommended_limited_policy,
        },
        "gates": {
            "at_least_one_limited_candidate_preserves_throughput_and_reduces_model_extra": (
                recommended_limited_policy is not None
            ),
        },
        "policies": policy_summaries,
        "policy_results": result_paths,
        "theoretical_work": {
            "batch_size": batch_size,
            "transform_output_blocks_per_image": blocks_per_image,
            "transform_output_blocks_per_batch": output_blocks,
            "device_sm_count": sm_count,
            "kernel_resources": resource_reference,
            "limited_candidates": {
                f"o{blocks}-c{ctas}": {
                    "configured_output_blocks_per_launch": blocks,
                    "ctas_per_launch": min(ctas, blocks),
                    "output_blocks_per_cta_upper_bound": math.ceil(blocks / min(ctas, blocks)),
                    "launches_per_batch_theoretical": limited_launches[blocks],
                    **_residency_bounds(
                        submitted_ctas=min(ctas, blocks),
                        sm_count=sm_count,
                        max_active_ctas_per_sm=resource_reference[
                            "planless_transform_max_active_ctas_per_sm"
                        ],
                        threads_per_cta=resource_reference[
                            "planless_transform_threads_per_cta"
                        ],
                        max_threads_per_sm=resource_reference["cuda_max_threads_per_sm"],
                    ),
                }
                for blocks, ctas in limited_candidates
            },
            "interpretation": (
                "Each 64-thread transform CTA processes one or more 8x8 DCT output blocks by grid stride. "
                "The output limit determines launch count as ceil(output_blocks/output_limit), while the "
                "independent CTA cap bounds submitted low-priority work. CUDA Runtime kernel attributes and "
                "occupancy APIs bound actual resident CTAs and thread occupancy; together these values trade "
                "transform progress and launch overhead against model isolation."
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
    parser.add_argument(
        "--transform-blocks",
        type=int,
        action="append",
        help="Transform blocks per limited-overlap launch; repeat to sweep multiple limits (default: 64).",
    )
    parser.add_argument(
        "--transform-ctas",
        type=int,
        action="append",
        help="CTA cap paired with each --transform-blocks value; one value broadcasts (default: 64).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run_matrix(_parse_args()), indent=2, sort_keys=True))
