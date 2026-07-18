#!/usr/bin/env python3
"""CPU-only planning audit for the compact Direct-DCT execution path."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TORCH_BINDING_DIR = REPO_ROOT / "build/galp/torch"
if DEFAULT_TORCH_BINDING_DIR.is_dir() and str(DEFAULT_TORCH_BINDING_DIR) not in sys.path:
    sys.path.append(str(DEFAULT_TORCH_BINDING_DIR))

TORCH_SOURCE_DIR = REPO_ROOT / "galp/torch"
if str(TORCH_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_SOURCE_DIR))

import _galp_direct_dct as galp_dct
from rgbnomore_dct_profile import RGBNOMORE_VAL_DCT_GRID_TRANSFORM


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[rank])


def _trace_ids(image_count: int, trace: str, seed: int) -> list[int]:
    image_ids = list(range(image_count))
    if trace == "shuffled":
        random.Random(seed).shuffle(image_ids)
    return image_ids


def _audit_trace(
    reader: Any,
    *,
    repeat: int,
    image_count: int,
    batch_size: int,
    trace: str,
    seed: int,
) -> dict[str, Any]:
    image_ids = _trace_ids(image_count, trace, seed)
    planning_ms: list[float] = []
    wall_ms: list[float] = []
    descriptor_count = 0
    rowgroup_count = 0
    source_block_count = 0
    output_block_count = 0
    started = time.perf_counter()
    for offset in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[offset : offset + batch_size]
        wall_started = time.perf_counter()
        preview = reader.plan_batch(
            batch_ids,
            crop=None,
            dct_coeffs="all",
            cache_capacity_mib=0,
            layout="transformed_dct_grid",
            grid_transform=RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
            # Exercise the production invariant: compact planning must ignore
            # a caller's historical nonzero exact-batch cache setting.
            plan_cache_capacity=128,
        )
        wall_ms.append((time.perf_counter() - wall_started) * 1000.0)
        planning_ms.append(float(preview["planning_ms"]))
        expected_descriptors = len(batch_ids)
        structural_failures = {
            "uses_planless_fixed_transform": bool(preview.get("uses_planless_fixed_transform", False)),
            "compact_image_descriptor_count": int(preview.get("compact_image_descriptor_count", -1))
            == expected_descriptors,
            "host_expanded_transform_items_created": int(
                preview.get("host_expanded_transform_items_created", -1)
            )
            == 0,
            "host_output_block_source_lists_created": int(
                preview.get("host_output_block_source_lists_created", -1)
            )
            == 0,
            "host_global_transform_sort_items": int(preview.get("host_global_transform_sort_items", -1)) == 0,
            "exact_batch_plan_cache_disabled": not bool(
                preview.get("exact_batch_plan_cache_enabled", True)
            ),
            "block_metadata_empty": len(preview.get("block_metadata", [])) == 0,
        }
        failed = [name for name, passed in structural_failures.items() if not passed]
        if failed:
            raise RuntimeError(
                f"planless planning audit failed for trace={trace} offset={offset}: {', '.join(failed)}"
            )
        descriptor_count += int(preview["compact_image_descriptor_count"])
        rowgroup_count += int(preview["rowgroup_count"])
        source_block_count += int(preview["fixed_transform_source_block_count"])
        output_block_count += int(preview["fixed_transform_output_block_count"])
    elapsed_seconds = time.perf_counter() - started
    return {
        "repeat": repeat,
        "trace": trace,
        "seed": seed if trace == "shuffled" else None,
        "images": len(image_ids),
        "batches": len(planning_ms),
        "batch_size": batch_size,
        "planning_ms": {
            "median": statistics.median(planning_ms) if planning_ms else 0.0,
            "p95": _percentile(planning_ms, 0.95),
            "min": min(planning_ms, default=0.0),
            "max": max(planning_ms, default=0.0),
        },
        "call_wall_ms": {
            "median": statistics.median(wall_ms) if wall_ms else 0.0,
            "p95": _percentile(wall_ms, 0.95),
            "min": min(wall_ms, default=0.0),
            "max": max(wall_ms, default=0.0),
        },
        "elapsed_seconds": elapsed_seconds,
        "compact_image_descriptors": descriptor_count,
        "rowgroups": rowgroup_count,
        "source_blocks_described_by_formula": source_block_count,
        "output_blocks_described_by_formula": output_block_count,
        "host_expanded_transform_items_created": 0,
        "host_output_block_source_lists_created": 0,
        "host_global_transform_sort_items": 0,
        "exact_batch_plan_cache_enabled_batches": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dataset-sizes", type=int, nargs="+", default=[1000, 50000])
    parser.add_argument("--traces", choices=("sequential", "shuffled"), nargs="+", default=["sequential", "shuffled"])
    parser.add_argument("--shuffle-seed", type=int, default=20260718)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    reader = galp_dct.DirectDctReader(str(args.manifest))
    results: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for requested_size in args.dataset_sizes:
            image_count = min(int(reader.image_count), requested_size)
            if image_count <= 0:
                raise RuntimeError("manifest has no images")
            for trace in args.traces:
                results.append(
                    _audit_trace(
                        reader,
                        repeat=repeat,
                        image_count=image_count,
                        batch_size=args.batch_size,
                        trace=trace,
                        seed=args.shuffle_seed,
                    )
                )
    by_key = {(item["repeat"], item["images"], item["trace"]): item for item in results}
    trace_checks: list[dict[str, Any]] = []
    if {"sequential", "shuffled"}.issubset(args.traces):
        for repeat in range(args.repeats):
            for image_count in sorted(set(item["images"] for item in results)):
                sequential = by_key[(repeat, image_count, "sequential")]["planning_ms"]["median"]
                shuffled = by_key[(repeat, image_count, "shuffled")]["planning_ms"]["median"]
                denominator = max(abs(sequential), 1e-12)
                trace_checks.append(
                    {
                        "repeat": repeat,
                        "images": image_count,
                        "sequential_median_ms": sequential,
                        "shuffled_median_ms": shuffled,
                        "median_difference_ms": abs(shuffled - sequential),
                        "median_difference_ratio": abs(shuffled - sequential) / denominator,
                    }
                )
    scale_checks: list[dict[str, Any]] = []
    if len(set(item["images"] for item in results)) >= 2:
        minimum = min(item["images"] for item in results)
        maximum = max(item["images"] for item in results)
        for repeat in range(args.repeats):
            for trace in args.traces:
                small = by_key[(repeat, minimum, trace)]["planning_ms"]["median"]
                large = by_key[(repeat, maximum, trace)]["planning_ms"]["median"]
                denominator = max(abs(small), 1e-12)
                scale_checks.append(
                    {
                        "repeat": repeat,
                        "trace": trace,
                        "small_images": minimum,
                        "large_images": maximum,
                        "median_difference_ms": abs(large - small),
                        "median_difference_ratio": abs(large - small) / denominator,
                    }
                )
    gates = {
        "planning_median_ms_max": 2.0,
        "planning_p95_ms_max": 3.0,
        "trace_median_difference_ratio_max": 0.10,
        "dataset_size_median_difference_ratio_max": 0.10,
    }
    gate_results = {
        "planning_median": all(item["planning_ms"]["median"] <= gates["planning_median_ms_max"] for item in results),
        "planning_p95": all(item["planning_ms"]["p95"] <= gates["planning_p95_ms_max"] for item in results),
        "trace_invariance": all(
            item["median_difference_ratio"] <= gates["trace_median_difference_ratio_max"]
            for item in trace_checks
        ),
        "dataset_size_invariance": all(
            item["median_difference_ratio"] <= gates["dataset_size_median_difference_ratio_max"]
            for item in scale_checks
        ),
    }
    gate_results["passed"] = all(gate_results.values())
    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "manifest_image_count": int(reader.image_count),
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "results": results,
        "trace_checks": trace_checks,
        "dataset_scale_checks": scale_checks,
        "gates": gates,
        "gate_results": gate_results,
    }
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if gate_results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
