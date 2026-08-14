#!/usr/bin/env python3
"""Select a manifest-shard tuning candidate using the locked cold-I/O rules."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from common import parse_manifest  # noqa: E402


ZERO_COUNTERS = (
    "segment_cross_shard_count",
    "shard_reactivation_count",
    "duplicate_physical_read_count",
    "rowgroup_revisit_count",
    "vector_run_revisit_count",
    "physical_read_order_inversions",
)
SCHEMA = "galp_manifest_shard_tuning_selection_v1"


def _load_candidate(path: Path) -> dict[str, Any]:
    result_path = path / "controlled_cold_results.json" if path.is_dir() else path
    result = json.loads(result_path.read_text(encoding="utf-8"))
    round_dir = Path(result["round_directories"][0])
    contract = json.loads((round_dir / "contract.json").read_text(encoding="utf-8"))
    name = "dct_major_pushdown"
    if name not in result["pipelines"]:
        raise RuntimeError(f"candidate does not contain {name}: {result_path}")
    pipeline = result["pipelines"][name]
    config = contract["pipelines"][name]
    manifest = parse_manifest(Path(config["manifest"]))
    native_repeats = pipeline["native_totals"]
    counter_failures = {
        counter: [int(native.get(counter, -1)) for native in native_repeats]
        for counter in ZERO_COUNTERS
        if any(int(native.get(counter, -1)) != 0 for native in native_repeats)
    }
    fallback_failures = {
        key: [float(native.get(key, 0.0)) for native in native_repeats]
        for key in sorted({key for native in native_repeats for key in native if "fallback" in key})
        if any(float(native.get(key, 0.0)) != 0.0 for native in native_repeats)
    }
    cv = float(pipeline["throughput_images_per_s"]["cv_population"])
    exclusions: list[str] = []
    if not result.get("ok"):
        exclusions.append("controlled-cold validation failed")
    if cv > 0.05:
        exclusions.append(f"throughput CV {cv:.6f} exceeds 0.05")
    if counter_failures:
        exclusions.append(f"repeat counters are nonzero or missing: {counter_failures}")
    if fallback_failures:
        exclusions.append(f"fallback counters are nonzero: {fallback_failures}")
    host_peak = float(pipeline["host_peak_rss_bytes"]["median"])
    torch_gpu_peak = float(pipeline["peak_torch_gpu_reserved_bytes"]["median"])
    native_gpu_values = [
        float(native.get("galp_native_device_peak_in_use_bytes", 0.0))
        for native in native_repeats
    ]
    native_gpu_peak = statistics.median(native_gpu_values)
    storage = contract["dataset"]["dct_major_storage"]
    descriptor_bytes = int((contract["dataset"].get("block_major_access") or {}).get("descriptor_bytes", 0))
    total_storage_bytes = (
        int(storage["persistent_bytes"])
        + int(storage["manifest"]["size_bytes"])
        + descriptor_bytes
    )
    return {
        "result": str(result_path.resolve()),
        "manifest": str(Path(config["manifest"]).resolve()),
        "access_dir": config.get("block_major_access_dir"),
        "shard_images": int(manifest["shards"][0]["image_count"]),
        "rowgroup_vectors": int(manifest["rowgroup_vectors"]),
        "rowgroups_per_shard": int(manifest["rowgroups_per_shard"]),
        "cold_throughput_images_per_s": float(pipeline["throughput_images_per_s"]["median"]),
        "cold_throughput_p95_images_per_s": float(pipeline["throughput_images_per_s"]["p95"]),
        "cold_throughput_cv": cv,
        "process_scope_ttfb_ms": float(
            pipeline["cold_time_to_first_batch_ms"]["median"]
        ),
        "host_peak_bytes": host_peak,
        "torch_gpu_reserved_peak_bytes": torch_gpu_peak,
        "native_gpu_peak_bytes": native_gpu_peak,
        "selection_peak_bytes": host_peak + max(torch_gpu_peak, native_gpu_peak),
        "total_storage_bytes": total_storage_bytes,
        "counter_failures": counter_failures,
        "fallback_failures": fallback_failures,
        "eligible": not exclusions,
        "exclusions": exclusions,
    }


def select(paths: Sequence[Path]) -> dict[str, Any]:
    candidates = [_load_candidate(path.resolve()) for path in paths]
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise RuntimeError("no tuning candidate passed validation, fallback, repeat, and CV gates")
    best_throughput = max(float(item["cold_throughput_images_per_s"]) for item in eligible)
    within_three_percent = [
        item
        for item in eligible
        if float(item["cold_throughput_images_per_s"]) >= 0.97 * best_throughput
    ]
    selected = min(
        within_three_percent,
        key=lambda item: (
            float(item["selection_peak_bytes"]),
            int(item["total_storage_bytes"]),
            float(item["process_scope_ttfb_ms"]),
            -float(item["cold_throughput_images_per_s"]),
        ),
    )
    return {
        "schema_version": SCHEMA,
        "selection_rule": (
            "exclude validation/fallback/repeat/CV failures; maximize controlled-cold throughput; "
            "within 3% choose lower peak memory, then storage, then TTFB"
        ),
        "best_observed_throughput_images_per_s": best_throughput,
        "selected": selected,
        "candidates": candidates,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = select(args.candidates)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["selected"], sort_keys=True))


if __name__ == "__main__":
    main()
