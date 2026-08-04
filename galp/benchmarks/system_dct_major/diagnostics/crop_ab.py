#!/usr/bin/env python3
"""Summarize the strict DCT-major full-versus-pushdown result pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from common import distribution, read_json, write_json  # noqa: E402


def _hot(result: dict) -> list[dict]:
    records = list(result["repeats"])
    return records[1:] if result["execution"].get("aggregate_exclude_first_repeat") and len(records) > 1 else records


def _mean_native(result: dict, *keys: str) -> float:
    records = _hot(result)
    values = []
    for item in records:
        native = item.get("native_totals", {})
        values.append(next((float(native[key]) for key in keys if key in native and float(native[key]) > 0.0), 0.0))
    return sum(values) / len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full = read_json(args.output_dir / "pipeline_dct_major_full.json")
    push = read_json(args.output_dir / "pipeline_dct_major_pushdown.json")
    full_median = float(distribution([item["throughput_images_per_s"] for item in _hot(full)])["p50"])
    push_median = float(distribution([item["throughput_images_per_s"] for item in _hot(push)])["p50"])
    keys = {
        "compressed_payload_bytes_read": ("compressed_payload_bytes_read", "rowgroup_storage_bytes_read"),
        "actual_vector_count": ("actual_vector_count", "selected_vector_count"),
        "requested_source_block_count": (
            "requested_source_block_count",
            "source_blocks_transformed",
            "fixed_transform_source_block_count",
            "block_count",
            "planned_selected_vector_count",
            "planned_vector_count",
        ),
        "pread_count": ("pread_count",),
    }
    metrics = {
        label: {"full": _mean_native(full, *aliases), "pushdown": _mean_native(push, *aliases)}
        for label, aliases in keys.items()
    }
    result = {
        "schema_version": "galp_dct_major_crop_ab_v1",
        "throughput": {
            "full_median_images_per_s": full_median,
            "pushdown_median_images_per_s": push_median,
            "pushdown_over_full": push_median / full_median,
        },
        "native": metrics,
    }
    if args.output_json:
        write_json(args.output_json, result)
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
