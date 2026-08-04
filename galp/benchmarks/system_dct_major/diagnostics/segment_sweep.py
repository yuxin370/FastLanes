#!/usr/bin/env python3
"""Measure CPU planning growth for contiguous DCT-major segments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
REPO_ROOT = BENCHMARK_ROOT.parents[2]
for path in (BENCHMARK_ROOT, REPO_ROOT / "galp/torch"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import parse_manifest, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--torch-binding-dir", type=Path, default=REPO_ROOT / "build/galp/torch")
    parser.add_argument("--block-major-access-dir", type=Path, required=True)
    parser.add_argument("--segment-sizes", type=int, nargs="+", default=(1, 8, 32, 50, 128, 256, 512, 1024))
    parser.add_argument(
        "--crop-execution-mode",
        choices=("auto", "full-rowgroup-decode", "rowgroup-read-selected-decode", "vector-range-read-selected-decode"),
        default="rowgroup-read-selected-decode",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.torch_binding_dir.resolve(),):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from _galp_direct_dct import DirectDctReader  # type: ignore  # noqa: PLC0415
    from rgbnomore_dct_profile import RGBNOMORE_VAL_DCT_GRID_TRANSFORM_FP32  # type: ignore  # noqa: PLC0415

    access_dir = args.block_major_access_dir.resolve()
    if not access_dir.is_dir():
        raise NotADirectoryError(access_dir)
    os.environ["GALP_BLOCK_MAJOR_ACCESS_DIR"] = str(access_dir)
    manifest = parse_manifest(args.manifest)
    reader = DirectDctReader(str(args.manifest.resolve()))
    records = []
    for size in args.segment_sizes:
        if size <= 0 or size > int(reader.image_count):
            raise ValueError(f"invalid segment size: {size}")
        preview = reader.plan_batch(
            list(range(size)),
            layout="transformed_dct_grid",
            grid_transform=RGBNOMORE_VAL_DCT_GRID_TRANSFORM_FP32,
            cache_capacity_mib=0,
            plan_cache_capacity=0,
            enable_planless_execution=True,
            crop_execution_mode=args.crop_execution_mode,
        )
        record = {
            "segment_size": size,
            **{
                key: preview[key]
                for key in (
                    "block_count",
                    "rowgroup_count",
                    "planned_selected_vector_count",
                    "full_vector_count",
                    "planned_selected_vector_ratio",
                    "planning_ms",
                    "uses_planless_fixed_transform",
                    "compact_image_descriptor_count",
                    "coordinate_group_lookup_count",
                    "coordinate_group_index_entries",
                    "coordinate_group_index_populated",
                    "coordinate_group_index_holes",
                    "coordinate_group_index_bytes",
                    "coordinate_group_index_density",
                    "host_expanded_transform_items_created",
                    "host_output_block_source_lists_created",
                    "host_global_transform_sort_items",
                )
            },
        }
        if not bool(record["uses_planless_fixed_transform"]):
            raise RuntimeError(f"segment={size}: block-major planless path was not selected")
        for counter in (
            "host_expanded_transform_items_created",
            "host_output_block_source_lists_created",
            "host_global_transform_sort_items",
        ):
            if int(record[counter]) != 0:
                raise RuntimeError(f"segment={size}: planless invariant failed: {counter}={record[counter]}")
        if int(record["compact_image_descriptor_count"]) != size:
            raise RuntimeError(
                f"segment={size}: compact descriptor count {record['compact_image_descriptor_count']} differs"
            )
        coordinate_entries = int(record["coordinate_group_index_entries"])
        coordinate_populated = int(record["coordinate_group_index_populated"])
        coordinate_holes = int(record["coordinate_group_index_holes"])
        if (
            int(record["coordinate_group_lookup_count"]) <= 0
            or coordinate_populated <= 0
            or coordinate_entries != coordinate_populated + coordinate_holes
            or int(record["coordinate_group_index_bytes"]) < coordinate_entries * 4
        ):
            raise RuntimeError(f"segment={size}: coordinate lookup accounting is inconsistent")
        expected_density = coordinate_populated / coordinate_entries
        if abs(float(record["coordinate_group_index_density"]) - expected_density) > 1.0e-12:
            raise RuntimeError(f"segment={size}: coordinate lookup density is inconsistent")
        records.append(record)
    result = {
        "schema_version": "galp_dct_major_segment_sweep_v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_version": manifest["version"],
        "physical_layout": manifest["physical_layout"],
        "block_major_access_dir": str(access_dir),
        "crop_execution_mode": args.crop_execution_mode,
        "records": records,
    }
    if args.output_json:
        write_json(args.output_json, result)
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
