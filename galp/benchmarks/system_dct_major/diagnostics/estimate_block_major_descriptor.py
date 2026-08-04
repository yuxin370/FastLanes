#!/usr/bin/env python3
"""Estimate a compact block-major rank/select descriptor from real metadata.

The estimate is deliberately conservative: sparse and missing image-id lists
use fixed uint16 IDs.  A real builder may delta-code those lists, but Phase 1
must satisfy the hard one-percent storage gate without relying on that win.
No coefficient payload or metadata file is changed by this diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


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
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--rank-checkpoint-images", type=int, default=256)
    parser.add_argument("--topology-checkpoint-groups", type=int, default=128)
    return parser.parse_args()


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _image_components(metadata: dict[str, Any]) -> dict[int, tuple[int, int]]:
    return {
        int(component["semantic_slot_id"]):
            (int(component["width_in_blocks"]), int(component["height_in_blocks"]))
        for component in metadata["components"]
        if bool(component["present"])
    }


def main() -> None:
    args = parse_args()
    if args.rank_checkpoint_images <= 0 or args.topology_checkpoint_groups <= 0:
        raise ValueError("checkpoint intervals must be positive")
    binding_dir = args.torch_binding_dir.resolve()
    if str(binding_dir) not in sys.path:
        sys.path.insert(0, str(binding_dir))
    from _galp_direct_dct import DirectDctReader  # type: ignore  # noqa: PLC0415

    manifest_path = args.manifest.resolve()
    manifest = parse_manifest(manifest_path)
    if int(manifest["version"]) != 1 or manifest["physical_layout"] != "dct-major/spatial-major-image-minor":
        raise ValueError("descriptor estimate requires a version-1 spatial-major/image-minor manifest")

    reader = DirectDctReader(str(manifest_path))
    metadata_started = time.perf_counter()
    images: list[dict[int, tuple[int, int]]] = []
    quant_tables: list[set[tuple[int, ...]]] = [set() for _ in manifest["shards"]]
    present_component_count = 0
    shard_cursor = 0
    shards = manifest["shards"]
    for image_index in range(int(manifest["image_count"])):
        while image_index >= int(shards[shard_cursor]["first_global_image_index"]) + int(
            shards[shard_cursor]["image_count"]
        ):
            shard_cursor += 1
        metadata = reader.image_metadata(image_index)
        components = _image_components(metadata)
        images.append(components)
        present_component_count += len(components)
        for table in metadata["quant_tables"]:
            quant_tables[shard_cursor].add(tuple(int(value) for value in table["values"]))
    metadata_seconds = time.perf_counter() - metadata_started

    totals = {
        "coordinate_positions": 0,
        "block_groups": 0,
        "source_rows": 0,
        "presence_cells": 0,
        "all_present_cells": 0,
        "sparse_cells": 0,
        "missing_cells": 0,
        "bitmap_cells": 0,
        "presence_payload_bytes_fixed_u16": 0,
        "threshold_array_bytes": 0,
        "semantic_slot_count": 0,
    }
    shard_records: list[dict[str, Any]] = []
    for shard in shards:
        begin = int(shard["first_global_image_index"])
        end = begin + int(shard["image_count"])
        shard_images = images[begin:end]
        image_count = len(shard_images)
        bitmap_bytes = _ceil_div(image_count, 8) + 2 * _ceil_div(image_count, args.rank_checkpoint_images)
        record = {
            "shard_id": int(shard["shard_id"]),
            "image_count": image_count,
            "coordinate_positions": 0,
            "block_groups": 0,
            "source_rows": 0,
            "presence_cells": 0,
            "all_present_cells": 0,
            "sparse_cells": 0,
            "missing_cells": 0,
            "bitmap_cells": 0,
            "presence_payload_bytes_fixed_u16": 0,
            "threshold_array_bytes": 0,
        }
        slots = sorted({slot for image in shard_images for slot in image})
        totals["semantic_slot_count"] += len(slots)
        for slot in slots:
            shapes = [image.get(slot, (0, 0)) for image in shard_images]
            nonzero_shapes = {shape for shape in shapes if shape != (0, 0)}
            widths = sorted({width for width, _ in nonzero_shapes})
            heights = sorted({height for _, height in nonzero_shapes})
            max_width = max(widths)
            max_height = max(heights)

            dense_histogram = np.zeros((max_width + 1, max_height + 1), dtype=np.int32)
            for width, height in shapes:
                dense_histogram[width, height] += 1
            dense_counts = (
                dense_histogram[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1][1:, 1:]
            ).reshape(-1).astype(np.int64)
            positive_dense = dense_counts > 0
            record["coordinate_positions"] += int(dense_counts.size)
            record["block_groups"] += int(positive_dense.sum())
            record["source_rows"] += int(dense_counts.sum())

            width_index = {value: index for index, value in enumerate(widths)}
            height_index = {value: index for index, value in enumerate(heights)}
            threshold_histogram = np.zeros((len(widths), len(heights)), dtype=np.int32)
            for width, height in shapes:
                if width != 0 and height != 0:
                    threshold_histogram[width_index[width], height_index[height]] += 1
            counts = (
                threshold_histogram[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]
            ).reshape(-1).astype(np.int64)
            positive = counts > 0
            all_present = counts == image_count
            sparse_bytes = counts * 2
            missing_bytes = (image_count - counts) * 2
            payload_bytes = np.minimum(np.minimum(sparse_bytes, missing_bytes), bitmap_bytes)
            payload_bytes[all_present] = 0
            payload_bytes[~positive] = 0

            sparse = (sparse_bytes <= missing_bytes) & (sparse_bytes <= bitmap_bytes) & ~all_present & positive
            missing = (missing_bytes < sparse_bytes) & (missing_bytes <= bitmap_bytes) & ~all_present & positive
            bitmap = (bitmap_bytes < sparse_bytes) & (bitmap_bytes < missing_bytes) & ~all_present & positive
            record["presence_cells"] += int(positive.sum())
            record["all_present_cells"] += int(all_present.sum())
            record["sparse_cells"] += int(sparse.sum())
            record["missing_cells"] += int(missing.sum())
            record["bitmap_cells"] += int(bitmap.sum())
            record["presence_payload_bytes_fixed_u16"] += int(payload_bytes.sum())
            record["threshold_array_bytes"] += 2 * (len(widths) + len(heights))

        if record["block_groups"] != int(shard["block_group_count"]):
            raise RuntimeError(
                f"shard {record['shard_id']} block-group reconstruction mismatch: "
                f"{record['block_groups']} != {shard['block_group_count']}"
            )
        if record["source_rows"] != int(shard["real_row_count"]):
            raise RuntimeError(
                f"shard {record['shard_id']} source-row reconstruction mismatch: "
                f"{record['source_rows']} != {shard['real_row_count']}"
            )
        for key in totals:
            if key in record:
                totals[key] += int(record[key])
        shard_records.append(record)

    block_groups = totals["block_groups"]
    cells = totals["presence_cells"]
    image_record_bytes = int(manifest["image_count"]) * 32
    component_record_bytes = present_component_count * 24
    quant_dictionary_bytes = sum(len(values) for values in quant_tables) * (8 + 64 * 2)
    cell_directory_bytes = (
        cells * 2
        + _ceil_div(cells, 4)
        + _ceil_div(cells, 256) * 4
        + totals["threshold_array_bytes"]
    )
    # Each topology checkpoint resolves logical group rank, physical row,
    # rowgroup, and row-in-rowgroup for the next bounded scan.
    topology_checkpoint_bytes = _ceil_div(totals["coordinate_positions"], args.topology_checkpoint_groups) * 24
    rowgroup_boundary_bytes = _ceil_div(block_groups, 8) + _ceil_div(block_groups, 256) * 4
    slot_directory_bytes = totals["semantic_slot_count"] * 64
    header_bytes = len(shards) * 4096
    descriptor_bytes = (
        totals["presence_payload_bytes_fixed_u16"]
        + image_record_bytes
        + component_record_bytes
        + quant_dictionary_bytes
        + cell_directory_bytes
        + topology_checkpoint_bytes
        + rowgroup_boundary_bytes
        + slot_directory_bytes
        + header_bytes
    )
    persistent_dataset_bytes = sum(
        int(shard["fls_file_size"]) + int(shard["metadata_file_size"]) for shard in shards
    )
    result = {
        "schema_version": "galp_block_major_descriptor_estimate_v1",
        "manifest": str(manifest_path),
        "metadata_scan_seconds": metadata_seconds,
        "persistent_dataset_bytes": persistent_dataset_bytes,
        "hard_limit_bytes_1_percent": persistent_dataset_bytes // 100,
        "target_bytes_0_5_percent": persistent_dataset_bytes // 200,
        "fixed_u16_upper_bound": {
            "descriptor_bytes": descriptor_bytes,
            "dataset_growth_ratio": descriptor_bytes / persistent_dataset_bytes,
            "passes_hard_one_percent": descriptor_bytes * 100 <= persistent_dataset_bytes,
            "passes_target_half_percent": descriptor_bytes * 200 <= persistent_dataset_bytes,
            "sections": {
                "presence_payload_bytes": totals["presence_payload_bytes_fixed_u16"],
                "cell_directory_bytes": cell_directory_bytes,
                "topology_checkpoint_bytes": topology_checkpoint_bytes,
                "rowgroup_boundary_bytes": rowgroup_boundary_bytes,
                "image_record_bytes": image_record_bytes,
                "component_record_bytes": component_record_bytes,
                "quant_dictionary_bytes": quant_dictionary_bytes,
                "slot_directory_bytes": slot_directory_bytes,
                "header_bytes": header_bytes,
            },
        },
        "counts": {
            **totals,
            "image_count": int(manifest["image_count"]),
            "present_component_count": present_component_count,
            "unique_quant_table_count": sum(len(values) for values in quant_tables),
        },
        "encoding_policy": {
            "all_present": "no payload",
            "sparse": "sorted uint16 image IDs (delta-ULEB128 is a Phase-2 size optimization)",
            "missing": "sorted uint16 missing image IDs (delta-ULEB128 is a Phase-2 size optimization)",
            "bitmap": f"image bitmap plus uint16 rank every {args.rank_checkpoint_images} images",
            "topology_checkpoint_groups": args.topology_checkpoint_groups,
        },
        "shards": shard_records,
    }
    if args.output_json is not None:
        write_json(args.output_json, result)
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
