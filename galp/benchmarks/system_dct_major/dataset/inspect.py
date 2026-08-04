#!/usr/bin/env python3
"""Inspect manifest layout and sequential crop planning without running CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from common import parse_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--torch-binding-dir", type=Path)
    parser.add_argument("--segment-size", type=int, default=50)
    parser.add_argument("--crop-execution-mode", default="rowgroup-read-selected-decode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = parse_manifest(args.manifest)
    summary: dict[str, object] = {
        "manifest": str(args.manifest.resolve()),
        "version": metadata["version"],
        "physical_layout": metadata["physical_layout"],
        "image_count": metadata["image_count"],
        "shard_count": metadata["shard_count"],
        "persistent_fls_bytes": sum(int(item["fls_file_size"]) for item in metadata["shards"]),
        "persistent_metadata_bytes": sum(int(item["metadata_file_size"]) for item in metadata["shards"]),
    }
    if args.torch_binding_dir is not None:
        binding = str(args.torch_binding_dir.resolve())
        torch_profile = str((BENCHMARK_ROOT.parents[1] / "torch").resolve())
        for path in (binding, torch_profile):
            if path not in sys.path:
                sys.path.insert(0, path)
        from _galp_direct_dct import DirectDctReader  # type: ignore  # noqa: PLC0415
        from rgbnomore_dct_profile import RGBNOMORE_VAL_DCT_GRID_TRANSFORM_FP32  # type: ignore  # noqa: PLC0415

        image_count = min(args.segment_size, int(metadata["image_count"]))
        reader = DirectDctReader(str(args.manifest.resolve()))
        preview = reader.plan_batch(
            list(range(image_count)),
            layout="transformed_dct_grid",
            grid_transform=RGBNOMORE_VAL_DCT_GRID_TRANSFORM_FP32,
            cache_capacity_mib=0,
            plan_cache_capacity=0,
            enable_planless_execution=True,
            crop_execution_mode=args.crop_execution_mode,
        )
        summary["plan"] = {
            key: preview[key]
            for key in (
                "image_count",
                "block_count",
                "rowgroup_count",
                "planned_selected_vector_count",
                "full_vector_count",
                "planned_selected_vector_ratio",
                "planning_ms",
                "uses_planless_fixed_transform",
                "compact_image_descriptor_count",
                "y_shape",
                "cbcr_shape",
            )
        }
    print("RESULT_JSON " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
