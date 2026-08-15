#!/usr/bin/env python3
"""Inspect manifest layout and sequential crop planning without running CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
REPO_ROOT = BENCHMARK_ROOT.parents[2]
for path in (BENCHMARK_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import parse_manifest  # noqa: E402
from galp.profiles.rgbnomore import VALIDATION_CENTER_CROP_512  # noqa: E402
from galp.torch import DirectDctReader  # noqa: E402
from galp.diagnostics.direct_dct import plan_preview  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--torch-binding-dir", type=Path)
    parser.add_argument("--segment-size", type=int, default=50)
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
        image_count = min(args.segment_size, int(metadata["image_count"]))
        reader = DirectDctReader(
            args.manifest,
            module_path=args.torch_binding_dir,
        )
        preview = plan_preview(
            reader,
            list(range(image_count)),
            VALIDATION_CENTER_CROP_512,
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
