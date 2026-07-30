#!/usr/bin/env python3
"""Run the FLS metadata analyzer over canonical shard_NNNNNN.fls files."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SHARD_PATTERN = re.compile(r"shard_[0-9]{6}\.fls")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tool",
        type=Path,
        default=Path("build/galp/tools/metadata/galp_fls_metadata_tool"),
    )
    args = parser.parse_args()
    shards = sorted(path for path in args.input_dir.iterdir() if SHARD_PATTERN.fullmatch(path.name))
    if not shards:
        parser.error("input directory contains no canonical shard_NNNNNN.fls files")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, Any]] = []
    analyses: list[dict[str, Any]] = []
    for shard in shards:
        output = args.output_dir / f"{shard.stem}.breakdown.json"
        command = [str(args.tool), "analyze", "--input", str(shard), "--output", str(output), "--pretty"]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        commands.append(
            {
                "argv": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(f"analyzer failed for {shard}: {completed.stderr}")
        analyses.append(json.loads(output.read_text()))

    def sum_path(*path: str) -> int:
        total = 0
        for analysis in analyses:
            value: Any = analysis
            for component in path:
                value = value[component]
            total += int(value)
        return total

    file_bytes = sum_path("file_layout", "file_bytes")
    descriptor_bytes = sum_path("file_layout", "table_descriptor_bytes")
    projected_bytes = sum_path("table_descriptor", "compact_v2_projection", "projected_bytes")
    summary = {
        "schema_version": "galp_fls_metadata_dataset_breakdown_v1",
        "classification": {
            "measured": "File regions, parsed counts, and per-shard canonical repetition",
            "inferred": "Compact V2 byte projection",
            "unknown": "Cross-shard canonical unique count (keys are intentionally not exported)",
        },
        "input_dir": str(args.input_dir.resolve()),
        "shards": [str(path.resolve()) for path in shards],
        "totals": {
            "shard_count": len(shards),
            "file_bytes": file_bytes,
            "payload_bytes": sum_path("file_layout", "payload_bytes"),
            "descriptor_bytes": descriptor_bytes,
            "file_header_bytes": sum_path("file_layout", "file_header_bytes"),
            "file_footer_bytes": sum_path("file_layout", "file_footer_bytes"),
            "rowgroups": sum_path("table_descriptor", "counts", "rowgroups"),
            "root_columns": sum_path("table_descriptor", "counts", "root_columns"),
            "expression_results": sum_path("table_descriptor", "counts", "expression_results"),
            "segment_descriptors": sum_path("table_descriptor", "counts", "segment_descriptors"),
            "compact_v2_projected_bytes": projected_bytes,
            "descriptor_fraction_of_fls": descriptor_bytes / file_bytes,
            "compact_v2_projected_descriptor_reduction": 1.0 - projected_bytes / descriptor_bytes,
            "compact_v2_projected_file_reduction": (descriptor_bytes - projected_bytes) / file_bytes,
        },
        "sidecar_meta_bin_bytes": sum(
            path.stat().st_size for path in args.input_dir.glob("shard_*.meta.bin") if path.is_file()
        ),
        "per_shard": analyses,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
