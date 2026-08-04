#!/usr/bin/env python3
"""Prepare a reconstructable GALP shard set with an explicit physical layout."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DEFAULT_TOOL = REPO_ROOT / "build/galp/tools/jpeg_dct/galp_jpeg_dct_tool"


def build_command(args: argparse.Namespace) -> list[str]:
    layout = "spatial-major" if args.layout == "dct-major" else "image-major"
    command = [
        str(args.tool.resolve()),
        "--shard",
        "--out-dir",
        str(args.output_dir.resolve()),
        "--preset",
        args.preset,
        "--physical-layout",
        layout,
        "--metadata-profile",
        "reconstruct",
    ]
    for flag, value in (
        ("--shard-images", args.shard_images),
        ("--rowgroup-vectors", args.rowgroup_vectors),
        ("--rowgroups-per-shard", args.rowgroups_per_shard),
        ("--threads", args.threads),
        ("--shard-workers", args.shard_workers),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    command.append(str(args.input_dir.resolve()))
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", choices=("dct-major", "image-major"), required=True)
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL)
    parser.add_argument("--preset", choices=("crop-latency", "balanced", "throughput", "random-access"), default="throughput")
    parser.add_argument("--shard-images", type=int)
    parser.add_argument("--rowgroup-vectors", type=int)
    parser.add_argument("--rowgroups-per-shard", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--shard-workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tool.is_file():
        raise FileNotFoundError(args.tool)
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    command = build_command(args)
    print("COMMAND " + shlex.join(command), flush=True)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True, cwd=REPO_ROOT)
    print(json.dumps({"layout": args.layout, "output_dir": str(args.output_dir.resolve()), "dry_run": args.dry_run}, sort_keys=True))


if __name__ == "__main__":
    main()

