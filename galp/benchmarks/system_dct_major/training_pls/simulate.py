#!/usr/bin/env python3
"""Generate one auditable PLS schedule without running a model."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from .artifacts import write_schedule_artifacts
from .schedule import (
    CROP_POLICIES,
    ORDER_POLICIES,
    ORGANIZATIONS,
    PlsScheduleConfig,
    build_schedule,
    load_training_samples,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segment-images", type=int, default=1024)
    parser.add_argument("--segments-per-pool", type=int, default=4)
    parser.add_argument("--optimizer-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--organization-seed", type=int, default=20260810)
    parser.add_argument("--organization", choices=ORGANIZATIONS, default="current")
    parser.add_argument("--crop-policy", choices=CROP_POLICIES, default="per-shard")
    parser.add_argument("--order-policy", choices=ORDER_POLICIES, default="pls-wave")
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distributed-rank", type=int, default=0)
    parser.add_argument("--distributed-world-size", type=int, default=1)
    parser.add_argument("--trace-detail", choices=("digests", "full"), default="digests")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = args.train_manifest.expanduser().resolve()
    if manifest.is_dir():
        parser.error(
            f"--train-manifest resolved to a directory: {manifest}. "
            "This commonly means a shell variable such as $TRAIN_JSON was unset; "
            "pass the path of the training JSON file itself."
        )
    if not manifest.is_file():
        parser.error(f"--train-manifest is not a file: {manifest}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(
            f"--output-dir already exists: {output_dir}. "
            "Choose a new directory so an earlier schedule is never overwritten."
        )
    samples = load_training_samples(manifest)
    config = PlsScheduleConfig(
        segment_images=args.segment_images,
        segments_per_pool=args.segments_per_pool,
        optimizer_batch_size=args.optimizer_batch_size,
        epochs=args.epochs,
        seed=args.seed,
        organization_seed=args.organization_seed,
        organization=args.organization,
        crop_policy=args.crop_policy,
        order_policy=args.order_policy,
        drop_last=args.drop_last,
        distributed_rank=args.distributed_rank,
        distributed_world_size=args.distributed_world_size,
    )
    result = build_schedule(samples, config)
    write_schedule_artifacts(
        output_dir,
        result,
        manifest_record={
            "path": str(manifest),
            "sha256": _sha256_file(manifest),
            "sample_count": len(samples),
        },
        trace_detail=args.trace_detail,
    )
    print(output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
