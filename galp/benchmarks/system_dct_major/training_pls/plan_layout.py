#!/usr/bin/env python3
"""Generate the immutable virtual Physical Load Segment sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .layout import create_layout_plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segment-images", type=int, default=1024)
    parser.add_argument("--organization", choices=("current",), default="current")
    parser.add_argument("--organization-seed", type=int, default=20260810)
    args = parser.parse_args(argv)
    plan = create_layout_plan(
        args.train_manifest,
        args.output_dir,
        segment_images=args.segment_images,
        organization=args.organization,
        organization_seed=args.organization_seed,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
