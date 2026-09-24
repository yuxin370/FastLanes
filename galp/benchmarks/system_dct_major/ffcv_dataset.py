#!/usr/bin/env python3
"""Convert the canonical RGB sample order to FFCV's lossless raw format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from galp.benchmarks.system_dct_major.common import (
    collect_sequential_samples,
    load_sample_manifest,
    parse_manifest,
)


class CanonicalImages:
    def __init__(self, samples: Sequence[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        sample = self.samples[index]
        with Image.open(sample["path"]) as image:
            return image.convert("RGB"), int(sample["ordinal"])


def write_beton(samples: Sequence[dict[str, Any]], output: Path, workers: int) -> None:
    from ffcv.fields import IntField, RGBImageField
    from ffcv.writer import DatasetWriter

    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = DatasetWriter(
        str(output), {"image": RGBImageField(write_mode="raw"), "ordinal": IntField()},
        num_workers=max(1, workers),
    )
    writer.from_indexed_dataset(CanonicalImages(samples))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-manifest", type=Path)
    source.add_argument("--data-root", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--dct-major-label-map", type=Path)
    parser.add_argument("--dct-major-manifest", type=Path)
    parser.add_argument("--sample-count", type=int)
    args = parser.parse_args()
    if args.sample_manifest is not None:
        samples = load_sample_manifest(args.sample_manifest)
    else:
        if args.dct_major_label_map is None or args.dct_major_manifest is None or args.sample_count is None:
            parser.error("--data-root requires --dct-major-label-map, --dct-major-manifest and --sample-count")
        image_count = int(parse_manifest(args.dct_major_manifest)["image_count"])
        samples, _ = collect_sequential_samples(
            data_root=args.data_root, split=args.split, label_map_json=args.dct_major_label_map,
            expected_images=image_count, sample_count=args.sample_count,
            hash_samples=False,
        )
    write_beton(samples, args.output, args.workers)


if __name__ == "__main__":
    main()
