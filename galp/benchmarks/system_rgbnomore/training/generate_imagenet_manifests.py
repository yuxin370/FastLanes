#!/usr/bin/env python3
"""Generate disjoint train/validation JSON files for the training benchmark.

The current four-pipeline runner accepts one GALP manifest.  This generator
therefore selects both splits from the JPEG population used to create that
manifest, preserving the original sorted-path GALP image IDs.  The validation
split is a deterministic held-out subset of ImageNet train, not official
ImageNet validation accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import struct
import sys
from itertools import chain
from pathlib import Path
from typing import Any, Sequence


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from dataset.manifest import jpeg_sampling


JPEG_SUFFIXES = {".jpg", ".jpeg", ".jpe"}
LABEL_MAPPING = "imagenet-1k-zero-based"
MANIFEST_FORMAT = "galp-rgbnomore-training-manifest-v1"
SUPPORTED_JPEG_SAMPLING = {"4:2:0", "4:4:4"}


def _normalize(value: str) -> str:
    normalized = value.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _load_labels(index_csv: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with index_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"Filepath", "Label"}.issubset(reader.fieldnames):
            raise ValueError(f"{index_csv} must contain Filepath and Label columns")
        for row_number, row in enumerate(reader, start=2):
            sample_id = _normalize(str(row["Filepath"]))
            label = int(row["Label"])
            if not 0 <= label < 1000:
                raise ValueError(f"{index_csv}:{row_number} has invalid label {label}")
            if sample_id in labels:
                raise ValueError(f"{index_csv}:{row_number} duplicates {sample_id!r}")
            labels[sample_id] = label
    if not labels:
        raise ValueError(f"{index_csv} contains no samples")
    return labels


def _collect_jpegs(jpeg_root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, subdirectories, filenames in os.walk(jpeg_root):
        subdirectories.sort()
        paths.extend(
            Path(directory) / filename
            for filename in filenames
            if Path(filename).suffix.lower() in JPEG_SUFFIXES
        )
    paths.sort()
    if not paths:
        raise ValueError(f"no JPEG files found below {jpeg_root}")
    return paths


def _galp_image_count(manifest: Path) -> int:
    data = manifest.read_bytes()[:30]
    if len(data) < 30 or data[:8] != b"GJDCTSH1":
        raise ValueError(f"unexpected GALP shard manifest format: {manifest}")
    _version, _reserved, _rowgroup_vectors, _rowgroups_per_shard, image_count = struct.unpack_from(
        "<IHIIQ", data, 8
    )
    return int(image_count)


def _selection_sha256(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for image_id in indices:
        digest.update(struct.pack("<Q", image_id))
    return digest.hexdigest()


def _select_indices(
    paths: Sequence[Path], *, train_count: int, val_count: int, seed: int
) -> tuple[list[int], list[int], dict[int, str]]:
    population = len(paths)
    if val_count <= 0 or val_count >= population:
        raise ValueError(f"val-count must be in [1,{population - 1}], got {val_count}")
    if train_count < 0:
        raise ValueError("train-count must be non-negative; use 0 for all remaining samples")
    rng = random.Random(seed)
    if train_count == 0:
        sampling = {image_id: jpeg_sampling(path) for image_id, path in enumerate(paths)}
        eligible = [
            image_id
            for image_id, mode in sampling.items()
            if mode in SUPPORTED_JPEG_SAMPLING
        ]
        if val_count >= len(eligible):
            raise ValueError(
                f"val-count must be smaller than the {len(eligible)} eligible JPEG population"
            )
        rng.shuffle(eligible)
        val_indices = eligible[:val_count]
        train_indices = eligible[val_count:]
    else:
        if train_count + val_count > population:
            raise ValueError(
                f"train-count + val-count exceeds the {population}-image population"
            )
        required = train_count + val_count
        selected: list[int] = []
        seen: set[int] = set()
        sampling: dict[int, str] = {}
        while len(selected) < required and len(seen) < population:
            image_id = rng.randrange(population)
            if image_id in seen:
                continue
            seen.add(image_id)
            mode = jpeg_sampling(paths[image_id])
            sampling[image_id] = mode
            if mode in SUPPORTED_JPEG_SAMPLING:
                selected.append(image_id)
        if len(selected) != required:
            raise ValueError(
                f"requested {required} samples, but found only {len(selected)} eligible JPEGs"
            )
        val_indices = selected[:val_count]
        train_indices = selected[val_count:]
    selected_sampling = {
        image_id: sampling[image_id] for image_id in chain(train_indices, val_indices)
    }
    return train_indices, val_indices, selected_sampling


def _sample_id(path: Path, jpeg_root: Path, source_split: str) -> str:
    return f"{source_split}/{path.relative_to(jpeg_root).as_posix()}"


def _build_samples(
    indices: Sequence[int],
    *,
    paths: Sequence[Path],
    labels: dict[str, int],
    jpeg_root: Path,
    source_split: str,
    sampling: dict[int, str],
    probe_dimensions: bool,
    progress_interval: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for ordinal, image_id in enumerate(indices, start=1):
        path = paths[image_id]
        sample_id = _sample_id(path, jpeg_root, source_split)
        sample: dict[str, Any] = {
            "logical_sample_id": sample_id,
            "path": str(path),
            "label": labels[sample_id],
            "galp_image_id": image_id,
            "jpeg_sampling": sampling[image_id],
        }
        if probe_dimensions:
            from PIL import Image

            with Image.open(path) as image:
                sample["width"], sample["height"] = map(int, image.size)
        samples.append(sample)
        if progress_interval > 0 and ordinal % progress_interval == 0:
            print(f"prepared {ordinal}/{len(indices)} samples", file=sys.stderr)
    return samples


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpeg-root", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--galp-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-split", default="train")
    parser.add_argument(
        "--train-count",
        type=int,
        default=100_000,
        help="Number of training samples; use 0 for every sample not held out for validation.",
    )
    parser.add_argument("--val-count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=11_997_733)
    parser.add_argument(
        "--probe-dimensions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read and store JPEG dimensions now instead of during benchmark startup.",
    )
    parser.add_argument("--progress-interval", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    jpeg_root = args.jpeg_root.resolve()
    index_csv = args.index_csv.resolve()
    galp_manifest = args.galp_manifest.resolve()
    if not jpeg_root.is_dir():
        raise FileNotFoundError(jpeg_root)
    if not index_csv.is_file():
        raise FileNotFoundError(index_csv)
    if not galp_manifest.is_file():
        raise FileNotFoundError(galp_manifest)

    paths = _collect_jpegs(jpeg_root)
    labels = _load_labels(index_csv)
    if len(labels) != len(paths):
        raise ValueError(
            f"JPEG/index population mismatch: {len(paths)} files versus {len(labels)} index rows"
        )
    for path in paths:
        sample_id = _sample_id(path, jpeg_root, args.source_split)
        if sample_id not in labels:
            raise ValueError(f"JPEG is missing from label index: {sample_id}")

    manifest_count = _galp_image_count(galp_manifest)
    if manifest_count != len(paths):
        raise ValueError(
            f"GALP/JPEG population mismatch: manifest has {manifest_count} images, "
            f"JPEG root has {len(paths)}"
        )

    train_indices, val_indices, sampling = _select_indices(
        paths, train_count=args.train_count, val_count=args.val_count, seed=args.seed
    )
    common = {
        "format": MANIFEST_FORMAT,
        "label_mapping": LABEL_MAPPING,
        "source_split": args.source_split,
        "validation_semantics": "deterministic-heldout-from-imagenet-train",
        "jpeg_root": str(jpeg_root),
        "source_index_csv": str(index_csv),
        "galp_manifest": str(galp_manifest),
        "population_count": len(paths),
        "selection_seed": args.seed,
        "jpeg_sampling_eligibility": sorted(SUPPORTED_JPEG_SAMPLING),
    }
    train_payload = {
        **common,
        "split": "train",
        "selection_sha256": _selection_sha256(train_indices),
        "samples": _build_samples(
            train_indices,
            paths=paths,
            labels=labels,
            jpeg_root=jpeg_root,
            source_split=args.source_split,
            sampling=sampling,
            probe_dimensions=args.probe_dimensions,
            progress_interval=args.progress_interval,
        ),
    }
    val_payload = {
        **common,
        "split": "val",
        "selection_sha256": _selection_sha256(val_indices),
        "samples": _build_samples(
            val_indices,
            paths=paths,
            labels=labels,
            jpeg_root=jpeg_root,
            source_split=args.source_split,
            sampling=sampling,
            probe_dimensions=args.probe_dimensions,
            progress_interval=args.progress_interval,
        ),
    }
    output_dir = args.output_dir.resolve()
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"
    _write_json(train_path, train_payload, overwrite=args.overwrite)
    _write_json(val_path, val_payload, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "train_json": str(train_path),
                "train_samples": len(train_indices),
                "val_json": str(val_path),
                "val_samples": len(val_indices),
                "validation_semantics": common["validation_semantics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
