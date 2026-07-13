#!/usr/bin/env python3
"""Create the canonical sample/label/order manifest shared by all four pipelines."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from common import MANIFEST_SCHEMA, fingerprint_file, sha256_file, sha256_json, write_json


JPEG_SUFFIXES = {".jpg", ".jpeg", ".jpe"}
SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _normalize_relative(value: str) -> str:
    normalized = value.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def load_label_index(index_csv: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with index_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"Filepath", "Label"}.issubset(reader.fieldnames):
            raise ValueError(f"{index_csv} must contain Filepath and Label columns")
        for row_number, row in enumerate(reader, start=2):
            key = _normalize_relative(str(row["Filepath"]))
            label = int(row["Label"])
            if not 0 <= label < 1000:
                raise ValueError(f"{index_csv}:{row_number} has invalid ImageNet-1K label {label}")
            if key in labels:
                raise ValueError(f"{index_csv}:{row_number} duplicates {key}")
            labels[key] = label
    if not labels:
        raise ValueError(f"{index_csv} contains no samples")
    return labels


def jpeg_sampling(path: Path) -> str:
    """Return the JPEG component sampling mode without a Pillow dependency."""
    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError(f"not a JPEG file: {path}")
        while True:
            byte = stream.read(1)
            while byte and byte != b"\xff":
                byte = stream.read(1)
            while byte == b"\xff":
                byte = stream.read(1)
            if not byte:
                break
            marker = byte[0]
            if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                break
            length = int.from_bytes(length_bytes, "big")
            if length < 2:
                raise ValueError(f"invalid JPEG marker length in {path}")
            if marker not in SOF_MARKERS:
                stream.seek(length - 2, 1)
                continue
            segment = stream.read(length - 2)
            if len(segment) < 6:
                raise ValueError(f"short JPEG SOF segment in {path}")
            component_count = int(segment[5])
            if component_count == 1:
                return "grayscale"
            if component_count != 3 or len(segment) < 6 + 3 * component_count:
                return f"components:{component_count}"
            factors = []
            for component in range(component_count):
                sampling = int(segment[6 + 3 * component + 1])
                factors.append((sampling >> 4, sampling & 0x0F))
            y, cb, cr = factors
            if cb != cr:
                return "mismatched_chroma"
            if y == cb:
                return "4:4:4"
            if y[0] == cb[0] * 2 and y[1] == cb[1] * 2:
                return "4:2:0"
            if y[0] == cb[0] * 2 and y[1] == cb[1]:
                return "4:2:2"
            return "unsupported"
    raise ValueError(f"JPEG SOF marker not found: {path}")


def collect_dataset(data_root: Path, split: str, index_csv: Path) -> list[dict[str, Any]]:
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)
    labels = load_label_index(index_csv)
    paths = sorted(path for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in JPEG_SUFFIXES)
    if not paths:
        raise ValueError(f"no JPEG files found below {split_dir}")

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for galp_image_id, path in enumerate(paths):
        sample_id = _normalize_relative(path.relative_to(data_root).as_posix())
        if sample_id not in labels:
            raise ValueError(f"dataset file is missing from RGB-no-more index: {sample_id}")
        seen.add(sample_id)
        entries.append(
            {
                "sample_id": sample_id,
                "path": str(path.resolve()),
                "label": labels[sample_id],
                "galp_image_id": galp_image_id,
                "size_bytes": path.stat().st_size,
                "jpeg_sampling": jpeg_sampling(path),
            }
        )
    missing = sorted(set(labels) - seen)
    if missing:
        raise ValueError(f"RGB-no-more index contains {len(missing)} files absent from the dataset; first={missing[:3]}")
    return entries


def validate_galp_label_map(label_map_json: Path, full_entries: list[dict[str, Any]]) -> dict[str, Any]:
    payload = json.loads(label_map_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != "galp_rgbnomore_label_map_v1":
        raise ValueError(f"unexpected GALP label sidecar format: {label_map_json}")
    labels = payload.get("labels")
    expected = [entry["label"] for entry in full_entries]
    if labels != expected:
        mismatch = next(
            (index for index, (actual, wanted) in enumerate(zip(labels or [], expected)) if actual != wanted),
            min(len(labels or []), len(expected)),
        )
        raise ValueError(f"GALP label sidecar/order mismatch at image_id {mismatch}: {label_map_json}")
    if payload.get("image_count") != len(full_entries):
        raise ValueError(f"GALP label sidecar image_count does not match dataset: {label_map_json}")
    sample_ids = payload.get("sample_ids")
    expected_sample_ids = [entry["sample_id"] for entry in full_entries]
    if sample_ids != expected_sample_ids:
        mismatch = next(
            (
                index
                for index, (actual, wanted) in enumerate(zip(sample_ids or [], expected_sample_ids))
                if actual != wanted
            ),
            min(len(sample_ids or []), len(expected_sample_ids)),
        )
        raise ValueError(f"GALP sample identity/order mismatch at image_id {mismatch}: {label_map_json}")
    return {
        "path": str(label_map_json.resolve()),
        "sha256": sha256_file(label_map_json),
        "format": payload["format"],
        "image_count": payload["image_count"],
        "order_validation": "all_sample_ids_and_labels_match_sorted_jpeg_order",
    }


def build_manifest(
    *,
    data_root: Path,
    split: str,
    index_csv: Path,
    galp_label_map_json: Path,
    sample_count: int,
    seed: int,
    output: Path,
) -> tuple[dict[str, Any], str]:
    full_entries = collect_dataset(data_root, split, index_csv)
    eligible_entries = [entry for entry in full_entries if entry["jpeg_sampling"] in ("4:2:0", "4:4:4")]
    if sample_count <= 0 or sample_count > len(eligible_entries):
        raise ValueError(f"sample_count must be in [1,{len(eligible_entries)}], got {sample_count}")
    label_map = validate_galp_label_map(galp_label_map_json, full_entries)

    full_index_rows = [
        {
            "sample_id": entry["sample_id"],
            "label": entry["label"],
            "galp_image_id": entry["galp_image_id"],
            "size_bytes": entry["size_bytes"],
            "jpeg_sampling": entry["jpeg_sampling"],
        }
        for entry in full_entries
    ]
    order = list(range(len(eligible_entries)))
    random.Random(seed).shuffle(order)
    selected: list[dict[str, Any]] = []
    for ordinal, full_index in enumerate(order[:sample_count]):
        entry = dict(eligible_entries[full_index])
        entry["ordinal"] = ordinal
        fingerprint = fingerprint_file(Path(entry["path"]))
        entry["size_bytes"] = fingerprint["size_bytes"]
        entry["sha256"] = fingerprint["sha256"]
        entry["file_identity"] = fingerprint["file_identity"]
        selected.append(entry)

    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_root": str(data_root.resolve()),
        "split": split,
        "source_index_csv": str(index_csv.resolve()),
        "source_index_sha256": sha256_file(index_csv),
        "full_dataset_size": len(full_entries),
        "eligible_dataset_size": len(eligible_entries),
        "full_index_sha256": sha256_json(full_index_rows),
        "galp_label_map": label_map,
        "selection": {
            "algorithm": "python_random_v1_full_permutation_prefix",
            "seed": seed,
            "sample_count": sample_count,
            "order": "manifest_ordinal",
            "eligibility": "three-component JPEG with 4:2:0 or 4:4:4 sampling (intersection supported by all four pipelines)",
        },
        "samples": selected,
    }
    manifest_hash = sha256_json(payload)
    write_json(output, payload)
    return payload, manifest_hash


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--galp-label-map-json", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload, manifest_hash = build_manifest(
        data_root=args.data_root,
        split=args.split,
        index_csv=args.index_csv,
        galp_label_map_json=args.galp_label_map_json,
        sample_count=args.sample_count,
        seed=args.seed,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "sample_manifest": str(args.output.resolve()),
                "manifest_sha256": manifest_hash,
                "samples": len(payload["samples"]),
                "full_dataset_size": payload["full_dataset_size"],
                "eligible_dataset_size": payload["eligible_dataset_size"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
