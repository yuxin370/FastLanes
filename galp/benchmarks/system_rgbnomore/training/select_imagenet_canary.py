#!/usr/bin/env python3
"""Extract a deterministic, sampling-compatible ImageNet training canary.

The selector reads the canonical RGB-no-more ImageNet index and one tar archive
per WordNet class.  For every label it keeps the first requested number of
JPEGs whose component sampling is supported by the Direct-DCT training
contract.  Selection happens before compression, so the physical GALP image
population and the 900/100 or 9000/1000 logical split have the same cardinality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Sequence


SCHEMA_VERSION = "galp-training-imagenet-canary-selection-v1"
SUPPORTED_SAMPLING = frozenset(("4:4:4", "4:2:0"))
SOF_MARKERS = frozenset(
    (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF)
)


def _jpeg_sampling(stream: BinaryIO) -> str:
    if stream.read(2) != b"\xff\xd8":
        raise ValueError("input is not a JPEG stream")
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
            raise ValueError("invalid JPEG marker length")
        if marker not in SOF_MARKERS:
            stream.seek(length - 2, os.SEEK_CUR)
            continue
        segment = stream.read(length - 2)
        if len(segment) < 6:
            raise ValueError("short JPEG SOF segment")
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
        if y[0] == cb[0] and y[1] == cb[1] * 2:
            return "4:4:0"
        if y[0] == cb[0] * 4 and y[1] == cb[1]:
            return "4:1:1"
        return "unsupported"
    raise ValueError("JPEG SOF marker not found")


def _load_candidates(
    index_csv: Path, *, class_count: int
) -> tuple[dict[int, list[tuple[str, str]]], dict[int, str]]:
    candidates: dict[int, list[tuple[str, str]]] = defaultdict(list)
    wnid_for_label: dict[int, str] = {}
    seen_paths: set[str] = set()
    with index_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"Filepath", "Label"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{index_csv} must contain Filepath and Label columns")
        for row_number, row in enumerate(reader, start=2):
            relative = PurePosixPath(str(row["Filepath"]).strip())
            if len(relative.parts) != 3 or relative.parts[0] != "train":
                continue
            label = int(row["Label"])
            if label < 0 or label >= class_count:
                raise ValueError(f"{index_csv}:{row_number} has invalid label {label}")
            normalized = relative.as_posix()
            if normalized in seen_paths:
                raise ValueError(f"{index_csv}:{row_number} duplicates {normalized!r}")
            seen_paths.add(normalized)
            _, wnid, filename = relative.parts
            prior = wnid_for_label.setdefault(label, wnid)
            if prior != wnid:
                raise ValueError(
                    f"label {label} maps to both {prior!r} and {wnid!r}"
                )
            candidates[label].append((wnid, filename))
    missing = sorted(set(range(class_count)) - set(candidates))
    if missing:
        raise ValueError(f"ImageNet index is missing labels: {missing}")
    return dict(candidates), wnid_for_label


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _selection_sha256(records: Sequence[dict[str, Any]]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_canary(
    *,
    class_tar_root: Path,
    index_csv: Path,
    output_dir: Path,
    selected_index_csv: Path,
    output_json: Path,
    image_count: int,
    class_count: int = 1000,
) -> dict[str, Any]:
    if image_count <= 0 or image_count % class_count:
        raise ValueError(
            f"image_count must be a positive multiple of class_count={class_count}"
        )
    per_class = image_count // class_count
    class_tar_root = class_tar_root.resolve()
    index_csv = index_csv.resolve()
    output_dir = output_dir.resolve()
    selected_index_csv = selected_index_csv.resolve()
    output_json = output_json.resolve()
    if not class_tar_root.is_dir():
        raise FileNotFoundError(class_tar_root)
    if not index_csv.is_file():
        raise FileNotFoundError(index_csv)
    for target in (output_dir, selected_index_csv, output_json):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")

    candidates, wnid_for_label = _load_candidates(index_csv, class_count=class_count)
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    rejected_sampling: dict[str, int] = defaultdict(int)
    try:
        for label in range(class_count):
            wnid = wnid_for_label[label]
            archive = class_tar_root / f"{wnid}.tar"
            if not archive.is_file():
                raise FileNotFoundError(archive)
            selected_for_class = 0
            class_output = temporary / wnid
            class_output.mkdir()
            with tarfile.open(archive, mode="r:*") as tar:
                members = {
                    PurePosixPath(member.name).name: member
                    for member in tar.getmembers()
                    if member.isfile()
                }
                for candidate_wnid, filename in candidates[label]:
                    if candidate_wnid != wnid:
                        raise AssertionError((candidate_wnid, wnid))
                    member = members.get(filename)
                    if member is None:
                        raise FileNotFoundError(f"{archive}: missing {filename}")
                    source = tar.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"cannot read {archive}:{filename}")
                    with source:
                        sampling = _jpeg_sampling(source)
                    if sampling not in SUPPORTED_SAMPLING:
                        rejected_sampling[sampling] += 1
                        continue
                    source = tar.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"cannot reread {archive}:{filename}")
                    destination = class_output / filename
                    with source, destination.open("xb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    os.utime(destination, (member.mtime, member.mtime))
                    records.append(
                        {
                            "Filepath": f"train/{wnid}/{filename}",
                            "Label": label,
                            "jpeg_sampling": sampling,
                            "sha256": _sha256_file(destination),
                        }
                    )
                    selected_for_class += 1
                    if selected_for_class == per_class:
                        break
            if selected_for_class != per_class:
                raise RuntimeError(
                    f"label {label} ({wnid}) has only {selected_for_class} compatible JPEGs; "
                    f"requires {per_class}"
                )
            if (label + 1) % 100 == 0 or label + 1 == class_count:
                print(
                    f"selected classes={label + 1}/{class_count} "
                    f"images={len(records)}/{image_count}",
                    flush=True,
                )
        if len(records) != image_count:
            raise AssertionError((len(records), image_count))
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    selected_index_csv.parent.mkdir(parents=True, exist_ok=True)
    with selected_index_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Filepath", "Label"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {"Filepath": record["Filepath"], "Label": record["Label"]}
            )
    result = {
        "schema_version": SCHEMA_VERSION,
        "selection_algorithm": "index-order-first-compatible-per-class-v1",
        "class_tar_root": str(class_tar_root),
        "source_index_csv": str(index_csv),
        "source_index_sha256": _sha256_file(index_csv),
        "output_dir": str(output_dir),
        "selected_index_csv": str(selected_index_csv),
        "image_count": image_count,
        "class_count": class_count,
        "images_per_class": per_class,
        "supported_sampling": sorted(SUPPORTED_SAMPLING),
        "selected_sampling_counts": {
            sampling: sum(record["jpeg_sampling"] == sampling for record in records)
            for sampling in sorted(SUPPORTED_SAMPLING)
        },
        "rejected_sampling_counts": dict(sorted(rejected_sampling.items())),
        "selection_sha256": _selection_sha256(records),
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-tar-root", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-index-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--image-count", type=int, choices=(1000, 10000), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = select_canary(
        class_tar_root=args.class_tar_root,
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        selected_index_csv=args.selected_index_csv,
        output_json=args.output_json,
        image_count=args.image_count,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
