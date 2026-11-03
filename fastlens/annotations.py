"""Task annotations bound to a FastLens store's immutable physical image order.

This module uses only the Python standard library. Labels remain JSON data;
classification and other task semantics belong to the application.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


SCHEMA = "fastlens-samples-v2"
DIRECTORY = "annotations"
SOURCE_ORDER = "geometry/source_order.sha256"
SHUFFLE_ALGORITHM = "fastlens-shuffle-v1"


@dataclass
class InputRecord:
    source_path: Path
    sample_id: str
    label: object


def shuffle_records(records: list[InputRecord], seed: int) -> None:
    """Versioned Fisher-Yates, independent of Python's random implementation.

    For bounds N..2, draw uint64 big-endian values from the first eight bytes
    of SHA256(b"fastlens-shuffle-v1\\0" + uint64_be(seed) + uint64_be(counter)).
    Counter starts at zero and advances for every draw, including rejections.
    Reject values >= floor(2**64 / bound) * bound, then swap bound-1 with
    value % bound. Seed is an unsigned 64-bit integer; the input order matters.
    """
    if not 0 <= seed < 1 << 64:
        raise ValueError("shuffle seed must be in [0, 2**64-1]")
    prefix = SHUFFLE_ALGORITHM.encode("ascii") + b"\0" + seed.to_bytes(8, "big")
    counter = 0
    for bound in range(len(records), 1, -1):
        limit = (1 << 64) - (1 << 64) % bound
        while True:
            value = int.from_bytes(hashlib.sha256(prefix + counter.to_bytes(8, "big")).digest()[:8], "big")
            counter += 1
            if value < limit:
                break
        other = value % bound
        records[bound - 1], records[other] = records[other], records[bound - 1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_order(store: Path) -> str:
    """Read the runtime order binding without requiring conversion records."""
    path = store / SOURCE_ORDER
    value = path.read_text(encoding="ascii").rstrip("\n")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"invalid SHA-256 in {path}")
    return value


def order_sha256(sample_ids) -> str:
    """Hash successive uint64 big-endian UTF-8 byte lengths and sample-ID bytes."""
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        data = sample_id.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _json(text: str):
    value = json.loads(text)
    # Also rejects overflow such as 1e999, in addition to NaN and Infinity.
    json.dumps(value, allow_nan=False)
    return value


def physical_paths(store: Path, image_count: int) -> list[Path]:
    source = store / "conversion/sources.txt"
    if not source.is_file():
        raise ValueError("missing conversion/sources.txt; building annotations requires the original physical-order source list")
    if file_sha256(source) != read_source_order(store):
        raise ValueError("conversion/sources.txt disagrees with the store's physical source order")
    with source.open(encoding="utf-8") as stream:
        lines = [line.rstrip("\r\n") for line in stream]
    paths = [Path(line) for line in lines]
    if len(paths) != image_count or not paths:
        raise ValueError("conversion/sources.txt image count disagrees with manifest")
    if any(not line or not path.is_absolute() for line, path in zip(lines, paths)):
        raise ValueError("conversion/sources.txt must contain one absolute source path per image")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate path in conversion/sources.txt")
    return paths


def _class_mapping(value) -> dict[str, int]:
    if (not isinstance(value, dict) or not value
            or any(type(label) is not int or label < 0 for label in value.values())
            or len(set(value.values())) != len(value)):
        raise ValueError("class map must contain distinct nonnegative integer labels")
    return value


def prepare_records(paths: list[Path], *, image_root: Path | None = None,
                    labels_from_folders: bool = False, labels_jsonl: Path | None = None,
                    labels_csv: Path | None = None, class_map: Path | None = None
                    ) -> tuple[list[InputRecord], dict | None]:
    """Join a complete annotation source to paths before any physical reordering.

    JSONL paths are relative to the JSONL file. CSV paths are relative to
    image_root, or to the CSV file when no root is supplied. Explicit sample_id
    fields allow sources spanning multiple roots. No JPEG content is read.
    """
    if sum((labels_from_folders, labels_jsonl is not None, labels_csv is not None)) > 1:
        raise ValueError("choose one annotation source")
    if class_map is not None and not labels_from_folders:
        raise ValueError("--class-map requires --labels-from-folders")
    root = image_root.resolve() if image_root is not None else None
    annotations = {}
    if labels_jsonl is not None or labels_csv is not None:
        source = labels_jsonl if labels_jsonl is not None else labels_csv
        base = source.resolve().parent if labels_jsonl is not None else (root or source.resolve().parent)
        with source.open(encoding="utf-8", newline="") as stream:
            if labels_jsonl is not None:
                rows = (_json(line) for line in stream if line.strip())
            else:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None or not {"Filepath", "Label"} <= set(reader.fieldnames):
                    raise ValueError("CSV requires Filepath,Label columns (optional sample_id)")
                rows = reader
            for number, row in enumerate(rows, 1):
                if labels_csv is not None:
                    try:
                        label = int(row["Label"])
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"{source}: row {number}: Label must be a nonnegative integer") from error
                    if label < 0:
                        raise ValueError(f"{source}: row {number}: Label must be nonnegative")
                    row = {"path": row["Filepath"], "label": label,
                           **({"sample_id": row["sample_id"]} if "sample_id" in row else {})}
                if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                        or not row["path"] or "label" not in row):
                    raise ValueError(f"{source}: row {number}: expected path and label fields")
                path = (base / row["path"]).resolve()
                if path in annotations:
                    raise ValueError(f"duplicate annotation path: {path}")
                annotations[path] = row
        unknown = annotations.keys() - set(paths)
        if unknown:
            raise ValueError(f"unknown annotation path: {min(unknown)}")
        missing = set(paths) - annotations.keys()
        if missing:
            raise ValueError(f"missing annotation for: {min(missing)}")

    classes = None
    if labels_from_folders:
        if root is None:
            raise ValueError("--labels-from-folders requires --image-root (or a conversion input directory)")
        folder_names = []
        for path in paths:
            if not path.is_relative_to(root) or len(path.relative_to(root).parts) < 2:
                raise ValueError(f"folder labels require paths beneath ROOT/CLASS/: {path}")
            folder_names.append(path.relative_to(root).parts[0])
        classes = (_class_mapping(_json(class_map.read_text(encoding="utf-8"))) if class_map is not None
                   else {name: index for index, name in enumerate(sorted(set(folder_names)))})
        missing = set(folder_names) - classes.keys()
        if missing:
            raise ValueError(f"classes absent from fixed class map: {', '.join(sorted(missing))}")

    records = []
    seen = set()
    for index, path in enumerate(paths):
        row = annotations.get(path, {})
        sample_id = row.get("sample_id")
        if "sample_id" not in row:
            if root is None or not path.is_relative_to(root):
                raise ValueError(f"provide --image-root or an explicit sample_id for {path}")
            sample_id = path.relative_to(root).as_posix()
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"sample_id must be a unique nonempty string: {sample_id!r}")
        seen.add(sample_id)
        label = classes[folder_names[index]] if labels_from_folders else row.get("label")
        records.append(InputRecord(path, sample_id, label))
    return records, classes


def publish_annotations(store: Path, records: list[InputRecord], classes: dict | None,
                        *, label_source: str | None, shuffle: dict | None = None) -> dict:
    """Publish a complete annotation directory once; never replace existing data."""
    destination = store / DIRECTORY
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"refusing to replace existing annotations: {destination}")
    paths = physical_paths(store, len(records))
    if paths != [record.source_path for record in records]:
        raise ValueError("annotation order disagrees with conversion/sources.txt")
    stage = Path(tempfile.mkdtemp(prefix=".annotations-staging-", dir=store))
    try:
        with (stage / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for image_id, record in enumerate(records):
                stream.write(json.dumps({"image_id": image_id, "sample_id": record.sample_id,
                                         "label": record.label}, allow_nan=False) + "\n")
        if classes is not None:
            (stage / "class_to_idx.json").write_text(json.dumps(classes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metadata = {
            "schema_version": SCHEMA, "image_count": len(records),
            "manifest_sha256": file_sha256(store / "manifest.bin"),
            "source_order_sha256": read_source_order(store),
            "samples_sha256": file_sha256(stage / "samples.jsonl"),
            "sample_order_sha256": order_sha256(record.sample_id for record in records),
            "shuffle": shuffle, "label_source": label_source,
            "class_mapping": "class_to_idx.json" if classes is not None else None,
            "class_mapping_sha256": file_sha256(stage / "class_to_idx.json") if classes is not None else None,
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Check the written representation before making the directory visible.
        load_annotations(store, directory=stage)
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"refusing to replace existing annotations: {destination}")
        stage.rename(destination)
        return metadata
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_annotations(store: Path, *, directory: Path | None = None) -> tuple[list[dict], dict]:
    """Validate bindings and return physical-order samples plus metadata.

    Source JPEGs, FLS payloads, torch and native tools are not needed here. These
    hashes detect accidental mismatches, not maliciously rewritten metadata.
    """
    directory = directory if directory is not None else store / DIRECTORY
    metadata = _json((directory / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA:
        raise ValueError("unsupported samples metadata schema")
    count = metadata.get("image_count")
    if type(count) is not int or count <= 0:
        raise ValueError("invalid annotation image_count")
    if metadata.get("source_order_sha256") != read_source_order(store):
        raise ValueError("annotation physical source order SHA-256 mismatch")
    for name, path in (("manifest", store / "manifest.bin"), ("samples", directory / "samples.jsonl")):
        if file_sha256(path) != metadata.get(name + "_sha256"):
            raise ValueError(f"annotation {name} SHA-256 mismatch")
    with (directory / "samples.jsonl").open(encoding="utf-8") as stream:
        samples = [_json(line) for line in stream]
    if len(samples) != count:
        raise ValueError("annotation sample count mismatch")
    seen = set()
    for image_id, sample in enumerate(samples):
        if (not isinstance(sample, dict) or type(sample.get("image_id")) is not int
                or sample["image_id"] != image_id or "label" not in sample):
            raise ValueError("samples must contain consecutive physical image_id values and labels")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("samples must contain unique nonempty sample_id strings")
        seen.add(sample_id)
    if order_sha256(sample["sample_id"] for sample in samples) != metadata.get("sample_order_sha256"):
        raise ValueError("annotation sample order SHA-256 mismatch")
    mapping = metadata.get("class_mapping")
    if mapping is not None:
        if mapping != "class_to_idx.json":
            raise ValueError("unsupported class_mapping path")
        if file_sha256(directory / mapping) != metadata.get("class_mapping_sha256"):
            raise ValueError("annotation class mapping SHA-256 mismatch")
        classes = _class_mapping(_json((directory / mapping).read_text(encoding="utf-8")))
        values = set(classes.values())
        if any(type(sample["label"]) is not int or sample["label"] not in values for sample in samples):
            raise ValueError("sample label is absent from class mapping")
    return samples, metadata
