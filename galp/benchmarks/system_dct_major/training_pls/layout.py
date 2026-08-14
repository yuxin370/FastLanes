#!/usr/bin/env python3
"""Frozen virtual Physical Load Segment layout sidecar."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .recipe import sha256_json


LAYOUT_SCHEMA = "galp-physical-layout-plan-v2"
LAYOUT_VERSION = "virtual-pls-current-v1"
WRITER_CONTRACT_VERSION = "galp-block-major-writer-mapping-v1"
MAPPING_SCHEMA = "galp-physical-layout-samples-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _optional_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return None
    return pa, pq


def _parquet_helper_python() -> Path:
    candidates = (
        Path("/home/tangyuxin/transformers/.venv_transformers/bin/python"),
        Path("/home/tangyuxin/langchain-playground/rag/.lcenv/bin/python"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        completed = subprocess.run(
            [str(candidate), "-c", "import pyarrow, numpy"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode == 0:
            return candidate
    raise RuntimeError(
        "physical_layout_samples.parquet requires pyarrow. The experiment Python "
        "does not provide it and no local pyarrow helper environment was found."
    )


def _manifest_samples(path: Path) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"training manifest contains no samples: {path}")
    population = payload.get("population_count")
    if population is not None and int(population) != len(samples):
        raise ValueError(
            f"training manifest population_count={population} but contains {len(samples)} samples"
        )
    return payload, samples


def _validated_order(samples: Sequence[Mapping[str, Any]]) -> list[int]:
    seen_logical: set[str] = set()
    seen_image: set[int] = set()
    indexed: list[tuple[int, str, int]] = []
    for manifest_index, raw in enumerate(samples):
        logical_id = str(raw.get("logical_sample_id", ""))
        if not logical_id or logical_id in seen_logical:
            raise ValueError(
                f"sample {manifest_index} has a missing or duplicate logical_sample_id"
            )
        image_id = int(raw.get("galp_image_id", -1))
        if image_id < 0 or image_id in seen_image:
            raise ValueError(
                f"sample {manifest_index} has a missing or duplicate galp_image_id"
            )
        label = int(raw.get("label", -1))
        if label < 0:
            raise ValueError(f"sample {manifest_index} has an invalid label")
        seen_logical.add(logical_id)
        seen_image.add(image_id)
        indexed.append((image_id, logical_id, manifest_index))
    indexed.sort()
    return [manifest_index for _image_id, _logical_id, manifest_index in indexed]


def _write_mapping(
    path: Path,
    samples: Sequence[Mapping[str, Any]],
    order: Sequence[int],
    *,
    segment_images: int,
) -> None:
    columns: dict[str, list[Any]] = {
        "logical_sample_id": [],
        "label": [],
        "galp_image_id": [],
        "manifest_index": [],
        "planned_physical_position": [],
        "virtual_pls_id": [],
        "position_in_pls": [],
        "width": [],
        "height": [],
    }
    for planned_position, manifest_index in enumerate(order):
        raw = samples[manifest_index]
        columns["logical_sample_id"].append(str(raw["logical_sample_id"]))
        columns["label"].append(int(raw["label"]))
        columns["galp_image_id"].append(int(raw["galp_image_id"]))
        columns["manifest_index"].append(int(manifest_index))
        columns["planned_physical_position"].append(planned_position)
        columns["virtual_pls_id"].append(planned_position // segment_images)
        columns["position_in_pls"].append(planned_position % segment_images)
        columns["width"].append(int(raw.get("width", 0)))
        columns["height"].append(int(raw.get("height", 0)))
    runtime_path = path.with_name("physical_layout_samples.runtime.npz")
    runtime_temporary = runtime_path.with_name(
        f".{runtime_path.name}.{os.getpid()}.tmp"
    )
    arrays = {
        "logical_sample_id": np.asarray(columns["logical_sample_id"], dtype=np.str_),
        "label": np.asarray(columns["label"], dtype=np.int32),
        "galp_image_id": np.asarray(columns["galp_image_id"], dtype=np.int64),
        "manifest_index": np.asarray(columns["manifest_index"], dtype=np.int64),
        "planned_physical_position": np.asarray(
            columns["planned_physical_position"], dtype=np.int64
        ),
        "virtual_pls_id": np.asarray(columns["virtual_pls_id"], dtype=np.int32),
        "position_in_pls": np.asarray(columns["position_in_pls"], dtype=np.int32),
        "width": np.asarray(columns["width"], dtype=np.int32),
        "height": np.asarray(columns["height"], dtype=np.int32),
        "schema_version": np.asarray(MAPPING_SCHEMA),
        "writer_contract_version": np.asarray(WRITER_CONTRACT_VERSION),
    }
    try:
        with runtime_temporary.open("wb") as output:
            np.savez_compressed(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        os.replace(runtime_temporary, runtime_path)
    finally:
        if runtime_temporary.exists():
            runtime_temporary.unlink()

    optional = _optional_pyarrow()
    if optional is None:
        helper = Path(__file__).with_name("parquet_helper.py")
        completed = subprocess.run(
            [
                str(_parquet_helper_python()),
                str(helper),
                "--runtime-mapping",
                str(runtime_path),
                "--output",
                str(path),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "local pyarrow helper failed to write the Parquet sidecar: "
                + completed.stderr.strip()
            )
        return
    pa, pq = optional
    schema = pa.schema(
        [
            ("logical_sample_id", pa.string()),
            ("label", pa.int32()),
            ("galp_image_id", pa.int64()),
            ("manifest_index", pa.int64()),
            ("planned_physical_position", pa.int64()),
            ("virtual_pls_id", pa.int32()),
            ("position_in_pls", pa.int32()),
            ("width", pa.int32()),
            ("height", pa.int32()),
        ],
        metadata={
            b"schema_version": MAPPING_SCHEMA.encode("ascii"),
            b"writer_contract_version": WRITER_CONTRACT_VERSION.encode("ascii"),
        },
    )
    table = pa.Table.from_pydict(columns, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=["logical_sample_id"],
            write_statistics=True,
            row_group_size=64 * 1024,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_layout_plan(
    train_manifest: Path,
    output_dir: Path,
    *,
    segment_images: int = 1024,
    organization: str = "current",
    organization_seed: int = 20260810,
) -> dict[str, Any]:
    if segment_images <= 0:
        raise ValueError("segment_images must be positive")
    if organization != "current":
        raise ValueError(
            "the core model-effect experiment freezes only organization='current'"
        )
    train_manifest = train_manifest.resolve()
    if not train_manifest.is_file():
        raise FileNotFoundError(train_manifest)
    manifest_payload, samples = _manifest_samples(train_manifest)
    order = _validated_order(samples)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "physical_layout_samples.parquet"
    _write_mapping(mapping_path, samples, order, segment_images=segment_images)
    runtime_mapping_path = output_dir / "physical_layout_samples.runtime.npz"
    sample_count = len(order)
    segment_count = (sample_count + segment_images - 1) // segment_images
    tail = sample_count - (segment_count - 1) * segment_images
    plan: dict[str, Any] = {
        "schema_version": LAYOUT_SCHEMA,
        "layout_version": LAYOUT_VERSION,
        "dataset_manifest": str(train_manifest),
        "dataset_manifest_hash": sha256_file(train_manifest),
        "dataset_manifest_format": manifest_payload.get("format"),
        "dataset_population_count": manifest_payload.get("population_count"),
        "galp_manifest": manifest_payload.get("galp_manifest"),
        "ordering_algorithm": "current-galp-image-id-then-logical-id-v1",
        "organization": organization,
        "organization_seed": int(organization_seed),
        "target_pls_size": int(segment_images),
        "sample_count": sample_count,
        "virtual_pls_count": segment_count,
        "full_virtual_pls_count": sample_count // segment_images,
        "tail_virtual_pls_samples": tail,
        "sample_mapping_file": mapping_path.name,
        "sample_mapping_hash": sha256_file(mapping_path),
        "sample_mapping_format": "parquet",
        "runtime_mapping_file": runtime_mapping_path.name,
        "runtime_mapping_hash": sha256_file(runtime_mapping_path),
        "runtime_mapping_format": "numpy-npz-exact-parquet-mirror",
        "writer_contract_version": WRITER_CONTRACT_VERSION,
        "writer_order_semantics": (
            "physical writer consumes rows in planned_physical_position order and "
            "preserves logical_sample_id-to-virtual_pls_id membership"
        ),
        "runtime_membership_policy": (
            "virtual_pls_id is loaded from the frozen sidecar; runtime index//G is forbidden"
        ),
    }
    plan["layout_hash"] = sha256_json(plan)
    plan_path = output_dir / "physical_layout_plan.json"
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise ValueError(
                f"refusing to replace a different frozen layout plan: {plan_path}"
            )
    else:
        _atomic_json(plan_path, plan)
    return plan


@dataclass(frozen=True)
class LayoutMapping:
    plan_path: Path
    plan: dict[str, Any]
    logical_sample_ids: tuple[str, ...]
    labels: np.ndarray
    galp_image_ids: np.ndarray
    manifest_indices: np.ndarray
    virtual_pls_ids: np.ndarray
    positions_in_pls: np.ndarray
    widths: np.ndarray
    heights: np.ndarray
    positions_by_pls: tuple[np.ndarray, ...]

    @property
    def sample_count(self) -> int:
        return len(self.logical_sample_ids)

    @property
    def segment_images(self) -> int:
        return int(self.plan["target_pls_size"])

    @property
    def layout_hash(self) -> str:
        return str(self.plan["layout_hash"])


def load_layout_plan(path: Path) -> dict[str, Any]:
    path = path.resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != LAYOUT_SCHEMA:
        raise ValueError(f"unsupported physical layout plan schema: {path}")
    expected_hash = sha256_json({key: value for key, value in plan.items() if key != "layout_hash"})
    if plan.get("layout_hash") != expected_hash:
        raise ValueError(f"physical layout plan hash mismatch: {path}")
    mapping = path.parent / str(plan["sample_mapping_file"])
    if not mapping.is_file() or sha256_file(mapping) != plan["sample_mapping_hash"]:
        raise ValueError(f"physical layout sample mapping hash mismatch: {mapping}")
    runtime_mapping = path.parent / str(plan["runtime_mapping_file"])
    if (
        not runtime_mapping.is_file()
        or sha256_file(runtime_mapping) != plan["runtime_mapping_hash"]
    ):
        raise ValueError(f"physical layout runtime mapping hash mismatch: {runtime_mapping}")
    return plan


def _numpy_column(table: Any, name: str, dtype: Any) -> np.ndarray:
    values = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype)


def load_layout_mapping(path: Path) -> LayoutMapping:
    path = path.resolve()
    plan = load_layout_plan(path)
    runtime_path = path.parent / str(plan["runtime_mapping_file"])
    archive = np.load(runtime_path, allow_pickle=False)
    required = {
        "logical_sample_id",
        "label",
        "galp_image_id",
        "manifest_index",
        "planned_physical_position",
        "virtual_pls_id",
        "position_in_pls",
        "width",
        "height",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise ValueError(f"layout mapping is missing columns: {missing}")
    count = len(archive["logical_sample_id"])
    if count != int(plan["sample_count"]):
        raise ValueError("layout mapping row count differs from physical layout plan")
    planned = np.asarray(archive["planned_physical_position"], dtype=np.int64)
    if not np.array_equal(planned, np.arange(count, dtype=np.int64)):
        raise ValueError("planned_physical_position must be contiguous and file-ordered")
    pls_ids = np.asarray(archive["virtual_pls_id"], dtype=np.int32)
    positions_in_pls = np.asarray(archive["position_in_pls"], dtype=np.int32)
    segment_count = int(plan["virtual_pls_count"])
    positions_by_pls: list[np.ndarray] = []
    for pls_id in range(segment_count):
        positions = np.flatnonzero(pls_ids == pls_id).astype(np.int64, copy=False)
        if not len(positions):
            raise ValueError(f"layout mapping has an empty virtual PLS {pls_id}")
        expected_positions = np.arange(len(positions), dtype=np.int32)
        if not np.array_equal(positions_in_pls[positions], expected_positions):
            raise ValueError(f"layout mapping position_in_pls is invalid for PLS {pls_id}")
        positions_by_pls.append(positions)
    logical_ids = tuple(str(value) for value in archive["logical_sample_id"].tolist())
    if len(set(logical_ids)) != count:
        raise ValueError("layout mapping contains duplicate logical_sample_id values")
    return LayoutMapping(
        plan_path=path,
        plan=plan,
        logical_sample_ids=logical_ids,
        labels=np.asarray(archive["label"], dtype=np.int32),
        galp_image_ids=np.asarray(archive["galp_image_id"], dtype=np.int64),
        manifest_indices=np.asarray(archive["manifest_index"], dtype=np.int64),
        virtual_pls_ids=pls_ids,
        positions_in_pls=positions_in_pls,
        widths=np.asarray(archive["width"], dtype=np.int32),
        heights=np.asarray(archive["height"], dtype=np.int32),
        positions_by_pls=tuple(positions_by_pls),
    )


def validate_manifest_against_layout(
    mapping: LayoutMapping,
    *,
    manifest_path: Path,
    samples: Sequence[Any],
) -> None:
    manifest_path = manifest_path.resolve()
    if sha256_file(manifest_path) != mapping.plan["dataset_manifest_hash"]:
        raise ValueError("training manifest hash differs from the frozen layout plan")
    if len(samples) != mapping.sample_count:
        raise ValueError("training sample count differs from the frozen layout mapping")
    for planned_position in range(mapping.sample_count):
        manifest_index = int(mapping.manifest_indices[planned_position])
        sample = samples[manifest_index]
        if str(sample.logical_sample_id) != mapping.logical_sample_ids[planned_position]:
            raise ValueError(
                f"layout logical identity mismatch at planned position {planned_position}"
            )
        if int(sample.galp_image_id) != int(mapping.galp_image_ids[planned_position]):
            raise ValueError(
                f"layout GALP image identity mismatch at planned position {planned_position}"
            )


def mapping_rows(mapping: LayoutMapping) -> Iterable[dict[str, Any]]:
    for position in range(mapping.sample_count):
        yield {
            "logical_sample_id": mapping.logical_sample_ids[position],
            "label": int(mapping.labels[position]),
            "galp_image_id": int(mapping.galp_image_ids[position]),
            "manifest_index": int(mapping.manifest_indices[position]),
            "planned_physical_position": position,
            "virtual_pls_id": int(mapping.virtual_pls_ids[position]),
            "position_in_pls": int(mapping.positions_in_pls[position]),
            "width": int(mapping.widths[position]),
            "height": int(mapping.heights[position]),
        }
