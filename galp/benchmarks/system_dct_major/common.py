#!/usr/bin/env python3
"""Dependency-light contracts and artifacts for the DCT-major benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import struct
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


CONTRACT_SCHEMA = "galp_dct_major_contract_v1"
SAMPLE_MANIFEST_SCHEMA = "galp_dct_major_samples_v1"
PIPELINE_RESULT_SCHEMA = "galp_dct_major_pipeline_v1"
SUMMARY_SCHEMA = "galp_dct_major_summary_v1"
BLOCK_MAJOR_RUNTIME_PROFILE = "block-major-p4-scheduled-bounded-110-v1"

PIPELINES = (
    "dct_major_pushdown",
    "rgbnomore",
    "dali",
    "pytorch",
)
DEFAULT_PIPELINES = (
    "dct_major_pushdown",
    "rgbnomore",
    "dali",
    "pytorch",
)
WORKLOADS = ("feature-extraction", "evaluation")
JPEG_SUFFIXES = {".jpg", ".jpeg", ".jpe"}

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RGBNOMORE_BENCHMARK_ROOT = REPO_ROOT / "galp/benchmarks/system_rgbnomore"
if str(RGBNOMORE_BENCHMARK_ROOT) not in sys.path:
    # Keep this benchmark's own ``diagnostics`` and other top-level modules
    # ahead of the sibling RGB-no-more package while exposing ``shared``.
    sys.path.append(str(RGBNOMORE_BENCHMARK_ROOT))

from shared.manifest_contract import (  # noqa: E402
    JPEG_DCT_MANIFEST_CONTRACTS,
    MANIFEST_MAGIC,
    canonical_extension_fields,
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def fingerprint_file(path: Path, *, hash_contents: bool = True) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    identity = file_identity(resolved)
    result: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": identity["size_bytes"],
        "file_identity": identity,
    }
    if hash_contents:
        result["sha256"] = sha256_file(resolved)
    return result


def source_fingerprints(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted({item.resolve(strict=True) for item in paths}):
        result[str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)] = fingerprint_file(path)
    return result


def parse_manifest(manifest_path: Path) -> dict[str, Any]:
    """Parse the stable sharded JPEG-DCT manifest without loading CUDA."""

    manifest_path = manifest_path.resolve(strict=True)
    data = manifest_path.read_bytes()
    require(data[: len(MANIFEST_MAGIC)] == MANIFEST_MAGIC, f"unexpected GALP manifest magic: {manifest_path}")
    offset = 8

    def take(fmt: str) -> tuple[int, ...]:
        nonlocal offset
        size = struct.calcsize(fmt)
        require(offset + size <= len(data), f"truncated GALP manifest: {manifest_path}")
        values = struct.unpack_from(fmt, data, offset)
        offset += size
        return values

    def take_string() -> str:
        nonlocal offset
        (size,) = take("<I")
        require(offset + size <= len(data), f"truncated GALP manifest string: {manifest_path}")
        value = data[offset : offset + size].decode("utf-8")
        offset += size
        return value

    version, _reserved, rowgroup_vectors, rowgroups_per_shard, image_count = take("<IHIIQ")
    (shard_count,) = take("<I")
    shards: list[dict[str, Any]] = []
    root = manifest_path.parent
    for _ in range(shard_count):
        (
            shard_id,
            first_global_image_index,
            shard_image_count,
            real_row_count,
            padding_row_count,
            physical_row_count,
            rowgroup_count,
            block_group_count,
        ) = take("<IQIQQQII")
        fls_file_size, metadata_file_size = take("<QQ")
        fls_file_name = take_string()
        metadata_file_name = take_string()
        fls_path = (root / fls_file_name).resolve()
        metadata_path = (root / metadata_file_name).resolve()
        require(fls_path.is_relative_to(root), f"FLS payload escapes manifest root: {fls_file_name}")
        require(metadata_path.is_relative_to(root), f"metadata payload escapes manifest root: {metadata_file_name}")
        shards.append(
            {
                "shard_id": shard_id,
                "first_global_image_index": first_global_image_index,
                "image_count": shard_image_count,
                "real_row_count": real_row_count,
                "padding_row_count": padding_row_count,
                "physical_row_count": physical_row_count,
                "rowgroup_count": rowgroup_count,
                "block_group_count": block_group_count,
                "fls_file_size": fls_file_size,
                "metadata_file_size": metadata_file_size,
                "fls_file_name": fls_file_name,
                "metadata_file_name": metadata_file_name,
                "fls_path": str(fls_path),
                "metadata_path": str(metadata_path),
            }
        )
    version_contract = JPEG_DCT_MANIFEST_CONTRACTS.get(version)
    physical_layout = version_contract.physical_layout if version_contract is not None else "unknown"
    compact: dict[str, Any] | None = None
    if version_contract is not None and version_contract.required_extension_magic is not None:
        require(
            offset < len(data),
            f"GALP manifest version {version} requires its canonical Compact-v3 descriptor extension: {manifest_path}",
        )
    if offset < len(data):
        require(
            version_contract is not None and version_contract.required_extension_magic is not None,
            f"unexpected GALP manifest extension for version {version}: {manifest_path}",
        )
        extension_magic = version_contract.required_extension_magic
        require(
            offset + len(extension_magic) <= len(data),
            f"truncated GALP Compact-v3 manifest extension: {manifest_path}",
        )
        require(
            data[offset : offset + len(extension_magic)] == extension_magic,
            f"unexpected GALP Compact-v3 manifest extension: {manifest_path}",
        )
        offset += len(extension_magic)
        declared_physical_layout = take_string()
        descriptor_kind = take_string()
        (vector_size,) = take("<I")
        spatial_order = take_string()
        (spatial_order_id,) = take("<H")
        (extension_shard_count,) = take("<I")
        require(
            extension_shard_count == shard_count,
            f"GALP Compact-v3 extension shard count mismatch: {manifest_path}",
        )
        shard_contracts: list[dict[str, int]] = []
        for expected in shards:
            shard_id, payload_size, payload_crc64, compact_size, source_size = take("<IQQQQ")
            require(
                shard_id == int(expected["shard_id"]),
                f"GALP Compact-v3 extension shard id mismatch: {manifest_path}",
            )
            shard_contracts.append(
                {
                    "shard_id": shard_id,
                    "compressed_payload_bytes": payload_size,
                    "compressed_payload_crc64": payload_crc64,
                    "compact_descriptor_bytes": compact_size,
                    "source_descriptor_bytes": source_size,
                }
            )
        for label, actual, expected_value in canonical_extension_fields(
            version_contract,
            physical_layout=declared_physical_layout,
            descriptor_kind=descriptor_kind,
            vector_size=vector_size,
            spatial_order=spatial_order,
            spatial_order_id=spatial_order_id,
            rowgroup_vectors=rowgroup_vectors,
        ):
            if expected_value is not None:
                require(
                    actual == expected_value,
                    f"GALP manifest version {version} has non-canonical {label}: "
                    f"expected {expected_value!r}, got {actual!r}: {manifest_path}",
                )
        compact = {
            "physical_layout": declared_physical_layout,
            "descriptor_kind": descriptor_kind,
            "vector_size": vector_size,
            "spatial_order": spatial_order,
            "spatial_order_id": spatial_order_id,
            "shards": shard_contracts,
        }
    require(offset == len(data), f"GALP manifest has trailing bytes: {manifest_path}")
    return {
        "path": str(manifest_path),
        "version": version,
        "physical_layout": physical_layout,
        "rowgroup_vectors": rowgroup_vectors,
        "rowgroups_per_shard": rowgroups_per_shard,
        "image_count": image_count,
        "shard_count": shard_count,
        "shards": shards,
        "compact": compact,
    }


def manifest_snapshot(manifest_path: Path, *, hash_payloads: bool) -> dict[str, Any]:
    parsed = parse_manifest(manifest_path)
    payloads: list[dict[str, Any]] = []
    for shard in parsed["shards"]:
        for kind, path_key, size_key in (
            ("fls", "fls_path", "fls_file_size"),
            ("metadata", "metadata_path", "metadata_file_size"),
        ):
            path = Path(shard[path_key])
            require(path.is_file(), f"GALP {kind} payload is missing: {path}")
            identity = file_identity(path)
            require(
                identity["size_bytes"] == int(shard[size_key]),
                f"GALP {kind} payload size differs from manifest: {path}",
            )
            item = fingerprint_file(path, hash_contents=hash_payloads)
            item.update({"kind": kind, "shard_id": int(shard["shard_id"])})
            payloads.append(item)
        if int(parsed["version"]) == 2:
            vector_bundle = Path(shard["fls_path"]).with_suffix(".svb")
            if vector_bundle.is_file():
                item = fingerprint_file(vector_bundle, hash_contents=hash_payloads)
                item.update(
                    {
                        "kind": "legacy-v2-sparse-vector-bundle",
                        "shard_id": int(shard["shard_id"]),
                    }
                )
                payloads.append(item)
    return {
        "manifest": fingerprint_file(Path(parsed["path"])),
        "header": {key: value for key, value in parsed.items() if key not in {"path", "shards"}},
        "payload_hash_policy": "sha256" if hash_payloads else "identity-and-manifest-size",
        "payloads": payloads,
        "persistent_bytes": sum(int(item["size_bytes"]) for item in payloads),
    }


def load_label_map(path: Path, expected_images: int | None = None) -> dict[str, Any]:
    payload = read_json(path)
    require(isinstance(payload, dict), f"label map is not an object: {path}")
    require(payload.get("format") == "galp_rgbnomore_label_map_v1", f"bad label map format: {path}")
    labels = payload.get("labels")
    require(isinstance(labels, list) and all(isinstance(item, int) for item in labels), f"bad labels: {path}")
    require(payload.get("image_count") == len(labels), f"label map image_count mismatch: {path}")
    if expected_images is not None:
        require(len(labels) == expected_images, f"label map has {len(labels)} images, expected {expected_images}")
    sample_ids = payload.get("sample_ids")
    require(
        sample_ids is None
        or (isinstance(sample_ids, list) and len(sample_ids) in {0, len(labels)}),
        f"bad sample_ids in label map: {path}",
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "labels": [int(item) for item in labels],
        "sample_ids": [str(item).replace("\\", "/") for item in (sample_ids or [])],
    }


def collect_sequential_samples(
    *,
    data_root: Path,
    split: str,
    label_map_json: Path,
    expected_images: int,
    sample_count: int,
    hash_samples: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the exact physical-image-order view; no sampler permutation exists."""

    data_root = data_root.resolve(strict=True)
    split_root = (data_root / split).resolve(strict=True)
    label_map = load_label_map(label_map_json, expected_images)
    logical_paths = sorted(
        path
        for path in split_root.rglob("*")
        if path.is_file() and path.suffix.lower() in JPEG_SUFFIXES
    )
    require(
        len(logical_paths) == expected_images,
        f"found {len(logical_paths)} JPEGs below {split_root}, expected {expected_images}",
    )
    require(0 < sample_count <= expected_images, f"sample_count must be in [1,{expected_images}]")
    samples: list[dict[str, Any]] = []
    label_sample_ids = label_map["sample_ids"]
    for image_id, logical_path in enumerate(logical_paths[:sample_count]):
        sample_id = logical_path.relative_to(data_root).as_posix()
        path = logical_path.resolve(strict=True)
        if label_sample_ids:
            require(
                sample_id == label_sample_ids[image_id],
                f"physical image order differs from label map at {image_id}: {sample_id} != {label_sample_ids[image_id]}",
            )
        identity = file_identity(path)
        item: dict[str, Any] = {
            "ordinal": image_id,
            "galp_image_id": image_id,
            "sample_id": sample_id,
            "path": str(path),
            "label": int(label_map["labels"][image_id]),
            "size_bytes": identity["size_bytes"],
            "file_identity": identity,
        }
        if hash_samples:
            item["sha256"] = sha256_file(path)
        samples.append(item)
    provenance = {
        "data_root": str(data_root),
        "split": split,
        "full_image_count": expected_images,
        "selected_image_count": sample_count,
        "order": "galp_image_id_ascending",
        "shuffle": False,
        "hash_samples": hash_samples,
        "label_map": {key: value for key, value in label_map.items() if key not in {"labels", "sample_ids"}},
    }
    return samples, provenance


def write_sample_manifest(path: Path, samples: Sequence[dict[str, Any]], provenance: dict[str, Any]) -> str:
    payload = {
        "schema_version": SAMPLE_MANIFEST_SCHEMA,
        "provenance": provenance,
        "samples": list(samples),
    }
    digest = sha256_json(payload)
    write_json(path, payload)
    return digest


def load_sample_manifest(path: Path, expected_sha256: str | None = None) -> list[dict[str, Any]]:
    payload = read_json(path)
    require(isinstance(payload, dict), "sample manifest must be an object")
    require(payload.get("schema_version") == SAMPLE_MANIFEST_SCHEMA, "bad sample manifest schema")
    if expected_sha256 is not None:
        require(sha256_json(payload) == expected_sha256, "sample manifest hash changed")
    samples = payload.get("samples")
    require(isinstance(samples, list) and samples, "sample manifest contains no samples")
    for index, sample in enumerate(samples):
        require(isinstance(sample, dict), f"sample {index} is not an object")
        require(int(sample.get("ordinal", -1)) == index, f"sample ordinal is not sequential at {index}")
        require(int(sample.get("galp_image_id", -1)) == index, f"GALP image id is not sequential at {index}")
    return samples


def write_canonical_index(path: Path, samples: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("Filepath", "Label"))
        writer.writeheader()
        writer.writerows({"Filepath": item["path"], "Label": item["label"]} for item in samples)


def chunked(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    require(size > 0, "chunk size must be positive")
    for begin in range(0, len(items), size):
        yield list(items[begin : begin + size])


def selected_batches(contract: dict[str, Any], samples: Sequence[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    execution = contract["execution"]
    batches = list(chunked(samples, int(execution["batch_size"])))
    warmup_count = int(execution["warmup_batches"])
    measurement_count = int(execution["measurement_batches"])
    require(len(batches) >= warmup_count + measurement_count, "sample manifest is too short for configured batches")
    return batches[:warmup_count], batches[warmup_count : warmup_count + measurement_count]


def sample_trace(batches: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, int]]:
    return [
        {
            "ordinal": int(sample["ordinal"]),
            "galp_image_id": int(sample["galp_image_id"]),
            "label": int(sample["label"]),
        }
        for batch in batches
        for sample in batch
    ]


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    require(bool(values), "cannot summarize an empty distribution")
    ordered = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    mean = statistics.fmean(ordered)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean,
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "cv_population": statistics.pstdev(ordered) / mean if len(ordered) > 1 and mean else 0.0,
    }


def load_contract(path: Path) -> dict[str, Any]:
    contract = read_json(path)
    require(isinstance(contract, dict), "contract must be an object")
    require(contract.get("schema_version") == CONTRACT_SCHEMA, f"contract schema must be {CONTRACT_SCHEMA}")
    for section in ("dataset", "execution", "workload", "models", "preprocess", "pipelines", "semantic_validation"):
        require(isinstance(contract.get(section), dict), f"contract.{section} must be an object")
    execution = contract["execution"]
    for name, minimum in (
        ("batch_size", 1),
        ("workers", 0),
        ("warmup_batches", 0),
        ("measurement_batches", 1),
        ("repeats", 1),
        ("seed", 0),
    ):
        value = execution.get(name)
        require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, f"execution.{name} must be >= {minimum}")
    require(execution.get("shuffle") is False, "DCT-major benchmark requires shuffle=false")
    require(execution.get("drop_last") is False, "DCT-major benchmark requires drop_last=false")
    require(execution.get("precision") == "fp32", "the canonical DCT-major contract is FP32")
    require(contract["workload"].get("kind") in WORKLOADS, f"workload.kind must be one of {WORKLOADS}")
    enabled = contract["pipelines"].get("enabled")
    require(isinstance(enabled, list) and enabled, "pipelines.enabled must be a non-empty list")
    require(len(enabled) == len(set(enabled)), "pipelines.enabled contains duplicates")
    require(all(item in PIPELINES for item in enabled), f"unknown pipeline; expected subset of {PIPELINES}")
    if "dct_major_pushdown" in enabled:
        galp = contract["pipelines"].get("dct_major_pushdown")
        require(isinstance(galp, dict), "pipelines.dct_major_pushdown must be an object")
        require(
            galp.get("runtime_profile") == BLOCK_MAJOR_RUNTIME_PROFILE,
            "dct_major_pushdown must use the canonical block-major runtime profile",
        )
        internal_fields = {
            "cache_capacity_mib",
            "plan_cache_capacity",
            "decode_batch_rowgroups",
            "decode_workset_capacity_mib",
            "rowgroup_prefetch_depth",
            "rowgroup_prefetch_workers",
            "rowgroup_prefetch_min_decode_batches",
            "enable_planless_execution",
            "scheduling_policy",
            "transform_blocks_per_launch",
            "transform_ctas_per_launch",
            "use_low_priority_streams",
            "block_major_double_buffer",
            "crop_execution_mode",
            "bounded_read_amplification_cap",
            "bounded_read_local_amplification_cap",
            "bounded_read_max_run_bytes",
            "segment_mode",
            "output_prefetch_policy",
        }
        leaked = sorted(internal_fields.intersection(galp))
        require(not leaked, f"dct_major_pushdown exposes native runtime fields: {leaked}")
    manifest_path = Path(str(contract["dataset"].get("sample_manifest", "")))
    require(manifest_path.is_file(), f"sample manifest does not exist: {manifest_path}")
    load_sample_manifest(manifest_path, str(contract["dataset"]["sample_manifest_sha256"]))
    return contract
