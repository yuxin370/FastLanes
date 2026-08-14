#!/usr/bin/env python3
"""Compact-v3 contract adapter and independent-process acceptance runner.

This tool lives in the GALP JPEG-DCT boundary, parses the Compact-v3 trailer
strictly, refreshes immutable file fingerprints in an existing same-dataset
contract, and can execute alternating v2/v3 pipeline legs while sampling the
complete Linux process tree. Production contracts expose one ``galp`` pipeline;
the native runtime profile owns the chosen execution representation.

It intentionally does not turn missing evidence into a pass.  In particular,
the current inference pipeline stores latency distributions rather than the
ordered first-batch latency; the resulting summary reports that gate as
unverified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = REPO_ROOT / "galp/benchmarks/system_rgbnomore"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from shared.common import (  # noqa: E402
    GALP_PIPELINES,
    GALP_RUNTIME_IMPLEMENTATION_FIELDS,
    GALP_RUNTIME_PROFILE,
    canonical_pipeline_name,
    contract_pipeline_name,
)

ACCEPTANCE_GALP_PIPELINES = GALP_PIPELINES
DEFAULT_PIPELINE = REPO_ROOT / "galp/benchmarks/system_rgbnomore/inference/pipeline.py"
DEFAULT_PYTHON = Path("/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python")
DEFAULT_RUNTIME_AUDIT = (
    REPO_ROOT / "build/galp/tools/jpeg_dct/galp_compact_v3_runtime_audit"
)
MANIFEST_MAGIC = b"GJDCTSH1"
COMPACT_EXTENSION_MAGIC = b"GJDCCV31"
EXPECTED_LAYOUT = "image-major-vector-rowgroups"
EXPECTED_DESCRIPTOR_KIND = "galp-compact-v1"
EXPECTED_SPATIAL_ORDER = "tiled-z32"
EXPECTED_VECTOR_SIZE = 1024
EXPECTED_SPATIAL_ORDER_ID = 3
FLS_HEADER_BYTES = 24
FLS_FOOTER_BYTES = 24
COMPACT_DESCRIPTOR_MAGIC = b"GALPCV3\0"
COMPACT_DESCRIPTOR_HEADER_BYTES = 256
COMPACT_IMAGE_RECORD_BYTES = 32
COMPACT_COMPONENT_RECORD_BYTES = 32
COMPACT_ROWGROUP_RECORD_BYTES = 48
COMPACT_REQUIRED_FLAGS = 0x07
COMPACT_KNOWN_FLAGS = 0x3F
COMPACT_OPTIMIZED_FLAGS = 0x38
COMPACT_DENSE_COEFFICIENT_RANGES_FLAG = 0x20
STRICT_REFERENCE_100_IMAGE_BYTES = 26_914_204
PROJECTED_50K_LIMIT_BYTES = 13_244_000_000
MIB = 1024 * 1024


def _make_crc64_ecma_table() -> tuple[int, ...]:
    polynomial = 0x42F0E1EBA9EA3693
    mask = (1 << 64) - 1
    table = []
    for value in range(256):
        crc = value << 56
        for _ in range(8):
            crc = ((crc << 1) ^ polynomial) & mask if crc & (1 << 63) else (crc << 1) & mask
        table.append(crc)
    return tuple(table)


CRC64_ECMA_TABLE = _make_crc64_ecma_table()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_passing_boolean_gates(gates: dict[str, Any], label: str) -> None:
    failed = sorted(name for name, value in gates.items() if isinstance(value, bool) and not value)
    require(not failed, f"{label} failed gates: {', '.join(failed)}")


class Cursor:
    def __init__(self, data: bytes, label: str) -> None:
        self.data = data
        self.label = label
        self.offset = 0

    def take(self, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        require(self.offset + size <= len(self.data), f"truncated {self.label}")
        result = struct.unpack_from(fmt, self.data, self.offset)
        self.offset += size
        return result

    def bytes(self, size: int) -> bytes:
        require(size >= 0 and self.offset + size <= len(self.data), f"truncated {self.label}")
        result = self.data[self.offset : self.offset + size]
        self.offset += size
        return result

    def string(self) -> str:
        (size,) = self.take("<I")
        try:
            return self.bytes(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"invalid UTF-8 in {self.label}") from error

    def eof(self) -> bool:
        return self.offset == len(self.data)


@dataclass(frozen=True)
class PayloadSpec:
    kind: str
    relative_path: str
    path: Path
    expected_size: int


def _safe_member(root: Path, member: str, kind: str) -> tuple[str, Path]:
    relative = Path(member)
    require(member != "" and not relative.is_absolute(), f"unsafe absolute/empty {kind} path: {member!r}")
    require(".." not in relative.parts, f"unsafe parent traversal in {kind} path: {member!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    require(resolved.is_relative_to(resolved_root), f"{kind} path escapes manifest root: {member!r}")
    return resolved.relative_to(resolved_root).as_posix(), resolved


def _crc64_ecma_file_range(path: Path, offset: int, size: int) -> int:
    mask = (1 << 64) - 1
    crc = 0
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            chunk = stream.read(min(4 * MIB, remaining))
            require(bool(chunk), f"truncated payload while hashing {path}")
            remaining -= len(chunk)
            for byte in chunk:
                crc = CRC64_ECMA_TABLE[((crc >> 56) ^ byte) & 0xFF] ^ ((crc << 8) & mask)
    return crc


def _crc64_ecma_bytes(data: bytes | bytearray) -> int:
    mask = (1 << 64) - 1
    crc = 0
    for byte in data:
        crc = CRC64_ECMA_TABLE[((crc >> 56) ^ byte) & 0xFF] ^ ((crc << 8) & mask)
    return crc


def _inspect_compact_descriptor(path: Path, shard: dict[str, Any]) -> dict[str, Any]:
    descriptor_offset = FLS_HEADER_BYTES + int(shard["payload_size"])
    descriptor_size = int(shard["compact_descriptor_size"])
    require(
        descriptor_size >= COMPACT_DESCRIPTOR_HEADER_BYTES,
        f"Compact-v3 descriptor is shorter than its header: {path}",
    )
    with path.open("rb") as stream:
        stream.seek(descriptor_offset)
        descriptor = bytearray(stream.read(descriptor_size))
    require(len(descriptor) == descriptor_size, f"truncated Compact-v3 descriptor: {path}")
    require(descriptor[:8] == COMPACT_DESCRIPTOR_MAGIC, f"Compact-v3 descriptor magic mismatch: {path}")
    version, header_bytes, flags = struct.unpack_from("<HHI", descriptor, 8)
    require(version == 3, f"unexpected Compact-v3 descriptor version: {path}")
    require(header_bytes == COMPACT_DESCRIPTOR_HEADER_BYTES, f"unexpected Compact-v3 header size: {path}")
    require(
        flags & COMPACT_REQUIRED_FLAGS == COMPACT_REQUIRED_FLAGS and not flags & ~COMPACT_KNOWN_FLAGS,
        f"unsupported Compact-v3 descriptor flags: {path}",
    )
    (
        declared_descriptor_size,
        payload_size,
        payload_crc64,
        rowgroup_count,
        column_count,
        vector_size,
        spatial_order_id,
        schema_count,
        image_count,
        component_count,
    ) = struct.unpack_from("<QQQQIIIIII", descriptor, 16)
    require(declared_descriptor_size == descriptor_size, f"descriptor size mismatch: {path}")
    require(payload_size == shard["payload_size"], f"descriptor payload size mismatch: {path}")
    require(payload_crc64 == shard["payload_crc64"], f"descriptor payload CRC mismatch: {path}")
    require(rowgroup_count == shard["rowgroup_count"], f"descriptor rowgroup count mismatch: {path}")
    require(column_count == 64, f"JPEG-DCT compact descriptor must have 64 columns: {path}")
    require(vector_size == EXPECTED_VECTOR_SIZE, f"descriptor vector size mismatch: {path}")
    require(spatial_order_id == EXPECTED_SPATIAL_ORDER_ID, f"descriptor spatial order mismatch: {path}")
    require(schema_count > 0 and image_count == shard["image_count"] and component_count > 0,
            f"descriptor declares an empty/inconsistent directory: {path}")
    section_names = ("schema", "image", "component", "rowgroup", "coefficient", "rowgroup_page")
    section_values = struct.unpack_from("<" + "QQ" * len(section_names), descriptor, 72)
    sections: dict[str, dict[str, int]] = {}
    canonical_end = COMPACT_DESCRIPTOR_HEADER_BYTES
    for index, name in enumerate(section_names):
        offset, size = section_values[index * 2 : index * 2 + 2]
        require(
            offset >= COMPACT_DESCRIPTOR_HEADER_BYTES
            and offset <= descriptor_size
            and size <= descriptor_size - offset,
            f"Compact-v3 {name} section exceeds descriptor bounds: {path}",
        )
        require(
            offset == ((canonical_end + 7) & ~7),
            f"non-canonical Compact-v3 section order: {path}",
        )
        canonical_end = offset + size
        sections[name] = {"offset": offset, "size": size}
    require(canonical_end == descriptor_size, f"unclassified Compact-v3 descriptor bytes: {path}")
    coefficient_record_bytes = 4 if flags & COMPACT_DENSE_COEFFICIENT_RANGES_FLAG else 8
    schema_directory_bytes = (schema_count + 1) * 8
    require(sections["schema"]["size"] >= schema_directory_bytes, f"bad schema directory size: {path}")
    require(
        sections["image"]["size"] == image_count * COMPACT_IMAGE_RECORD_BYTES,
        f"bad image directory size: {path}",
    )
    require(
        sections["component"]["size"] == component_count * COMPACT_COMPONENT_RECORD_BYTES,
        f"bad component directory size: {path}",
    )
    require(
        sections["rowgroup"]["size"] == rowgroup_count * COMPACT_ROWGROUP_RECORD_BYTES,
        f"bad rowgroup directory size: {path}",
    )
    require(
        sections["coefficient"]["size"] == rowgroup_count * column_count * coefficient_record_bytes,
        f"bad coefficient directory size: {path}",
    )
    expected_crc64 = struct.unpack_from("<Q", descriptor, 168)[0]
    descriptor[168:176] = bytes(8)
    require(_crc64_ecma_bytes(descriptor) == expected_crc64, f"descriptor CRC64 mismatch: {path}")
    require(not any(descriptor[176:256]), f"non-zero Compact-v3 reserved header bytes: {path}")

    schema_base = sections["schema"]["offset"]
    schema_offsets = struct.unpack_from("<" + "Q" * (schema_count + 1), descriptor, schema_base)
    require(schema_offsets[0] == schema_directory_bytes, f"bad first schema offset: {path}")
    require(
        all(left < right for left, right in zip(schema_offsets, schema_offsets[1:])),
        f"schema dictionary is not a dense non-empty partition: {path}",
    )
    require(schema_offsets[-1] == sections["schema"]["size"], f"schema section has trailing bytes: {path}")

    rowgroup_base = sections["rowgroup"]["offset"]
    coefficient_base = sections["coefficient"]["offset"]
    expected_payload_offset = FLS_HEADER_BYTES
    expected_page_offset = 0
    zero_payload_rowgroup_count = 0
    rowgroups: list[dict[str, int]] = []
    for rowgroup_index in range(rowgroup_count):
        base = rowgroup_base + rowgroup_index * COMPACT_ROWGROUP_RECORD_BYTES
        payload_offset, rowgroup_payload_size, real_row_count = struct.unpack_from("<QII", descriptor, base)
        page_offset, page_size = struct.unpack_from("<QI", descriptor, base + 16)
        local_image_index, image_local_vector_index = struct.unpack_from("<II", descriptor, base + 28)
        (rowgroup_payload_crc64,) = struct.unpack_from("<Q", descriptor, base + 40)
        require(
            payload_offset == expected_payload_offset
            and 0 < real_row_count <= vector_size
            and rowgroup_payload_size <= FLS_HEADER_BYTES + payload_size - payload_offset,
            f"rowgroup payload directory is not a dense valid partition: {path}",
        )
        require(
            rowgroup_payload_size != 0 or rowgroup_payload_crc64 == 0,
            f"metadata-only rowgroup has a non-empty payload CRC64: {path}",
        )
        require(
            page_offset == expected_page_offset
            and page_size > 0
            and page_size <= sections["rowgroup_page"]["size"] - page_offset,
            f"rowgroup page directory is not a dense valid partition: {path}",
        )
        coefficient_end = 0
        for coefficient_index in range(column_count):
            record_index = rowgroup_index * column_count + coefficient_index
            record_offset = coefficient_base + record_index * coefficient_record_bytes
            if coefficient_record_bytes == 4:
                coefficient_offset = coefficient_end
                (coefficient_size,) = struct.unpack_from("<I", descriptor, record_offset)
            else:
                coefficient_offset, coefficient_size = struct.unpack_from("<II", descriptor, record_offset)
            require(
                coefficient_offset == coefficient_end
                and coefficient_size <= rowgroup_payload_size - coefficient_end,
                f"coefficient directory is not a dense exact rowgroup partition: {path}",
            )
            coefficient_end += coefficient_size
        require(coefficient_end == rowgroup_payload_size, f"coefficient directory has a payload gap: {path}")
        if rowgroup_payload_size == 0:
            zero_payload_rowgroup_count += 1
        rowgroups.append(
            {
                "real_row_count": real_row_count,
                "local_image_index": local_image_index,
                "image_local_vector_index": image_local_vector_index,
            }
        )
        expected_payload_offset += rowgroup_payload_size
        expected_page_offset += page_size
    require(
        expected_payload_offset == FLS_HEADER_BYTES + payload_size,
        f"rowgroups do not exactly cover the compressed payload: {path}",
    )
    require(
        expected_page_offset == sections["rowgroup_page"]["size"],
        f"rowgroups do not exactly cover the page section: {path}",
    )

    image_base = sections["image"]["offset"]
    component_base = sections["component"]["offset"]
    expected_first_rowgroup = 0
    expected_first_component = 0
    expected_first_physical_row = 0
    for image_index in range(image_count):
        base = image_base + image_index * COMPACT_IMAGE_RECORD_BYTES
        first_rowgroup, image_rowgroup_count, image_real_rows, first_component = struct.unpack_from(
            "<IIII", descriptor, base
        )
        (image_component_count,) = struct.unpack_from("<H", descriptor, base + 16)
        (image_spatial_order,) = struct.unpack_from("<I", descriptor, base + 20)
        (first_physical_row,) = struct.unpack_from("<Q", descriptor, base + 24)
        require(
            first_rowgroup == expected_first_rowgroup
            and image_rowgroup_count > 0
            and first_rowgroup + image_rowgroup_count <= rowgroup_count
            and image_real_rows > 0
            and first_component == expected_first_component
            and image_component_count > 0
            and first_component + image_component_count <= component_count
            and image_spatial_order == spatial_order_id
            and first_physical_row == expected_first_physical_row,
            f"image directory is not a dense valid partition: {path}",
        )
        image_rows = 0
        for local_vector in range(image_rowgroup_count):
            rowgroup = rowgroups[first_rowgroup + local_vector]
            require(
                rowgroup["local_image_index"] == image_index
                and rowgroup["image_local_vector_index"] == local_vector,
                f"rowgroup image mapping disagrees with the image directory: {path}",
            )
            image_rows += rowgroup["real_row_count"]
        require(image_rows == image_real_rows, f"image row count mismatch: {path}")
        component_rows = 0
        for local_component in range(image_component_count):
            component_record = component_base + (first_component + local_component) * COMPACT_COMPONENT_RECORD_BYTES
            _, width, height, padded_width, padded_height, row_offset = struct.unpack_from(
                "<IIIIII", descriptor, component_record
            )
            require(
                width > 0
                and height > 0
                and width <= padded_width
                and height <= padded_height
                and row_offset == component_rows,
                f"component directory contains invalid geometry: {path}",
            )
            component_rows += width * height
        require(component_rows == image_real_rows, f"component grids do not cover the image: {path}")
        expected_first_rowgroup += image_rowgroup_count
        expected_first_component += image_component_count
        expected_first_physical_row += image_real_rows
    require(
        expected_first_rowgroup == rowgroup_count and expected_first_component == component_count,
        f"image directory does not cover all rowgroups/components: {path}",
    )
    return {
        "flags": flags,
        "optimized_runtime_schema": flags & COMPACT_OPTIMIZED_FLAGS == COMPACT_OPTIMIZED_FLAGS,
        "schema_count": schema_count,
        "image_count": image_count,
        "component_count": component_count,
        "zero_payload_rowgroup_count": zero_payload_rowgroup_count,
        "sections": sections,
        "descriptor_crc64": expected_crc64,
    }


def parse_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    cursor = Cursor(path.read_bytes(), f"JPEG-DCT shard manifest {path}")
    require(cursor.bytes(8) == MANIFEST_MAGIC, f"unexpected manifest magic: {path}")
    version, validation_mode, rowgroup_vectors, rowgroups_per_shard, image_count = cursor.take("<IHIIQ")
    (shard_count,) = cursor.take("<I")
    root = path.parent
    shards: list[dict[str, Any]] = []
    payloads: list[PayloadSpec] = []
    for _ in range(shard_count):
        (
            shard_id,
            first_image,
            shard_image_count,
            real_rows,
            padding_rows,
            physical_rows,
            rowgroup_count,
            block_group_count,
        ) = cursor.take("<IQIQQQII")
        fls_size, metadata_size = cursor.take("<QQ")
        fls_name = cursor.string()
        metadata_name = cursor.string()
        fls_relative, fls_path = _safe_member(root, fls_name, "FLS")
        metadata_relative, metadata_path = _safe_member(root, metadata_name, "metadata")
        payloads.extend(
            (
                PayloadSpec("fls", fls_relative, fls_path, fls_size),
                PayloadSpec("metadata", metadata_relative, metadata_path, metadata_size),
            )
        )
        shards.append(
            {
                "shard_id": shard_id,
                "first_image": first_image,
                "image_count": shard_image_count,
                "real_rows": real_rows,
                "padding_rows": padding_rows,
                "physical_rows": physical_rows,
                "rowgroup_count": rowgroup_count,
                "block_group_count": block_group_count,
                "fls_size": fls_size,
                "metadata_size": metadata_size,
                "fls_name": fls_relative,
                "metadata_name": metadata_relative,
            }
        )

    compact: dict[str, Any] | None = None
    if not cursor.eof():
        require(cursor.bytes(8) == COMPACT_EXTENSION_MAGIC, "unknown JPEG-DCT manifest trailer")
        physical_layout = cursor.string()
        descriptor_kind = cursor.string()
        (vector_size,) = cursor.take("<I")
        spatial_order = cursor.string()
        (spatial_order_id,) = cursor.take("<H")
        (extension_shard_count,) = cursor.take("<I")
        require(extension_shard_count == shard_count, "Compact-v3 extension shard count mismatch")
        extension_shards: list[dict[str, int]] = []
        for index in range(extension_shard_count):
            shard_id, payload_size, payload_crc64, compact_size, source_size = cursor.take("<IQQQQ")
            require(shard_id == shards[index]["shard_id"], "Compact-v3 extension shard id mismatch")
            extension = {
                "shard_id": shard_id,
                "payload_size": payload_size,
                "payload_crc64": payload_crc64,
                "compact_descriptor_size": compact_size,
                "source_descriptor_size": source_size,
            }
            shards[index].update(extension)
            extension_shards.append(extension)
        compact = {
            "physical_layout": physical_layout,
            "descriptor_kind": descriptor_kind,
            "vector_size": vector_size,
            "spatial_order": spatial_order,
            "spatial_order_id": spatial_order_id,
            "shards": extension_shards,
        }
    require(cursor.eof(), "JPEG-DCT manifest has trailing bytes")

    require(version == 3, f"Compact-v3 acceptance requires manifest version 3, got {version}")
    require(compact is not None, "manifest version 3 has no Compact-v3 extension")
    require(compact["physical_layout"] == EXPECTED_LAYOUT, "unexpected Compact-v3 physical layout")
    require(compact["descriptor_kind"] == EXPECTED_DESCRIPTOR_KIND, "unexpected descriptor kind")
    require(compact["vector_size"] == EXPECTED_VECTOR_SIZE, "unexpected Compact-v3 vector size")
    require(compact["spatial_order"] == EXPECTED_SPATIAL_ORDER, "unexpected Compact-v3 spatial order")
    require(compact["spatial_order_id"] == EXPECTED_SPATIAL_ORDER_ID, "unexpected spatial-order id")
    require(rowgroup_vectors == 1, "Compact-v3 manifest must use one vector per rowgroup")
    require(shards, "Compact-v3 manifest has no shards")

    expected_first_image = 0
    runtime_total = path.stat().st_size
    source_descriptor_total = 0
    compact_descriptor_total = 0
    compressed_payload_total = 0
    for shard in shards:
        require(shard["first_image"] == expected_first_image, "manifest images are not a dense partition")
        expected_first_image += shard["image_count"]
        require(shard["image_count"] > 0 and shard["rowgroup_count"] > 0, "empty Compact-v3 shard")
        require(shard["compact_descriptor_size"] > 0, "missing compact descriptor size")
        require(shard["source_descriptor_size"] > 0, "missing source descriptor size")
        require(
            shard["fls_size"]
            == FLS_HEADER_BYTES
            + shard["payload_size"]
            + shard["compact_descriptor_size"]
            + FLS_FOOTER_BYTES,
            "Compact-v3 FLS size disagrees with header/payload/descriptor/footer geometry",
        )
        fls_path = (root / shard["fls_name"]).resolve()
        actual_payload_crc64 = _crc64_ecma_file_range(
            fls_path, FLS_HEADER_BYTES, shard["payload_size"]
        )
        require(
            actual_payload_crc64 == shard["payload_crc64"],
            f"Compact-v3 aggregate payload CRC64 mismatch: {fls_path}",
        )
        shard["actual_payload_crc64"] = actual_payload_crc64
        shard["compact_descriptor"] = _inspect_compact_descriptor(fls_path, shard)
        source_descriptor_total += shard["source_descriptor_size"]
        compact_descriptor_total += shard["compact_descriptor_size"]
        compressed_payload_total += shard["payload_size"]
    require(expected_first_image == image_count, "manifest image count disagrees with its shard partition")

    for spec in payloads:
        require(spec.path.is_file(), f"missing {spec.kind} payload: {spec.path}")
        actual_size = spec.path.stat().st_size
        require(actual_size == spec.expected_size, f"{spec.kind} size mismatch: {spec.path}")
        require(spec.path.suffix != ".svb", "Compact-v3 runtime payload list contains .svb")
        runtime_total += actual_size
    svb_files = sorted(root.rglob("*.svb"))
    require(not svb_files, f"Compact-v3 dataset contains forbidden .svb files: {svb_files[:4]}")

    descriptor_reduction = 1.0 - compact_descriptor_total / source_descriptor_total
    metadata_ratio = (
        compact_descriptor_total / compressed_payload_total
        if compressed_payload_total
        else None
    )
    projected_50k = math.ceil(runtime_total * 50_000 / image_count)
    strict_reference_applicable = image_count == 100
    optimized_runtime_schema = all(
        shard["compact_descriptor"]["optimized_runtime_schema"] for shard in shards
    )
    return {
        "manifest": str(path),
        "version": version,
        "validation_mode": validation_mode,
        "rowgroup_vectors": rowgroup_vectors,
        "rowgroups_per_shard": rowgroups_per_shard,
        "image_count": image_count,
        "shard_count": shard_count,
        "runtime_total_bytes": runtime_total,
        "compressed_payload_bytes": compressed_payload_total,
        "source_descriptor_bytes": source_descriptor_total,
        "compact_descriptor_bytes": compact_descriptor_total,
        "descriptor_reduction": descriptor_reduction,
        "compact_metadata_to_payload_ratio": metadata_ratio,
        "optimized_runtime_schema": optimized_runtime_schema,
        "projected_50k_runtime_bytes": projected_50k,
        "storage_gates": {
            "strict_reference_100_image_bytes": STRICT_REFERENCE_100_IMAGE_BYTES,
            "strict_reference_applicable": strict_reference_applicable,
            "runtime_at_most_strict_reference": (
                runtime_total <= STRICT_REFERENCE_100_IMAGE_BYTES
                if strict_reference_applicable
                else None
            ),
            "projected_50k_limit_bytes": PROJECTED_50K_LIMIT_BYTES,
            "projected_50k_at_most_limit": projected_50k <= PROJECTED_50K_LIMIT_BYTES,
            "descriptor_reduction_at_least_80pct": descriptor_reduction >= 0.80,
            "recommended_metadata_at_most_10pct_payload": (
                metadata_ratio <= 0.10 if metadata_ratio is not None else None
            ),
            "optimized_runtime_schema_flags_present": optimized_runtime_schema,
        },
        "compact": compact,
        "shards": shards,
        "payloads": payloads,
    }


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * MIB)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    identity = _file_identity(path)
    return {
        "path": str(path),
        "size_bytes": identity["size_bytes"],
        "sha256": _sha256_file(path),
        "file_identity": identity,
    }


def _load_cached_fingerprints(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.is_file():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("format") != "galp-compact-v3-acceptance-fingerprints-v1":
        return {}
    return {
        str(item.get("path")): item
        for item in payload.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }


def _payload_fingerprints(manifest: dict[str, Any], cache_path: Path) -> list[dict[str, Any]]:
    cached = _load_cached_fingerprints(cache_path)
    fingerprints: list[dict[str, Any]] = []
    for spec in manifest["payloads"]:
        key = str(spec.path.resolve())
        identity = _file_identity(spec.path)
        candidate = cached.get(key)
        if (
            candidate is not None
            and candidate.get("file_identity") == identity
            and isinstance(candidate.get("sha256"), str)
            and len(candidate["sha256"]) == 64
        ):
            fingerprint = dict(candidate)
        else:
            fingerprint = _fingerprint(spec.path)
        fingerprint["kind"] = spec.kind
        fingerprint["relative_path"] = spec.relative_path
        fingerprints.append(fingerprint)
    _write_json(
        cache_path,
        {"format": "galp-compact-v3-acceptance-fingerprints-v1", "files": fingerprints},
        replace=True,
    )
    return fingerprints


def _write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_validation_evidence(validation_path: Path, contract_path: Path) -> dict[str, Any]:
    validation_path = validation_path.resolve()
    contract_path = contract_path.resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    require(validation.get("ok") is True, f"semantic validation did not pass: {validation_path}")
    require(not validation.get("failures"), f"semantic validation reports failures: {validation_path}")
    evidence_contract_path = validation_path.parent / "contract.json"
    require(
        evidence_contract_path.is_file(),
        f"semantic validation directory has no contract.json: {validation_path.parent}",
    )
    expected_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    evidence_contract = json.loads(evidence_contract_path.read_text(encoding="utf-8"))
    require(
        _sha256_json(evidence_contract) == _sha256_json(expected_contract),
        f"semantic validation contract does not match A/B contract: {validation_path}",
    )
    return {
        "validation": str(validation_path),
        "validation_sha256": _sha256_file(validation_path),
        "contract": str(evidence_contract_path.resolve()),
        "contract_sha256": _sha256_json(evidence_contract),
        "ok": True,
    }


def _binding_binary(config: dict[str, Any]) -> Path:
    binding_dir = Path(config["torch_binding_dir"]).resolve()
    candidates = sorted(binding_dir.glob("_galp_direct_dct*.so"))
    require(len(candidates) == 1, f"expected one GALP Torch binding in {binding_dir}, found {len(candidates)}")
    return candidates[0]


def _canonicalize_contract_pipeline_names(contract: dict[str, Any]) -> None:
    """Require the single production GALP pipeline and strip native knobs."""
    pipelines = contract.get("pipelines")
    require(isinstance(pipelines, dict), "bad contract.pipelines section")
    enabled = pipelines.get("enabled")
    require(isinstance(enabled, list), "bad contract.pipelines.enabled section")

    require(all(str(name) in GALP_PIPELINES or str(name) in {"rgbnomore", "dali", "pytorch"} for name in enabled),
            "contract contains a removed historical pipeline")
    config = pipelines.get("galp")
    require(isinstance(config, dict), "contract has no pipelines.galp configuration")
    for field in GALP_RUNTIME_IMPLEMENTATION_FIELDS:
        config.pop(field, None)
    config["runtime_profile"] = GALP_RUNTIME_PROFILE


def adapt_contract(args: argparse.Namespace) -> None:
    manifest = parse_manifest(args.manifest)
    contract = json.loads(args.template.read_text(encoding="utf-8"))
    require(isinstance(contract, dict) and isinstance(contract.get("pipelines"), dict), "bad template contract")
    _canonicalize_contract_pipeline_names(contract)
    contract["execution"]["warmup_batches"] = 0
    if args.measurement_batches is not None:
        require(args.measurement_batches >= 1, "--measurement-batches must be positive")
        contract["execution"]["measurement_batches"] = args.measurement_batches
    required_images = int(contract["execution"]["batch_size"]) * (
        int(contract["execution"]["warmup_batches"]) + int(contract["execution"]["measurement_batches"])
    )
    require(manifest["image_count"] >= required_images, "Compact-v3 dataset is smaller than the contract sample set")
    if args.repeats is not None:
        require(args.repeats >= 1, "--repeats must be positive")
        contract["execution"]["repeats"] = args.repeats
    contract["execution"]["aggregate_exclude_first_repeat"] = False
    sample_manifest_path = Path(contract["dataset"]["sample_manifest"])
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    samples = sample_manifest.get("samples")
    require(isinstance(samples, list) and len(samples) >= required_images, "template sample manifest is too small")
    if len(samples) != required_images:
        sample_manifest = dict(sample_manifest)
        sample_manifest["samples"] = samples[:required_images]
        adapted_sample_manifest = args.output.with_name(
            args.output.stem + ".sample_manifest.json"
        ).resolve()
        _write_json(adapted_sample_manifest, sample_manifest)
        contract["dataset"]["sample_manifest"] = str(adapted_sample_manifest)
        contract["dataset"]["manifest_sha256"] = _sha256_json(sample_manifest)
        semantic = contract.get("semantic_validation", {})
        if isinstance(semantic, dict):
            if isinstance(semantic.get("prediction_agreement_sample_count"), int):
                semantic["prediction_agreement_sample_count"] = min(
                    semantic["prediction_agreement_sample_count"], required_images
                )
            if isinstance(semantic.get("sample_count"), int):
                semantic["sample_count"] = min(semantic["sample_count"], required_images)
            for group in semantic.get("comparison_groups", []):
                thresholds = group.get("thresholds", {}) if isinstance(group, dict) else {}
                if isinstance(thresholds.get("full_prediction_sample_count"), int):
                    thresholds["full_prediction_sample_count"] = min(
                        thresholds["full_prediction_sample_count"], required_images
                    )
    enabled = [canonical_pipeline_name(name) for name in args.enabled_pipelines]
    require(enabled, "at least one enabled pipeline is required")
    require(len(set(enabled)) == len(enabled), "enabled pipelines contain duplicate canonical names")
    require(
        all(name in GALP_PIPELINES for name in enabled),
        "adapter only supports the GALP production pipeline",
    )
    contract["pipelines"]["enabled"] = enabled

    cache_path = args.fingerprint_cache or args.output.with_suffix(".payload_fingerprints.json")
    payload_fingerprints = _payload_fingerprints(manifest, cache_path.resolve())
    manifest_fingerprint = _fingerprint(args.manifest)
    for name in enabled:
        contract_name = contract_pipeline_name(contract, name, require_enabled=False)
        config = contract["pipelines"].get(contract_name)
        require(isinstance(config, dict), f"template has no pipelines.{name} configuration")
        config["manifest"] = str(args.manifest.resolve())
        config["manifest_fingerprint"] = manifest_fingerprint
        config["manifest_sha256"] = manifest_fingerprint["sha256"]
        config["manifest_version"] = 3
        config["payload_fingerprint_cache"] = str(cache_path.resolve())
        config["payload_fingerprints"] = payload_fingerprints
        config["native_binary_fingerprint"] = _fingerprint(_binding_binary(config))
        config["compact_descriptor_contract"] = {
            "physical_layout": EXPECTED_LAYOUT,
            "descriptor_kind": EXPECTED_DESCRIPTOR_KIND,
            "vector_size": EXPECTED_VECTOR_SIZE,
            "spatial_order": EXPECTED_SPATIAL_ORDER,
            "runtime_total_bytes": manifest["runtime_total_bytes"],
            "compressed_payload_bytes": manifest["compressed_payload_bytes"],
            "compact_descriptor_bytes": manifest["compact_descriptor_bytes"],
        }
    contract["compact_v3_acceptance"] = {
        "adapter": str(Path(__file__).resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_fingerprint["sha256"],
        "first_repeat_included": True,
        "fresh_process_required_per_leg": True,
        "ordered_first_batch_latency_available": int(
            contract["execution"]["measurement_batches"]
        )
        == 1,
    }
    _write_json(args.output, contract)
    print(json.dumps({"output": str(args.output.resolve()), "manifest": _manifest_json(manifest)}, indent=2))


def _manifest_json(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "payloads"}


def _read_kib_fields(path: Path) -> dict[str, int]:
    fields: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return fields
    for line in lines:
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        tokens = rest.split()
        if tokens and tokens[0].isdigit():
            multiplier = 1024 if len(tokens) > 1 and tokens[1] == "kB" else 1
            fields[key] = int(tokens[0]) * multiplier
    return fields


def _process_children(pid: int) -> list[int]:
    try:
        text = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8")
    except OSError:
        return []
    return [int(value) for value in text.split() if value.isdigit()]


def _process_tree(root_pid: int) -> set[int]:
    discovered: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in discovered or not Path(f"/proc/{pid}").exists():
            continue
        discovered.add(pid)
        pending.extend(_process_children(pid))
    return discovered


def _process_io(pid: int) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        lines = Path(f"/proc/{pid}/io").read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and value.strip().isdigit():
            result[key] = int(value.strip())
    return result


_SMAPS_HEADER = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s")


def _mapped_fls_resident(pid: int, fls_paths: set[str]) -> dict[str, int]:
    result = {"rss_bytes": 0, "pss_bytes": 0}
    try:
        lines = Path(f"/proc/{pid}/smaps").read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    selected = False
    for line in lines:
        if _SMAPS_HEADER.match(line):
            fields = line.split(maxsplit=5)
            selected = len(fields) == 6 and fields[5] in fls_paths
            continue
        if not selected:
            continue
        if line.startswith("Rss:"):
            result["rss_bytes"] += int(line.split()[1]) * 1024
        elif line.startswith("Pss:"):
            result["pss_bytes"] += int(line.split()[1]) * 1024
    return result


class ProcessTreeMonitor:
    def __init__(self, root_pid: int, fls_paths: Iterable[Path]) -> None:
        self.root_pid = root_pid
        self.fls_paths = {str(path.resolve()) for path in fls_paths}
        self.samples = 0
        self.peak_process_count = 0
        self.peak_rss_bytes = 0
        self.peak_pss_bytes = 0
        self.peak_pss_file_bytes = 0
        self.peak_pss_anon_bytes = 0
        self.peak_fls_mmap_rss_bytes = 0
        self.peak_fls_mmap_pss_bytes = 0
        self.io_first: dict[int, dict[str, int]] = {}
        self.io_last: dict[int, dict[str, int]] = {}

    def sample(self) -> None:
        pids = _process_tree(self.root_pid)
        if not pids:
            return
        rss = pss = pss_file = pss_anon = mapped_rss = mapped_pss = 0
        for pid in pids:
            memory = _read_kib_fields(Path(f"/proc/{pid}/smaps_rollup"))
            rss += memory.get("Rss", 0)
            pss += memory.get("Pss", 0)
            pss_file += memory.get("Pss_File", 0)
            pss_anon += memory.get("Pss_Anon", 0)
            mapped = _mapped_fls_resident(pid, self.fls_paths)
            mapped_rss += mapped["rss_bytes"]
            mapped_pss += mapped["pss_bytes"]
            io = _process_io(pid)
            if io:
                self.io_first.setdefault(pid, io)
                self.io_last[pid] = io
        self.samples += 1
        self.peak_process_count = max(self.peak_process_count, len(pids))
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.peak_pss_bytes = max(self.peak_pss_bytes, pss)
        self.peak_pss_file_bytes = max(self.peak_pss_file_bytes, pss_file)
        self.peak_pss_anon_bytes = max(self.peak_pss_anon_bytes, pss_anon)
        self.peak_fls_mmap_rss_bytes = max(self.peak_fls_mmap_rss_bytes, mapped_rss)
        self.peak_fls_mmap_pss_bytes = max(self.peak_fls_mmap_pss_bytes, mapped_pss)

    def result(self) -> dict[str, Any]:
        io_delta: dict[str, int] = {}
        for pid, last in self.io_last.items():
            first = self.io_first.get(pid, {})
            for key, value in last.items():
                io_delta[key] = io_delta.get(key, 0) + max(0, value - first.get(key, 0))
        return {
            "sample_count": self.samples,
            "peak_process_count": self.peak_process_count,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_pss_bytes": self.peak_pss_bytes,
            "peak_pss_file_bytes": self.peak_pss_file_bytes,
            "peak_pss_anon_bytes": self.peak_pss_anon_bytes,
            "peak_fls_mmap_rss_bytes": self.peak_fls_mmap_rss_bytes,
            "peak_fls_mmap_pss_bytes": self.peak_fls_mmap_pss_bytes,
            "process_tree_io_delta": io_delta,
            "actual_block_device_read_bytes": io_delta.get("read_bytes", 0),
            "logical_read_characters": io_delta.get("rchar", 0),
            "read_syscalls": io_delta.get("syscr", 0),
        }


def _contract_fls_paths(contract_path: Path, pipeline: str) -> list[Path]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_name = contract_pipeline_name(contract, pipeline)
    return [
        Path(item["path"])
        for item in contract["pipelines"][contract_name]["payload_fingerprints"]
        if item.get("kind") == "fls"
    ]


def run_leg(
    *,
    contract: Path,
    pipeline: str,
    output: Path,
    resource_output: Path,
    log_path: Path,
    python: Path,
    pipeline_script: Path,
    poll_ms: float,
) -> dict[str, Any]:
    for path in (output, resource_output, log_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    contract_payload = json.loads(contract.read_text(encoding="utf-8"))
    contract_pipeline = contract_pipeline_name(contract_payload, pipeline)
    command = [
        str(python),
        str(pipeline_script),
        "--pipeline",
        contract_pipeline,
        "--contract",
        str(contract.resolve()),
        "--output",
        str(output.resolve()),
    ]
    environment = dict(os.environ)
    binding_dir = contract_payload["pipelines"][contract_pipeline]["torch_binding_dir"]
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(Path(binding_dir).resolve()) + (":" + existing_pythonpath if existing_pythonpath else "")
    started = time.time_ns()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        monitor = ProcessTreeMonitor(process.pid, _contract_fls_paths(contract, pipeline))
        while process.poll() is None:
            monitor.sample()
            time.sleep(max(0.001, poll_ms / 1000.0))
        monitor.sample()
        return_code = process.wait()
    finished = time.time_ns()
    resource = {
        "command": command,
        "contract": str(contract.resolve()),
        "pipeline": canonical_pipeline_name(pipeline),
        "contract_pipeline": contract_pipeline,
        "return_code": return_code,
        "started_unix_ns": started,
        "finished_unix_ns": finished,
        "wall_seconds": (finished - started) / 1e9,
        "monitor": monitor.result(),
        "log": str(log_path.resolve()),
        "pipeline_output": str(output.resolve()),
    }
    _write_json(resource_output, resource)
    if return_code != 0:
        raise RuntimeError(f"pipeline leg failed with exit code {return_code}; see {log_path}")
    return resource


def run_one(args: argparse.Namespace) -> None:
    resource = run_leg(
        contract=args.contract,
        pipeline=args.pipeline,
        output=args.output,
        resource_output=args.resource_output,
        log_path=args.log,
        python=args.python,
        pipeline_script=args.pipeline_script,
        poll_ms=args.poll_ms,
    )
    print(json.dumps(resource, indent=2, sort_keys=True))


def _cv(values: list[float]) -> float | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else math.inf


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "cv_population": _cv(values),
    }


def _bootstrap_lower(values: list[float], seed: int, samples: int = 20000) -> float | None:
    if not values:
        return None
    generator = random.Random(seed)
    medians = []
    for _ in range(samples):
        medians.append(statistics.median(generator.choice(values) for _ in values))
    medians.sort()
    return medians[max(0, math.floor(0.025 * (len(medians) - 1)))]


def _native_peak(record: dict[str, Any], key: str) -> int:
    return max((int(repeat.get("native_counters", {}).get(key, 0)) for repeat in record["repeats"]), default=0)


def _leg_metrics(record_path: Path, resource_path: Path) -> dict[str, Any]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    contract = json.loads(Path(resource["contract"]).read_text(encoding="utf-8"))
    source_revisions = contract.get("source_revisions", {})
    source_revision_policy = contract.get("source_revision_policy", {})
    contract_path = Path(resource["contract"]).resolve()
    require(
        record.get("contract_sha256") == _sha256_json(contract),
        f"pipeline record contract hash mismatch: {record_path}",
    )
    require(
        Path(record.get("contract", "")).resolve() == contract_path,
        f"pipeline record contract path mismatch: {record_path}",
    )
    execution = record.get("execution", {})
    require(execution.get("warmup_batches") == 0, f"pipeline leg used warmup batches: {record_path}")
    require(
        execution.get("aggregate_exclude_first_repeat") is False,
        f"pipeline leg excluded repeat 0: {record_path}",
    )
    clean_contract_sources = (
        isinstance(source_revisions, dict)
        and bool(source_revisions)
        and all(
            isinstance(revision, dict)
            and revision.get("benchmark_source_clean") is True
            and bool(revision.get("runtime_file_sha256"))
            for revision in source_revisions.values()
        )
        and source_revision_policy.get("runtime_file_changes_during_benchmark_are_errors")
        is True
        and not source_revision_policy.get("dirty_sources_at_contract_creation")
    )
    repeats = record.get("repeats", [])
    require(repeats, f"pipeline record has no repeats: {record_path}")
    cold = repeats[0]
    hot = repeats[-1] if len(repeats) > 1 else None
    comparison_signature = _sha256_json(
        {
            "dataset_full_size": record.get("dataset_full_size"),
            "device_metadata": record.get("device_metadata"),
            "execution": execution,
            "model": record.get("model"),
            "preprocess": record.get("preprocess"),
            "sample_manifest_sha256": record.get("sample_manifest_sha256"),
        }
    )
    monitor = resource["monitor"]
    ordered_first_batch = (
        float(cold["end_to_end_latency_ms"]["mean"])
        if int(record.get("execution", {}).get("measurement_batches", 0)) == 1
        else None
    )
    metrics = {
        "record": str(record_path.resolve()),
        "resource": str(resource_path.resolve()),
        "cold_throughput_images_per_s": float(cold["throughput_images_per_s"]),
        "hot_throughput_images_per_s": float(hot["throughput_images_per_s"]) if hot else None,
        "cold_latency_mean_ms": float(cold["end_to_end_latency_ms"]["mean"]),
        "ordered_first_batch_latency_ms": ordered_first_batch,
        "peak_process_tree_rss_bytes": int(monitor["peak_rss_bytes"]),
        "peak_process_tree_pss_bytes": int(monitor["peak_pss_bytes"]),
        "peak_fls_mmap_rss_bytes": int(monitor["peak_fls_mmap_rss_bytes"]),
        "peak_fls_mmap_pss_bytes": int(monitor["peak_fls_mmap_pss_bytes"]),
        "actual_block_device_read_bytes": int(monitor["actual_block_device_read_bytes"]),
        "logical_read_characters": int(monitor["logical_read_characters"]),
        "process_tree_read_syscalls": int(monitor["read_syscalls"]),
        "torch_peak_gpu_allocated_bytes": max(int(item.get("peak_gpu_memory_allocated_bytes", 0)) for item in repeats),
        "torch_peak_gpu_reserved_bytes": max(int(item.get("peak_gpu_memory_reserved_bytes", 0)) for item in repeats),
        "galp_native_device_peak_in_use_bytes": _native_peak(record, "galp_native_device_peak_in_use_bytes"),
        "galp_native_pinned_peak_in_use_bytes": _native_peak(record, "galp_native_pinned_peak_in_use_bytes"),
        "native_pread_count": _native_peak(record, "pread_count"),
        "compressed_payload_bytes_read": _native_peak(record, "compressed_payload_bytes_read"),
        "full_compressed_payload_bytes": _native_peak(record, "full_compressed_payload_bytes"),
        "contract_sources_clean_and_hashed": clean_contract_sources,
        "comparison_signature": comparison_signature,
        "cold_sample_trace_sha256": cold.get("sample_trace", {}).get("sha256"),
    }
    full_payload = metrics["full_compressed_payload_bytes"]
    metrics["compressed_payload_ratio"] = (
        metrics["compressed_payload_bytes_read"] / full_payload if full_payload else None
    )
    return metrics


def _summarize_ab(legs: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    by_variant = {
        variant: [leg for leg in legs if leg["variant"] == variant]
        for variant in ("A", "B")
    }
    cold_a = [item["cold_throughput_images_per_s"] for item in by_variant["A"]]
    cold_b = [item["cold_throughput_images_per_s"] for item in by_variant["B"]]
    hot_a = [item["hot_throughput_images_per_s"] for item in by_variant["A"] if item["hot_throughput_images_per_s"]]
    hot_b = [item["hot_throughput_images_per_s"] for item in by_variant["B"] if item["hot_throughput_images_per_s"]]
    pair_count = min(len(cold_a), len(cold_b))
    cold_ratios = [cold_b[index] / cold_a[index] for index in range(pair_count)]
    hot_pair_count = min(len(hot_a), len(hot_b))
    hot_ratios = [hot_b[index] / hot_a[index] for index in range(hot_pair_count)]
    torch_allocated_a = max((item["torch_peak_gpu_allocated_bytes"] for item in by_variant["A"]), default=0)
    torch_allocated_b = max((item["torch_peak_gpu_allocated_bytes"] for item in by_variant["B"]), default=0)
    torch_reserved_a = max((item.get("torch_peak_gpu_reserved_bytes", 0) for item in by_variant["A"]), default=0)
    torch_reserved_b = max((item.get("torch_peak_gpu_reserved_bytes", 0) for item in by_variant["B"]), default=0)
    native_device_a = max((item.get("galp_native_device_peak_in_use_bytes", 0) for item in by_variant["A"]), default=0)
    native_device_b = max((item.get("galp_native_device_peak_in_use_bytes", 0) for item in by_variant["B"]), default=0)
    native_pinned_a = max((item.get("galp_native_pinned_peak_in_use_bytes", 0) for item in by_variant["A"]), default=0)
    native_pinned_b = max((item.get("galp_native_pinned_peak_in_use_bytes", 0) for item in by_variant["B"]), default=0)
    gpu_a = torch_reserved_a + native_device_a
    gpu_b = torch_reserved_b + native_device_b
    pss_a = max((item["peak_process_tree_pss_bytes"] for item in by_variant["A"]), default=0)
    pss_b = max((item["peak_process_tree_pss_bytes"] for item in by_variant["B"]), default=0)
    first_batch_a = [
        item["ordered_first_batch_latency_ms"]
        for item in by_variant["A"]
        if item["ordered_first_batch_latency_ms"] is not None
    ]
    first_batch_b = [
        item["ordered_first_batch_latency_ms"]
        for item in by_variant["B"]
        if item["ordered_first_batch_latency_ms"] is not None
    ]
    clean_baseline_contracts = bool(by_variant["A"]) and all(
        item.get("contract_sources_clean_and_hashed") is True for item in by_variant["A"]
    )
    comparison_signatures = {item.get("comparison_signature") for item in legs}
    sample_traces = {item.get("cold_sample_trace_sha256") for item in legs}
    return {
        "process_count": len(legs),
        "variant_A": {
            "cold_throughput": _distribution(cold_a),
            "hot_throughput": _distribution(hot_a),
            "ordered_first_batch_latency_ms": _distribution(first_batch_a),
            "peak_process_tree_pss_bytes": pss_a,
            "peak_torch_gpu_allocated_bytes": torch_allocated_a,
            "peak_torch_gpu_reserved_bytes": torch_reserved_a,
            "peak_galp_native_device_bytes": native_device_a,
            "peak_galp_native_pinned_bytes": native_pinned_a,
            "peak_total_gpu_bytes": gpu_a,
        },
        "variant_B": {
            "cold_throughput": _distribution(cold_b),
            "hot_throughput": _distribution(hot_b),
            "ordered_first_batch_latency_ms": _distribution(first_batch_b),
            "peak_process_tree_pss_bytes": pss_b,
            "peak_torch_gpu_allocated_bytes": torch_allocated_b,
            "peak_torch_gpu_reserved_bytes": torch_reserved_b,
            "peak_galp_native_device_bytes": native_device_b,
            "peak_galp_native_pinned_bytes": native_pinned_b,
            "peak_total_gpu_bytes": gpu_b,
            "peak_fls_mmap_pss_bytes": max((item["peak_fls_mmap_pss_bytes"] for item in by_variant["B"]), default=0),
        },
        "paired_cold_B_over_A": {
            "distribution": _distribution(cold_ratios),
            "bootstrap_95pct_lower_median": _bootstrap_lower(cold_ratios, seed),
        },
        "paired_hot_B_over_A": {
            "distribution": _distribution(hot_ratios),
            "bootstrap_95pct_lower_median": _bootstrap_lower(hot_ratios, seed + 1),
        },
        "gates": {
            "at_least_five_independent_processes_per_variant": (
                len(by_variant["A"]) >= 5 and len(by_variant["B"]) >= 5
            ),
            "cold_B_median_at_least_A": bool(cold_a and cold_b and statistics.median(cold_b) >= statistics.median(cold_a)),
            "hot_B_median_at_least_A": bool(hot_a and hot_b and statistics.median(hot_b) >= statistics.median(hot_a)),
            "cold_throughput_cv_at_most_5pct": _cv(cold_b) is not None and _cv(cold_b) <= 0.05,
            "hot_throughput_cv_at_most_5pct": _cv(hot_b) is not None and _cv(hot_b) <= 0.05,
            "paired_hot_ratio_95pct_lower_at_least_0_98": (
                (_bootstrap_lower(hot_ratios, seed + 1) or -math.inf) >= 0.98
            ),
            "total_host_peak_within_A_plus_512MiB": pss_b <= pss_a + 512 * MIB,
            "total_gpu_peak_within_1_05x_A": gpu_a > 0 and gpu_b <= gpu_a * 1.05,
            "ordered_first_batch_latency_at_most_A": (
                statistics.median(first_batch_b) <= statistics.median(first_batch_a)
                if first_batch_a and first_batch_b
                else "unverified_requires_measurement_batches_1"
            ),
            "clean_hashed_baseline_contract": clean_baseline_contracts,
            "same_hardware_data_model_and_execution": len(comparison_signatures) == 1
            and None not in comparison_signatures,
            "same_sample_order": len(sample_traces) == 1 and None not in sample_traces,
            "semantic_validation": "unverified_requires_validator_evidence",
        },
    }


def run_ab(args: argparse.Namespace) -> None:
    require(args.legs >= 10 and args.legs % 2 == 0, "--legs must be an even number of at least 10")
    validation_evidence = {
        "A": _load_validation_evidence(args.validation_a, args.contract_a),
        "B": _load_validation_evidence(args.validation_b, args.contract_b),
    }
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"A/B output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    legs: list[dict[str, Any]] = []
    for index in range(args.legs):
        variant = "A" if index % 2 == 0 else "B"
        contract = args.contract_a if variant == "A" else args.contract_b
        leg_dir = output_dir / f"leg_{index:02d}_{variant}"
        record_path = leg_dir / "pipeline.json"
        resource_path = leg_dir / "resources.json"
        run_leg(
            contract=contract,
            pipeline=args.pipeline,
            output=record_path,
            resource_output=resource_path,
            log_path=leg_dir / "pipeline.log",
            python=args.python,
            pipeline_script=args.pipeline_script,
            poll_ms=args.poll_ms,
        )
        metrics = _leg_metrics(record_path, resource_path)
        metrics.update({"leg": index, "variant": variant, "contract": str(contract.resolve())})
        legs.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
    ab_summary = _summarize_ab(legs, args.seed)
    ab_summary["gates"]["semantic_validation"] = all(
        evidence["ok"] for evidence in validation_evidence.values()
    )
    summary = {
        "order": [item["variant"] for item in legs],
        "legs": legs,
        "validation_evidence": validation_evidence,
        "summary": ab_summary,
    }
    _write_json(output_dir / "ab_results.json", summary)
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))
    _require_passing_boolean_gates(summary["summary"]["gates"], "A/B acceptance")


def _run_runtime_audit_process(
    *,
    manifest: dict[str, Any],
    audit_binary: Path,
    audit_arguments: list[str],
    output: Path,
    resource_output: Path,
    log_path: Path,
    poll_ms: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (output, resource_output, log_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    require(audit_binary.is_file(), f"runtime-audit binary does not exist: {audit_binary}")
    command = [
        str(audit_binary.resolve()),
        "--manifest",
        manifest["manifest"],
        *audit_arguments,
        "--output",
        str(output.resolve()),
    ]
    fls_paths = [spec.path for spec in manifest["payloads"] if spec.kind == "fls"]
    started = time.time_ns()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        monitor = ProcessTreeMonitor(process.pid, fls_paths)
        while process.poll() is None:
            monitor.sample()
            time.sleep(max(0.001, poll_ms / 1000.0))
        monitor.sample()
        return_code = process.wait()
    finished = time.time_ns()
    resource = {
        "command": command,
        "return_code": return_code,
        "started_unix_ns": started,
        "finished_unix_ns": finished,
        "wall_seconds": (finished - started) / 1e9,
        "monitor": monitor.result(),
        "log": str(log_path.resolve()),
        "audit_output": str(output.resolve()),
    }
    _write_json(resource_output, resource)
    if return_code != 0:
        raise RuntimeError(
            f"Compact-v3 runtime audit failed with exit code {return_code}; see {log_path}"
        )
    record = json.loads(output.read_text(encoding="utf-8"))
    require(
        record.get("schema") == "galp-compact-v3-runtime-audit-v1",
        f"unexpected runtime-audit schema: {output}",
    )
    return record, resource


def _io_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["workload"]), int(record["coefficient_count"])


def summarize_io_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {_io_key(record): record for record in records}
    require(len(by_key) == len(records), "duplicate workload/K runtime-audit records")
    coefficient_counts = (1, 4, 8, 16, 32, 64)
    required = {("full-all", 64), ("crop-all", 64)}
    required.update(("full-prefix", count) for count in coefficient_counts)
    required.update(("crop-prefix", count) for count in coefficient_counts)
    missing = sorted(required - set(by_key))
    require(not missing, f"runtime I/O matrix is incomplete: {missing}")

    full_all = by_key[("full-all", 64)]
    crop_all = by_key[("crop-all", 64)]
    full_pages = int(full_all["physical_page_bytes_covered"])
    full_payload = int(full_all["physical_range_bytes_read"])
    require(full_pages > 0 and full_payload > 0, "Full-All reported no physical I/O")

    annotated: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["page_ratio_to_full_all"] = (
            int(record["physical_page_bytes_covered"]) / full_pages
        )
        item["payload_ratio_to_full_all"] = (
            int(record["physical_range_bytes_read"]) / full_payload
        )
        annotated.append(item)

    full_prefix_page_ratios = [
        int(by_key[("full-prefix", count)]["physical_page_bytes_covered"]) / full_pages
        for count in coefficient_counts
    ]
    full_prefix_payload_ratios = [
        int(by_key[("full-prefix", count)]["physical_range_bytes_read"]) / full_payload
        for count in coefficient_counts
    ]
    prefix_monotonic_pages = all(
        left <= right + 1e-12
        for left, right in zip(full_prefix_page_ratios, full_prefix_page_ratios[1:])
    )
    prefix_monotonic_payload = all(
        left <= right + 1e-12
        for left, right in zip(full_prefix_payload_ratios, full_prefix_payload_ratios[1:])
    )
    combined_strict = all(
        int(by_key[("crop-prefix", count)]["physical_page_bytes_covered"])
        < int(crop_all["physical_page_bytes_covered"])
        and int(by_key[("crop-prefix", count)]["physical_page_bytes_covered"])
        < int(by_key[("full-prefix", count)]["physical_page_bytes_covered"])
        for count in coefficient_counts
        if count < 64
    )
    coefficient_ratios_exact = all(
        math.isclose(
            float(by_key[(workload, count)]["selected_coefficient_ratio"]),
            count / 64.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for workload in ("full-prefix", "crop-prefix")
        for count in coefficient_counts
    )
    accounting_consistent = all(
        int(record["native_pread_count"]) == int(record["coalesced_run_count"])
        and int(record["logical_compressed_bytes"])
        <= int(record["physical_range_bytes_read"])
        <= int(record["full_compressed_payload_bytes"])
        for record in records
    )
    crop_vector_ratio = float(crop_all["selected_vector_ratio"])
    crop_page_ratio = int(crop_all["physical_page_bytes_covered"]) / full_pages
    full_all_vector_count = int(full_all["selected_vector_count"])
    full_all_pread_count = int(full_all["native_pread_count"])
    full_all_preadv_count = int(full_all.get("native_preadv_count", 0))
    full_all_scatter_coalescing = (
        full_all_vector_count > 1
        and full_all_preadv_count > 0
        and full_all_pread_count < full_all_vector_count
    )
    full_prefix_k64 = by_key[("full-prefix", 64)]
    crop_prefix_k64 = by_key[("crop-prefix", 64)]
    actual_device_reads = {
        f"{record['workload']}:K{record['coefficient_count']}": int(
            record["actual_block_device_read_bytes"]
        )
        for record in records
    }
    return {
        "schema": "galp-compact-v3-io-matrix-v1",
        "records": sorted(
            annotated,
            key=lambda record: (str(record["workload"]), int(record["coefficient_count"])),
        ),
        "full_prefix_page_ratios": dict(zip(map(str, coefficient_counts), full_prefix_page_ratios)),
        "full_prefix_payload_ratios": dict(
            zip(map(str, coefficient_counts), full_prefix_payload_ratios)
        ),
        "actual_block_device_read_bytes": actual_device_reads,
        "gates": {
            "crop_all_vector_ratio_below_one": 0.0 < crop_vector_ratio < 1.0,
            "crop_all_page_ratio_below_one": 0.0 < crop_page_ratio < 1.0,
            "full_prefix_pages_monotonic_with_k": prefix_monotonic_pages,
            "full_prefix_payload_monotonic_with_k": prefix_monotonic_payload,
            "full_prefix_k_below_64_reads_fewer_pages": all(
                ratio < 1.0 for ratio in full_prefix_page_ratios[:-1]
            ),
            "full_prefix_k64_matches_full_all": (
                int(full_prefix_k64["physical_page_bytes_covered"]) == full_pages
                and int(full_prefix_k64["physical_range_bytes_read"]) == full_payload
            ),
            "crop_prefix_k64_matches_crop_all": (
                int(crop_prefix_k64["physical_page_bytes_covered"])
                == int(crop_all["physical_page_bytes_covered"])
                and int(crop_prefix_k64["physical_range_bytes_read"])
                == int(crop_all["physical_range_bytes_read"])
            ),
            "crop_prefix_below_crop_all_and_full_prefix_for_k_below_64": combined_strict,
            "coefficient_ratios_equal_k_over_64": coefficient_ratios_exact,
            "native_range_accounting_consistent": accounting_consistent,
            "full_all_preadv_coalescing_avoids_per_rowgroup_syscalls": full_all_scatter_coalescing,
            "actual_block_device_reads_reported_per_workload": len(actual_device_reads)
            == len(required),
            "formal_cold_cache_state": "requires_external_cache_control_and_recording",
        },
    }


def run_runtime_matrix(args: argparse.Namespace) -> None:
    require(args.first_image >= 0, "--first-image must be non-negative")
    require(args.batch_size > 0, "--batch-size must be positive")
    require(args.poll_ms > 0.0, "--poll-ms must be positive")
    require(args.hold_ms >= 0, "--hold-ms must be non-negative")
    manifest = parse_manifest(args.manifest)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"runtime-audit output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [("full-all", 64), ("crop-all", 64)]
    cases.extend(("full-prefix", count) for count in (1, 4, 8, 16, 32, 64))
    cases.extend(("crop-prefix", count) for count in (1, 4, 8, 16, 32, 64))
    records: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    memory_record: dict[str, Any] | None = None
    memory_resource: dict[str, Any] | None = None
    if args.include_memory_probe:
        record, resource = _run_runtime_audit_process(
            manifest=manifest,
            audit_binary=args.audit_binary,
            audit_arguments=[
                "--memory-probe",
                "--first-image",
                str(args.first_image),
                "--batch-size",
                str(args.batch_size),
                "--hold-ms",
                str(args.hold_ms),
            ],
            output=output_dir / "memory.json",
            resource_output=output_dir / "memory.resources.json",
            log_path=output_dir / "memory.log",
            poll_ms=args.poll_ms,
        )
        _write_json(
            output_dir / "memory.combined.json",
            {"audit": record, "external_process_tree": resource},
        )
        memory_record = record
        memory_resource = resource
    for workload, coefficient_count in cases:
        stem = f"{workload.replace('-', '_')}_k{coefficient_count}"
        record, resource = _run_runtime_audit_process(
            manifest=manifest,
            audit_binary=args.audit_binary,
            audit_arguments=[
                "--workload",
                workload,
                "--coefficients",
                str(coefficient_count),
                "--first-image",
                str(args.first_image),
                "--batch-size",
                str(args.batch_size),
                "--hold-ms",
                str(args.hold_ms),
            ],
            output=output_dir / f"{stem}.json",
            resource_output=output_dir / f"{stem}.resources.json",
            log_path=output_dir / f"{stem}.log",
            poll_ms=args.poll_ms,
        )
        record["external_process_tree"] = resource["monitor"]
        records.append(record)
        resources.append(resource)
        print(json.dumps(record, sort_keys=True), flush=True)
    summary = summarize_io_records(records)
    summary["fresh_process_count"] = len(resources) + int(args.include_memory_probe)
    summary["manifest"] = _manifest_json(manifest)
    if memory_record is not None and memory_resource is not None:
        summary["memory_probe"] = {
            "audit": memory_record,
            "external_process_tree": memory_resource["monitor"],
        }
        summary["gates"]["descriptor_and_index_incremental_rss_at_most_256MiB"] = bool(
            memory_record["descriptor_rss_gate_pass"]
        )
    _write_json(output_dir / "io_matrix.json", summary)
    print(json.dumps(summary["gates"], indent=2, sort_keys=True))
    _require_passing_boolean_gates(summary["gates"], "runtime I/O matrix")


def summarize_io_directory(args: argparse.Namespace) -> None:
    records = []
    for path in sorted(args.input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") == "galp-compact-v3-runtime-audit-v1" and payload.get("mode") == "io":
            records.append(payload)
    summary = summarize_io_records(records)
    if args.output is not None:
        _write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    _require_passing_boolean_gates(summary["gates"], "runtime I/O matrix")


def inspect_manifest(args: argparse.Namespace) -> None:
    print(json.dumps(_manifest_json(parse_manifest(args.manifest)), indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-manifest", help="strictly parse and audit a Compact-v3 manifest")
    inspect_parser.add_argument("manifest", type=Path)
    inspect_parser.set_defaults(function=inspect_manifest)

    adapt = subparsers.add_parser("adapt-contract", help="replace GALP payloads in a same-dataset system contract")
    adapt.add_argument("--template", type=Path, required=True)
    adapt.add_argument("--manifest", type=Path, required=True)
    adapt.add_argument("--output", type=Path, required=True)
    adapt.add_argument("--fingerprint-cache", type=Path)
    adapt.add_argument(
        "--enabled-pipelines",
        nargs="+",
        default=("galp",),
        choices=ACCEPTANCE_GALP_PIPELINES,
        help="GALP production pipeline",
    )
    adapt.add_argument("--repeats", type=int)
    adapt.add_argument("--measurement-batches", type=int)
    adapt.set_defaults(function=adapt_contract)

    run_parser = subparsers.add_parser("run-leg", help="run one fresh pipeline process with process-tree monitoring")
    run_parser.add_argument("--contract", type=Path, required=True)
    run_parser.add_argument(
        "--pipeline",
        choices=ACCEPTANCE_GALP_PIPELINES,
        default="galp",
    )
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--resource-output", type=Path, required=True)
    run_parser.add_argument("--log", type=Path, required=True)
    run_parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    run_parser.add_argument("--pipeline-script", type=Path, default=DEFAULT_PIPELINE)
    run_parser.add_argument("--poll-ms", type=float, default=10.0)
    run_parser.set_defaults(function=run_one)

    ab = subparsers.add_parser("run-ab", help="run strict alternating A/B legs in independent processes")
    ab.add_argument("--contract-a", type=Path, required=True)
    ab.add_argument("--contract-b", type=Path, required=True)
    ab.add_argument("--validation-a", type=Path, required=True)
    ab.add_argument("--validation-b", type=Path, required=True)
    ab.add_argument("--output-dir", type=Path, required=True)
    ab.add_argument(
        "--pipeline",
        choices=ACCEPTANCE_GALP_PIPELINES,
        default="galp",
    )
    ab.add_argument("--legs", type=int, default=10)
    ab.add_argument("--seed", type=int, default=11997733)
    ab.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    ab.add_argument("--pipeline-script", type=Path, default=DEFAULT_PIPELINE)
    ab.add_argument("--poll-ms", type=float, default=10.0)
    ab.set_defaults(function=run_ab)

    runtime = subparsers.add_parser(
        "run-runtime-matrix",
        help="run memory and four-class Compact-v3 I/O audits in fresh processes",
    )
    runtime.add_argument("--manifest", type=Path, required=True)
    runtime.add_argument("--output-dir", type=Path, required=True)
    runtime.add_argument("--audit-binary", type=Path, default=DEFAULT_RUNTIME_AUDIT)
    runtime.add_argument("--first-image", type=int, default=0)
    runtime.add_argument("--batch-size", type=int, default=64)
    runtime.add_argument("--poll-ms", type=float, default=10.0)
    runtime.add_argument("--hold-ms", type=int, default=100)
    runtime.add_argument(
        "--include-memory-probe",
        action="store_true",
        default=True,
        help="include the required fresh-process descriptor/index memory probe (default: enabled)",
    )
    runtime.set_defaults(function=run_runtime_matrix)

    summarize_io = subparsers.add_parser(
        "summarize-io",
        help="strictly summarize an existing Compact-v3 per-process I/O matrix",
    )
    summarize_io.add_argument("--input-dir", type=Path, required=True)
    summarize_io.add_argument("--output", type=Path)
    summarize_io.set_defaults(function=summarize_io_directory)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
