#!/usr/bin/env python3
"""Thin GALP JPEG-DCT manifest preflight for the training benchmark.

This module deliberately understands only the shard-manifest envelope.  It
does not open FLS descriptors, inspect rowgroups, or derive planner state.  All
version/layout handling used by the Python training layer lives here.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from shared.manifest_contract import (  # noqa: E402
    COMPACT_V3_EXTENSION_MAGIC,
    JPEG_DCT_MANIFEST_CONTRACTS,
    MANIFEST_MAGIC,
    canonical_extension_fields,
)

DATASET_KIND = "jpeg-dct-sharded"

_VERSION_CONTRACTS = {
    version: JPEG_DCT_MANIFEST_CONTRACTS[version] for version in (2, 3)
}
SUPPORTED_LAYOUTS = {
    version: contract.physical_layout
    for version, contract in _VERSION_CONTRACTS.items()
}


class ManifestPreflightError(ValueError):
    """Raised before training when the GALP dataset contract is invalid."""


class _Cursor:
    def __init__(self, data: bytes, path: Path) -> None:
        self.data = data
        self.path = path
        self.offset = 0

    def take_bytes(self, size: int, label: str) -> bytes:
        end = self.offset + size
        if size < 0 or end > len(self.data):
            raise ManifestPreflightError(
                f"truncated GALP manifest while reading {label}: {self.path}"
            )
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def take(self, fmt: str, label: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        raw = self.take_bytes(size, label)
        return struct.unpack(fmt, raw)

    def string(self, label: str) -> str:
        (size,) = self.take("<I", f"{label} length")
        raw = self.take_bytes(size, label)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ManifestPreflightError(
                f"GALP manifest {label} is not UTF-8: {self.path}"
            ) from error

    @property
    def eof(self) -> bool:
        return self.offset == len(self.data)


@dataclass(frozen=True)
class _Payload:
    shard_id: int
    kind: str
    path: Path
    relative_path: str
    expected_size: int

    def fingerprint_input(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "shard_id": self.shard_id,
            "path": self.path,
            "relative_path": self.relative_path,
            "expected_size": self.expected_size,
        }


@dataclass(frozen=True)
class ManifestPreflight:
    path: Path
    magic: str
    dataset_kind: str
    version: int
    physical_layout: str
    physical_layout_source: str
    declared_physical_layout: str | None
    descriptor_kind: str | None
    vector_size: int | None
    spatial_order: str | None
    spatial_order_id: int | None
    validation_mode_id: int
    rowgroup_vectors: int
    rowgroups_per_shard: int
    image_count: int
    shard_count: int
    extension_present: bool
    payloads: tuple[_Payload, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "galp-training-manifest-preflight-v1",
            "path": str(self.path),
            "magic": self.magic,
            "dataset_kind": self.dataset_kind,
            "supported": True,
            "version": self.version,
            "physical_layout": self.physical_layout,
            "physical_layout_source": self.physical_layout_source,
            "declared_physical_layout": self.declared_physical_layout,
            "descriptor_kind": self.descriptor_kind,
            "vector_size": self.vector_size,
            "spatial_order": self.spatial_order,
            "spatial_order_id": self.spatial_order_id,
            "validation_mode_id": self.validation_mode_id,
            "rowgroup_vectors": self.rowgroup_vectors,
            "rowgroups_per_shard": self.rowgroups_per_shard,
            "image_count": self.image_count,
            "shard_count": self.shard_count,
            "extension_present": self.extension_present,
            "payload_file_count": len(self.payloads),
            "legacy_v2_implicit_payload_file_count": sum(
                payload.kind == "legacy-v2-sparse-vector-bundle"
                for payload in self.payloads
            ),
            "declared_payload_bytes": sum(
                payload.expected_size
                for payload in self.payloads
                if payload.kind != "legacy-v2-sparse-vector-bundle"
            ),
            "fingerprinted_payload_bytes": sum(
                payload.expected_size for payload in self.payloads
            ),
            "optional_field_policy": (
                "version-required fields are enforced; missing optional fields remain null"
            ),
        }

    def fingerprint_inputs(self) -> list[dict[str, Any]]:
        return [payload.fingerprint_input() for payload in self.payloads]


def _resolve_member(root: Path, name: str, label: str) -> Path:
    if not name:
        raise ManifestPreflightError(f"GALP manifest has an empty {label} path")
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ManifestPreflightError(
            f"GALP manifest {label} escapes the dataset root: {name!r}"
        ) from error
    return candidate


def _check_payload(payload: _Payload) -> None:
    if not payload.path.is_file():
        raise ManifestPreflightError(
            f"GALP manifest payload is missing ({payload.kind}, shard {payload.shard_id}): "
            f"{payload.path}"
        )
    actual_size = payload.path.stat().st_size
    if actual_size != payload.expected_size:
        raise ManifestPreflightError(
            f"GALP manifest payload size mismatch ({payload.kind}, shard {payload.shard_id}): "
            f"declared {payload.expected_size}, actual {actual_size}, path {payload.path}"
        )


def preflight_manifest(
    manifest_path: Path,
    *,
    expected_manifest_version: int | None = None,
    expected_physical_layout: str | None = None,
    expected_spatial_order: str | None = None,
    expected_image_count: int | None = None,
) -> ManifestPreflight:
    """Validate the stable manifest envelope without reading storage internals."""

    path = manifest_path.resolve()
    if not path.is_file():
        raise ManifestPreflightError(f"GALP manifest does not exist: {path}")
    cursor = _Cursor(path.read_bytes(), path)
    magic = cursor.take_bytes(len(MANIFEST_MAGIC), "magic")
    if magic != MANIFEST_MAGIC:
        raise ManifestPreflightError(
            f"GALP manifest magic mismatch: expected {MANIFEST_MAGIC!r}, got {magic!r}: {path}"
        )
    (version,) = cursor.take("<I", "version")
    if version not in SUPPORTED_LAYOUTS:
        raise ManifestPreflightError(
            f"unsupported GALP training manifest version {version}; supported versions are "
            f"{sorted(SUPPORTED_LAYOUTS)}: {path}"
        )
    (validation_mode_id,) = cursor.take("<H", "validation mode")
    rowgroup_vectors, rowgroups_per_shard = cursor.take("<II", "rowgroup geometry")
    (image_count,) = cursor.take("<Q", "image count")
    (shard_count,) = cursor.take("<I", "shard count")
    if image_count <= 0 or shard_count <= 0:
        raise ManifestPreflightError(
            f"GALP training manifest must contain images and shards: images={image_count}, "
            f"shards={shard_count}: {path}"
        )

    root = path.parent.resolve()
    payloads: list[_Payload] = []
    shard_ids: list[int] = []
    declared_shard_images = 0
    for ordinal in range(shard_count):
        (
            shard_id,
            _first_global_image_index,
            shard_images,
            _real_rows,
            _padding_rows,
            _physical_rows,
            rowgroup_count,
            _block_group_count,
        ) = cursor.take("<IQIQQQII", f"shard {ordinal} geometry")
        fls_size, metadata_size = cursor.take("<QQ", f"shard {ordinal} payload sizes")
        fls_name = cursor.string(f"shard {ordinal} FLS path")
        metadata_name = cursor.string(f"shard {ordinal} metadata path")
        if shard_id in shard_ids:
            raise ManifestPreflightError(f"duplicate GALP shard id {shard_id}: {path}")
        if shard_images <= 0 or rowgroup_count <= 0:
            raise ManifestPreflightError(
                f"GALP shard {shard_id} has invalid image/rowgroup counts: "
                f"images={shard_images}, rowgroups={rowgroup_count}"
            )
        shard_ids.append(shard_id)
        declared_shard_images += shard_images
        fls_path = _resolve_member(root, fls_name, "FLS payload")
        metadata_path = _resolve_member(root, metadata_name, "metadata payload")
        payloads.extend(
            (
                _Payload(
                    shard_id,
                    "fls",
                    fls_path,
                    fls_path.relative_to(root).as_posix(),
                    fls_size,
                ),
                _Payload(
                    shard_id,
                    "metadata",
                    metadata_path,
                    metadata_path.relative_to(root).as_posix(),
                    metadata_size,
                ),
            )
        )
        # Compact-v2 predates manifest-declared auxiliary payloads. Its public
        # reader uses this documented sibling-name convention when the sparse
        # vector bundle exists, so it is an input dependency and must be
        # fingerprinted. Compact-v3 does not inherit this convention: future
        # v3 auxiliary payloads must be declared by its manifest extension.
        if version == 2:
            bundle_path = fls_path.with_suffix(".svb")
            if bundle_path.is_file():
                payloads.append(
                    _Payload(
                        shard_id,
                        "legacy-v2-sparse-vector-bundle",
                        bundle_path,
                        bundle_path.relative_to(root).as_posix(),
                        bundle_path.stat().st_size,
                    )
                )
    if declared_shard_images != image_count:
        raise ManifestPreflightError(
            f"GALP manifest image count {image_count} disagrees with shard total "
            f"{declared_shard_images}: {path}"
        )

    version_contract = _VERSION_CONTRACTS[version]
    extension_present = False
    declared_physical_layout: str | None = None
    descriptor_kind: str | None = None
    vector_size: int | None = None
    spatial_order: str | None = None
    spatial_order_id: int | None = None
    if cursor.eof and version_contract.required_extension_magic is not None:
        raise ManifestPreflightError(
            f"GALP manifest version {version} requires its canonical "
            f"Compact-v3 descriptor extension: {path}"
        )
    if not cursor.eof:
        if version_contract.required_extension_magic is None:
            raise ManifestPreflightError(
                f"unexpected GALP manifest extension for version {version}: {path}"
            )
        extension_present = True
        extension_magic = cursor.take_bytes(
            len(version_contract.required_extension_magic),
            "Compact-v3 extension magic",
        )
        if extension_magic != version_contract.required_extension_magic:
            raise ManifestPreflightError(
                f"unexpected GALP manifest extension for version {version}: {path}"
            )
        declared_physical_layout = cursor.string("physical layout") or None
        descriptor_kind = cursor.string("descriptor kind") or None
        (vector_size,) = cursor.take("<I", "vector size")
        spatial_order = cursor.string("spatial order") or None
        (spatial_order_id,) = cursor.take("<H", "spatial order id")
        (extension_shard_count,) = cursor.take("<I", "extension shard count")
        if extension_shard_count != shard_count:
            raise ManifestPreflightError(
                f"GALP Compact-v3 extension shard count {extension_shard_count} "
                f"does not match base count {shard_count}: {path}"
            )
        for expected_shard_id in shard_ids:
            shard_id, _payload_size, _payload_crc64, _compact_size, _source_size = cursor.take(
                "<IQQQQ", f"Compact-v3 shard {expected_shard_id} contract"
            )
            if shard_id != expected_shard_id:
                raise ManifestPreflightError(
                    f"GALP Compact-v3 extension shard id {shard_id} does not match "
                    f"base id {expected_shard_id}: {path}"
                )
    if not cursor.eof:
        raise ManifestPreflightError(f"GALP manifest has trailing bytes: {path}")

    physical_layout = version_contract.physical_layout
    physical_layout_source = (
        "declared-extension" if declared_physical_layout is not None else "manifest-version-contract"
    )
    if declared_physical_layout is not None and declared_physical_layout != physical_layout:
        raise ManifestPreflightError(
            f"GALP manifest version {version} declares physical layout "
            f"{declared_physical_layout!r}, expected {physical_layout!r}: {path}"
        )
    canonical_fields = canonical_extension_fields(
        version_contract,
        physical_layout=declared_physical_layout or physical_layout,
        descriptor_kind=descriptor_kind,
        vector_size=vector_size,
        spatial_order=spatial_order,
        spatial_order_id=spatial_order_id,
        rowgroup_vectors=rowgroup_vectors,
    )
    for label, actual, expected in canonical_fields:
        if expected is not None and actual != expected:
            raise ManifestPreflightError(
                f"GALP manifest version {version} has non-canonical {label}: "
                f"expected {expected!r}, got {actual!r}: {path}"
            )
    if expected_manifest_version is not None and version != expected_manifest_version:
        raise ManifestPreflightError(
            f"GALP manifest version mismatch: expected {expected_manifest_version}, got {version}: {path}"
        )
    if expected_physical_layout is not None and physical_layout != expected_physical_layout:
        raise ManifestPreflightError(
            f"GALP physical layout mismatch: expected {expected_physical_layout!r}, "
            f"got {physical_layout!r}: {path}"
        )
    if expected_spatial_order is not None and spatial_order != expected_spatial_order:
        raise ManifestPreflightError(
            f"GALP spatial order mismatch: expected {expected_spatial_order!r}, "
            f"got {spatial_order!r}: {path}"
        )
    if expected_image_count is not None and image_count != expected_image_count:
        raise ManifestPreflightError(
            f"GALP image count mismatch: expected {expected_image_count}, got {image_count}: {path}"
        )

    for payload in payloads:
        _check_payload(payload)
    return ManifestPreflight(
        path=path,
        magic=magic.decode("ascii"),
        dataset_kind=DATASET_KIND,
        version=version,
        physical_layout=physical_layout,
        physical_layout_source=physical_layout_source,
        declared_physical_layout=declared_physical_layout,
        descriptor_kind=descriptor_kind,
        vector_size=vector_size,
        spatial_order=spatial_order,
        spatial_order_id=spatial_order_id,
        validation_mode_id=validation_mode_id,
        rowgroup_vectors=rowgroup_vectors,
        rowgroups_per_shard=rowgroups_per_shard,
        image_count=image_count,
        shard_count=shard_count,
        extension_present=extension_present,
        payloads=tuple(payloads),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-manifest-version", type=int, choices=tuple(sorted(SUPPORTED_LAYOUTS)))
    parser.add_argument(
        "--expected-physical-layout",
        choices=tuple(SUPPORTED_LAYOUTS[version] for version in sorted(SUPPORTED_LAYOUTS)),
    )
    parser.add_argument("--expected-spatial-order")
    parser.add_argument("--expected-image-count", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = preflight_manifest(
        args.manifest,
        expected_manifest_version=args.expected_manifest_version,
        expected_physical_layout=args.expected_physical_layout,
        expected_spatial_order=args.expected_spatial_order,
        expected_image_count=args.expected_image_count,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
