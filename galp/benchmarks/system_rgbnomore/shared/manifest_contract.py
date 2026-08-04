"""Canonical contracts for the stable GALP JPEG-DCT manifest envelope."""

from __future__ import annotations

from dataclasses import dataclass


MANIFEST_MAGIC = b"GJDCTSH1"
COMPACT_V3_EXTENSION_MAGIC = b"GJDCCV31"


@dataclass(frozen=True)
class ManifestVersionContract:
    physical_layout: str
    required_extension_magic: bytes | None = None
    descriptor_kind: str | None = None
    vector_size: int | None = None
    spatial_order: str | None = None
    spatial_order_id: int | None = None
    rowgroup_vectors: int | None = None


JPEG_DCT_MANIFEST_CONTRACTS = {
    1: ManifestVersionContract(
        physical_layout="dct-major/spatial-major-image-minor"
    ),
    2: ManifestVersionContract(physical_layout="image-major"),
    3: ManifestVersionContract(
        physical_layout="image-major-vector-rowgroups",
        required_extension_magic=COMPACT_V3_EXTENSION_MAGIC,
        descriptor_kind="galp-compact-v1",
        vector_size=1024,
        spatial_order="tiled-z32",
        spatial_order_id=3,
        rowgroup_vectors=1,
    ),
}


def canonical_extension_fields(
    contract: ManifestVersionContract,
    *,
    physical_layout: str | None,
    descriptor_kind: str | None,
    vector_size: int | None,
    spatial_order: str | None,
    spatial_order_id: int | None,
    rowgroup_vectors: int,
) -> tuple[tuple[str, object, object], ...]:
    """Return every version-constrained field for uniform caller validation."""

    return (
        ("physical layout", physical_layout, contract.physical_layout),
        ("descriptor kind", descriptor_kind, contract.descriptor_kind),
        ("vector size", vector_size, contract.vector_size),
        ("spatial order", spatial_order, contract.spatial_order),
        ("spatial order id", spatial_order_id, contract.spatial_order_id),
        ("rowgroup vectors", rowgroup_vectors, contract.rowgroup_vectors),
    )
