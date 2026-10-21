"""Raw PCR records: protobuf metadata followed by cross-image scan groups.

The payload follows PCR_Release's raw_bytes=True layout. The external index is
JSON instead of the original database; no TensorFlow fork is required.
"""
from __future__ import annotations

import re
from pathlib import Path

from google.protobuf import descriptor_pb2, message_factory

UPSTREAM_COMMIT = "aa76741f5dc6dcb6d7676c62e2fd53a78f43f142"
EOI = b"\xff\xd9"
MARKER = re.compile(b"\xff[^\x00\xff\xd0-\xd7]")

# LightMLRecords.proto: MetadataRecord, including its proto2 field cardinalities.
_descriptor = descriptor_pb2.FileDescriptorProto(name="pcr_metadata.proto", syntax="proto2")
_message = _descriptor.message_type.add(name="MetadataRecord")
for _name, _number, _label in (("labels", 1, 3), ("progressive_levels", 2, 2), ("version", 3, 1)):
    _message.field.add(name=_name, number=_number, label=_label, type=5)
MetadataRecord = message_factory.GetMessages([_descriptor])["MetadataRecord"]


def split_scans(blob: bytes) -> list[bytes]:
    """Split at entropy-scan ends, omitting EOI as the upstream writer does."""
    if not blob.startswith(b"\xff\xd8"):
        raise ValueError("Expected a JPEG SOI marker")
    cursor, first = 2, 0
    scans = []
    while cursor < len(blob):
        if blob[cursor:cursor + 2] == EOI:
            if not scans or cursor != len(blob) - 2:
                raise ValueError("Unexpected JPEG end marker")
            return scans
        if blob[cursor] != 255 or cursor + 4 > len(blob):
            raise ValueError("Invalid JPEG marker")
        marker = blob[cursor + 1]
        length = int.from_bytes(blob[cursor + 2:cursor + 4], "big")
        if length < 2 or cursor + 2 + length > len(blob):
            raise ValueError("Invalid JPEG segment length")
        cursor += 2 + length
        if marker == 0xDA:
            end = MARKER.search(blob, cursor)
            if end is None:
                raise ValueError("Truncated JPEG entropy scan")
            cursor = end.start()
            scans.append(blob[first:cursor])
            first = cursor
    raise ValueError("Missing JPEG end marker")


def write_record(path: Path, images: list[list[bytes]], labels: list[int]) -> dict:
    levels = max(map(len, images))
    metadata = MetadataRecord(labels=labels, progressive_levels=levels, version=2).SerializeToString()
    lengths = []
    with path.open("xb") as output:
        output.write(metadata)
        for level in range(levels):
            group = [image[level] if level < len(image) else b"" for image in images]
            lengths.append([len(fragment) for fragment in group])
            output.write(b"".join(group))
    return dict(path=path.name, metadata_bytes=len(metadata), scan_lengths=lengths)


def read_size(record: dict, scans: int) -> int:
    if scans < 1:
        raise ValueError("Scan count must be positive")
    return record["metadata_bytes"] + sum(map(sum, record["scan_lengths"][:scans]))


def read_record(root: Path, record: dict, scans: int, limiter=None):
    size = read_size(record, scans)
    # Unbuffered, bounded read: omitted scan groups must never be read first.
    with (root / record["path"]).open("rb", buffering=0) as source:
        if limiter is not None:
            deadline = limiter.reserve(size)
        blob = source.read(size)
    if len(blob) != size:
        raise ValueError("Truncated PCR record")
    if limiter is not None:
        limiter.finish(deadline)
    header = MetadataRecord.FromString(blob[:record["metadata_bytes"]])
    images = [bytearray() for _ in header.labels]
    offset = record["metadata_bytes"]
    for group in record["scan_lengths"][:scans]:
        for image, length in zip(images, group, strict=True):
            image.extend(blob[offset:offset + length])
            offset += length
    return [bytes(image) + EOI for image in images], list(header.labels), size
