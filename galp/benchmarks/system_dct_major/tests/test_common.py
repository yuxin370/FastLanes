from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from common import (  # noqa: E402
    SAMPLE_MANIFEST_SCHEMA,
    chunked,
    collect_sequential_samples,
    load_sample_manifest,
    manifest_snapshot,
    parse_manifest,
    sample_trace,
    write_sample_manifest,
)


def _manifest_bytes(
    version: int,
    image_count: int,
    fls_name: str,
    meta_name: str,
    *,
    rowgroup_vectors: int = 64,
) -> bytes:
    payload = bytearray(b"GJDCTSH1")
    payload += struct.pack("<IHIIQ", version, 0, rowgroup_vectors, 256, image_count)
    payload += struct.pack("<I", 1)
    payload += struct.pack("<IQIQQQII", 0, 0, image_count, 6, 0, 6, 1, 6)
    payload += struct.pack("<QQ", 4, 3)
    for value in (fls_name, meta_name):
        encoded = value.encode("utf-8")
        payload += struct.pack("<I", len(encoded)) + encoded
    return bytes(payload)


def _compact_v3_manifest_bytes(
    image_count: int,
    fls_name: str,
    meta_name: str,
    *,
    rowgroup_vectors: int = 1,
    physical_layout: str = "image-major-vector-rowgroups",
    descriptor_kind: str = "galp-compact-v1",
    vector_size: int = 1024,
    spatial_order: str = "tiled-z32",
    spatial_order_id: int = 3,
) -> bytes:
    payload = bytearray(
        _manifest_bytes(
            3,
            image_count,
            fls_name,
            meta_name,
            rowgroup_vectors=rowgroup_vectors,
        )
    )

    def encoded(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<I", len(raw)) + raw

    payload += b"GJDCCV31"
    payload += encoded(physical_layout)
    payload += encoded(descriptor_kind)
    payload += struct.pack("<I", vector_size)
    payload += encoded(spatial_order)
    payload += struct.pack("<HI", spatial_order_id, 1)
    payload += struct.pack("<IQQQQ", 0, 123, 456, 789, 987)
    return bytes(payload)


class CommonTest(unittest.TestCase):
    def test_manifest_version_maps_to_physical_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version, expected in ((1, "dct-major/spatial-major-image-minor"), (2, "image-major")):
                path = root / f"manifest-{version}.bin"
                path.write_bytes(_manifest_bytes(version, 5, "data.fls", "data.meta.bin"))
                parsed = parse_manifest(path)
                self.assertEqual(parsed["physical_layout"], expected)
                self.assertEqual(parsed["image_count"], 5)

    def test_compact_v3_extension_is_parsed_and_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest-v3.bin"
            path.write_bytes(_compact_v3_manifest_bytes(5, "data.fls", "data.meta.bin"))
            parsed = parse_manifest(path)
            self.assertEqual(parsed["physical_layout"], "image-major-vector-rowgroups")
            self.assertEqual(parsed["compact"]["descriptor_kind"], "galp-compact-v1")
            self.assertEqual(parsed["compact"]["vector_size"], 1024)
            self.assertEqual(parsed["compact"]["spatial_order"], "tiled-z32")
            self.assertEqual(parsed["compact"]["shards"][0]["compact_descriptor_bytes"], 789)

    def test_compact_v3_requires_its_canonical_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest-v3.bin"
            path.write_bytes(
                _manifest_bytes(
                    3,
                    5,
                    "data.fls",
                    "data.meta.bin",
                    rowgroup_vectors=1,
                )
            )
            with self.assertRaisesRegex(ValueError, "requires its canonical Compact-v3"):
                parse_manifest(path)

    def test_compact_v3_rejects_every_noncanonical_field(self) -> None:
        cases = (
            ({"physical_layout": "image-major"}, "physical layout"),
            ({"descriptor_kind": "future-descriptor"}, "descriptor kind"),
            ({"vector_size": 512}, "vector size"),
            ({"spatial_order": "raster"}, "spatial order"),
            ({"spatial_order_id": 0}, "spatial order id"),
            ({"rowgroup_vectors": 2}, "rowgroup vectors"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest-v3.bin"
            for override, label in cases:
                with self.subTest(label=label):
                    path.write_bytes(
                        _compact_v3_manifest_bytes(
                            5,
                            "data.fls",
                            "data.meta.bin",
                            **override,
                        )
                    )
                    with self.assertRaisesRegex(ValueError, f"non-canonical {label}"):
                        parse_manifest(path)

    def test_v2_snapshot_includes_existing_sparse_vector_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest-v2.bin"
            manifest.write_bytes(_manifest_bytes(2, 5, "data.fls", "data.meta.bin"))
            (root / "data.fls").write_bytes(b"fls0")
            (root / "data.meta.bin").write_bytes(b"abc")
            (root / "data.svb").write_bytes(b"bundle")
            snapshot = manifest_snapshot(manifest, hash_payloads=False)
            self.assertEqual(
                [item["kind"] for item in snapshot["payloads"]],
                ["fls", "metadata", "legacy-v2-sparse-vector-bundle"],
            )
            self.assertEqual(snapshot["persistent_bytes"], 4 + 3 + 6)

    def test_sample_manifest_rejects_nonsequential_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            samples = [
                {"ordinal": 0, "galp_image_id": 0, "label": 1},
                {"ordinal": 1, "galp_image_id": 1, "label": 2},
            ]
            digest = write_sample_manifest(path, samples, {"shuffle": False})
            self.assertEqual(load_sample_manifest(path, digest), samples)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], SAMPLE_MANIFEST_SCHEMA)
            payload["samples"][1]["galp_image_id"] = 7
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not sequential"):
                load_sample_manifest(path)

    def test_collect_samples_preserves_logical_ids_for_symlink_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.JPEG"
            source.write_bytes(b"jpeg-placeholder")
            logical = root / "prefix" / "val" / "n00000001" / "sample.JPEG"
            logical.parent.mkdir(parents=True)
            logical.symlink_to(source)
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "format": "galp_rgbnomore_label_map_v1",
                        "image_count": 1,
                        "labels": [7],
                        "sample_ids": ["val/n00000001/sample.JPEG"],
                    }
                ),
                encoding="utf-8",
            )

            samples, _ = collect_sequential_samples(
                data_root=root / "prefix",
                split="val",
                label_map_json=labels,
                expected_images=1,
                sample_count=1,
                hash_samples=False,
            )

            self.assertEqual(samples[0]["sample_id"], "val/n00000001/sample.JPEG")
            self.assertEqual(samples[0]["path"], str(source.resolve()))

    def test_chunk_and_trace_preserve_partial_tail(self) -> None:
        samples = [
            {"ordinal": index, "galp_image_id": index, "label": index + 10}
            for index in range(5)
        ]
        batches = list(chunked(samples, 2))
        self.assertEqual([len(item) for item in batches], [2, 2, 1])
        self.assertEqual([item["ordinal"] for item in sample_trace(batches)], list(range(5)))


if __name__ == "__main__":
    unittest.main()
