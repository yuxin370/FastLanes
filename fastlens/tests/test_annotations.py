"""Annotation identity, completeness and publication contracts; CPU/stdlib only."""

import csv
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastlens import annotations as ann


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fastlens-annotations-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.paths = [self.images / "zebra/a.jpg", self.images / "ant/b.jpg", self.images / "zebra/c.jpg"]
        self.store = self.root / "store"
        self.store.mkdir()
        (self.store / "manifest.bin").write_bytes(b"test manifest identity")
        (self.store / "conversion").mkdir()
        (self.store / "conversion/sources.txt").write_text("".join(f"{path}\n" for path in self.paths))
        (self.store / "geometry").mkdir()
        (self.store / ann.SOURCE_ORDER).write_text(ann.file_sha256(self.store / "conversion/sources.txt") + "\n")
        self.rows = [{"path": str(path), "sample_id": f"sample-{index}", "label": index}
                     for index, path in enumerate(self.paths)]
        self.table = self.root / "labels.jsonl"

    def write_rows(self, rows):
        self.table.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_folder_labels_and_fixed_mapping_across_splits(self):
        records, classes = ann.prepare_records(self.paths, image_root=self.images, labels_from_folders=True)
        self.assertEqual(classes, {"ant": 0, "zebra": 1})
        self.assertEqual([row.label for row in records], [1, 0, 1])
        mapping = self.root / "classes.json"
        mapping.write_text('{"ant": 17, "zebra": 999, "unused": 2}')
        for paths in (self.paths, self.paths[:1]):
            records, classes = ann.prepare_records(paths, image_root=self.images,
                                                  labels_from_folders=True, class_map=mapping)
            self.assertEqual(records[0].label, 999)
            self.assertEqual(records[0].sample_id, "zebra/a.jpg")
            self.assertEqual(classes["unused"], 2)
        mapping.write_text('{"ant": 17}')
        with self.assertRaisesRegex(ValueError, "absent from fixed"):
            ann.prepare_records(self.paths, image_root=self.images, labels_from_folders=True, class_map=mapping)
        mapping.write_text('{"ant": 17, "zebra": 17}')
        with self.assertRaisesRegex(ValueError, "distinct nonnegative"):
            ann.prepare_records(self.paths, image_root=self.images, labels_from_folders=True, class_map=mapping)
        with self.assertRaisesRegex(ValueError, "ROOT/CLASS"):
            ann.prepare_records([self.images / "flat.jpg"], image_root=self.images, labels_from_folders=True)

    def test_csv_retains_class_ids_and_matches_paths_not_rows(self):
        table = self.root / "labels.csv"
        with table.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Filepath", "Label"])
            for index in reversed(range(3)):
                writer.writerow([self.paths[index].relative_to(self.images).as_posix(), 990 + index])
        records, _ = ann.prepare_records(self.paths, image_root=self.images, labels_csv=table)
        self.assertEqual([record.label for record in records], [990, 991, 992])
        self.assertEqual(records[0].sample_id, "zebra/a.jpg")
        table.write_text("Filepath,Label,sample_id\nimages/zebra/a.jpg,999,external-id\n")
        records, _ = ann.prepare_records(self.paths[:1], labels_csv=table)
        self.assertEqual((records[0].sample_id, records[0].label), ("external-id", 999))

    def test_json_labels_include_explicit_null_and_external_identities(self):
        self.rows[0]["label"] = None
        self.rows[1]["label"] = {"classes": [2, 5], "boxes": [[1, 2, 3, 4]], "caption": "猫"}
        self.rows[2]["path"] = str(self.paths[2].relative_to(self.root))
        self.write_rows(reversed(self.rows))
        records, classes = ann.prepare_records(self.paths, labels_jsonl=self.table)
        self.assertIsNone(classes)
        self.assertEqual([record.label for record in records], [row["label"] for row in self.rows])
        self.assertEqual([record.sample_id for record in records], [f"sample-{index}" for index in range(3)])
        with self.assertRaisesRegex(ValueError, "--image-root or an explicit"):
            ann.prepare_records(self.paths)

    def test_incomplete_ambiguous_and_non_json_annotations_fail(self):
        for rows, message in (
            (self.rows[:2], "missing annotation"),
            (self.rows + [{"path": "unknown.jpg", "label": 1}], "unknown annotation"),
            (self.rows + [self.rows[0]], "duplicate annotation"),
            ([dict(row, sample_id="same") for row in self.rows], "unique nonempty"),
            ([{"path": str(self.paths[0])}], "path and label"),
            ([dict(row, label=float("nan")) for row in self.rows], "Out of range"),
        ):
            self.write_rows(rows)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                ann.prepare_records(self.paths, labels_jsonl=self.table)
        for token in ("Infinity", "1e999"):
            self.table.write_text('{"path":"x.jpg","label":' + token + '}\n')
            with self.assertRaises(ValueError):
                ann.prepare_records(self.paths, labels_jsonl=self.table)

    def publish(self):
        records, classes = ann.prepare_records(self.paths, image_root=self.images, labels_from_folders=True)
        return ann.publish_annotations(self.store, records, classes, label_source="folders")

    def test_publish_without_jpegs_and_relocate_store(self):
        before = {path.relative_to(self.store): path.read_bytes() for path in self.store.rglob("*") if path.is_file()}
        metadata = self.publish()
        self.assertEqual({name: (self.store / name).read_bytes() for name in before}, before)
        moved = self.root / "moved"
        self.store.rename(moved)
        samples, loaded = ann.load_annotations(moved)
        self.assertEqual(metadata, loaded)
        self.assertEqual([row["label"] for row in samples], [1, 0, 1])
        self.assertEqual([row["image_id"] for row in samples], [0, 1, 2])
        self.assertFalse(self.images.exists())
        with self.assertRaisesRegex(ValueError, "refusing to replace"):
            records, classes = ann.prepare_records(self.paths, image_root=self.images, labels_from_folders=True)
            ann.publish_annotations(moved, records, classes, label_source="folders")

    def test_binding_detects_wrong_store_order_samples_and_class_map(self):
        self.publish()
        for relative, message in (("manifest.bin", "manifest"),
                                  (ann.SOURCE_ORDER, "physical source order"),
                                  ("annotations/samples.jsonl", "samples"),
                                  ("annotations/class_to_idx.json", "class mapping")):
            path = self.store / relative
            original = path.read_bytes()
            path.write_bytes(original + b"!")
            with self.subTest(path=relative), self.assertRaisesRegex(ValueError, message):
                ann.load_annotations(self.store)
            path.write_bytes(original)
        metadata_path = self.store / "annotations/metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["sample_order_sha256"] = "wrong"
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "sample order"):
            ann.load_annotations(self.store)

    def test_runtime_annotations_do_not_require_conversion_records(self):
        metadata = self.publish()
        shutil.rmtree(self.store / "conversion")
        samples, loaded = ann.load_annotations(self.store)
        self.assertEqual(loaded, metadata)
        self.assertEqual([row["label"] for row in samples], [1, 0, 1])
        with self.assertRaisesRegex(ValueError, "missing conversion/sources.txt"):
            ann.physical_paths(self.store, 3)
        metadata["schema_version"] = "fastlens-samples-v1"
        (self.store / "annotations/metadata.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "unsupported samples metadata schema"):
            ann.load_annotations(self.store)

    def test_copied_annotations_reject_different_physical_order_with_identical_manifest(self):
        self.publish()
        other = self.root / "other-store"
        shutil.copytree(self.store, other)
        source = other / "conversion/sources.txt"
        source.write_text("".join(f"{path}\n" for path in reversed(self.paths)))
        (other / ann.SOURCE_ORDER).write_text(ann.file_sha256(source) + "\n")
        shutil.rmtree(other / "conversion")
        self.assertEqual((other / "manifest.bin").read_bytes(), (self.store / "manifest.bin").read_bytes())
        with self.assertRaisesRegex(ValueError, "physical source order"):
            ann.load_annotations(other)

    def test_changed_source_list_cannot_rebind_annotations(self):
        (self.store / "conversion/sources.txt").write_text("".join(f"{path}\n" for path in reversed(self.paths)))
        records, classes = ann.prepare_records(list(reversed(self.paths)), image_root=self.images, labels_from_folders=True)
        with self.assertRaisesRegex(ValueError, "physical source order"):
            ann.publish_annotations(self.store, records, classes, label_source="folders")
        self.assertFalse((self.store / "annotations").exists())

    def test_wrong_physical_order_and_failed_publication_leave_no_sidecar(self):
        records, classes = ann.prepare_records(self.paths, image_root=self.images, labels_from_folders=True)
        with self.assertRaisesRegex(ValueError, "order disagrees"):
            ann.publish_annotations(self.store, list(reversed(records)), classes, label_source="folders")
        with patch.object(Path, "rename", side_effect=OSError("injected publication failure")), self.assertRaises(OSError):
            ann.publish_annotations(self.store, records, classes, label_source="folders")
        self.assertFalse((self.store / "annotations").exists())
        self.assertEqual(list(self.store.glob(".annotations-staging-*")), [])

    def test_shuffle_v1_fixed_vectors_and_record_identity(self):
        for seed, expected in (
            (0, [8, 1, 5, 0, 6, 7, 2, 3, 9, 4]),
            (42, [2, 0, 5, 9, 4, 8, 3, 6, 1, 7]),
            (2**64 - 1, [3, 6, 8, 9, 0, 5, 2, 1, 4, 7]),
        ):
            records = [ann.InputRecord(Path(f"{i}.jpg"), str(i), {"class": i}) for i in range(10)]
            ann.shuffle_records(records, seed)
            self.assertEqual([int(record.sample_id) for record in records], expected)
            for record in records:
                self.assertEqual(record.source_path, Path(f"{record.sample_id}.jpg"))
                self.assertEqual(record.label, {"class": int(record.sample_id)})
        for seed in (-1, 2**64):
            with self.assertRaisesRegex(ValueError, "shuffle seed"):
                ann.shuffle_records([], seed)
        self.assertEqual(ann.order_sha256(["2", "0", "5", "9", "4", "8", "3", "6", "1", "7"]),
                         "10a73d595d3a64b3f6c5022ab3d2e838546ae05a9835b4e0f8d95a4ea4065f4c")


if __name__ == "__main__":
    unittest.main()
