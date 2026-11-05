"""Native regression coverage for rebuilding the paper's existing stores."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from fastlens import annotations, cli
from fastlens.benchmarks.paper.rebuild_store import rebuild
from fastlens.benchmarks.system_dct_major.common import parse_manifest
from fastlens.tests.test_cli_native import NativeCliFixture


class PaperRebuildTests(NativeCliFixture):
    def prepare_old(self, split):
        root = self.root / "old"
        root.mkdir()
        store = root / "dct" if split == "train" else root
        paths = list(reversed(self.paths))
        listing = root / ("ordered_inputs.txt" if split == "train" else "source_paths.txt")
        listing.write_text("".join(f"{path}\n" for path in paths))
        subprocess.run([
            str(self.tools[cli.TOOL_NAMES[0]]), "--shard", "--out-dir", str(store),
            "--physical-layout", "spatial-major", "--metadata-profile", "reconstruct",
            "--shard-images", "2", "--rowgroup-vectors", "128", "--rowgroups-per-shard", "64",
            "--input-list", str(listing),
        ], check=True, capture_output=True)
        if split == "train":
            mapping = root / "ordered_mapping.csv"
            with mapping.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["planned_physical_position", "virtual_pls_id", "position_in_pls",
                                 "galp_image_id", "logical_sample_id", "label", "source_path"])
                for index, path in enumerate(paths):
                    writer.writerow([index, index // 2, index % 2, 90 + index,
                                     f"sample-{index}", 23 + index, path])
            (root / "materialization_contract.json").write_text(json.dumps(dict(
                ordered_input_list=str(listing), ordered_input_list_sha256=annotations.file_sha256(listing),
                ordered_mapping=str(mapping), ordered_mapping_sha256=annotations.file_sha256(mapping),
                organization_seed=20260810, runtime_mapping_hash="original-runtime-mapping")))
        else:
            (root / "samples.json").write_text(json.dumps([
                dict(galp_image_id=i, path=str(path), logical_sample_id=f"sample-{i}", label=23 + i, model_label=i)
                for i, path in enumerate(paths)]))
            (root / "profile.json").write_text('{"input_geometry":"fixture"}')
        return root, store

    def check_rebuild(self, split):
        source, old = self.prepare_old(split)
        original = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            rebuild(source, self.output, split=split, workers=2)
        new = self.output / "dct" if split == "train" else self.output
        before, after = parse_manifest(old / "manifest.bin"), parse_manifest(new / "manifest.bin")
        self.assertEqual((after["rowgroup_vectors"], after["rowgroups_per_shard"]), (128, 64))
        for a, b in zip(before["shards"], after["shards"], strict=True):
            for key in ("fls_path", "metadata_path"):
                self.assertEqual(Path(a[key]).read_bytes(), Path(b[key]).read_bytes())
                self.assertEqual(Path(b[key]).parent, new / "partitions")
            self.assertEqual(a["first_global_image_index"], b["first_global_image_index"])
            self.assertEqual(a["image_count"], b["image_count"])
        samples, _ = annotations.load_annotations(new)
        self.assertEqual([(s["sample_id"], s["label"]) for s in samples],
                         [(f"sample-{i}", 23 + i) for i in range(3)])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["inspect", str(new)]), 0)
            self.assertEqual(cli.main(["annotations", "inspect", str(new)]), 0)
        if split == "train":
            contract = json.loads((self.output / "materialization_contract.json").read_text())
            self.assertEqual(contract["runtime_mapping_hash"], "original-runtime-mapping")
            self.assertEqual(Path(contract["ordered_mapping"]).read_bytes(), original[Path("ordered_mapping.csv")])
            self.assertEqual(annotations.file_sha256(Path(contract["ordered_mapping"])), contract["ordered_mapping_sha256"])
            self.assertEqual(annotations.file_sha256(Path(contract["ordered_input_list"])), contract["ordered_input_list_sha256"])
        else:
            for name in ("samples.json", "profile.json"):
                self.assertEqual((new / name).read_bytes(), original[Path(name)])
        self.assertEqual({p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}, original)
        with self.assertRaises(cli.ConversionError):
            rebuild(source, self.output, split=split, workers=2)

    def test_validation_preserves_payloads_and_both_label_conventions(self):
        self.check_rebuild("val")

    def test_training_preserves_premix_partitions_and_mapping_contract(self):
        self.check_rebuild("train")

    def test_wrong_source_order_and_verification_failure_never_publish(self):
        source, _ = self.prepare_old("val")
        listing = source / "source_paths.txt"
        original = listing.read_text()
        listing.write_text("".join(f"{path}\n" for path in self.paths))
        with self.assertRaisesRegex(ValueError, "source order"):
            rebuild(source, self.output, split="val", workers=2)
        self.assertFalse(self.output.exists())
        listing.write_text(original)
        native = cli.run_native

        def fail(command, log, phase):
            if phase == "Exact DCT verification":
                raise cli.ConversionError("injected verification failure")
            native(command, log, phase)

        with contextlib.redirect_stdout(io.StringIO()), patch.object(cli, "run_native", side_effect=fail):
            with self.assertRaisesRegex(cli.ConversionError, "verification failure"):
                rebuild(source, self.output, split="val", workers=2)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(f".{self.output.name}.rebuild-*")), [])

    def test_changed_training_mapping_rejects_original_contract(self):
        source, _ = self.prepare_old("train")
        mapping = source / "ordered_mapping.csv"
        mapping.write_text(mapping.read_text().replace(",23,", ",24,"))
        with self.assertRaisesRegex(ValueError, "materialization contract"):
            rebuild(source, self.output, split="train", workers=2)
        self.assertFalse(self.output.exists())
