"""Native CLI integration; set FASTLENS_REQUIRE_NATIVE_TESTS=1 to require prerequisites."""

import contextlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastlens import annotations, cli


class NativeCliFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from PIL import Image
            cls.Image = Image
            cls.tools = cli.find_tools(None)
        except (ImportError, cli.ConversionError) as error:
            if os.environ.get("FASTLENS_REQUIRE_NATIVE_TESTS") == "1":
                raise RuntimeError(f"required native integration prerequisite unavailable: {error}") from error
            raise unittest.SkipTest(f"native integration prerequisite unavailable: {error}") from error

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fastlens-native-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source images"
        self.source.mkdir()
        self.paths = [self.source / name for name in ("z space.jpg", "A.JPEG", "middle.jpe")]
        for seed, path in enumerate(self.paths):
            mode = "L" if seed == 2 else "RGB"
            width, height = (31, 25) if seed == 2 else (32, 32)
            data = random.Random(seed).randbytes(width * height * (1 if mode == "L" else 3))
            self.Image.frombytes(mode, (width, height), data).save(path, format="JPEG", quality=85 + seed, progressive=seed == 1)
        self.output = self.root / "store"
        self.listing = self.root / "ordered.txt"
        self.listing.write_text("\n".join(str(path.relative_to(self.root)) for path in self.paths) + "\n")

    def convert(self, *, directory=False, extra=()):
        inputs = [str(self.source)] if directory else ["--input-list", str(self.listing)]
        args = cli.make_parser().parse_args(["convert", *inputs, "--output", str(self.output), "--partition-images", "2",
                                            "--workers", "2", *extra])
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.convert(args)

    def verify(self, listing=None):
        return subprocess.run([str(self.tools[cli.TOOL_NAMES[0]]), "--verify-manifest", str(self.output / "manifest.bin"),
                               "--verify-workers", "2", "--input-list", str(listing or self.output / "conversion/sources.txt")],
                              capture_output=True, text=True)

    def extend_inputs(self, image_count):
        for index in range(len(self.paths), image_count):
            path = self.source / f"image-{index:04d}.jpg"
            self.Image.new("RGB", (32, 32), (index % 256, (index * 7) % 256, (index * 13) % 256)).save(path)
            self.paths.append(path)
        self.listing.write_text("\n".join(str(path) for path in self.paths) + "\n")


class NativeCliTests(NativeCliFixture):
    def test_add_annotations_without_source_jpegs_or_payload_access(self):
        self.convert()
        original = {path.relative_to(self.output): path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
        table = self.root / "labels.csv"
        table.write_text("Filepath,Label\n" + "".join(f"{path.name},{997 + index}\n" for index, path in enumerate(self.paths)))
        for path in self.paths:
            path.unlink()
        command = ["annotations", "build", str(self.output), "--image-root", str(self.source), "--labels-csv", str(table)]
        with contextlib.redirect_stdout(io.StringIO()), patch.object(cli, "run_native", side_effect=AssertionError("compression called")):
            self.assertEqual(cli.main([*command, "--dry-run"]), 0)
            self.assertFalse((self.output / "annotations").exists())
            # The manifest-only native parser must work even without shard files.
            generation = self.output / "partitions"
            parked = self.root / "parked-generation"
            generation.rename(parked)
            try:
                self.assertEqual(cli.main(command), 0)
                self.assertEqual(cli.main(["annotations", "inspect", str(self.output), "--json"]), 0)
            finally:
                parked.rename(generation)
        samples, _ = annotations.load_annotations(self.output)
        self.assertEqual([sample["label"] for sample in samples], [997, 998, 999])
        for relative, data in original.items():
            self.assertEqual((self.output / relative).read_bytes(), data)

    def test_complete_conversion_inspection_relocation_and_order(self):
        external = self.root / "report.json"
        self.assertEqual(self.convert(extra=("--report", str(external))), 0)
        self.assertEqual((self.output / "conversion/sources.txt").read_text().splitlines(), [str(path) for path in self.paths])
        moved = self.root / "relocated store"
        self.output.rename(moved)
        self.output = moved
        result = self.verify()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("exact: true", result.stdout)
        # The single-image mode exercises JpegDctShardDatasetReader itself.
        reader_check = subprocess.run(
            [str(self.tools[cli.TOOL_NAMES[0]]), "--verify-manifest", str(moved / "manifest.bin"),
             "--image-index", "1", str(self.paths[1])], capture_output=True, text=True,
        )
        self.assertEqual(reader_check.returncode, 0, reader_check.stdout + reader_check.stderr)
        info = cli.native_inspect(moved, self.tools[cli.TOOL_NAMES[1]])
        self.assertEqual(info["image_count"], 3)
        self.assertEqual(info["shard_count"], 2)
        self.assertEqual(info["manifest_version"], 1)
        self.assertTrue(info["index"]["compatible"])
        self.assertEqual({path.name for path in moved.iterdir()}, {"manifest.bin", "partitions", "geometry", "conversion"})
        self.assertEqual({path.name for path in (moved / "partitions").iterdir()},
                         {"000000.fls", "000000.images.bin", "000001.fls", "000001.images.bin"})
        for field, pattern in (("coefficient", "*.fls"), ("metadata", "*.images.bin")):
            paths = list(moved.rglob(pattern))
            self.assertEqual(info[f"{field}_file_count"], len(paths))
            self.assertEqual(info[f"{field}_bytes"], sum(path.stat().st_size for path in paths))
        self.assertEqual(info["index_file_count"], 3)
        report = json.loads((moved / cli.REPORT_NAME).read_text())
        self.assertEqual(external.read_bytes(), (moved / cli.REPORT_NAME).read_bytes())
        self.assertEqual(report["original_jpeg_bytes"], sum(path.stat().st_size for path in self.paths))
        self.assertEqual(report["total_output_bytes"], cli.stored_bytes(moved))
        self.assertEqual(report["verification"], "passed")
        self.assertEqual(report["physical_layout"], cli.PHYSICAL_LAYOUT)
        inspected = subprocess.run([sys.executable, "-m", "fastlens", "inspect", str(moved), "--json"],
                                   cwd=cli.REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(inspected.returncode, 0, inspected.stderr)
        sizes = json.loads(inspected.stdout)
        self.assertEqual(sizes["total_stored_bytes"], report["total_output_bytes"])
        self.assertEqual(sizes["core_storage_bytes"] + sizes["annotation_bytes"] + sizes["conversion_record_bytes"],
                         sizes["total_stored_bytes"])
        before = (moved / "manifest.bin").read_bytes()
        with self.assertRaisesRegex(cli.ConversionError, "refusing to replace"):
            self.convert()
        self.assertEqual((moved / "manifest.bin").read_bytes(), before)

    def test_directory_conversion_order_and_none_verification(self):
        self.convert(directory=True, extra=("--verify", "none"))
        self.assertFalse((self.output / "annotations").exists())
        self.assertEqual((self.output / "conversion/sources.txt").read_text().splitlines(), [str(path) for path in sorted(self.paths)])
        report = json.loads((self.output / cli.REPORT_NAME).read_text())
        self.assertEqual(report["verification"], "not_requested")
        self.assertEqual(report["wall_times"]["verification_seconds"], 0)
        self.assertEqual(self.verify().returncode, 0)

    def test_premixing_keeps_annotations_and_coefficients_aligned_across_shards(self):
        self.extend_inputs(17)
        table = self.root / "annotations.jsonl"
        rows = [{"path": str(path), "sample_id": f"original-{index}",
                 "label": {"category": index, "caption": path.stem}}
                for index, path in enumerate(self.paths)]
        table.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
        self.convert(extra=("--shuffle-seed", "42", "--labels-jsonl", str(table), "--partition-images", "8"))
        samples, metadata = annotations.load_annotations(self.output)
        ordered = (self.output / "conversion/sources.txt").read_text().splitlines()
        self.assertNotEqual(ordered, [str(path) for path in self.paths])
        self.assertEqual(len(ordered), 17)
        for sample, path in zip(samples, ordered, strict=True):
            source = rows[self.paths.index(Path(path))]
            self.assertEqual(sample["sample_id"], source["sample_id"])
            self.assertEqual(sample["label"], source["label"])
        report = json.loads((self.output / cli.REPORT_NAME).read_text())
        self.assertEqual(report["shard_count"], 3)
        self.assertEqual(report["annotations"], "annotations")
        self.assertEqual(metadata["shuffle"], {"algorithm": "fastlens-shuffle-v1", "seed": 42})
        self.assertEqual(report["total_output_bytes"], cli.stored_bytes(self.output))
        self.assertEqual(self.verify().returncode, 0)
        self.assertNotEqual(self.verify(self.listing).returncode, 0)

    def test_annotation_publication_failure_does_not_publish_conversion(self):
        with patch.object(annotations, "publish_annotations", side_effect=ValueError("annotation failure")), \
                self.assertRaisesRegex(ValueError, "annotation failure"):
            self.convert(directory=True, extra=("--shuffle-seed", "42"))
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".*fastlens-staging-*")), [])

    def test_single_jpeg_and_dry_run(self):
        self.listing.write_text(str(self.paths[0]) + "\n")
        self.convert(extra=("--dry-run",))
        self.assertFalse(self.output.exists())
        self.convert(extra=("--workers", "auto", "--verify-workers", "1"))
        info = cli.native_inspect(self.output, self.tools[cli.TOOL_NAMES[1]])
        self.assertEqual(info["image_count"], 1)
        self.assertTrue(info["index"]["compatible"])

    def test_invalid_and_truncated_jpegs_do_not_publish(self):
        original = self.paths[0].read_bytes()
        for data in (b"this is not a JPEG", original[:-20]):
            self.paths[0].write_bytes(data)
            with self.subTest(length=len(data)), self.assertRaisesRegex(cli.ConversionError, "JPEG|jpeg"):
                self.convert(extra=("--verify", "none"))
            self.assertFalse(self.output.exists())
            self.assertEqual(list(self.root.glob(".*fastlens-staging-*")), [])

    def test_full_verifier_detects_wrong_order_and_corrupt_fls(self):
        self.convert()
        wrong = self.root / "wrong.txt"
        wrong.write_text("\n".join(str(path) for path in reversed(self.paths)) + "\n")
        self.assertNotEqual(self.verify(wrong).returncode, 0)
        fls = next(self.output.rglob("*.fls"))
        fls.write_bytes(b"corrupt")
        self.assertNotEqual(self.verify().returncode, 0)

    def test_index_and_verification_failure_never_publish(self):
        original = cli.run_native
        for failing in ("Store layout", "Geometry index", "Exact DCT verification"):
            def fail(command, log, phase):
                if phase == failing:
                    if phase == "Exact DCT verification":
                        # Corrupt the actual native dataset before calling the verifier.
                        next(log.parent.rglob("*.fls")).write_bytes(b"broken")
                    else:
                        if phase == "Store layout":
                            original(command, log, phase)
                            self.assertTrue((log.parent / "partitions/000000.fls").exists())
                        raise cli.ConversionError("injected index failure")
                original(command, log, phase)
            with self.subTest(phase=failing), patch.object(cli, "run_native", side_effect=fail), self.assertRaises(cli.ConversionError):
                self.convert()
            self.assertFalse(self.output.exists())
            self.assertEqual(list(self.root.glob(".*fastlens-staging-*")), [])

    def test_inspect_missing_and_incompatible_index(self):
        self.convert()
        index = self.output / "geometry"
        for path in (index / "index.bin", index / "000000.bin"):
            original = path.read_bytes()
            for corrupted in (None, original[:-1] + bytes([original[-1] ^ 1])):
                with self.subTest(path=path.name, missing=corrupted is None):
                    if corrupted is None:
                        path.unlink()
                    else:
                        path.write_bytes(corrupted)
                    with self.assertRaisesRegex(cli.ConversionError, "inspection failed"):
                        cli.native_inspect(self.output, self.tools[cli.TOOL_NAMES[1]])
                    # This mode constructs the native Reader, unlike the
                    # streaming full verifier which does not use the index.
                    result = subprocess.run([
                        str(self.tools[cli.TOOL_NAMES[0]]), "--verify-manifest", str(self.output / "manifest.bin"),
                        "--image-index", "0", str(self.paths[0]),
                    ], capture_output=True, text=True)
                    # Descriptor contents are validated lazily by selective
                    # reads (covered by the GPU test), not CPU materialization.
                    if corrupted is None or path.name == "index.bin":
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    path.write_bytes(original)

    def test_conversion_records_are_optional_for_inspect_and_labels(self):
        self.convert(directory=True, extra=("--shuffle-seed", "42"))
        expected, _ = annotations.load_annotations(self.output)
        shutil.rmtree(self.output / "conversion")
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(cli.main(["inspect", str(self.output), "--json"]), 0)
        self.assertIsNone(json.loads(stream.getvalue())["original_jpeg_bytes"])
        self.assertEqual(json.loads(stream.getvalue())["conversion_record_bytes"], 0)
        self.assertEqual(annotations.load_annotations(self.output)[0], expected)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["annotations", "inspect", str(self.output)]), 0)

    def test_inspect_requires_valid_order_binding_and_matching_sources(self):
        self.convert()
        digest = self.output / annotations.SOURCE_ORDER
        original = digest.read_bytes()
        for value in (None, b"", b"g" * 64, b"0" * 63):
            with self.subTest(value=value):
                if value is None:
                    digest.unlink()
                else:
                    digest.write_bytes(value)
                with self.assertRaises((OSError, ValueError)):
                    cli.inspect_standard_store(self.output, self.tools[cli.TOOL_NAMES[1]])
                digest.write_bytes(original)
        sources = self.output / "conversion/sources.txt"
        sources.write_text("".join(f"{path}\n" for path in reversed(self.paths)))
        with self.assertRaisesRegex(cli.ConversionError, "physical source order"):
            cli.inspect_standard_store(self.output, self.tools[cli.TOOL_NAMES[1]])

    def test_paper_and_cli_core_sizes_exclude_auxiliary_files_and_require_geometry(self):
        from fastlens.benchmarks.paper.section_6_3.storage.storage import standard_store_indexes, measured_files

        self.convert()
        tool = self.tools[cli.TOOL_NAMES[1]]
        before = cli.inspect_standard_store(self.output, tool)
        for member in ("geometry/build.json", "geometry/cache.bin", "partitions/notes.txt"):
            (self.output / member).write_text("auxiliary data")
        indexes, core_bytes = standard_store_indexes(self.output / "manifest.bin")
        self.assertEqual(core_bytes, before["core_storage_bytes"])
        self.assertIn(self.output / annotations.SOURCE_ORDER, indexes)
        measured_core = sum(row["logical_bytes"] for row in measured_files(indexes)) + sum(
            before[key] for key in ("manifest_bytes", "coefficient_bytes", "metadata_bytes"))
        self.assertEqual(core_bytes, measured_core)
        self.assertGreater(cli.stored_bytes(self.output), core_bytes)
        for member in ("geometry/index.bin", "geometry/000000.bin", annotations.SOURCE_ORDER):
            path = self.output / member
            data = path.read_bytes()
            path.unlink()
            with self.subTest(member=member), self.assertRaises((cli.ConversionError, OSError)):
                standard_store_indexes(self.output / "manifest.bin")
            path.write_bytes(data)

    def test_standard_store_contract_is_separate_from_native_manifest_reading(self):
        self.convert()
        manifest = self.output / "manifest.bin"
        original = manifest.read_bytes()
        # The generic native reader still supports safe, nonstandard member paths.
        (self.output / "partitions").rename(self.output / "native-set")
        manifest.write_bytes(original.replace(b"partitions/", b"native-set/"))
        self.assertEqual(self.verify().returncode, 0)
        with self.assertRaisesRegex(cli.ConversionError, "nonstandard Store"):
            cli.native_inspect(self.output, self.tools[cli.TOOL_NAMES[1]])
        (self.output / "native-set").rename(self.output / "partitions")
        # Check ranges through manifest-only inspection, before index bindings.
        for fmt, offset, value in (("I", 34, 1), ("Q", 38, 1), ("I", 46, 0),
                                   ("I", 46, 4), ("Q", 22, 4)):
            changed = bytearray(original)
            struct.pack_into("<" + fmt, changed, offset, value)
            manifest.write_bytes(changed)
            result = subprocess.run([str(self.tools[cli.TOOL_NAMES[1]]), "--inspect-manifest", str(manifest)],
                                    capture_output=True, text=True)
            with self.subTest(offset=offset, value=value):
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Store", result.stderr)
        manifest.write_bytes(original)

    def test_annotations_cannot_be_copied_between_different_physical_orders(self):
        # Equal JPEG payloads deliberately make the native manifests identical;
        # the samples still have distinct identities and externally supplied labels.
        for path in self.paths[1:]:
            path.write_bytes(self.paths[0].read_bytes())
        table = self.root / "labels.csv"
        table.write_text("Filepath,Label\n" + "".join(f"{path.name},{index}\n" for index, path in enumerate(self.paths)))
        options = ("--image-root", str(self.source), "--labels-csv", str(table))
        self.convert(extra=options)
        first = self.output
        self.output = self.root / "reversed-store"
        self.listing.write_text("".join(f"{path}\n" for path in reversed(self.paths)))
        self.convert(extra=options)
        self.assertEqual((first / "manifest.bin").read_bytes(), (self.output / "manifest.bin").read_bytes())
        self.assertEqual([row["label"] for row in annotations.load_annotations(self.output)[0]], [2, 1, 0])
        shutil.rmtree(self.output / "conversion")
        shutil.rmtree(self.output / "annotations")
        shutil.copytree(first / "annotations", self.output / "annotations")
        with self.assertRaisesRegex(ValueError, "physical source order"):
            annotations.load_annotations(self.output)
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertNotEqual(cli.main(["annotations", "inspect", str(self.output)]), 0)
        self.assertIn("physical source order", errors.getvalue())

    def test_manifest_binding_and_member_paths(self):
        self.convert()
        manifest = self.output / "manifest.bin"
        original = manifest.read_bytes()
        member = b"partitions/000000.fls"
        encoded = struct.pack("<I", len(member)) + member
        for target in (str(self.output / "partitions/000000.fls"), "../outside.fls"):
            value = target.encode()
            manifest.write_bytes(original.replace(encoded, struct.pack("<I", len(value)) + value))
            with self.subTest(target=target), self.assertRaisesRegex(cli.ConversionError, "unsafe"):
                cli.native_inspect(self.output, self.tools[cli.TOOL_NAMES[1]])
            result = self.verify()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsafe", result.stderr)
        manifest.write_bytes(original)
        partition = self.output / "partitions/000000.fls"
        outside = self.root / "outside.fls"
        partition.rename(outside)
        partition.symlink_to(outside)
        with self.assertRaisesRegex(cli.ConversionError, "escapes"):
            cli.native_inspect(self.output, self.tools[cli.TOOL_NAMES[1]])
        self.assertNotEqual(self.verify().returncode, 0)
        partition.unlink()
        outside.rename(partition)
        # A valid-looking manifest edit still invalidates the geometry binding.
        changed = bytearray(original)
        changed[14] ^= 1  # rowgroup_vectors, keeping the manifest parseable
        manifest.write_bytes(changed)
        with self.assertRaisesRegex(cli.ConversionError, "inspection failed"):
            cli.native_inspect(self.output, self.tools[cli.TOOL_NAMES[1]])
        manifest.write_bytes(original)
        first, second = self.output / "partitions/000000.images.bin", self.output / "partitions/000001.images.bin"
        first_data, second_data = first.read_bytes(), second.read_bytes()
        first.write_bytes(second_data)
        second.write_bytes(first_data)
        with self.assertRaisesRegex(cli.ConversionError, "inspection failed"):
            cli.native_inspect(self.output, self.tools[cli.TOOL_NAMES[1]])

    def test_single_image_ordered_list_verifies_entire_manifest(self):
        self.convert()
        single = self.root / "one.txt"
        single.write_text(str(self.paths[0]) + "\n")
        result = subprocess.run([str(self.tools[cli.TOOL_NAMES[0]]), "--verify-manifest", str(self.output / "manifest.bin"),
                                 "--verify-workers", "1", "--input-list", str(single)], capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_multiple_partitions_with_mixed_geometry_and_tail(self):
        self.extend_inputs(17)
        self.convert(extra=("--partition-images", "8"))
        info = cli.native_inspect(self.output, self.tools[cli.TOOL_NAMES[1]])
        self.assertEqual(info["image_count"], 17)
        self.assertEqual(info["shard_count"], 3)
        self.assertEqual(info["index_file_count"], 4)
        self.assertTrue(info["index"]["compatible"])
        self.assertEqual((self.output / "conversion/sources.txt").read_text().splitlines(), [str(path) for path in self.paths])
        result = self.verify()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_full_verification_rejects_quantization_corruption_before_publication(self):
        original = cli.run_native
        phases = []

        def corrupt_before_index(command, log, phase):
            phases.append(phase)
            if phase == "Geometry index":
                metadata = sorted(log.parent.rglob("*.images.bin"))[0]
                data = bytearray(metadata.read_bytes())
                self.assertEqual(data[:8], b"GJDCTMD3")
                # Sectioned metadata v3: change the last quantization entry of
                # the first image, leaving the FLS coefficients and file sizes intact.
                offset = 38
                while offset < len(data):
                    section, size = struct.unpack_from("<HQ", data, offset)
                    offset += 10
                    if section == 3:  # ReconstructableImageInfo
                        value_offset = offset + 8 + 17 + 1 + 63 * 2
                        value = struct.unpack_from("<H", data, value_offset)[0]
                        struct.pack_into("<H", data, value_offset, value + 1)
                        metadata.write_bytes(data)
                        break
                    offset += size
                else:
                    self.fail("reconstructable quantization metadata missing")
            original(command, log, phase)

        with patch.object(cli, "run_native", side_effect=corrupt_before_index), self.assertRaisesRegex(
            cli.ConversionError, "metadata mismatch.*quantization"
        ):
            self.convert()
        self.assertIn("Exact DCT verification", phases)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".*fastlens-staging-*")), [])


class DefaultPartitionCanary(NativeCliFixture):
    """Run separately for the default 1024-image partition boundary."""

    def test_1025_images_with_default_partition_size(self):
        self.extend_inputs(1025)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main([
                "convert", "--input-list", str(self.listing), "--output", str(self.output), "--workers", "4",
            ]), 0)
        report = json.loads((self.output / cli.REPORT_NAME).read_text())
        self.assertEqual(report["partition_images"], 1024)
        self.assertEqual(report["input_image_count"], 1025)
        self.assertEqual(report["shard_count"], 2)
        self.assertEqual(report["verification"], "passed")
        info = cli.native_inspect(self.output, self.tools[cli.TOOL_NAMES[1]])
        self.assertTrue(info["index"]["compatible"])
        self.assertEqual(info["index_file_count"], 3)
        for image_id in (1023, 1024):
            result = subprocess.run([
                str(self.tools[cli.TOOL_NAMES[0]]), "--verify-manifest", str(self.output / "manifest.bin"),
                "--image-index", str(image_id), str(self.paths[image_id]),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
