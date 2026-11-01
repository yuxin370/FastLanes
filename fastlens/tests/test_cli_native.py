"""Native CLI integration; set FASTLENS_REQUIRE_NATIVE_TESTS=1 to require prerequisites."""

import contextlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastlens import cli


class NativeCliTests(unittest.TestCase):
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
                               "--verify-workers", "2", "--input-list", str(listing or self.output / "source_paths.txt")],
                              capture_output=True, text=True)

    def test_complete_conversion_inspection_relocation_and_order(self):
        external = self.root / "report.json"
        self.assertEqual(self.convert(extra=("--report", str(external))), 0)
        self.assertEqual((self.output / "source_paths.txt").read_text().splitlines(), [str(path) for path in self.paths])
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
        for field, pattern in (("coefficient", "*.fls"), ("metadata", "*.meta.bin")):
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
        self.assertEqual(json.loads(inspected.stdout)["total_stored_bytes"], report["total_output_bytes"])
        before = (moved / "manifest.bin").read_bytes()
        with self.assertRaisesRegex(cli.ConversionError, "refusing to replace"):
            self.convert()
        self.assertEqual((moved / "manifest.bin").read_bytes(), before)

    def test_directory_conversion_order_and_none_verification(self):
        self.convert(directory=True, extra=("--verify", "none"))
        self.assertEqual((self.output / "source_paths.txt").read_text().splitlines(), [str(path) for path in sorted(self.paths)])
        report = json.loads((self.output / cli.REPORT_NAME).read_text())
        self.assertEqual(report["verification"], "not_requested")
        self.assertEqual(report["wall_times"]["verification_seconds"], 0)
        self.assertEqual(self.verify().returncode, 0)

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
        for failing in ("Access index", "Exact DCT verification"):
            def fail(command, log, phase):
                if phase == failing:
                    if phase == "Exact DCT verification":
                        # Corrupt the actual native dataset before calling the verifier.
                        next(log.parent.rglob("*.fls")).write_bytes(b"broken")
                    else:
                        raise cli.ConversionError("injected index failure")
                original(command, log, phase)
            with self.subTest(phase=failing), patch.object(cli, "run_native", side_effect=fail), self.assertRaises(cli.ConversionError):
                self.convert()
            self.assertFalse(self.output.exists())
            self.assertEqual(list(self.root.glob(".*fastlens-staging-*")), [])

    def test_inspect_missing_and_incompatible_index(self):
        self.convert()
        index = self.output / "block_major_access_v1"
        descriptor = next(index.glob("shard_*.bin"))
        data = bytearray(descriptor.read_bytes())
        data[-1] ^= 1
        descriptor.write_bytes(data)
        with self.assertRaisesRegex(cli.ConversionError, "inspection failed"):
            cli.native_inspect(self.output, self.tools[cli.TOOL_NAMES[1]])
        shutil.rmtree(index)
        self.assertFalse(cli.native_inspect(self.output, self.tools[cli.TOOL_NAMES[1]])["index"]["present"])
        (self.output / cli.REPORT_NAME).unlink()
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(cli.main(["inspect", str(self.output), "--json"]), 0)
        self.assertIsNone(json.loads(stream.getvalue())["original_jpeg_bytes"])

    def test_single_image_ordered_list_verifies_entire_manifest(self):
        self.convert()
        single = self.root / "one.txt"
        single.write_text(str(self.paths[0]) + "\n")
        result = subprocess.run([str(self.tools[cli.TOOL_NAMES[0]]), "--verify-manifest", str(self.output / "manifest.bin"),
                                 "--verify-workers", "1", "--input-list", str(single)], capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
