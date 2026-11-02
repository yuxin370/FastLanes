"""CPU-only CLI contracts; no native binaries, Pillow, torch or CUDA required."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastlens import cli


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fastlens-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "images"
        self.source.mkdir()
        self.output = self.root / "store"

    def args(self, *extra):
        return cli.make_parser().parse_args(["convert", str(self.source), "--output", str(self.output), *extra])

    def inputs(self):
        paths = [self.source / name for name in ("z space.jpg", "A.JPEG", "nested/b.jpe")]
        for path in paths:
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"fixture")
        return paths

    def test_defaults_and_help_without_native_imports(self):
        args = self.args()
        self.assertEqual((args.layout, args.partition_images, args.preset, args.workers, args.verify),
                         ("block-major", 1024, "balanced", "auto", "full"))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
            cli.main(["convert", "--help"])
        self.assertEqual(result.exception.code, 0)

    def test_invalid_arguments(self):
        for extra in (("--layout", "image-major"), ("--workers", "0"), ("--partition-images", "-1"),
                      ("--verify", "quick"), ("--preset", "random-access"), ("--layout-threads", "no")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*extra)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.make_parser().parse_args(["convert", str(self.source)])

    def test_parallel_limits_and_overrides(self):
        for budget in (1, 2, 8, 256):
            config = cli.parallel_config(self.args("--workers", str(budget)))
            self.assertLessEqual(config["shard_workers"] * max(config["shard_decode_threads"], config["encoding_workers_per_shard"]), budget)
        for options in (("--workers", "257"), ("--layout-threads", "257"), ("--verify-workers", "33"),
                        ("--shard-workers", "129", "--shard-decode-threads", "2")):
            with self.subTest(options=options), self.assertRaises(cli.ConversionError):
                cli.parallel_config(self.args(*options))
        config = cli.parallel_config(self.args("--workers", "1", "--layout-threads", "3", "--encoding-workers-per-shard", "2"))
        self.assertEqual(config["layout_threads"], 3)
        self.assertEqual(config["encoding_workers_per_shard"], 2)
        self.assertEqual(cli.parallel_config(self.args("--verify-workers", "32"))["verify_workers"], 32)

    def test_auto_uses_affinity_conservatively(self):
        with patch.object(os, "sched_getaffinity", return_value=set(range(512))):
            config = cli.parallel_config(self.args())
        self.assertEqual(config["layout_threads"], 32)
        self.assertEqual(config["shard_workers"], 4)
        self.assertLessEqual(config["verify_workers"], 16)

    def test_directory_order_is_stable_and_recursive(self):
        paths = self.inputs()
        (self.source / "ignored.png").write_bytes(b"ignored")
        actual, size, _ = cli.discover_inputs(self.source, None)
        self.assertEqual(actual, [paths[1], paths[2], paths[0]])
        self.assertEqual(size, 21)

    def test_ordered_list_relative_paths_spaces_and_blank_lines(self):
        paths = self.inputs()
        listing = self.root / "ordered.txt"
        listing.write_bytes(b"images/z space.jpg\r\n\r\nimages/nested/b.jpe\r\nimages/A.JPEG\r\n")
        self.assertEqual(cli.discover_inputs(None, listing)[0], [paths[0], paths[2], paths[1]])

    def test_rejects_missing_duplicate_empty_and_conflicting_inputs(self):
        listing = self.root / "ordered.txt"
        for content, error in (("", "no JPEG"), ("missing.jpg\n", "missing")):
            listing.write_text(content)
            with self.assertRaisesRegex(cli.ConversionError, error):
                cli.discover_inputs(None, listing)
        with self.assertRaisesRegex(cli.ConversionError, "no JPEG"):
            cli.discover_inputs(self.source, None)
        paths = self.inputs()
        alias = self.source / "alias.jpg"
        alias.symlink_to(paths[0])
        listing.write_text(f"{paths[0]}\n{alias}\n")
        with self.assertRaisesRegex(cli.ConversionError, "duplicate"):
            cli.discover_inputs(None, listing)
        for inputs in ((self.source, listing), (None, None)):
            with self.assertRaisesRegex(cli.ConversionError, "exactly one"):
                cli.discover_inputs(*inputs)

    def test_tool_discovery_precedence_and_missing_error(self):
        build = self.root / "build"
        directory = build / "fastlens/tools/jpeg_dct"
        directory.mkdir(parents=True)
        for name in cli.TOOL_NAMES:
            path = directory / name
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
        with patch.dict(os.environ, {"FASTLENS_BUILD_DIR": str(build)}):
            self.assertEqual(cli.find_tools(None)[cli.TOOL_NAMES[0]], directory / cli.TOOL_NAMES[0])
            with self.assertRaisesRegex(cli.ConversionError, "cmake --build"):
                cli.find_tools(self.root / "missing")

    def test_nonempty_and_symlink_destinations_are_protected(self):
        self.output.mkdir()
        marker = self.output / "manifest.bin"
        marker.write_bytes(b"existing dataset")
        with self.assertRaises(cli.ConversionError):
            cli.convert(self.args())
        self.assertEqual(marker.read_bytes(), b"existing dataset")
        link = self.root / "link"
        link.symlink_to(self.output, target_is_directory=True)
        with self.assertRaises(cli.ConversionError):
            cli.check_destination(link)

    @patch.object(cli, "find_tools", return_value={name: Path("/unused") / name for name in cli.TOOL_NAMES})
    def test_dry_run_writes_nothing(self, _):
        self.inputs()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(cli, "run_native") as native:
            self.assertEqual(cli.convert(self.args("--dry-run")), 0)
        native.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".*fastlens-staging-*")), [])

    @patch.object(cli, "find_tools", return_value={name: Path("/unused") / name for name in cli.TOOL_NAMES})
    def test_shuffle_dry_run_and_annotation_errors_precede_compression(self, _):
        self.inputs()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(cli, "run_native") as native:
            self.assertEqual(cli.convert(self.args("--shuffle-seed", "42", "--dry-run")), 0)
            with self.assertRaisesRegex(ValueError, "shuffle seed"):
                cli.convert(self.args("--shuffle-seed", "-1"))
            labels = self.root / "labels.jsonl"
            labels.write_text('{"path":"images/A.JPEG","label":null}\n')
            with self.assertRaisesRegex(ValueError, "missing annotation"):
                cli.convert(self.args("--labels-jsonl", str(labels)))
        native.assert_not_called()
        self.assertFalse(self.output.exists())

    @patch.object(cli, "find_tools", return_value={name: Path("/unused") / name for name in cli.TOOL_NAMES})
    def test_each_failed_phase_cleans_staging_and_preserves_empty_output(self, _):
        self.inputs()
        self.output.mkdir()
        info = {"manifest_version": 1, "physical_layout": cli.PHYSICAL_LAYOUT,
                "index": {"compatible": True}, "image_count": 3, "jpeg_warning_image_ids": []}
        for failure in ("Compression", "Access index", "Exact DCT verification"):
            def run(command, log, phase):
                self.assertTrue(self.output.is_dir())
                self.assertEqual(list(self.output.iterdir()), [])
                if phase == failure:
                    raise cli.ConversionError("injected phase failure")
            with self.subTest(failure=failure), patch.object(cli, "run_native", side_effect=run), \
                    patch.object(cli, "native_inspect", return_value=info), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(cli.ConversionError, "injected"):
                    cli.convert(self.args())
            self.assertEqual(list(self.output.iterdir()), [])
            self.assertEqual(list(self.root.glob(".*fastlens-staging-*")), [])

    def test_report_counts_its_own_bytes(self):
        path = self.root / cli.REPORT_NAME
        report = {"schema_version": 1, "original_jpeg_bytes": 321}
        cli.write_report(path, report, 100)
        actual = json.loads(path.read_text())
        self.assertEqual(actual["total_output_bytes"], 100 + path.stat().st_size)
        self.assertEqual(actual["source_to_output_ratio"], 321 / actual["total_output_bytes"])


if __name__ == "__main__":
    unittest.main()
