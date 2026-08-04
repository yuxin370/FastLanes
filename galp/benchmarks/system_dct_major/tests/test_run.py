from __future__ import annotations

import sys
import tempfile
import unittest
import os
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from run import (  # noqa: E402
    _block_major_access_contract,
    _prepare_output_dir,
    _resolve_torch_binding_artifact,
    _run_streamed,
    _sample_plan,
    parse_args,
)


class RunContractTest(unittest.TestCase):
    def test_descriptor_contract_freezes_every_shard_and_storage_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.block_major_access.bin").write_bytes(b"index")
            for shard_id in (0, 4):
                (root / f"shard_{shard_id:06d}.block_major_access.bin").write_bytes(
                    bytes(100 + shard_id)
                )
            storage = {
                "persistent_bytes": 1_000_000,
                "manifest": {"size_bytes": 1000},
            }
            result = _block_major_access_contract(root, storage, (0, 4))
            self.assertEqual(result["shard_count"], 2)
            self.assertEqual(len(result["shards"]), 2)
            self.assertTrue(result["passes_one_percent"])
            self.assertTrue(result["passes_half_percent"])
            self.assertTrue(all("sha256" in item for item in result["shards"]))

    def test_descriptor_contract_rejects_storage_above_one_percent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.block_major_access.bin").write_bytes(bytes(100))
            (root / "shard_000000.block_major_access.bin").write_bytes(bytes(1000))
            storage = {"persistent_bytes": 10_000, "manifest": {"size_bytes": 100}}
            with self.assertRaisesRegex(ValueError, "exceeds the 1% storage gate"):
                _block_major_access_contract(root, storage, (0,))

    def test_default_matrix_includes_same_crop_legacy_baseline(self) -> None:
        args = parse_args(
            ["--output-dir", "/tmp/unused-dct-major-test-output"]
        )
        self.assertIn("dct_major_legacy_pushdown", args.pipelines)
        self.assertIn("dct_major_pushdown", args.pipelines)
        self.assertEqual(args.decode_workset_capacity_mib, 512)
        self.assertEqual(args.block_major_double_buffer, "auto")
        self.assertTrue(args.cold_start_model_prime)

    def test_unified_layout_matrix_accepts_explicit_v2_and_v3_pipelines(self) -> None:
        args = parse_args(
            [
                "--output-dir",
                "/tmp/unused-dct-major-test-output",
                "--pipelines",
                "dct_major_pushdown",
                "image_major_v2_pushdown",
                "image_major_v3_pushdown",
                "rgbnomore",
                "dali",
                "pytorch",
            ]
        )
        self.assertEqual(
            args.pipelines,
            [
                "dct_major_pushdown",
                "image_major_v2_pushdown",
                "image_major_v3_pushdown",
                "rgbnomore",
                "dali",
                "pytorch",
            ],
        )
        self.assertEqual(args.image_major_manifest_version, 2)
        self.assertEqual(args.image_major_v3_manifest.name, "manifest.bin")
        self.assertEqual(args.image_major_v3_manifest.parent.name, "compact_v3_tiled_z32")

    def test_explicit_sample_count_preserves_partial_tail(self) -> None:
        args = parse_args(
            [
                "--output-dir",
                "/tmp/unused-dct-major-test-output",
                "--preset",
                "smoke",
                "--sample-count",
                "5",
            ]
        )
        self.assertEqual(_sample_plan(args), (2, 1, 2, 5))

    def test_explicit_batch_count_must_cover_sample_count(self) -> None:
        args = parse_args(
            [
                "--output-dir",
                "/tmp/unused-dct-major-test-output",
                "--preset",
                "smoke",
                "--measurement-batches",
                "2",
                "--sample-count",
                "7",
            ]
        )
        with self.assertRaisesRegex(ValueError, "does not consume sample_count"):
            _sample_plan(args)

    def test_nonempty_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            _prepare_output_dir(output)
            (output / "occupied.txt").write_text("existing\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _prepare_output_dir(output)

    def test_torch_binding_artifact_must_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "found 0"):
                _resolve_torch_binding_artifact(root)
            expected = root / "_galp_direct_dct.cpython-test.so"
            expected.touch()
            self.assertEqual(_resolve_torch_binding_artifact(root), expected.resolve())
            (root / "_galp_direct_dct.second.so").touch()
            with self.assertRaisesRegex(ValueError, "found 2"):
                _resolve_torch_binding_artifact(root)

    def test_spawn_to_first_output_marker_excludes_later_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "child.log"
            code, wall_seconds, first_output_seconds = _run_streamed(
                [
                    sys.executable,
                    "-c",
                    (
                        "import time; "
                        "print('GALP_FIRST_OUTPUT_READY {}', flush=True); "
                        "time.sleep(0.05); "
                        "print('artifact serialization complete', flush=True)"
                    ),
                ],
                env=os.environ.copy(),
                log=log,
                dry_run=False,
            )
            self.assertEqual(code, 0)
            self.assertIsNotNone(first_output_seconds)
            assert first_output_seconds is not None
            self.assertLess(first_output_seconds, wall_seconds)
            self.assertGreaterEqual(wall_seconds - first_output_seconds, 0.04)


if __name__ == "__main__":
    unittest.main()
