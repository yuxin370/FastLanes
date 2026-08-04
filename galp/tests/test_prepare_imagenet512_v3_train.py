from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/system_rgbnomore/training/prepare_imagenet512_v3_train.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_imagenet512_v3_train", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


class PrepareImagenet512ParallelismTest(unittest.TestCase):
    @staticmethod
    def _args(*argv: str):
        args = prepare._parse_args(list(argv))
        prepare._resolve_parallelism(args)
        return args

    def test_independent_defaults_are_explicit(self) -> None:
        args = self._args()
        self.assertIsNone(args.legacy_compress_threads)
        self.assertEqual(args.layout_threads, 32)
        self.assertEqual(args.shard_decode_threads, 4)
        self.assertEqual(args.shard_workers, 4)
        self.assertEqual(args.encoding_workers_per_shard, 1)
        self.assertEqual(args.verify_workers, 16)

    def test_legacy_control_maps_to_layout_and_decode(self) -> None:
        args = self._args("--compress-threads", "7")
        self.assertEqual(args.legacy_compress_threads, 7)
        self.assertEqual(args.layout_threads, 7)
        self.assertEqual(args.shard_decode_threads, 7)

    def test_conflicting_legacy_and_independent_values_fail(self) -> None:
        args = prepare._parse_args(
            ["--compress-threads", "7", "--layout-threads", "8"]
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            prepare._resolve_parallelism(args)

    def test_commands_record_all_effective_controls(self) -> None:
        args = self._args(
            "--layout-threads",
            "8",
            "--shard-decode-threads",
            "3",
            "--shard-workers",
            "5",
            "--encoding-workers-per-shard",
            "2",
            "--verify-workers",
            "11",
        )
        compress = prepare._compress_command(args)
        verify = prepare._coefficient_verify_command(
            args, args.dct_root / "manifest.bin"
        )
        self.assertNotIn("--threads", compress)
        for option, value in (
            ("--layout-threads", "8"),
            ("--shard-decode-threads", "3"),
            ("--shard-workers", "5"),
            ("--encoding-workers-per-shard", "2"),
        ):
            self.assertEqual(compress[compress.index(option) + 1], value)
        self.assertEqual(verify[verify.index("--verify-workers") + 1], "11")


if __name__ == "__main__":
    unittest.main()
