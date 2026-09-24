from __future__ import annotations

import unittest
from pathlib import Path

from galp.benchmarks.system_rgbnomore.training import prepare_imagenet512_v3_train as prepare


class PrepareImagenet512ParallelismTest(unittest.TestCase):
    @staticmethod
    def _args(*argv: str):
        args = prepare._parse_args(list(argv))
        return args

    def test_independent_defaults_are_explicit(self) -> None:
        args = self._args()
        self.assertEqual(args.layout_threads, 32)
        self.assertEqual(args.shard_decode_threads, 4)
        self.assertEqual(args.shard_workers, 4)
        self.assertEqual(args.encoding_workers_per_shard, 1)
        self.assertEqual(args.verify_workers, 16)



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
