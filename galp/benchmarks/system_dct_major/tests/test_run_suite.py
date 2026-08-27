from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from run_suite import (  # noqa: E402
    _contract_phase,
    _formal_phases,
    _initial_phases,
    _semantic_phase,
    _volume,
    parse_args,
    run,
)


class CompleteSuiteTest(unittest.TestCase):
    @staticmethod
    def _args(*extra: str):
        return parse_args(
            [
                "--output-dir",
                "/tmp/unused-dct-major-suite",
                "--block-major-access-dir",
                "/tmp/unused-block-major-access",
                *extra,
            ]
        )

    def test_default_volume_and_phase_matrix(self) -> None:
        args = self._args()
        output = Path("/tmp/unused-dct-major-suite")
        initial = _initial_phases(args, output)
        formal = _formal_phases(args, output)

        self.assertEqual(len(initial), 2)
        self.assertEqual(len(formal), 6)
        self.assertEqual(_volume(args)["total_model_invocations"], 2_570_496)
        feature = next(phase for phase in formal if phase.name == "06_formal_feature_extraction")
        for removed in (
            "--dct-major-segment-size",
            "--decode-workset-capacity-mib",
            "--block-major-double-buffer",
            "--dct-major-crop-execution-mode",
        ):
            self.assertNotIn(removed, feature.command)
        self.assertNotIn("dct_major_full", feature.command)
        self.assertNotIn("dct_major_legacy_pushdown", feature.command)
        self.assertIn("dct_major_coefficient_pushdown", feature.command)
        self.assertEqual(
            feature.command[feature.command.index("--dct-coeffs") + 1],
            "first:32",
        )
        semantic = _semantic_phase(args, output)
        contract = _contract_phase(args, output)
        self.assertFalse(contract.gpu)
        self.assertEqual(contract.command[-1], "--dry-run")
        self.assertIn("00_semantic_contract/contract.json", semantic.command[3])
        self.assertIn("verify_coefficient_semantics.py", semantic.command[1])
        self.assertEqual(semantic.command[-1], "32")

    def test_removed_single_choice_options_are_rejected(self) -> None:
        for option in (
            "--segment-sizes",
            "--decode-workset-capacity-mib",
            "--block-major-double-buffer",
            "--include-full-in-formal",
            "--gates-only",
            "--plan-audit-tool",
        ):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self._args(option)

    def test_dry_run_writes_only_the_production_phase_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            access_dir = root / "access"
            access_dir.mkdir()
            output = root / "output"
            args = parse_args(
                [
                    "--output-dir",
                    str(output),
                    "--block-major-access-dir",
                    str(access_dir),
                    "--python",
                    sys.executable,
                    "--dry-run",
                ]
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
            plan = json.loads((output / "suite_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["runtime_policy"], "native block-major production profile")
            self.assertEqual(plan["dct_coeffs"], "first:32")
            self.assertEqual(len(plan["phases"]), 10)
            self.assertEqual(plan["phases"][0]["name"], "00_semantic_contract")
            self.assertEqual(plan["phases"][1]["name"], "01_coefficient_semantics")
            commands = [item for phase in plan["phases"] for item in phase["command"]]
            self.assertNotIn("--compare-legacy", commands)
            self.assertNotIn("--segment-sizes", commands)

    def test_execution_runs_the_ten_production_phases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            access_dir = root / "access"
            access_dir.mkdir()
            output = root / "output"
            args = parse_args(
                [
                    "--output-dir",
                    str(output),
                    "--block-major-access-dir",
                    str(access_dir),
                    "--python",
                    sys.executable,
                ]
            )

            with mock.patch("run_suite._execute_phase") as execute_phase:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run(args), 0)
            self.assertEqual(execute_phase.call_count, 10)
            result = json.loads((output / "suite_results.json").read_text(encoding="utf-8"))
            self.assertTrue(result["ok"])
            self.assertNotIn("selected_segment_size", result)
            self.assertNotIn("crop_abba", result)


if __name__ == "__main__":
    unittest.main()
