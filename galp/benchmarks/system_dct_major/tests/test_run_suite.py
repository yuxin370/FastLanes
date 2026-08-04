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
    _initial_phases,
    _parse_segment_sizes,
    _preflight_phases,
    _select_best_segment,
    _selected_phases,
    _volume,
    parse_args,
    run,
)


class CompleteSuiteTest(unittest.TestCase):
    def test_default_volume_and_phase_matrix(self) -> None:
        args = parse_args(
            [
                "--output-dir",
                "/tmp/unused-dct-major-suite",
                "--block-major-access-dir",
                "/tmp/unused-block-major-access",
            ]
        )
        output = Path("/tmp/unused-dct-major-suite")
        initial = _initial_phases(args, output)
        selected = _selected_phases(args, output, 1024)

        self.assertEqual(args.segment_sizes, (50, 250, 500, 1000, 1024))
        self.assertEqual(len(initial), 13)
        self.assertEqual(len(selected), 8)
        self.assertEqual(_volume(args)["total_model_invocations"], 3_139_084)
        formal = next(phase for phase in selected if phase.name == "06_formal_feature_extraction")
        self.assertIn("1024", formal.command)
        self.assertIn("--decode-workset-capacity-mib", formal.command)
        self.assertIn("512", formal.command)
        self.assertIn("--block-major-double-buffer", formal.command)
        self.assertIn("auto", formal.command)
        self.assertNotIn("dct_major_full", formal.command[formal.command.index("--pipelines") + 1 :])
        gate = next(phase for phase in initial if phase.name == "00_planless_gpu_gate_sequential")
        self.assertTrue(gate.gpu)
        self.assertIn("--execute", gate.command)
        self.assertIn("--compare-legacy", gate.command)
        self.assertIn("/tmp/unused-block-major-access", gate.command)
        gates = [phase for phase in initial if phase.name.startswith("00_planless_gpu_gate_")]
        self.assertEqual(len(gates), 5)
        self.assertTrue(any("--pattern" in phase.command and "random" in phase.command for phase in gates))
        self.assertTrue(any("--explicit-crops" in phase.command for phase in gates))
        self.assertTrue(any("--require-cross-shard" in phase.command for phase in gates))
        self.assertTrue(any("--require-grayscale" in phase.command for phase in gates))
        planning = next(phase for phase in initial if phase.name == "00_segment_planning")
        self.assertIn("--block-major-access-dir", planning.command)
        self.assertIn("/tmp/unused-block-major-access", planning.command)

    def test_gates_only_dry_run_contains_no_model_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            access_dir = root / "access"
            access_dir.mkdir()
            audit_tool = root / "audit-tool"
            audit_tool.touch()
            output = root / "output"
            args = parse_args(
                [
                    "--output-dir",
                    str(output),
                    "--block-major-access-dir",
                    str(access_dir),
                    "--plan-audit-tool",
                    str(audit_tool),
                    "--python",
                    sys.executable,
                    "--gates-only",
                    "--dry-run",
                ]
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
            plan = json.loads((output / "suite_plan.json").read_text(encoding="utf-8"))
            self.assertTrue(plan["gates_only"])
            self.assertEqual(plan["volume"]["gpu_gate_requests"], 23)
            self.assertEqual(plan["volume"]["total_model_invocations"], 0)
            self.assertEqual(len(plan["initial_phases"]), 6)
            self.assertEqual(plan["selected_phase_template"], [])
            commands = [item for phase in plan["initial_phases"] for item in phase["command"]]
            self.assertNotIn(str(BENCHMARK_ROOT / "run.py"), commands)
            self.assertNotIn(str(BENCHMARK_ROOT / "diagnostics/model_ceiling.py"), commands)

            preflight = _preflight_phases(args, output)
            self.assertEqual(sum(phase.gpu for phase in preflight), 5)

    def test_gates_only_execution_exits_after_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            access_dir = root / "access"
            access_dir.mkdir()
            audit_tool = root / "audit-tool"
            audit_tool.touch()
            output = root / "output"
            args = parse_args(
                [
                    "--output-dir",
                    str(output),
                    "--block-major-access-dir",
                    str(access_dir),
                    "--plan-audit-tool",
                    str(audit_tool),
                    "--python",
                    sys.executable,
                    "--gates-only",
                ]
            )

            with mock.patch("run_suite._execute_phase") as execute_phase:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run(args), 0)
            self.assertEqual(execute_phase.call_count, 6)
            executed_names = [call.args[0].name for call in execute_phase.call_args_list]
            self.assertEqual(executed_names[0], "00_segment_planning")
            self.assertTrue(all(name.startswith("00_") for name in executed_names))
            result = json.loads((output / "suite_results.json").read_text(encoding="utf-8"))
            self.assertTrue(result["ok"])
            self.assertTrue(result["gates_only"])
            self.assertEqual(len(result["gpu_gate_outputs"]), 5)
            self.assertEqual(result["volume"]["total_model_invocations"], 0)

    def test_exhaustive_formal_adds_full_pipeline_volume(self) -> None:
        args = parse_args(
            [
                "--output-dir",
                "/tmp/unused-dct-major-suite",
                "--block-major-access-dir",
                "/tmp/unused-block-major-access",
                "--include-full-in-formal",
            ]
        )
        self.assertEqual(_volume(args)["formal_pipeline_images"], 3_500_000)

    def test_segment_parser_rejects_duplicates(self) -> None:
        self.assertEqual(_parse_segment_sizes("64,1000,1024"), (64, 1000, 1024))
        with self.assertRaises(Exception):
            _parse_segment_sizes("50,50")

    def test_selects_highest_cold_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for segment, throughput, cold_throughput, steady in (
                (1000, 200.0, 220.0, 400.0),
                (1024, 210.0, 180.0, 390.0),
            ):
                phase = output / f"03_locality_segment_{segment:04d}"
                phase.mkdir(parents=True)
                (phase / "results.json").write_text(
                    json.dumps(
                        {
                            "ok": True,
                            "aggregates": [
                                {
                                    "pipeline": "dct_major_pushdown",
                                    "cold_start": {
                                        "throughput_images_per_s": cold_throughput,
                                        "time_to_first_batch_ms": 12.0,
                                    },
                                    "throughput_images_per_s": {"p50": throughput},
                                    "steady_throughput_images_per_s": {"p50": steady},
                                    "time_to_first_batch_ms": {"p50": 12.0},
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            selected, records = _select_best_segment(output, (1000, 1024))
            self.assertEqual(selected, 1000)
            self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
