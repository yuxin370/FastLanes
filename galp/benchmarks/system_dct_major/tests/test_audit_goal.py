#!/usr/bin/env python3
"""Tests for the read-only PLS goal audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training_pls.audit_goal import build_audit, render_markdown, scan_run, write_audit


class GoalAuditTests(unittest.TestCase):
    def test_scan_run_ignores_partial_tail_and_rejects_status_only_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run_status.json").write_text(
                json.dumps({"state": "running", "completed_epoch": 300}) + "\n",
                encoding="utf-8",
            )
            (run / "checkpoint_epoch_300.pt").write_bytes(b"checkpoint")
            (run / "metrics.jsonl").write_text(
                json.dumps({
                    "record_type": "validation", "condition": "A0", "seed": 7,
                    "epoch": 300, "validation_top1": 70.0, "validation_loss": 1.0,
                }) + "\n{" ,
                encoding="utf-8",
            )
            result = scan_run(run, condition="A0", seed=7, active={})
            self.assertEqual(result["completed_epoch"], 300)
            self.assertEqual(result["validation_epochs"], [300])
            self.assertEqual(result["malformed_metrics_lines"], 1)
            self.assertFalse(result["recorded_running_is_active"])
            self.assertFalse(result["epoch_300_complete"])

    def test_limited_pid_namespace_does_not_call_recorded_run_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run_status.json").write_text(
                '{"state":"running","completed_epoch":2}\n', encoding="utf-8"
            )
            result = scan_run(
                run,
                condition="A0",
                seed=7,
                active={},
                process_scan_authoritative=False,
            )
            self.assertIsNone(result["recorded_running_is_active"])
            self.assertEqual(result["activity_evidence"], "unavailable-limited-pid-namespace")

    def test_scan_run_accepts_matching_nested_final_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "final_result.json").write_text(
                json.dumps(
                    {
                        "state": "completed",
                        "condition": "A0",
                        "seed": 7,
                        "checkpoint": {"final_epoch": 300},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = scan_run(run, condition="A0", seed=7, active={})
            self.assertEqual(result["completed_epoch"], 300)
            self.assertEqual(result["final_epoch"], 300)
            self.assertTrue(result["final_result_identity_matches"])
            self.assertTrue(result["final_result_valid"])
            self.assertTrue(result["epoch_300_complete"])

    def test_scan_run_rejects_swapped_final_result_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "final_result.json").write_text(
                json.dumps(
                    {
                        "state": "completed",
                        "condition": "A1",
                        "seed": 13,
                        "checkpoint": {"final_epoch": 300},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = scan_run(run, condition="A0", seed=7, active={})
            self.assertEqual(result["final_result_condition"], "A1")
            self.assertEqual(result["final_result_seed"], 13)
            self.assertFalse(result["final_result_identity_matches"])
            self.assertFalse(result["final_result_valid"])
            self.assertFalse(result["epoch_300_complete"])

    def test_audit_requires_all_registered_final_results_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout_dir = root / "layout"
            report_dir = root / "report"
            layout_dir.mkdir()
            report_dir.mkdir()
            layout = {
                "layout_hash": "layout-hash",
                "sample_mapping_file": "physical_layout_samples.parquet",
                "sample_count": 4,
                "target_pls_size": 1024,
                "virtual_pls_count": 1,
            }
            layout_path = layout_dir / "physical_layout_plan.json"
            layout_path.write_text(json.dumps(layout) + "\n", encoding="utf-8")
            (layout_dir / "physical_layout_samples.parquet").write_bytes(b"mapping")
            run = root / "runs/A0/seed_7"
            run.mkdir(parents=True)
            plan = {
                "layout_hash": "layout-hash",
                "conditions": ["A0"],
                "seeds": [7],
                "commands": [{"condition": "A0", "seed": 7, "output_dir": str(run)}],
            }
            (root / "matrix_execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            for name in (
                "condition_execution_order.json", "recipe_contract.json",
                "analysis_contract.json", "condition_contract_diff.json", "environment.json",
            ):
                (root / name).write_text("{}\n", encoding="utf-8")
            (root / "run_status.csv").write_text("condition\n", encoding="utf-8")
            (root / "failures.jsonl").write_text("", encoding="utf-8")
            audit = build_audit(root, layout_path, report_dir)
            self.assertTrue(audit["formal_matrix"]["registered_matrix_exact"])
            self.assertFalse(audit["goal_complete"])
            self.assertIn("Formal E300 runs: 0/1", render_markdown(audit))

    def test_atomic_outputs_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            audit = {
                "goal_complete": False,
                "layout": {"plan": {"exists": True}, "mapping": {"exists": True}, "matrix_layout_hash_matches": True},
                "formal_matrix": {
                    "registered_command_count": 0, "expected_identity_count": 0,
                    "runs": [], "milestones": [], "epoch_300_complete_count": 0,
                },
                "comparison": None,
                "final_report_artifacts": {},
                "physical_canary_artifacts": {},
                "claim_boundaries": [],
            }
            write_audit(audit, output)
            self.assertTrue((output / "goal_completion_audit.json").is_file())
            self.assertTrue((output / "goal_completion_audit.md").is_file())


if __name__ == "__main__":
    unittest.main()
