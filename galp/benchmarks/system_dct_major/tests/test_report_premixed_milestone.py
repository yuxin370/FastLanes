#!/usr/bin/env python3
"""Tests for strict Premixed single-seed milestone reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training_pls.matrix import CORE_CONDITION_IDS
from training_pls.report_premixed_milestone import (
    EXPECTED_POLICIES,
    effect_rows,
    generate_report,
    partial_normalized_auc,
)


class PremixedMilestoneReportTests(unittest.TestCase):
    def make_runs(self, root: Path, *, omit_validation: str | None = None) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for index, condition in enumerate(CORE_CONDITION_IDS):
            run = root / condition
            run.mkdir(parents=True)
            crop, order, segments = EXPECTED_POLICIES[condition]
            contract = {
                "condition_id": condition,
                "training_seed": 7,
                "recipe_hash": "recipe",
                "layout_hash": "layout",
                "initial_model_hash": "initial",
                "backend_implementation": "backend",
                "execution_mode": "semantic_emulation",
                "epochs": 300,
                "microbatch_size": 64,
                "gradient_accumulation": 16,
                "effective_update_batch": 1024,
                "crop_policy": crop,
                "order_policy": order,
                "segments_per_pool": segments,
                "condition_hash": f"hash-{condition}",
            }
            (run / "condition_contract.json").write_text(json.dumps(contract) + "\n", encoding="utf-8")
            (run / "run_status.json").write_text('{"completed_epoch":10}\n', encoding="utf-8")
            (run / "checkpoint_epoch_010.pt").write_bytes(b"checkpoint")
            metrics = [{
                "record_type": "validation", "condition": condition, "seed": 7,
                "epoch": 0, "optimizer_update": 0, "processed_images": 0,
                "validation_top1": float(index), "validation_top5": float(index + 20),
                "validation_loss": float(5 - index), "validation_latency_seconds": 1.0,
                "validation_samples": 50,
            }]
            if condition != omit_validation:
                metrics.append({
                    "record_type": "validation", "condition": condition, "seed": 7,
                    "epoch": 10, "optimizer_update": 100, "processed_images": 1000,
                    "validation_top1": float(10 + index), "validation_top5": float(30 + index),
                    "validation_loss": float(4 - index), "validation_latency_seconds": 1.0,
                    "validation_samples": 50,
                })
            (run / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8"
            )
            result[condition] = run
        return result

    def test_factorial_effect_formulas(self) -> None:
        values = {"A0": 1.0, "A1": 3.0, "B2": 5.0, "B6": 11.0}
        fixed = [{
            "condition": condition,
            "validation_top1": value,
            "validation_top5": value,
            "validation_loss": value,
        } for condition, value in values.items()]
        rows = {row["effect_id"]: row for row in effect_rows(fixed, epoch=10, seed=7) if row["endpoint"] == "validation_top1"}
        self.assertEqual(rows["crop_global"]["single_seed_difference"], 2.0)
        self.assertEqual(rows["crop_closed_pool"]["single_seed_difference"], 6.0)
        self.assertEqual(rows["crop_main"]["single_seed_difference"], 4.0)
        self.assertEqual(rows["shuffle_main"]["single_seed_difference"], 6.0)
        self.assertEqual(rows["crop_x_shuffle"]["single_seed_difference"], 4.0)
        self.assertEqual(rows["combined_vs_standard"]["single_seed_difference"], 10.0)

    def test_partial_auc_uses_processed_images_and_normalizes_horizon(self) -> None:
        records = [
            {"processed_images": 0, "validation_top1": 10.0},
            {"processed_images": 25, "validation_top1": 20.0},
            {"processed_images": 100, "validation_top1": 40.0},
        ]
        self.assertAlmostEqual(partial_normalized_auc(records), 26.25)

    def test_complete_fixture_generates_strict_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self.make_runs(root / "runs")
            output = root / "report"
            result = generate_report(runs, output, seed=7, epoch=10)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["fixed_budget_metrics"]), 4)
            for name in (
                "fixed_budget_metrics_epoch_010.csv",
                "convergence_curves_to_epoch_010.csv",
                "single_seed_effects_epoch_010.csv",
                "partial_auc_epoch_010.csv",
                "top1_convergence_to_epoch_010.png",
                "top5_convergence_to_epoch_010.png",
                "validation_loss_convergence_to_epoch_010.png",
                "fixed_budget_top1_epoch_010.png",
                "single_seed_effects_epoch_010.png",
                "partial_auc_epoch_010.png",
                "milestone_report_epoch_010.md",
                "milestone_results_epoch_010.json",
            ):
                self.assertTrue((output / name).is_file(), name)

    def test_missing_exact_validation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs = self.make_runs(Path(directory) / "runs", omit_validation="B6")
            with self.assertRaisesRegex(ValueError, "B6.*lacks exact epoch-10 validation"):
                generate_report(runs, Path(directory) / "report", seed=7, epoch=10)


if __name__ == "__main__":
    unittest.main()
