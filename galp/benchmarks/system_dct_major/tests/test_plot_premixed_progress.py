#!/usr/bin/env python3
"""Tests for the live premixed-progress plotter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from training_pls.matrix import CORE_CONDITION_IDS
from training_pls.plot_premixed_progress import (
    generate_plots,
    parse_run_overrides,
    read_validation_metrics,
)


def validation(condition: str, epoch: int, top1: float) -> dict[str, object]:
    return {
        "record_type": "validation",
        "condition": condition,
        "seed": 11997733,
        "epoch": epoch,
        "optimizer_update": epoch * 1252,
        "processed_images": epoch * 1281167,
        "validation_top1": top1,
        "validation_top5": top1 + 20.0,
        "validation_loss": 6.0 - top1 / 20.0,
        "validation_latency_seconds": 1.0,
        "validation_samples": 50000,
    }


class PremixedProgressPlotTests(unittest.TestCase):
    def test_live_reader_ignores_partial_tail_and_uses_latest_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            rows = [validation("A0", 0, 0.1), validation("A0", 5, 10.0)]
            with (run / "metrics.jsonl").open("w", encoding="utf-8") as output:
                for row in rows:
                    output.write(json.dumps(row) + "\n")
                output.write(json.dumps(validation("A0", 5, 11.0)) + "\n")
                output.write('{"record_type":"validation"')
            records, metadata = read_validation_metrics(
                run, condition="A0", seed=11997733
            )
            self.assertEqual([row["epoch"] for row in records], [0, 5])
            self.assertEqual(records[-1]["validation_top1"], 11.0)
            self.assertEqual(metadata["ignored_json_lines"], 1)
            self.assertEqual(metadata["duplicate_validation_epochs"], 1)

    def test_generate_refreshes_all_plots_and_audit_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for index, condition in enumerate(CORE_CONDITION_IDS):
                run = root / condition
                run.mkdir()
                paths[condition] = run
                with (run / "metrics.jsonl").open("w", encoding="utf-8") as output:
                    output.write(json.dumps(validation(condition, 0, 0.1)) + "\n")
                    output.write(
                        json.dumps(validation(condition, 10, 30.0 + index)) + "\n"
                    )
            output_dir = root / "plots"
            result = generate_plots(paths, output_dir, seed=11997733)
            expected = (
                "premixed_top1.png",
                "premixed_top5.png",
                "premixed_validation_loss.png",
                "premixed_validation_curves.png",
                "premixed_validation_progress.csv",
                "premixed_progress_summary.json",
            )
            for filename in expected:
                self.assertTrue((output_dir / filename).is_file(), filename)
            with Image.open(output_dir / "premixed_validation_curves.png") as image:
                self.assertGreater(image.width, 1000)
                self.assertGreater(image.height, 2000)
            self.assertEqual(result["missing_conditions"], [])

    def test_run_overrides_are_explicit_and_unique(self) -> None:
        parsed = parse_run_overrides(["A0=/tmp/a0", "B6=/tmp/b6"])
        self.assertEqual(parsed["A0"], Path("/tmp/a0"))
        with self.assertRaises(ValueError):
            parse_run_overrides(["A0=/tmp/one", "A0=/tmp/two"])
        with self.assertRaises(ValueError):
            parse_run_overrides(["unknown=/tmp/value"])


if __name__ == "__main__":
    unittest.main()
