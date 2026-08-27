#!/usr/bin/env python3
"""Tests for the live premixed-progress plotter."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from training_pls.matrix import CORE_CONDITION_IDS
from training_pls.plot_premixed_progress import (
    generate_plots,
    parse_run_overrides,
    read_canonical_validation_csv,
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

    def test_canonical_csv_replaces_live_metrics_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical.csv"
            fields = (
                "condition",
                "seed",
                "epoch",
                "optimizer_update",
                "processed_images",
                "validation_top1",
                "validation_top5",
                "validation_loss",
                "checkpoint_sha256",
                "validation_gpu",
                "canonical_fresh_process",
                "inline_superseded",
            )
            with canonical.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for index, condition in enumerate(CORE_CONDITION_IDS):
                    for epoch in (0, 10):
                        writer.writerow(
                            {
                                "condition": condition,
                                "seed": 11997733,
                                "epoch": epoch,
                                "optimizer_update": epoch * 1252,
                                "processed_images": epoch * 1281167,
                                "validation_top1": 40.0 + index + epoch,
                                "validation_top5": 60.0 + index + epoch,
                                "validation_loss": 2.0 - epoch / 100.0,
                                "checkpoint_sha256": f"sha-{condition}-{epoch}",
                                "validation_gpu": "NVIDIA GeForce RTX 4090",
                                "canonical_fresh_process": "True",
                                "inline_superseded": "True",
                            }
                        )

            records, sources = read_canonical_validation_csv(
                canonical, seed=11997733
            )
            self.assertEqual(records["B6"][-1]["validation_top1"], 53.0)
            self.assertTrue(sources["A0"]["all_rows_canonical_fresh_process"])

            output_dir = root / "plots"
            result = generate_plots(
                {condition: root / condition for condition in CORE_CONDITION_IDS},
                output_dir,
                seed=11997733,
                canonical_csv=canonical,
            )
            self.assertEqual(result["data_mode"], "fresh-process-canonical")
            self.assertEqual(result["missing_conditions"], [])
            with (output_dir / "premixed_validation_progress.csv").open(
                newline="", encoding="utf-8"
            ) as source:
                plotted = list(csv.DictReader(source))
            self.assertEqual(len(plotted), 8)
            self.assertEqual(plotted[0]["source_validation"], str(canonical.resolve()))
            self.assertEqual(plotted[0]["canonical_fresh_process"], "True")

    def test_canonical_csv_rejects_non_fresh_process_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            canonical = Path(temporary) / "canonical.csv"
            row = {
                **validation("A0", 10, 30.0),
                "checkpoint_sha256": "sha-a0-10",
                "validation_gpu": "NVIDIA GeForce RTX 4090",
                "canonical_fresh_process": "False",
            }
            fields = tuple(row)
            with canonical.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "canonical_fresh_process"):
                read_canonical_validation_csv(canonical, seed=11997733)

    def test_run_overrides_are_explicit_and_unique(self) -> None:
        parsed = parse_run_overrides(["A0=/tmp/a0", "B6=/tmp/b6"])
        self.assertEqual(parsed["A0"], Path("/tmp/a0"))
        with self.assertRaises(ValueError):
            parse_run_overrides(["A0=/tmp/one", "A0=/tmp/two"])
        with self.assertRaises(ValueError):
            parse_run_overrides(["unknown=/tmp/value"])


if __name__ == "__main__":
    unittest.main()
