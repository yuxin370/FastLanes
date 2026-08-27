#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path


from galp.experiments.coefficient_mask_evaluator.evaluate import (
    load_progress,
    make_run_signature,
)
from galp.experiments.coefficient_mask_evaluator.masks import build_conditions


class ResumeContractTest(unittest.TestCase):
    @staticmethod
    def args(root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            manifest=root / "manifest.json",
            checkpoint=root / "checkpoint.pt",
            rgbnomore_root=root / "rgbnomore",
            random_seed=17,
            device="cuda:0",
            preprocess_device=None,
            expected_device_name="test-gpu",
            batch_size=8,
            condition_chunk_size=4,
            torch_cpu_threads=2,
            workers=2,
            prefetch_factor=2,
            no_wide_csv=False,
        )

    def test_source_fingerprints_are_part_of_resume_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            samples = [{"logical_sample_id": "sample-0"}]
            conditions = build_conditions()[:1]
            first = make_run_signature(
                args,
                samples,
                conditions,
                "manifest-hash",
                "checkpoint-hash",
                {"evaluator/evaluate.py": {"path": "/evaluate.py", "sha256": "old"}},
                {"model": "test-gpu"},
            )
            second = make_run_signature(
                args,
                samples,
                conditions,
                "manifest-hash",
                "checkpoint-hash",
                {"evaluator/evaluate.py": {"path": "/evaluate.py", "sha256": "new"}},
                {"model": "test-gpu"},
            )
            self.assertNotEqual(first, second)
            self.assertIn("execution_fingerprint", first)

    def test_resume_loads_accumulated_inference_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "progress.json").write_text(
                json.dumps({"completed_samples": 23, "inference_seconds": 4.5}),
                encoding="utf-8",
            )
            progress = load_progress(output_dir, resume=True)
            self.assertEqual(progress.completed_samples, 23)
            self.assertEqual(progress.inference_seconds, 4.5)

    def test_legacy_resume_time_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "progress.json").write_text(
                json.dumps(
                    {"completed_samples": 23, "elapsed_seconds_this_process": 4.5}
                ),
                encoding="utf-8",
            )
            progress = load_progress(output_dir, resume=True)
            self.assertEqual(progress.inference_seconds, 4.5)


if __name__ == "__main__":
    unittest.main()
