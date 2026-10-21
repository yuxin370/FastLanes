"""Small behavioral checks for aggregation and adaptive trial decisions."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from fastlens.benchmarks.development.reports.summarize import correctness_passed
from fastlens.benchmarks.development.reports.summarize import summarize
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.development.diagnostics.analyze_resources import intersection


class SummaryTest(unittest.TestCase):
    def test_resumed_command_metadata_is_not_a_pipeline_result(self):
        import fastlens.benchmarks.development.reports.summarize as module
        run = module.RESULTS / "raw/H100/vitti/inference"
        pipeline = run / "pipeline_coordl.json"
        metadata = run / "pipeline_coordl_resumed.command.json"
        fixtures = {
            module.RESULTS / "inventory/environment.json": dict(evidence=dict(
                cpu=dict(stdout=json.dumps(dict(lscpu=[dict(field="Model name:", data="test CPU")]))),
                gpu=dict(stdout="<nvidia_smi_log><gpu><product_name>4090</product_name></gpu>"
                                "<gpu><product_name>H100</product_name></gpu></nvidia_smi_log>"))),
            run / "results.json": dict(failures=[]),
            run / "contract.json": dict(execution=dict(batch_size=50)),
            run.with_suffix(".command.json"): dict(timestamp="original", git_commit="test"),
            metadata: dict(timestamp="resumed", git_commit="test"),
            pipeline: dict(domain="rgb", pipeline="coordl", repeats=[{}, dict(
                repeat=1, images=50000, repeat_scope_seconds=10,
                peak_torch_gpu_allocated_bytes=100, peak_torch_gpu_reserved_bytes=200,
                accuracy_top1=.7)]),
        }
        def glob(path, pattern):
            if path == module.RESULTS / "raw" and pattern == "*/*/inference*":
                return [run]
            if path == run and pattern == "pipeline_*.json":
                return [pipeline, metadata]
            return []

        with patch.object(module, "read", side_effect=fixtures.__getitem__), \
             patch.object(Path, "glob", autospec=True, side_effect=glob), \
             patch.object(Path, "rglob", return_value=[]), \
             patch.object(Path, "exists", autospec=True, side_effect=lambda p: p in fixtures), \
             patch.object(Path, "write_text"), patch.object(module, "write_csv") as write_csv, \
             patch("builtins.print"):
            module.main()
        rows = next(call.args[1] for call in write_csv.call_args_list
                    if call.args[0] == module.RESULTS / "e2e/inference_raw.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pipeline"], "coordl")
        self.assertEqual(rows[0]["timestamp"], "resumed")

    def rows(self, elapsed):
        return [dict(gpu="test GPU", model="test model", domain="dct", pipeline="test",
                     task="inference", precision="fp32", batch_size=50,
                     samples=50000, elapsed_s=value, raw_result=f"test fixture {index}")
                for index, value in enumerate(elapsed)]

    def test_stable_three_trials_complete(self):
        result, = summarize(self.rows([10, 10.01, 9.99]))
        self.assertEqual(result["trial_status"], "COMPLETE")
        self.assertEqual(result["median_images_per_s"], 5000)
        self.assertLess(result["min_images_per_s"], result["median_images_per_s"])
        self.assertGreater(result["max_images_per_s"], result["median_images_per_s"])

    def test_variable_trials_require_both_additional_observations(self):
        for times in ([10, 12, 9], [10, 12, 9, 10], [10, 10, 10, 10]):
            result, = summarize(self.rows(times))
            self.assertIn("additional trials", result["trial_status"])
        result, = summarize(self.rows([10, 12, 9, 10, 11]))
        self.assertEqual(result["trial_status"], "COMPLETE")
        self.assertGreater(result["cv"], .02)

    def test_semantic_failures_are_not_accepted_as_timing_noise(self):
        self.assertTrue(correctness_passed({"failures": ["ffcv: hot throughput CV 0.2 exceeds 0.05"]}))
        self.assertFalse(correctness_passed({"failures": ["sample identity mismatch"]}))

    def test_slow_valid_results_and_available_arms_remain_reportable(self):
        self.assertTrue(correctness_passed(dict(validation_failures=[],
            failures=['pipeline fastlens: median throughput below required target'])))
        self.assertTrue(correctness_passed(dict(failures=['missing pipeline result: pipeline_coordl.json'])))
        self.assertFalse(correctness_passed(dict(validation_failures=['tensor mismatch'], failures=['tensor mismatch'])))

    def test_gpu_selection_uses_identity_not_device_ordinal(self):
        with patch('fastlens.benchmarks.common.subprocess.check_output', return_value=
                   'NVIDIA RTX PRO 6000, GPU-third\nNVIDIA H100 80GB HBM3, GPU-target\nNVIDIA GeForce RTX 4090, GPU-first\n'):
            self.assertEqual(gpu_uuid('H100'), 'GPU-target')
            self.assertEqual(gpu_uuid('4090'), 'GPU-first')

    def test_overlap_does_not_double_count_concurrent_events(self):
        self.assertEqual(intersection([(0, 10), (5, 15)], [(7, 12), (11, 20)]), 8)


class PredictionAgreementTest(unittest.TestCase):
    def test_matching_samples_are_compared(self):
        from fastlens.benchmarks.dct_models.evaluate import prediction_agreement
        self.assertEqual(prediction_agreement(
            dict(sample_ids=["a", "b"], predictions=[1, 2]),
            dict(sample_ids=["a", "b"], predictions=[1, 3])), .5)

    def test_different_order_or_subset_is_rejected(self):
        from fastlens.benchmarks.dct_models.evaluate import prediction_agreement
        result = dict(sample_ids=["a", "b"], predictions=[1, 2])
        for ids in (["b", "a"], ["a", "c"]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "sample IDs or order differ"):
                prediction_agreement(result, dict(sample_ids=ids, predictions=[1, 2]))


if __name__ == "__main__":
    unittest.main()
