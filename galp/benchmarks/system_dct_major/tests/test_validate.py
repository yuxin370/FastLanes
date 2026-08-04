from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from common import PIPELINE_RESULT_SCHEMA, sha256_json  # noqa: E402
from validate import (  # noqa: E402
    _compact_plan_memory_is_consistent,
    _physical_evidence,
    _planless_resource_evidence,
    _semantic_compare,
    _validate_result,
    _write_csv,
)


class PlanlessResultGateTest(unittest.TestCase):
    def _contract(self) -> dict:
        return {
            "dataset": {
                "sample_manifest_sha256": "sample-hash",
                "block_major_access": {
                    "index_sha256": "descriptor-hash",
                    "passes_one_percent": True,
                },
            },
            "execution": {"repeats": 1},
            "pipelines": {
                "dct_major_pushdown": {
                    "decode_workset_capacity_mib": 1,
                    "crop_execution_mode": "auto",
                    "cache_capacity_mib": 0,
                    "plan_cache_capacity": 0,
                }
            },
        }

    def _result(self, contract: dict) -> dict:
        return {
            "schema_version": PIPELINE_RESULT_SCHEMA,
            "pipeline": "dct_major_pushdown",
            "contract_sha256": sha256_json(contract),
            "sample_manifest_sha256": "sample-hash",
            "repeats": [
                {
                    "sample_trace": [{"ordinal": 0, "label": 7}],
                    "images": 1,
                    "batches": 1,
                    "time_to_first_batch_ms": 1.0,
                    "native_totals": {
                        "host_expanded_transform_items_created": 0,
                        "host_output_block_source_lists_created": 0,
                        "host_global_transform_sort_items": 0,
                        "planless_transform_kernel_launch_count": 1,
                        "planless_image_descriptor_count": 1,
                        "segment_count": 1,
                        "workset_count": 2,
                        "planless_transform_full_scan_output_block_count": 10,
                        "planless_transform_output_block_count": 4,
                        "planless_transform_skipped_output_block_count": 6,
                        "planless_transform_active_output_index_bytes": 16,
                        "planless_transform_active_output_offset_bytes": 24,
                        "planless_transform_active_output_schedule_peak_bytes": 128,
                        "planless_transform_source_contribution_count": 8,
                        "planless_transform_source_contribution_visit_count": 16,
                        "planless_transform_output_workset_ownership_count": 4,
                        "planless_transform_active_output_workset_count": 2,
                        "planless_transform_active_output_schedule_build_count": 1,
                        "planless_transform_active_output_offsets_valid": 1,
                        "planless_transform_active_output_planning_ms": 0.01,
                        "planless_transform_group_workset_build_ms": 0.001,
                        "planless_transform_active_output_count_ms": 0.004,
                        "planless_transform_active_output_prefix_ms": 0.001,
                        "planless_transform_active_output_fill_ms": 0.003,
                        "planless_transform_gpu_kernel_ms": 0.02,
                        "coordinate_group_lookup_count": 2,
                        "coordinate_group_index_entries": 10,
                        "coordinate_group_index_populated": 8,
                        "coordinate_group_index_holes": 2,
                        "coordinate_group_index_bytes": 128,
                        "coordinate_group_index_density": 0.8,
                        "rowgroup_count": 3,
                        "run_interval_exact_rowgroup_count": 1,
                        "bitmap_exact_rowgroup_count": 1,
                        "full_rowgroup_strategy_count": 1,
                        "automatic_sparse_storage_candidate_rowgroup_count": 2,
                        "adaptive_run_interval_estimated_ns": 10.0,
                        "adaptive_bitmap_estimated_ns": 11.0,
                        "adaptive_full_rowgroup_estimated_ns": 12.0,
                        "decode_workset_capacity_bytes": 1024 * 1024,
                        "max_estimated_decode_workset_bytes": 768,
                        "bounded_double_buffer_peak_estimated_bytes": 896,
                        "oversized_decode_rowgroup_count": 0,
                        "host_io_staged_rowgroups": 0,
                        "compact_plan_bytes": 600,
                        "compact_plan_peak_bytes": 800,
                        "exact_batch_plan_cache_enabled": 0,
                        "plan_cache_hits": 0,
                        "plan_cache_misses": 0,
                        "plan_cache_evictions": 0,
                        "sparse_vector_cache_hits": 0,
                        "sparse_vector_cache_misses": 0,
                        "dct_resize_weight_cache_hits": 0,
                        "dct_resize_weight_cache_misses": 0,
                        "dct_conversion_matrix_cache_hits": 0,
                        "dct_conversion_matrix_cache_misses": 0,
                        "decoded_rowgroup_cache_capacity_bytes": 0,
                        "decoded_rowgroup_cache_current_bytes": 0,
                        "decoded_rowgroup_cache_peak_bytes": 0,
                        "decoded_rowgroup_cache_current_rowgroups": 0,
                        "decoded_rowgroup_cache_peak_rowgroups": 0,
                        "decoded_rowgroup_cache_hits": 0,
                        "decoded_rowgroup_cache_misses": 0,
                        "decoded_rowgroup_cache_inserts": 0,
                        "decoded_rowgroup_cache_evictions": 0,
                    },
                }
            ],
        }

    def test_accepts_bounded_compact_planless_execution(self) -> None:
        contract = self._contract()
        failures: list[str] = []
        _validate_result(
            "dct_major_pushdown",
            self._result(contract),
            contract,
            [{"ordinal": 0, "label": 7}],
            failures,
        )
        self.assertEqual(failures, [])

    def test_accepts_cumulative_compact_bytes_across_segments(self) -> None:
        native = {
            "segment_count": 50,
            "compact_plan_bytes": 581_599_456,
            "compact_plan_peak_bytes": 36_315_692,
        }
        self.assertTrue(_compact_plan_memory_is_consistent(native))

    def test_rejects_cumulative_compact_bytes_above_segment_bound(self) -> None:
        native = {
            "segment_count": 2,
            "compact_plan_bytes": 1_601,
            "compact_plan_peak_bytes": 800,
        }
        self.assertFalse(_compact_plan_memory_is_consistent(native))

    def test_rejects_expansion_missing_strategy_and_unbounded_peak(self) -> None:
        contract = self._contract()
        result = self._result(contract)
        native = result["repeats"][0]["native_totals"]
        native["host_expanded_transform_items_created"] = 1
        native["full_rowgroup_strategy_count"] = 0
        native["bounded_double_buffer_peak_estimated_bytes"] = 2 * 1024 * 1024
        native["host_io_staged_rowgroups"] = 3
        failures: list[str] = []
        _validate_result(
            "dct_major_pushdown",
            result,
            contract,
            [{"ordinal": 0, "label": 7}],
            failures,
        )
        self.assertTrue(any("must be zero" in item for item in failures))
        self.assertTrue(any("do not cover every rowgroup" in item for item in failures))
        self.assertTrue(any("exceeds capacity" in item for item in failures))
        self.assertTrue(any("host I/O staging" in item for item in failures))

    def test_rejects_repeated_workset_descriptor_accounting(self) -> None:
        contract = self._contract()
        result = self._result(contract)
        result["repeats"][0]["native_totals"]["planless_image_descriptor_count"] = 4
        failures: list[str] = []
        _validate_result(
            "dct_major_pushdown",
            result,
            contract,
            [{"ordinal": 0, "label": 7}],
            failures,
        )
        self.assertTrue(any("descriptor count must equal measured images" in item for item in failures))

    def test_rejects_inconsistent_active_output_schedule_and_missing_split_timing(self) -> None:
        contract = self._contract()
        result = self._result(contract)
        native = result["repeats"][0]["native_totals"]
        native["planless_transform_skipped_output_block_count"] = 5
        native["planless_transform_active_output_offset_bytes"] = 16
        native["planless_transform_active_output_schedule_peak_bytes"] = 8
        native["planless_transform_active_output_schedule_build_count"] = 2
        native["planless_transform_active_output_planning_ms"] = 0.0
        native["planless_transform_gpu_kernel_ms"] = 0.0
        failures: list[str] = []
        _validate_result(
            "dct_major_pushdown",
            result,
            contract,
            [{"ordinal": 0, "label": 7}],
            failures,
        )
        self.assertTrue(any("skip accounting" in item for item in failures))
        self.assertTrue(any("exactly once per segment" in item for item in failures))
        self.assertTrue(any("offset byte accounting" in item for item in failures))
        self.assertTrue(any("schedule peak" in item for item in failures))
        self.assertTrue(any("CPU schedule timing" in item for item in failures))
        self.assertTrue(any("CUDA kernel elapsed" in item for item in failures))

    def test_rejects_count_bounded_plan_cache_and_unbounded_decoded_cache(self) -> None:
        contract = self._contract()
        contract["pipelines"]["dct_major_pushdown"]["plan_cache_capacity"] = 2
        result = self._result(contract)
        native = result["repeats"][0]["native_totals"]
        native["exact_batch_plan_cache_enabled"] = 1
        native["plan_cache_hits"] = 1
        native["dct_resize_weight_cache_hits"] = 1
        native["decoded_rowgroup_cache_peak_bytes"] = 1
        native["decoded_rowgroup_cache_current_rowgroups"] = 1
        native["decoded_rowgroup_cache_peak_rowgroups"] = 1
        failures: list[str] = []
        _validate_result(
            "dct_major_pushdown",
            result,
            contract,
            [{"ordinal": 0, "label": 7}],
            failures,
        )
        self.assertTrue(any("count-bounded" in item for item in failures))
        self.assertTrue(any("exact expanded-plan cache" in item for item in failures))
        self.assertTrue(any("persistent helper cache" in item for item in failures))
        self.assertTrue(any("cache current/peak" in item for item in failures))
        self.assertTrue(any("retained entries" in item for item in failures))


class PlanlessResourceGateTest(unittest.TestCase):
    def _aggregates(self, *, planless_multiplier: float) -> dict:
        names = (
            "host_peak_rss_bytes",
            "galp_native_pinned_peak_in_use_bytes",
            "galp_native_device_peak_in_use_bytes",
            "peak_torch_gpu_allocated_bytes",
            "peak_torch_gpu_reserved_bytes",
        )
        legacy = {name: {"max": 1000.0} for name in names}
        planless = {name: {"max": 1000.0 * planless_multiplier} for name in names}
        return {
            "dct_major_legacy_pushdown": legacy,
            "dct_major_pushdown": planless,
        }

    def test_accepts_planless_peaks_no_higher_than_legacy(self) -> None:
        failures: list[str] = []
        evidence = _planless_resource_evidence(
            self._aggregates(planless_multiplier=0.8), failures
        )
        self.assertEqual(failures, [])
        assert evidence is not None
        self.assertTrue(evidence["ok"])

    def test_rejects_any_planless_peak_above_legacy(self) -> None:
        aggregates = self._aggregates(planless_multiplier=0.8)
        aggregates["dct_major_pushdown"]["galp_native_pinned_peak_in_use_bytes"]["max"] = 1001.0
        failures: list[str] = []
        evidence = _planless_resource_evidence(aggregates, failures)
        assert evidence is not None
        self.assertFalse(evidence["ok"])
        self.assertTrue(any("pinned" in item and "exceeds" in item for item in failures))


class CompleteCsvTest(unittest.TestCase):
    def test_writes_physical_memory_plan_cache_and_storage_columns(self) -> None:
        repeat = {
            "repeat": 0,
            "images": 2,
            "seconds": 1.0,
            "throughput_images_per_s": 2.0,
            "time_to_first_batch_ms": 4.0,
            "steady_throughput_images_per_s": 3.0,
            "latency_ms": {"mean": 5.0},
            "loader_submit_ms": {"mean": 2.0},
            "model_ms": {"mean": 3.0},
            "host_peak_rss_bytes": 100,
            "peak_torch_gpu_allocated_bytes": 200,
            "peak_torch_gpu_reserved_bytes": 300,
            "native_totals": {
                "planning_ms": 7.0,
                "compact_plan_peak_bytes": 11,
                "compressed_payload_bytes_read": 25,
                "full_compressed_payload_bytes": 100,
                "decoded_coefficient_bytes": 4096,
                "galp_native_pinned_peak_in_use_bytes": 50,
                "galp_native_device_peak_in_use_bytes": 60,
                "decoded_rowgroup_cache_hits": 3,
                "decoded_rowgroup_cache_misses": 1,
                "decoded_rowgroup_cache_current_rowgroups": 2,
                "decoded_rowgroup_cache_peak_rowgroups": 3,
                "decoded_rowgroup_cache_inserts": 4,
                "dct_resize_weight_cache_hits": 0,
                "dct_resize_weight_cache_misses": 0,
            },
        }
        result = {"pipeline": "dct_major_pushdown", "domain": "dct", "repeats": [repeat]}
        storage = {
            "descriptor_bytes": 10,
            "base_storage_bytes": 1000,
            "total_storage_bytes_with_descriptor": 1010,
            "storage_increase_percent": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            _write_csv(path, [result], storage)
            with path.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual(row["compact_plan_peak_bytes"], "11")
        self.assertEqual(row["decoded_coefficient_bytes"], "4096")
        self.assertEqual(row["native_pinned_peak_bytes"], "50")
        self.assertEqual(row["read_amplification"], "0.25")
        self.assertEqual(row["decoded_rowgroup_cache_hit_rate"], "0.75")
        self.assertEqual(row["decoded_rowgroup_cache_current_rowgroups"], "2")
        self.assertEqual(row["decoded_rowgroup_cache_peak_rowgroups"], "3")
        self.assertEqual(row["decoded_rowgroup_cache_inserts"], "4")
        self.assertEqual(row["dct_resize_weight_cache_hits"], "0")
        self.assertEqual(row["dct_resize_weight_cache_misses"], "0")
        self.assertEqual(row["total_storage_bytes_with_descriptor"], "1010")


class PhysicalEvidenceTest(unittest.TestCase):
    def test_requires_bytes_vectors_and_blocks_to_fall(self) -> None:
        aggregates = {
            "dct_major_full": {
                "native_hot_mean": {
                    "compressed_payload_bytes_read": 1000,
                    "actual_vector_count": 100,
                    "requested_source_block_count": 10000,
                    "pread_count": 10,
                }
            },
            "dct_major_pushdown": {
                "native_hot_mean": {
                    "compressed_payload_bytes_read": 600,
                    "actual_vector_count": 70,
                    "requested_source_block_count": 5000,
                    "pread_count": 8,
                }
            },
        }
        failures: list[str] = []
        evidence = _physical_evidence({}, aggregates, failures)
        self.assertEqual(failures, [])
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertTrue(evidence["ok"])
        self.assertAlmostEqual(evidence["physical_bytes_saved_percent"], 40.0)

    def test_logical_only_crop_is_rejected(self) -> None:
        aggregates = {
            "dct_major_full": {
                "native_hot_mean": {
                    "compressed_payload_bytes_read": 1000,
                    "actual_vector_count": 100,
                    "requested_source_block_count": 10000,
                }
            },
            "dct_major_pushdown": {
                "native_hot_mean": {
                    "compressed_payload_bytes_read": 1000,
                    "actual_vector_count": 70,
                    "requested_source_block_count": 5000,
                }
            },
        }
        failures: list[str] = []
        evidence = _physical_evidence({}, aggregates, failures)
        self.assertIsNotNone(evidence)
        self.assertTrue(any("physical bytes" in item for item in failures))

    def test_full_control_accepts_generic_block_count_alias(self) -> None:
        aggregates = {
            "dct_major_full": {
                "native_hot_mean": {
                    "compressed_payload_bytes_read": 1000,
                    "actual_vector_count": 100,
                    "block_count": 900,
                }
            },
            "dct_major_pushdown": {
                "native_hot_mean": {
                    "compressed_payload_bytes_read": 600,
                    "actual_vector_count": 50,
                    "requested_source_block_count": 200,
                }
            },
        }
        failures: list[str] = []
        evidence = _physical_evidence({}, aggregates, failures)
        self.assertEqual(failures, [])
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence["full_source_blocks"], 900)
        self.assertEqual(evidence["pushdown_source_blocks"], 200)

    def test_full_control_skips_zero_counter_and_uses_planned_vectors(self) -> None:
        aggregates = {
            "dct_major_full": {
                "native_hot_mean": {
                    "compressed_payload_bytes_read": 1000,
                    "actual_vector_count": 100,
                    "requested_source_block_count": 0,
                    "fixed_transform_source_block_count": 0,
                    "planned_selected_vector_count": 900,
                }
            },
            "dct_major_pushdown": {
                "native_hot_mean": {
                    "compressed_payload_bytes_read": 600,
                    "actual_vector_count": 50,
                    "requested_source_block_count": 200,
                }
            },
        }
        failures: list[str] = []
        evidence = _physical_evidence({}, aggregates, failures)
        self.assertEqual(failures, [])
        assert evidence is not None
        self.assertEqual(evidence["full_source_blocks"], 900)


class SemanticToleranceTest(unittest.TestCase):
    def _write_artifact(self, path: Path, *, inputs: np.ndarray, output: np.ndarray, predictions: np.ndarray | None = None) -> None:
        payload = {
            "ordinals": np.arange(inputs.shape[0], dtype=np.int64),
            "labels": np.zeros(inputs.shape[0], dtype=np.int64),
            "input_0": inputs.astype(np.float32),
            "output": output.astype(np.float32),
        }
        if predictions is not None:
            payload["top1_predictions"] = predictions.astype(np.int64)
        np.savez(path, **payload)

    def test_one_normalized_dct_level_is_accepted_for_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_input = np.ones((2, 100), dtype=np.float32)
            right_input = left_input.copy()
            right_input.reshape(-1)[:10] += np.float32(1.0 / 1020.0)
            rng = np.random.default_rng(7)
            left_output = rng.normal(size=(2, 192)).astype(np.float32)
            right_output = left_output + np.float32(0.01)
            self._write_artifact(root / "left.npz", inputs=left_input, output=left_output)
            self._write_artifact(root / "right.npz", inputs=right_input, output=right_output)
            contract = {
                "workload": {"kind": "feature-extraction"},
                "semantic_validation": {
                    "input_max_abs": 1.0e-3,
                    "input_mean_abs": 1.0e-4,
                    "feature_cosine_min": 0.999,
                },
            }
            failures: list[str] = []
            result = _semantic_compare(
                "left",
                "right",
                {"semantic_artifact": str(root / "left.npz")},
                {"semantic_artifact": str(root / "right.npz")},
                contract,
                strict=True,
                failures=failures,
            )
            self.assertEqual(failures, [])
            self.assertTrue(result["ok"])

    def test_full_predictions_use_contract_agreement_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = np.ones((2, 10), dtype=np.float32)
            output = np.asarray([[3.0, 2.0], [4.0, 1.0]], dtype=np.float32)
            left_predictions = np.zeros(1000, dtype=np.int64)
            right_predictions = left_predictions.copy()
            right_predictions[0] = 1
            self._write_artifact(root / "left.npz", inputs=inputs, output=output, predictions=left_predictions)
            self._write_artifact(root / "right.npz", inputs=inputs, output=output, predictions=right_predictions)
            contract = {
                "workload": {"kind": "evaluation"},
                "semantic_validation": {
                    "input_max_abs": 1.0e-3,
                    "input_mean_abs": 1.0e-4,
                    "logit_cosine_min": 0.999,
                    "semantic_top1_agreement_min": 1.0,
                    "full_prediction_top1_agreement_min": 0.999,
                },
            }
            failures: list[str] = []
            result = _semantic_compare(
                "left",
                "right",
                {"semantic_artifact": str(root / "left.npz")},
                {"semantic_artifact": str(root / "right.npz")},
                contract,
                strict=True,
                failures=failures,
            )
            self.assertEqual(failures, [])
            self.assertTrue(result["ok"])
            self.assertEqual(result["full_top1_prediction_agreement"], 0.999)


if __name__ == "__main__":
    unittest.main()
