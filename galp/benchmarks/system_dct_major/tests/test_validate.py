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

from common import BLOCK_MAJOR_RUNTIME_PROFILE, PIPELINE_RESULT_SCHEMA, sha256_json  # noqa: E402
from validate import (  # noqa: E402
    _compact_plan_memory_is_consistent,
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
                    "runtime_profile": BLOCK_MAJOR_RUNTIME_PROFILE,
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
                    "first_shard_ready_ms": 0.5,
                    "native_segments": [{"segment_shard_id": 0}],
                    "native_totals": {
                        "host_expanded_transform_items_created": 0,
                        "host_output_block_source_lists_created": 0,
                        "host_global_transform_sort_items": 0,
                        "planless_transform_kernel_launch_count": 1,
                        "planless_image_descriptor_count": 1,
                        "segment_count": 1,
                        "segment_cross_shard_count": 0,
                        "shard_reactivation_count": 0,
                        "duplicate_physical_read_count": 0,
                        "rowgroup_revisit_count": 0,
                        "vector_run_revisit_count": 0,
                        "physical_read_order_inversions": 0,
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
                        "planless_transform_active_output_schedule_build_count": 0,
                        "active_output_schedule_sidecar_hit_count": 1,
                        "active_output_schedule_sidecar_miss_count": 0,
                        "active_output_schedule_sidecar_reject_count": 0,
                        "active_output_schedule_sidecar_persist_count": 0,
                        "active_output_schedule_sidecar_bytes": 512,
                        "active_output_schedule_interval_count": 4,
                        "active_output_schedule_mapped_bytes_peak": 512,
                        "active_output_schedule_mmap_capacity_bytes": 16 * 1024 * 1024,
                        "active_output_schedule_mmap_window_count": 2,
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
                        "run_interval_exact_rowgroup_count": 0,
                        "run_interval_bounded_rowgroup_count": 3,
                        "bitmap_exact_rowgroup_count": 0,
                        "full_rowgroup_strategy_count": 0,
                        "planned_vector_count": 57,
                        "actual_vector_count": 57,
                        "sparse_read_fallback_rowgroup_count": 0,
                        "bounded_read_amplification_ppm": 1_100_000,
                        "bounded_read_local_amplification_ppm": 0,
                        "bounded_read_max_run_bytes": 0,
                        "bounded_io_backend": "io-uring",
                        "bounded_io_uring_queue_depth": 256,
                        "bounded_exact_storage_bytes": 100,
                        "bounded_physical_storage_bytes": 110,
                        "bounded_merged_gap_bytes": 10,
                        "full_compressed_payload_bytes": 150,
                        "selected_compressed_payload_bytes": 100,
                        "compressed_payload_bytes_read": 110,
                        "read_amplification": 1.10,
                        "merged_gap_bytes": 10,
                        "hole_clear_bytes": 10,
                        "static_prefix_restore_bytes": 10,
                        "bounded_exact_extent_count": 10,
                        "bounded_physical_run_count": 8,
                        "bounded_selected_gap_count": 2,
                        "pread_count": 0,
                        "io_uring_read_request_count": 8,
                        "io_uring_completion_count": 8,
                        "io_uring_submit_syscall_count": 2,
                        "io_uring_setup_count": 1,
                        "io_uring_ring_mapped_bytes": 16384,
                        "io_uring_fallback_count": 0,
                        "storage_read_granularity": "bounded-selected-vector-range",
                        "decode_granularity": "selected-vector",
                        "decode_workset_capacity_bytes": 512 * 1024 * 1024,
                        "max_estimated_decode_workset_bytes": 768,
                        "bounded_double_buffer_peak_estimated_bytes": 896,
                        "oversized_decode_rowgroup_count": 0,
                        "host_io_staged_rowgroups": 0,
                        "actual_transient_total_used_high_water_bytes": 700,
                        "actual_transient_total_allocated_high_water_bytes": 900,
                        "actual_transient_memory_gate_passed": True,
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

    def test_accepts_current_native_profile_execution(self) -> None:
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
        native["run_interval_bounded_rowgroup_count"] = 2
        native["bounded_double_buffer_peak_estimated_bytes"] = 513 * 1024 * 1024
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

    def test_rejects_bounded_read_above_whole_run_cap(self) -> None:
        contract = self._contract()
        result = self._result(contract)
        native = result["repeats"][0]["native_totals"]
        native["bounded_physical_storage_bytes"] = 111
        native["bounded_merged_gap_bytes"] = 11
        native["compressed_payload_bytes_read"] = 111
        native["read_amplification"] = 1.11
        native["merged_gap_bytes"] = 11
        native["hole_clear_bytes"] = 11
        failures: list[str] = []
        _validate_result(
            "dct_major_pushdown",
            result,
            contract,
            [{"ordinal": 0, "label": 7}],
            failures,
        )
        self.assertTrue(any("whole-run cap" in item for item in failures))

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

    def test_rejects_runtime_cache_activity_and_unbounded_decoded_cache(self) -> None:
        contract = self._contract()
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
        self.assertTrue(any("exact expanded-plan cache" in item for item in failures))
        self.assertTrue(any("persistent helper cache" in item for item in failures))
        self.assertTrue(any("cache current/peak" in item for item in failures))
        self.assertTrue(any("retained entries" in item for item in failures))


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
            "process_io": {
                "logical_read_bytes": 120,
                "storage_read_bytes": 8192,
                "read_syscalls": 4,
            },
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
        self.assertEqual(row["process_io_logical_read_bytes"], "120")
        self.assertEqual(row["process_io_storage_read_bytes"], "8192")
        self.assertEqual(row["process_io_read_syscalls"], "4")
        self.assertEqual(row["decoded_rowgroup_cache_hit_rate"], "0.75")
        self.assertEqual(row["decoded_rowgroup_cache_current_rowgroups"], "2")
        self.assertEqual(row["decoded_rowgroup_cache_peak_rowgroups"], "3")
        self.assertEqual(row["decoded_rowgroup_cache_inserts"], "4")
        self.assertEqual(row["dct_resize_weight_cache_hits"], "0")
        self.assertEqual(row["dct_resize_weight_cache_misses"], "0")
        self.assertEqual(row["total_storage_bytes_with_descriptor"], "1010")


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
