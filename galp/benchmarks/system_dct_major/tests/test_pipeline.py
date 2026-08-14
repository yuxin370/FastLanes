from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from galp.torch import DirectDctFuture


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from pipeline import (  # noqa: E402
    GalpAdapter,
    LoadedBatch,
    _accumulate_native,
    _manifest_shard_segments,
    _process_io_delta,
    _process_io_snapshot,
    _validate_identity,
)


class PipelineControlTest(unittest.TestCase):
    def test_manifest_shard_segments_use_exact_manifest_ranges(self) -> None:
        samples = [{"galp_image_id": image_id} for image_id in range(9)]
        manifest = {
            "shards": [
                {"shard_id": 4, "first_global_image_index": 0, "image_count": 4},
                {"shard_id": 8, "first_global_image_index": 4, "image_count": 5},
            ]
        }
        segments = _manifest_shard_segments(samples, manifest)
        self.assertEqual(
            [[item["galp_image_id"] for item in segment] for segment in segments],
            [[0, 1, 2, 3], [4, 5, 6, 7, 8]],
        )

    def test_manifest_shard_segments_reject_partial_tail(self) -> None:
        samples = [{"galp_image_id": image_id} for image_id in range(6)]
        manifest = {
            "shards": [
                {"shard_id": 0, "first_global_image_index": 0, "image_count": 4},
                {"shard_id": 1, "first_global_image_index": 4, "image_count": 5},
            ]
        }
        with self.assertRaisesRegex(ValueError, "truncates a physical shard"):
            _manifest_shard_segments(samples, manifest)

    def test_native_allocator_snapshots_are_not_summed_across_segments(self) -> None:
        totals: dict[str, object] = {}
        _accumulate_native(
            totals,
            {
                "galp_native_device_in_use_bytes": 100,
                "galp_native_device_peak_in_use_bytes": 140,
                "galp_native_device_cached_bytes": 20,
                "galp_native_device_cuda_allocation_count": 7,
                "galp_native_pinned_in_use_bytes": 50,
                "galp_native_pinned_peak_in_use_bytes": 80,
                "galp_native_pinned_cached_bytes": 10,
                "galp_native_pinned_cuda_allocation_bytes": 4096,
                "planning_ms": 1.25,
            },
        )
        _accumulate_native(
            totals,
            {
                "galp_native_device_in_use_bytes": 60,
                "galp_native_device_peak_in_use_bytes": 135,
                "galp_native_device_cached_bytes": 40,
                "galp_native_device_cuda_allocation_count": 9,
                "galp_native_pinned_in_use_bytes": 30,
                "galp_native_pinned_peak_in_use_bytes": 75,
                "galp_native_pinned_cached_bytes": 25,
                "galp_native_pinned_cuda_allocation_bytes": 8192,
                "planning_ms": 2.75,
            },
        )

        self.assertEqual(totals["segment_count"], 2)
        self.assertEqual(totals["galp_native_device_in_use_bytes"], 60)
        self.assertEqual(totals["galp_native_device_peak_in_use_bytes"], 140)
        self.assertEqual(totals["galp_native_device_cached_bytes"], 40)
        self.assertEqual(totals["galp_native_device_cuda_allocation_count"], 9)
        self.assertEqual(totals["galp_native_pinned_in_use_bytes"], 30)
        self.assertEqual(totals["galp_native_pinned_peak_in_use_bytes"], 80)
        self.assertEqual(totals["galp_native_pinned_cached_bytes"], 25)
        self.assertEqual(totals["galp_native_pinned_cuda_allocation_bytes"], 8192)
        self.assertEqual(totals["planning_ms"], 4.0)

    def test_process_io_snapshot_and_delta_keep_storage_reads_separate(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "io"
            path.write_text("rchar: 100\nsyscr: 7\nread_bytes: 4096\n", encoding="utf-8")
            before = _process_io_snapshot(path)
            path.write_text("rchar: 250\nsyscr: 11\nread_bytes: 12288\n", encoding="utf-8")
            after = _process_io_snapshot(path)

        self.assertEqual(
            _process_io_delta(before, after),
            {
                "logical_read_bytes": 150,
                "storage_read_bytes": 8192,
                "read_syscalls": 4,
            },
        )
        self.assertIsNone(_process_io_delta(None, after))

    def test_actual_transient_high_water_is_maximized_across_segments(self) -> None:
        totals: dict[str, object] = {}
        _accumulate_native(
            totals,
            {
                "actual_transient_total_used_high_water_bytes": 96,
                "actual_transient_total_allocated_high_water_bytes": 128,
                "actual_transient_memory_gate_passed": True,
            },
        )
        _accumulate_native(
            totals,
            {
                "actual_transient_total_used_high_water_bytes": 80,
                "actual_transient_total_allocated_high_water_bytes": 112,
                "actual_transient_memory_gate_passed": True,
            },
        )

        self.assertEqual(totals["segment_count"], 2)
        self.assertEqual(totals["actual_transient_total_used_high_water_bytes"], 96)
        self.assertEqual(totals["actual_transient_total_allocated_high_water_bytes"], 128)
        self.assertIs(totals["actual_transient_memory_gate_passed"], True)

    def test_native_ratios_are_recomputed_from_whole_run_totals(self) -> None:
        totals: dict[str, object] = {}
        _accumulate_native(
            totals,
            {
                "compressed_payload_bytes_read": 102,
                "selected_compressed_payload_bytes": 100,
                "read_amplification": 1.02,
                "selected_coefficient_count": 1,
                "full_coefficient_count": 4,
                "selected_coefficient_ratio": 0.25,
                "physical_page_bytes_covered": 5,
                "full_physical_page_bytes": 10,
                "physical_page_coverage_ratio": 0.5,
            },
        )
        _accumulate_native(
            totals,
            {
                "compressed_payload_bytes_read": 50,
                "selected_compressed_payload_bytes": 50,
                "read_amplification": 1.0,
                "selected_coefficient_count": 3,
                "full_coefficient_count": 6,
                "selected_coefficient_ratio": 0.5,
                "physical_page_bytes_covered": 2,
                "full_physical_page_bytes": 10,
                "physical_page_coverage_ratio": 0.2,
            },
        )

        self.assertAlmostEqual(totals["read_amplification"], 152 / 150)
        self.assertAlmostEqual(totals["selected_coefficient_ratio"], 4 / 10)
        self.assertAlmostEqual(totals["physical_page_coverage_ratio"], 7 / 20)

    def test_bounded_configuration_is_constant_not_summed_across_segments(self) -> None:
        totals: dict[str, object] = {}
        configuration = {
            "bounded_read_amplification_ppm": 1_020_000,
            "bounded_read_local_amplification_ppm": 1_050_000,
            "bounded_read_max_run_bytes": 4 * 1024 * 1024,
            "bounded_io_backend": "io-uring",
            "bounded_io_uring_queue_depth": 256,
            "cuda_warp_size": 32,
            "cuda_least_stream_priority": 0,
            "cuda_greatest_stream_priority": -5,
            "direct_dct_low_priority_streams": True,
            "fixed_grid_output_float32": True,
            "fixed_grid_output_affine_applied": True,
            "fixed_grid_output_add": 4.0,
            "fixed_grid_output_scale": 1.0 / 1020.0,
        }
        _accumulate_native(totals, configuration)
        _accumulate_native(totals, configuration)
        self.assertEqual(totals["segment_count"], 2)
        for key, value in configuration.items():
            self.assertEqual(totals[key], value)

        with self.assertRaisesRegex(RuntimeError, "invariant changed across segments"):
            _accumulate_native(
                totals,
                {
                    **configuration,
                    "bounded_read_amplification_ppm": 1_050_000,
                },
            )

    def test_galp_measurement_does_not_reuse_warmup_segment(self) -> None:
        adapter = object.__new__(GalpAdapter)
        adapter._warmup_segments = [[{"ordinal": 0}]]
        adapter._measurement_segments = [[{"ordinal": 1}, {"ordinal": 2}]]
        adapter.begin_repeat()
        self.assertEqual(adapter.segments, adapter._warmup_segments)
        adapter.begin_measurement()
        self.assertEqual(adapter.segments, adapter._measurement_segments)
        self.assertEqual(adapter._next_segment, 0)
        self.assertIsNone(adapter._pending)
        self.assertIsNone(adapter._current)

    def test_galp_cold_prime_is_reused_by_first_measurement(self) -> None:
        adapter = object.__new__(GalpAdapter)
        adapter._warmup_segments = []
        adapter._measurement_segments = [[{"ordinal": 1}, {"ordinal": 2}]]
        adapter._cold_measurement_primed = False
        adapter._reuse_cold_measurement = False
        adapter._prefetch = lambda segment: ("prefetch", tuple(item["ordinal"] for item in segment))

        adapter.prime_cold_start()
        pending = adapter._pending
        self.assertEqual(pending, ("prefetch", (1, 2)))

        adapter.begin_repeat()
        adapter.begin_measurement()
        self.assertIs(adapter._pending, pending)
        self.assertEqual(adapter._next_segment, 1)

    def test_manifest_shard_prefetch_overlaps_and_model_batches_cross_boundary(self) -> None:
        class FakePending:
            producer_active_ms = 1.0
            planning_ms = 2.0
            io_staging_ms = 3.0
            ordered_submission_ms = 4.0

            def __init__(self, image_ids: list[int], value: float) -> None:
                self.read_calls = 0
                self.batch = SimpleNamespace(
                    global_image_ids=image_ids,
                    y=torch.full((len(image_ids), 1, 28, 28, 8, 8), value),
                    cbcr=torch.full((len(image_ids), 2, 14, 14, 8, 8), value),
                    layout="transformed_dct_grid",
                    execution_stats={"actual_vector_count": len(image_ids)},
                    cache_stats={},
                )

            def read(self):
                self.read_calls += 1
                return self.batch

        segments = [
            [{"galp_image_id": image_id} for image_id in range(3)],
            [{"galp_image_id": image_id} for image_id in range(3, 6)],
        ]
        native_pending = [
            FakePending([0, 1, 2], 1.0),
            FakePending([3, 4, 5], 2.0),
        ]
        pending = [
            DirectDctFuture(item, "test-profile") for item in native_pending
        ]

        class FakeReader:
            pass

        adapter = object.__new__(GalpAdapter)
        adapter.segment_mode = "manifest-shard"
        adapter.device = torch.device("cpu")
        adapter.reader = FakeReader()
        adapter._shard_by_image_id = {image_id: image_id // 3 for image_id in range(6)}
        adapter._process_scope_started_ns = None
        adapter._prefetch = lambda segment: pending[int(segment[0]["galp_image_id"]) // 3]
        adapter._activate_segments(segments)
        adapter._start_next_prefetch()

        first = adapter._load_pushdown(
            [
                {"galp_image_id": 0, "label": 10, "ordinal": 0},
                {"galp_image_id": 1, "label": 11, "ordinal": 1},
            ]
        )
        self.assertEqual(native_pending[0].read_calls, 1)
        self.assertEqual(native_pending[1].read_calls, 0)
        self.assertIs(adapter._pending, pending[1])
        self.assertEqual(first.native_stats[0]["segment_mode"], "manifest-shard")
        self.assertEqual(first.native_stats[0]["segment_shard_id"], 0)
        self.assertEqual(first.native_stats[0]["segment_cross_shard_count"], 0)
        self.assertGreaterEqual(first.native_stats[0]["prefetch_consumer_wait_ms"], 0.0)

        boundary = adapter._load_pushdown(
            [
                {"galp_image_id": 2, "label": 12, "ordinal": 2},
                {"galp_image_id": 3, "label": 13, "ordinal": 3},
            ]
        )
        self.assertEqual(native_pending[1].read_calls, 1)
        self.assertEqual(tuple(boundary.inputs[0].shape), (2, 1, 28, 28, 8, 8))
        self.assertEqual(boundary.ordinals, [2, 3])
        self.assertEqual(boundary.label_values, [12, 13])
        self.assertEqual(float(boundary.inputs[0][0, 0, 0, 0, 0, 0]), 1.0)
        self.assertEqual(float(boundary.inputs[0][1, 0, 0, 0, 0, 0]), 2.0)
        self.assertEqual(boundary.native_stats[0]["segment_shard_id"], 1)
        self.assertEqual(boundary.native_stats[0]["segment_cross_shard_count"], 0)
        self.assertEqual(boundary.native_stats[0]["shard_reactivation_count"], 0)

    def test_identity_validation_uses_host_labels(self) -> None:
        batch = LoadedBatch(
            inputs=(torch.zeros(2, 1),),
            labels=torch.tensor([4, 5]),
            ordinals=[2, 3],
            label_values=[4, 5],
            on_device=False,
        )
        expected = [
            {"ordinal": 2, "label": 4},
            {"ordinal": 3, "label": 5},
        ]
        _validate_identity(batch, expected)
        batch.label_values[1] = 9
        with self.assertRaisesRegex(RuntimeError, "label mismatch"):
            _validate_identity(batch, expected)

if __name__ == "__main__":
    unittest.main()
