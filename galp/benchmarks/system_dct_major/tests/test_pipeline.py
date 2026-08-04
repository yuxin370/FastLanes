from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from pipeline import (  # noqa: E402
    GalpAdapter,
    LoadedBatch,
    _accumulate_native,
    _is_grayscale_metadata,
    _load_direct_dct_modules,
    _rebuild_grayscale_full_grid,
    _validate_identity,
)


class PipelineControlTest(unittest.TestCase):
    @staticmethod
    def _grayscale_metadata(height: int = 2, width: int = 3) -> dict:
        return {
            "components": [
                {
                    "present": True,
                    "semantic_slot_id": 0,
                    "local_component_index": 0,
                    "height_in_blocks": height,
                    "width_in_blocks": width,
                },
                {"present": False, "semantic_slot_id": 1},
                {"present": False, "semantic_slot_id": 2},
            ]
        }

    def test_pushdown_module_loader_skips_postdecode_diagnostics(self) -> None:
        modules = {
            "_galp_direct_dct": SimpleNamespace(name="binding"),
            "rgbnomore_dct_profile": SimpleNamespace(
                RGBNOMORE_VAL_DCT_GRID_TRANSFORM_FP32={"output": "fp32"}
            ),
        }
        with patch("pipeline.importlib.import_module", side_effect=lambda name: modules[name]) as imported:
            binding, diagnostics, profile, timings = _load_direct_dct_modules(
                Path("/tmp/binding"), load_postdecode_diagnostics=False
            )
        self.assertEqual(binding.name, "binding")
        self.assertIsNone(diagnostics)
        self.assertEqual(profile, {"output": "fp32"})
        self.assertFalse(timings["postdecode_diagnostics_imported"])
        self.assertEqual(
            [call.args[0] for call in imported.call_args_list],
            ["_galp_direct_dct", "rgbnomore_dct_profile"],
        )

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

    def test_grayscale_metadata_requires_y_without_chroma(self) -> None:
        metadata = self._grayscale_metadata()
        self.assertTrue(_is_grayscale_metadata(metadata))
        metadata["components"][1] = {"present": True, "semantic_slot_id": 1}
        self.assertFalse(_is_grayscale_metadata(metadata))

    def test_rebuild_grayscale_full_grid_preserves_block_coordinates(self) -> None:
        coordinates = [(1, 2), (0, 1), (1, 0), (0, 0), (1, 1), (0, 2)]
        coefficients = torch.arange(6 * 64, dtype=torch.int16).reshape(6, 64)
        batch = type(
            "FakeBatch",
            (),
            {
                "layout": "compact",
                "selected_coefficients": list(range(64)),
                "coefficients": coefficients,
                "block_metadata": [
                    {
                        "request_index": 0,
                        "global_image_index": 239,
                        "semantic_slot_id": 0,
                        "block_y": block_y,
                        "block_x": block_x,
                    }
                    for block_y, block_x in coordinates
                ],
            },
        )()

        grid, block_count = _rebuild_grayscale_full_grid(
            batch,
            self._grayscale_metadata(),
            239,
        )

        self.assertEqual(block_count, 6)
        self.assertEqual(tuple(grid.y.shape), (1, 1, 2, 3, 8, 8))
        self.assertEqual(tuple(grid.cbcr.shape), (1, 2, 1, 1, 8, 8))
        self.assertTrue(torch.count_nonzero(grid.cbcr).item() == 0)
        for source_index, (block_y, block_x) in enumerate(coordinates):
            torch.testing.assert_close(
                grid.y[0, 0, block_y, block_x],
                coefficients[source_index].reshape(8, 8),
            )

    def test_full_adapter_falls_back_to_compact_and_preserves_native_stats(self) -> None:
        coefficients = torch.arange(6 * 64, dtype=torch.int16).reshape(6, 64)
        coordinates = [(y, x) for y in range(2) for x in range(3)]
        compact_batch = SimpleNamespace(
            layout="compact",
            selected_coefficients=list(range(64)),
            coefficients=coefficients,
            block_metadata=[
                {
                    "request_index": 0,
                    "global_image_index": 239,
                    "semantic_slot_id": 0,
                    "block_y": block_y,
                    "block_x": block_x,
                }
                for block_y, block_x in coordinates
            ],
            execution_stats={"compressed_payload_bytes_read": 1234, "actual_vector_count": 11},
        )

        class FakeReader:
            def image_metadata(self, image_id: int) -> dict:
                self.image_id = image_id
                return PipelineControlTest._grayscale_metadata()

            def read_batch(self, image_ids: list[int], **kwargs):
                self.read_image_ids = image_ids
                self.read_kwargs = kwargs
                return compact_batch

        class FakeDirectDct:
            @staticmethod
            def read_and_adapt_batch(*args, **kwargs):
                raise RuntimeError("YCbCr DCT grid layout requires Y, Cb, and Cr components per image")

            @staticmethod
            def adapt_galp_batch_to_rgbnomore(reader, grid, image_ids, **kwargs):
                self_grid = grid
                self_image_ids = image_ids
                assert torch.count_nonzero(self_grid.cbcr).item() == 0
                assert self_image_ids == [239]
                return grid.y, grid.cbcr

        adapter = object.__new__(GalpAdapter)
        adapter.reader = FakeReader()
        adapter.direct_dct = FakeDirectDct()
        adapter.args = SimpleNamespace(
            cache_capacity_mib=0,
            decode_batch_rowgroups=64,
            rowgroup_prefetch_depth=16,
            rowgroup_prefetch_workers=4,
            rowgroup_prefetch_min_decode_batches=1,
            plan_cache_capacity=0,
            enable_planless_execution=True,
            scheduling_policy="limited-overlap",
            transform_blocks_per_launch=0,
            transform_ctas_per_launch=0,
            use_low_priority_streams=True,
            no_dequantize=False,
            no_scale=False,
            preprocess="rgbnomore-val",
            block_major_double_buffer="auto",
        )
        adapter.rgbnomore_transform = object()
        adapter.device = torch.device("cpu")

        loaded = adapter._load_postdecode([{"galp_image_id": 239, "label": 7, "ordinal": 239}])

        self.assertEqual(adapter.reader.read_kwargs["layout"], "compact")
        self.assertEqual(adapter.reader.read_kwargs["crop_execution_mode"], "full-rowgroup-decode")
        self.assertEqual(loaded.label_values, [7])
        self.assertEqual(loaded.native_stats[0]["compressed_payload_bytes_read"], 1234)
        self.assertEqual(loaded.native_stats[0]["actual_vector_count"], 11)
        self.assertEqual(loaded.native_stats[0]["grayscale_full_y_block_count"], 6)
        self.assertEqual(loaded.native_stats[0]["grayscale_full_fallback_count"], 1)


if __name__ == "__main__":
    unittest.main()
