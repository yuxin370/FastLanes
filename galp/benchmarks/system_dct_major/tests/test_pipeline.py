from __future__ import annotations

import sys
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from galp.torch import DirectDctBatch


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]

from galp.benchmarks.system_dct_major.pipeline import (  # noqa: E402
    GalpAdapter,
    CoorDLAdapter,
    FfcvAdapter,
    LoadedBatch,
    _accumulate_native,
    _finalize_native_stats,
    _process_io_delta,
    _process_io_snapshot,
    _rgbnomore_fixed_validation_transform,
    _validate_identity,
    _training_step,
    _training_probe,
)
from galp.benchmarks.system_dct_major.common import BLOCK_MAJOR_RUNTIME_PROFILE


class PipelineControlTest(unittest.TestCase):
    def test_training_step_updates_weights_and_retains_optimizer_state(self) -> None:
        torch.manual_seed(17)
        model = torch.nn.Linear(4, 3).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = LoadedBatch(
            inputs=(torch.randn(5, 4),), labels=torch.tensor([0, 1, 2, 0, 1]),
            ordinals=list(range(5)), label_values=[0, 1, 2, 0, 1], on_device=True,
        )
        before = [p.detach().clone() for p in model.parameters()]
        output, loss = _training_step(model, optimizer, batch, 3)
        self.assertEqual(tuple(output.shape), (5, 3))
        self.assertFalse(output.requires_grad)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(_training_probe(model, before).values()))
        # A second, partial batch must perform another update without resetting Adam.
        batch.inputs = (batch.inputs[0][:2],)
        batch.labels = batch.labels[:2]
        _training_step(model, optimizer, batch, 3)
        self.assertTrue(all(int(state["step"]) == 2 for state in optimizer.state.values()))

    def test_training_probe_rejects_missing_update(self) -> None:
        model = torch.nn.Linear(4, 3)
        model(torch.ones(2, 4)).sum().backward()
        before = [p.detach().clone() for p in model.parameters()]
        self.assertFalse(_training_probe(model, before)["parameters_updated"])

    def test_coordl_retains_reader_and_sample_order_across_partial_epochs(self) -> None:
        modules = {
            name: mock.MagicMock() for name in (
                "nvidia", "nvidia.dali", "nvidia.dali.pipeline", "nvidia.dali.plugin",
                "nvidia.dali.plugin.pytorch",
            )
        }
        dali = modules["nvidia.dali"]
        dali.backend.GetSchema.return_value.GetArgumentNames.return_value = ["cache_size"]

        # CoorDL 0.20 uses output_dtype, unlike the modern DALI fn API.
        def normalize(*, device, output_dtype, output_layout, crop, crop_pos_x, crop_pos_y, mean, std):
            self.assertEqual(output_dtype, dali.types.FLOAT)
            self.assertEqual(output_layout, "CHW")
            self.assertEqual(crop, (224, 224))
            self.assertEqual((crop_pos_x, crop_pos_y), (0.5, 0.5))
            self.assertEqual(mean, [127.5] * 3)
            self.assertEqual(std, [127.5] * 3)
            return mock.Mock()

        dali.ops.CropMirrorNormalize.side_effect = normalize
        pipeline_initializations = []

        class Pipeline:
            def __init__(self, **kwargs):
                pipeline_initializations.append(kwargs)

        modules["nvidia.dali.pipeline"].Pipeline = Pipeline
        factory = modules["nvidia.dali.plugin.pytorch"].DALIGenericIterator
        iterator = factory.return_value
        samples = [{"path": f"/dataset/image{index}.jpg", "ordinal": index, "label": index + 10} for index in range(3)]
        batches = [
            [{"image": torch.zeros(2, 3, 224, 224), "ordinal": torch.tensor([0, 1])}],
            [{"image": torch.zeros(1, 3, 224, 224), "ordinal": torch.tensor([2])}],
        ]
        iterator.__next__.side_effect = batches * 2
        original_tensor = torch.tensor
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(sys.modules, modules), mock.patch(
            "galp.benchmarks.system_dct_major.pipeline.torch.tensor",
            side_effect=lambda values, **kwargs: original_tensor(values, dtype=kwargs["dtype"]),
        ):
            file_list = Path(temporary) / "files.txt"
            contract = {
                "execution": {"batch_size": 2, "workers": 3, "seed": 17},
                "preprocess": {"rgb": {"resize_shorter": None, "crop_size": [224, 224]}},
                "pipelines": {"coordl": {
                    "cache_size": 2, "file_list": str(file_list), "device_id": 0, "prefetch_queue_depth": 2,
                }},
            }
            adapter = CoorDLAdapter(contract, samples, torch.device("cuda"), "coordl")
            adapter.prime_cold_start()
            for _ in range(2):
                adapter.begin_repeat()
                first = adapter.load(samples[:2])
                tail = adapter.load(samples[2:])
                self.assertEqual(first.ordinals, [0, 1])
                self.assertEqual(tail.ordinals, [2])
                self.assertEqual(tail.label_values, [12])
                self.assertEqual(tail.inputs[0].shape[0], 1)
                adapter.end_repeat()
            self.assertEqual(file_list.read_text(), "dataset/image0.jpg 0\ndataset/image1.jpg 1\ndataset/image2.jpg 2\n")
            self.assertEqual(len(pipeline_initializations), 1)
            factory.assert_called_once()
            self.assertEqual(iterator.reset.call_count, 2)
            self.assertFalse(factory.call_args.kwargs["fill_last_batch"])
            self.assertTrue(factory.call_args.kwargs["last_batch_padded"])
            self.assertEqual(factory.call_args.kwargs["size"], 3)
            reader_args = dali.ops.FileReader.call_args.kwargs
            self.assertEqual(reader_args["cache_size"], 2)
            self.assertFalse(reader_args["random_shuffle"])
            self.assertFalse(reader_args["shuffle_after_epoch"])
            self.assertTrue(reader_args["pad_last_batch"])
            with mock.patch.object(Path, "unlink") as unlink:
                adapter.close()
                self.assertEqual(unlink.call_count, 6)

    def test_coordl_rejects_standard_dali_instead_of_disabling_cache(self) -> None:
        modules = {name: mock.MagicMock() for name in (
            "nvidia", "nvidia.dali", "nvidia.dali.pipeline", "nvidia.dali.plugin",
            "nvidia.dali.plugin.pytorch",
        )}
        modules["nvidia.dali"].backend.GetSchema.return_value.GetArgumentNames.return_value = []
        with mock.patch.dict(sys.modules, modules):
            adapter = CoorDLAdapter({}, [], torch.device("cuda"), "coordl")
            with self.assertRaisesRegex(RuntimeError, "not standard NVIDIA DALI"):
                adapter.begin_repeat()

    def test_coordl_rejects_existing_cache_before_creating_reader(self) -> None:
        modules = {name: mock.MagicMock() for name in (
            "nvidia", "nvidia.dali", "nvidia.dali.pipeline", "nvidia.dali.plugin",
            "nvidia.dali.plugin.pytorch",
        )}
        dali = modules["nvidia.dali"]
        dali.backend.GetSchema.return_value.GetArgumentNames.return_value = ["cache_size"]
        contract = {
            "execution": {}, "pipelines": {"coordl": {}},
            "preprocess": {"rgb": {"resize_shorter": None}},
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.object(Path, "exists", return_value=True):
            adapter = CoorDLAdapter(contract, [{"path": "/dataset/image.jpg"}], torch.device("cuda"), "coordl")
            with self.assertRaisesRegex(FileExistsError, "fresh cache"):
                adapter.begin_repeat()
        dali.ops.FileReader.assert_not_called()

    def test_ffcv_loader_preserves_order_and_partial_tail(self) -> None:
        modules = {
            name: mock.MagicMock() for name in (
                "ffcv", "ffcv.fields", "ffcv.fields.decoders", "ffcv.loader", "ffcv.transforms",
            )
        }
        batches = [
            (torch.full((2, 3, 224, 224), 255, dtype=torch.uint8), torch.tensor([0, 1])),
            (torch.zeros((1, 3, 224, 224), dtype=torch.uint8), torch.tensor([2])),
        ]
        loader = modules["ffcv.loader"].Loader
        loader.return_value.__iter__.return_value = iter(batches)
        samples = [{"ordinal": index, "label": index + 10} for index in range(3)]
        contract = {
            "execution": {"batch_size": 2, "workers": 2},
            "pipelines": {"ffcv": {"beton": "dataset.beton"}},
        }
        original_tensor = torch.tensor
        with mock.patch.dict(sys.modules, modules), mock.patch(
            "galp.benchmarks.system_dct_major.pipeline.torch.tensor",
            side_effect=lambda values, **kwargs: original_tensor(values, dtype=kwargs["dtype"]),
        ):
            adapter = FfcvAdapter(contract, samples, torch.device("cuda"), "ffcv")
            adapter.begin_repeat()
            first = adapter.load(samples[:2])
            last = adapter.load(samples[2:])
        self.assertEqual(loader.call_args.kwargs["indices"], [0, 1, 2])
        self.assertEqual(loader.call_args.kwargs["drop_last"], False)
        self.assertEqual(first.ordinals, [0, 1])
        self.assertEqual(last.label_values, [12])
        self.assertEqual(first.inputs[0].shape, (2, 3, 224, 224))
        self.assertEqual(first.inputs[0][0, 0, 0, 0].item(), 1.0)
        self.assertEqual(last.inputs[0][0, 0, 0, 0].item(), -1.0)

    def test_native_physical_orchestration_is_enabled_before_reader_creation(self) -> None:
        def make_reader(*args: object, **kwargs: object) -> None:
            self.assertEqual(os.environ["GALP_PHASE6_NATIVE_PHYSICAL"], "1")
            raise RuntimeError("reader reached")

        contract = {
            "pipelines": {
                "dct_major_coefficient_pushdown": {
                    "preprocess": "native",
                    "runtime_profile": BLOCK_MAJOR_RUNTIME_PROFILE,
                    "manifest": "unused",
                    "torch_binding_dir": "unused",
                }
            }
        }
        with mock.patch.dict(os.environ, {"GALP_PHASE6_NATIVE_PHYSICAL": "0"}):
            with mock.patch("galp.benchmarks.system_dct_major.pipeline.DirectDctReader", side_effect=make_reader):
                with self.assertRaisesRegex(RuntimeError, "reader reached"):
                    GalpAdapter(contract, [], torch.device("cuda"), "dct_major_coefficient_pushdown")

    def test_native_logical_hot_path_has_no_python_physical_stitch_or_sync(self) -> None:
        source = inspect.getsource(GalpAdapter.load)
        for forbidden in ("torch.cat", ".clone(", ".synchronize(", "_load_next_segment"):
            self.assertNotIn(forbidden, source)

    def test_native_physical_mode_schedules_logical_batches_without_shard_activation(self) -> None:
        class Pipeline:
            def start(self, batches):
                self.batches = batches
                return self

        adapter = object.__new__(GalpAdapter)
        adapter.pipeline = Pipeline()
        adapter._logical_batches = [
            [{"galp_image_id": 1000}, {"galp_image_id": 1001}],
            [{"galp_image_id": 1024}],
        ]
        adapter._cold_measurement_primed = False
        adapter.begin_repeat()
        self.assertEqual(adapter.pipeline.batches, [[1000, 1001], [1024]])

    def test_fixed_rgbnomore_transform_preserves_published_resize_reference(self) -> None:
        class Transform(torch.nn.Module):
            def __init__(self, name: str, *args: object, **kwargs: object) -> None:
                super().__init__()
                self.name = name
                self.args = args
                self.kwargs = kwargs

            def forward(self, value: object) -> object:
                return value

        ctrans = SimpleNamespace(
            ResizedCenterCrop_DCT=lambda *args: Transform("resize_crop", *args),
            ToRange=lambda **kwargs: Transform("range", **kwargs),
        )
        transform = _rgbnomore_fixed_validation_transform(ctrans)
        self.assertEqual(transform[0].name, "resize_crop")
        self.assertEqual(transform[0].args, (32, 28))
        self.assertEqual(transform[1].name, "range")
        self.assertEqual(transform[1].kwargs["orig_min"], -1024)
        self.assertEqual(transform[1].kwargs["orig_max"], 1016)



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

    def test_logical_views_without_new_physical_work_do_not_add_native_segments(self) -> None:
        physical = {
            "rowgroup_count": 48,
            "bounded_read_amplification_ppm": 1_100_000,
            "active_output_schedule_mmap_capacity_bytes": 16 * 1024 * 1024,
            "active_output_schedule_mmap_window_count": 2,
        }
        empty = {
            "rowgroup_count": 0,
            "bounded_read_amplification_ppm": 1_000_000,
            "active_output_schedule_mmap_capacity_bytes": 0,
            "active_output_schedule_mmap_window_count": 0,
        }
        totals: dict[str, object] = {}
        without_sidecar = {
            **physical,
            "active_output_schedule_mmap_capacity_bytes": 0,
            "active_output_schedule_mmap_window_count": 0,
        }
        for observed in (physical, empty, empty, without_sidecar):
            batch = LoadedBatch(
                inputs=(), labels=torch.empty(0), ordinals=[], label_values=[], on_device=True,
                native_stats=[{"_native_batch": object(), "segment_mode": "native-logical-batch"}],
            )
            with mock.patch(
                "galp.benchmarks.system_dct_major.pipeline._batch_native_stats", return_value=observed
            ):
                _finalize_native_stats(batch)
            self.assertEqual(len(batch.native_stats), int(observed["rowgroup_count"] > 0))
            for stats in batch.native_stats:
                _accumulate_native(totals, stats)
        self.assertEqual(totals["segment_count"], 2)
        self.assertEqual(totals["rowgroup_count"], 96)
        for key in ("bounded_read_amplification_ppm", "active_output_schedule_mmap_capacity_bytes",
                    "active_output_schedule_mmap_window_count"):
            value = physical[key]
            self.assertEqual(totals[key], value)
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                _accumulate_native(dict(totals), {key: value * 2})

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


    def test_galp_cold_prime_is_reused_by_first_measurement(self) -> None:
        class Pipeline:
            def __init__(self):
                self.starts = []

            def start(self, batches):
                self.starts.append(batches)
                return self

        adapter = object.__new__(GalpAdapter)
        adapter.pipeline = Pipeline()
        adapter._logical_batches = [[
            {"ordinal": 1, "galp_image_id": 1},
            {"ordinal": 2, "galp_image_id": 2},
        ]]
        adapter._cold_measurement_primed = False
        adapter._reuse_cold_measurement = False

        adapter.prime_cold_start()
        self.assertEqual(adapter.pipeline.starts, [[[1, 2]]])

        adapter.begin_repeat()
        adapter.begin_measurement()
        self.assertEqual(adapter.pipeline.starts, [[[1, 2]]])


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
