"""GPU regressions for tensor ownership, consumer streams and metrics completion."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import _fastlens_direct_dct as native
import _fastlens_test_consumer as consumer

from fastlens import cli
from fastlens.profiles.rgbnomore import VALIDATION
from fastlens.torch import DirectDctReader
from fastlens.torch.create_pls_smoke_fixture import create_fixture


class DirectDctLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            if os.environ.get("FASTLENS_REQUIRE_CUDA_SMOKE") == "1":
                raise RuntimeError("CUDA device/driver unavailable but FASTLENS_REQUIRE_CUDA_SMOKE=1")
            raise unittest.SkipTest("CUDA device/driver unavailable")
        if os.environ.get("CUDA_LAUNCH_BLOCKING") not in (None, "", "0"):
            raise RuntimeError("lifetime tests require asynchronous CUDA submission")
        temporary = tempfile.TemporaryDirectory(prefix="fastlens-lifetime-")
        cls.addClassCleanup(temporary.cleanup)
        tools = cli.find_tools(None)
        cls.manifest, _, _ = create_fixture(
            Path(temporary.name), *(str(tools[name]) for name in cli.TOOL_NAMES)
        )

    def setUp(self):
        backend = mock.patch.dict(os.environ, {"FASTLENS_PHASE4_NATIVE_LIFETIME": "1"})
        backend.start()
        self.addCleanup(backend.stop)
        self.addCleanup(native.manual_reclaim)
        self.addCleanup(torch.cuda.synchronize)

    def open_batch(self):
        pipeline = DirectDctReader(self.manifest).pipeline(VALIDATION).start([[0, 1, 2]])
        self.addCleanup(pipeline.close)
        return pipeline, next(pipeline)

    def test_backing_survives_until_every_tensor_storage_is_released(self):
        pipeline, batch = self.open_batch()
        state = batch._native._enable_lifetime_shadow_for_test()
        y, cbcr = batch.tensors
        expected = y.square().mean() + cbcr.square().mean()
        torch.cuda.current_stream().synchronize()
        _ = pipeline.metrics
        del batch
        pipeline.close()
        torch.testing.assert_close(y.square().mean() + cbcr.square().mean(), expected, rtol=0, atol=0)
        torch.cuda.current_stream().synchronize()

        del y
        native.manual_reclaim()
        self.assertFalse(state.snapshot["all_storage_references_released"])
        self.assertTrue(state.snapshot["backing_storage_present"])
        self.assertFalse(state.snapshot["reclaim_executed"])

        del cbcr
        torch.cuda.current_stream().synchronize()
        native.manual_reclaim()
        self.assertTrue(state.snapshot["all_storage_references_released"])
        self.assertTrue(state.snapshot["reclaim_executed"])
        self.assertFalse(state.snapshot["backing_storage_present"])

    def test_side_stream_blocks_reclaim_and_reuse_until_consumer_completes(self):
        pipeline, batch = self.open_batch()
        state = batch._native._enable_lifetime_shadow_for_test()
        y = batch.y
        batch._native._wait_for_producer_completion_for_test()
        _ = pipeline.metrics
        expected = y.reshape(-1)[0].item()
        pointer = y.data_ptr()
        backing_bytes = y.numel() * y.element_size()
        side = torch.cuda.Stream()
        batch.record_stream(side)
        handle = consumer.arm_outstanding_consumer(pointer, side.cuda_stream, side.device_index, 5000)
        try:
            self.assertFalse(handle.consumer_complete)
            del y, batch
            pipeline.close()
            native.manual_reclaim()
            before = state.snapshot
            self.assertTrue(before["producer_complete"])
            self.assertTrue(before["all_storage_references_released"])
            self.assertFalse(before["native_reclaim_eligible"])
            self.assertFalse(before["reclaim_executed"])
            self.assertTrue(before["backing_storage_present"])
            self.assertGreater(before["pending_consumer_dependency_count"], 0)
            self.assertNotIn(pointer, native._device_pool_reuse_probe_for_test(backing_bytes, 3))
            self.assertFalse(handle.consumer_complete)

            handle.release_gate()
            self.assertTrue(handle.wait_consumer(7000))
            self.assertFalse(handle.gate_timed_out)
            self.assertEqual(handle.consumer_checksum, expected)
            side.synchronize()
            native.manual_reclaim()
            after = state.snapshot
            self.assertTrue(after["reclaim_executed"])
            self.assertFalse(after["backing_storage_present"])
            self.assertEqual(after["pending_consumer_dependency_count"], 0)
        finally:
            handle.cleanup(7000)

    def test_metrics_finalize_after_consumer_completion(self):
        pipeline, batch = self.open_batch()
        value = batch.y.square().mean() + batch.cbcr.square().mean()
        completed = torch.cuda.Event()
        completed.record()
        completed.synchronize()
        self.assertTrue(batch._native._metrics_completion["host_snapshot_taken"])
        self.assertTrue(batch._native._metrics_completion["gpu_timings_finalized"])
        self.assertTrue(pipeline._native._metrics_completion["gpu_timings_finalized"])
        self.assertTrue(pipeline.metrics.complete)
        self.assertTrue(torch.isfinite(value).item())

    def test_legacy_backend_preserves_tensors_and_reclaims_after_close(self):
        with mock.patch.dict(os.environ, {"FASTLENS_PHASE4_NATIVE_LIFETIME": "0"}):
            pipeline, batch = self.open_batch()
        self.assertEqual(pipeline._native._lifetime_backend_for_test, "legacy")
        baseline = native._lifetime_reclaim_stats_for_test()["legacy"]["pending_reclaim_count"]
        y = batch.y
        expected = y.square().mean()
        torch.cuda.current_stream().synchronize()
        del batch
        pipeline.close()
        torch.testing.assert_close(y.square().mean(), expected, rtol=0, atol=0)
        del y
        torch.cuda.current_stream().synchronize()
        native.manual_reclaim()
        self.assertEqual(
            native._lifetime_reclaim_stats_for_test()["legacy"]["pending_reclaim_count"], baseline
        )


if __name__ == "__main__":
    unittest.main()
