from __future__ import annotations

import unittest
from types import SimpleNamespace

import galp.torch
from galp.torch.experimental import DirectDctPlsPipeline


class _NativeMicrobatch:
    y = "pls-y"
    cbcr = "pls-cbcr"
    targets = "pls-targets"
    epoch = 7
    pool_index = 2
    microbatch_index_in_pool = 0
    pool_offset = 0
    image_count = 2
    is_pool_end = True
    global_image_ids = [9, 3]
    labels = [4, 1]

    def __init__(self) -> None:
        self.record_stream_calls: list[tuple[int, ...]] = []

    def record_stream(self, *values: int) -> None:
        self.record_stream_calls.append(values)


class _NativePool:
    epoch = 7
    pool_index = 2
    image_count = 2
    microbatch_count = 1
    virtual_pls_ids = [11, 5, 8, 2]
    execution_stats = {"selected_vector_count": 7}

    def __init__(self) -> None:
        self._done = False
        self.retired = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return _NativeMicrobatch()

    def microbatch(self, _index: int):
        return _NativeMicrobatch()

    def retire(self) -> None:
        self.retired = True


class _NativePipeline:
    sample_count = 1281167
    pls_count = 1252
    segment_images = 1024
    has_next_pool = True
    prefetch_stats = {
        "context_capacity": 2,
        "live_context_count": 1,
        "peak_live_context_count": 2,
    }

    def __init__(self, manifest, mapping, seed, mapping_sha256, **options) -> None:
        self.arguments = (manifest, mapping, seed, mapping_sha256, options)
        self.epoch = None

    def start_epoch(self, epoch):
        self.epoch = epoch

    def next_pool(self):
        return _NativePool()

    def __next__(self):
        return _NativeMicrobatch()

    def close(self):
        self.closed = True


def _native_module():
    return SimpleNamespace(
        DirectDctPlsPipeline=_NativePipeline,
        reclaim_direct_dct_pls_pools=lambda: 3,
    )


class ExperimentalDirectDctPlsApiTest(unittest.TestCase):
    def test_stable_namespace_does_not_export_experimental_pls(self) -> None:
        self.assertNotIn("DirectDctPlsPipeline", galp.torch.__all__)
        self.assertFalse(hasattr(galp.torch, "DirectDctPlsPipeline"))

    def test_adapter_only_forwards_model_and_lifetime_contracts(self) -> None:
        pipeline = DirectDctPlsPipeline(
            "dataset/manifest.bin",
            "dataset/ordered_mapping.csv",
            training_seed=11997733,
            expected_mapping_sha256="0" * 64,
            native_module=_native_module(),
        ).start_epoch(7)

        self.assertEqual(pipeline.sample_count, 1281167)
        self.assertEqual(pipeline.pls_count, 1252)
        self.assertEqual(pipeline.reclaim_finished_pools(), 3)
        pool = pipeline.next_pool()
        self.assertEqual(pool.virtual_pls_ids, [11, 5, 8, 2])
        self.assertEqual(pool.execution_stats["selected_vector_count"], 7)
        self.assertEqual(pipeline.prefetch_stats["context_capacity"], 2)
        microbatch = next(pool)
        self.assertEqual(microbatch.tensors, ("pls-y", "pls-cbcr", "pls-targets"))
        self.assertEqual(microbatch.global_image_ids, [9, 3])
        microbatch.record_stream()
        stream = SimpleNamespace(cuda_stream=4321, device_index=0)
        microbatch.record_stream(stream)
        self.assertEqual(microbatch._native.record_stream_calls, [(), (4321, 0)])
        pool.retire()
        self.assertTrue(pool._native.retired)


if __name__ == "__main__":
    unittest.main()
