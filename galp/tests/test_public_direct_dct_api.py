from __future__ import annotations

import unittest
from types import SimpleNamespace

import galp.torch
from galp.profiles import DirectDctProfile
from galp.profiles.rgbnomore import VALIDATION, VALIDATION_CENTER_CROP_512
from galp.torch import DirectDctReader
from galp.diagnostics.direct_dct import (
    cache_stats,
    execution_stats,
    execution_stats_snapshot,
    image_metadata,
    pipeline_stats,
    plan_preview,
    rowgroup_storage_bytes,
)


class _NativeBatch:
    coefficients = None
    y = "y"
    cbcr = "cbcr"
    global_image_ids = [4, 7]
    transform_descriptors = [{"global_image_id": 4}, {"global_image_id": 7}]
    layout = "transformed_dct_grid"
    execution_stats = {"decode_ms": 1.0}
    execution_stats_snapshot = {"ready": True}
    cache_stats = {"hits": 0}
    metrics = {
        "schema": "galp-direct-dct-metrics-v2",
        "complete": True,
        "consumer_wait_ms": 0.1,
        "submit_to_ready_ms": 2.0,
        "producer_ms": 1.5,
        "planning_ms": 0.3,
        "io_ms": 0.7,
        "decode_ms": 0.4,
        "transform_ms": 0.2,
        "logical_bytes": 100,
        "physical_bytes": 110,
        "peak_transient_bytes": 4096,
    }

class _NativePipeline:
    ready = True
    started = True
    prefetch_metrics = {
        "producer_ms": 2.0,
        "planning_ms": 0.3,
        "io_ms": 0.7,
        "ordered_submission_ms": 1.0,
    }
    prefetched_batch_count = 0

    def __init__(self, reader, profile_id: str) -> None:
        self.reader = reader
        self.profile_id = profile_id
        self.batches: list[list[int]] = []
        self.transforms = None
        self.offset = 0

    def reset(self, batches, *, transforms_by_batch):
        self.batches = [list(batch) for batch in batches]
        self.transforms = transforms_by_batch
        self.offset = 0
        self.prefetched_batch_count = min(2, len(self.batches))
        self.reader.calls.append(("pipeline", self.batches, self.profile_id, transforms_by_batch))

    def __iter__(self):
        return self

    def __next__(self):
        if self.offset >= len(self.batches):
            raise StopIteration
        self.offset += 1
        self.prefetched_batch_count = min(len(self.batches), self.offset + 2)
        return _NativeBatch()

    @property
    def metrics(self):
        return _NativeBatch.metrics

    def close(self) -> int:
        return 0


class _NativeReader:
    def __init__(self, manifest: str) -> None:
        self.manifest = manifest
        self.image_count = 12
        self.initialization_stats = {"manifest_load_ms": 1.0}
        self.calls: list[tuple[str, list[int], str, object]] = []

    def plan(self, image_ids, profile_id, *, transforms):
        self.calls.append(("plan", image_ids, profile_id, transforms))
        return {"layout": "transformed_dct_grid", "image_count": len(image_ids)}

    def pipeline(self, profile_id):
        return _NativePipeline(self, profile_id)

    def read(self, image_ids, profile_id, *, transforms):
        self.calls.append(("read", image_ids, profile_id, transforms))
        return _NativeBatch()

    def image_metadata(self, image_id):
        return {"image_id": image_id}

    def rowgroup_storage_bytes(self, shard_id, rowgroups):
        return shard_id + sum(rowgroups)


def _profile_info(profile_id: str) -> dict[str, object]:
    runtime = {
        VALIDATION.id: "compact-v3-planless-limited-o512-c512-v1",
        VALIDATION_CENTER_CROP_512.id: "block-major-p4-scheduled-bounded-110-v1",
    }[profile_id]
    return {
        "schema": "galp-direct-dct-profile-v1",
        "id": profile_id,
        "runtime_policy_id": runtime,
        "layout": "transformed_dct_grid",
    }


def _native_module():
    return SimpleNamespace(
        DIRECT_DCT_PROFILE_SCHEMA="galp-direct-dct-profile-v1",
        DIRECT_DCT_METRICS_SCHEMA="galp-direct-dct-metrics-v2",
        DirectDctReader=_NativeReader,
        direct_dct_profile_info=_profile_info,
    )


class PublicDirectDctApiTest(unittest.TestCase):
    def test_public_torch_namespace_contains_only_model_facing_types(self) -> None:
        self.assertEqual(
            set(galp.torch.__all__),
            {
                "DirectDctBatch",
                "DirectDctMetrics",
                "DirectDctPipeline",
                "DirectDctReader",
            },
        )
        self.assertFalse(hasattr(galp.torch, "DirectDctFuture"))

    def test_profile_driven_pipeline_and_read(self) -> None:
        reader = DirectDctReader("dataset/manifest.bin", native_module=_native_module())
        transforms = [{"crop": (0, 0, 224, 224)}, {"horizontal_flip": True}]

        preview = plan_preview(reader, [4, 7], VALIDATION, transforms=transforms)
        self.assertEqual(preview["image_count"], 2)
        pipeline = reader.pipeline(VALIDATION).start(
            [[4, 7]], transforms_by_batch=[transforms]
        )
        self.assertEqual(pipeline_stats(pipeline)["planning_ms"], 0.3)
        self.assertEqual(pipeline_stats(pipeline)["prefetched_batch_count"], 1)
        batch = next(pipeline)
        self.assertEqual(batch.tensors, ("y", "cbcr"))
        self.assertEqual(batch.global_image_ids, [4, 7])
        self.assertEqual(batch.profile_id, VALIDATION.id)
        self.assertTrue(batch.metrics.complete)
        self.assertEqual(batch.metrics.physical_bytes, 110)
        self.assertTrue(pipeline.metrics.complete)
        self.assertFalse(hasattr(batch, "execution_stats"))
        self.assertEqual(execution_stats(batch), {"decode_ms": 1.0})
        self.assertEqual(execution_stats_snapshot(batch), {"ready": True})
        self.assertEqual(cache_stats(batch), {"hits": 0})

        sync_batch = reader.read([4, 7], VALIDATION_CENTER_CROP_512)
        self.assertEqual(sync_batch.profile_id, VALIDATION_CENTER_CROP_512.id)
        self.assertEqual(image_metadata(reader, 4), {"image_id": 4})
        self.assertEqual(rowgroup_storage_bytes(reader, 2, [3, 5]), 10)
        self.assertFalse(hasattr(reader, "plan"))
        self.assertFalse(hasattr(reader, "prefetch"))
        self.assertFalse(hasattr(reader, "image_metadata"))
        self.assertFalse(hasattr(reader, "initialization_stats"))
        self.assertFalse(hasattr(batch, "record_stream"))
        self.assertFalse(hasattr(reader, "prefetch_rgbnomore_val_batch"))
        self.assertFalse(hasattr(reader, "prefetch_batch"))

    def test_python_profile_contains_no_runtime_tuning(self) -> None:
        reader = DirectDctReader("manifest.bin", native_module=_native_module())
        self.assertEqual(
            reader.profile_info(VALIDATION)["runtime_policy_id"],
            "compact-v3-planless-limited-o512-c512-v1",
        )
        self.assertFalse(hasattr(VALIDATION, "runtime_policy_id"))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            DirectDctProfile("")

    def test_old_binding_schema_is_rejected(self) -> None:
        module = SimpleNamespace(DirectDctReader=_NativeReader)
        with self.assertRaisesRegex(RuntimeError, "rebuild the binding"):
            DirectDctReader("manifest.bin", native_module=module)


if __name__ == "__main__":
    unittest.main()
