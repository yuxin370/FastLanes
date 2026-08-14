from __future__ import annotations

import unittest
from types import SimpleNamespace

from galp.profiles import DirectDctProfile
from galp.profiles.rgbnomore import VALIDATION, VALIDATION_CENTER_CROP_512
from galp.torch import DirectDctReader
from galp.torch.diagnostics import (
    cache_stats,
    execution_stats,
    execution_stats_snapshot,
    prefetch_stats,
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

    def __init__(self) -> None:
        self.recorded = False

    def record_stream(self) -> None:
        self.recorded = True


class _NativeFuture:
    ready = True
    started = True
    active = False
    finished = True
    producer_active_ms = 2.0
    planning_ms = 0.3
    io_staging_ms = 0.7
    ordered_submission_ms = 1.0

    def __init__(self) -> None:
        self.released = False

    def read(self):
        return _NativeBatch()

    def cancel(self) -> bool:
        return False

    def release_submission(self) -> bool:
        self.released = True
        return True


class _NativeReader:
    def __init__(self, manifest: str) -> None:
        self.manifest = manifest
        self.image_count = 12
        self.initialization_stats = {"manifest_load_ms": 1.0}
        self.calls: list[tuple[str, list[int], str, object]] = []

    def plan(self, image_ids, profile_id, *, transforms):
        self.calls.append(("plan", image_ids, profile_id, transforms))
        return {"layout": "transformed_dct_grid", "image_count": len(image_ids)}

    def prefetch(self, image_ids, profile_id, *, transforms):
        self.calls.append(("prefetch", image_ids, profile_id, transforms))
        return _NativeFuture()

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
        DirectDctReader=_NativeReader,
        direct_dct_profile_info=_profile_info,
    )


class PublicDirectDctApiTest(unittest.TestCase):
    def test_profile_driven_plan_prefetch_and_read(self) -> None:
        reader = DirectDctReader("dataset/manifest.bin", native_module=_native_module())
        transforms = [{"crop": (0, 0, 224, 224)}, {"horizontal_flip": True}]

        preview = reader.plan([4, 7], VALIDATION, transforms=transforms)
        self.assertEqual(preview["image_count"], 2)
        future = reader.prefetch([4, 7], VALIDATION, transforms=transforms)
        self.assertEqual(prefetch_stats(future)["planning_ms"], 0.3)
        self.assertFalse(hasattr(future, "telemetry"))
        batch = future.read()
        self.assertEqual(batch.tensors, ("y", "cbcr"))
        self.assertEqual(batch.global_image_ids, [4, 7])
        self.assertEqual(batch.profile_id, VALIDATION.id)
        batch.record_stream()
        self.assertTrue(batch._native.recorded)
        self.assertFalse(hasattr(batch, "execution_stats"))
        self.assertEqual(execution_stats(batch), {"decode_ms": 1.0})
        self.assertEqual(execution_stats_snapshot(batch), {"ready": True})
        self.assertEqual(cache_stats(batch), {"hits": 0})

        sync_batch = reader.read([4, 7], VALIDATION_CENTER_CROP_512)
        self.assertEqual(sync_batch.profile_id, VALIDATION_CENTER_CROP_512.id)
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
