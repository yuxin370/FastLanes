from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import galp.torch
from galp.profiles import DirectDctProfile
from galp.profiles.rgbnomore import VALIDATION, VALIDATION_CENTER_CROP_512, SWINV2_VALIDATION
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


@dataclass(frozen=True)
class _FakeTensor:
    values: tuple[object, ...]
    shape: tuple[int, ...]
    dtype: str = "torch.float32"

    def stride(self) -> tuple[int, ...]:
        return tuple(
            1 if index == len(self.shape) - 1 else self.shape[index + 1]
            for index in range(len(self.shape))
        )


class _NativeBatch:
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

    def __init__(
        self,
        image_ids=None,
        *,
        dct_coeffs: str = "all",
        transforms=None,
    ) -> None:
        self.global_image_ids = list(image_ids or [4, 7])
        self.y = _FakeTensor(
            ("y", *self.global_image_ids),
            (len(self.global_image_ids), 1),
        )
        self.cbcr = _FakeTensor(
            ("cbcr", *self.global_image_ids),
            (len(self.global_image_ids), 2),
        )
        self.coefficients = {
            "canonical_selection": dct_coeffs,
            "global_image_ids": tuple(self.global_image_ids),
        }
        self.transform_descriptors = (
            [{"global_image_id": value} for value in self.global_image_ids]
            if transforms is None
            else [dict(value) for value in transforms]
        )
        self.record_stream_calls: list[tuple[int, ...]] = []

    def record_stream(self, *values: int) -> None:
        self.record_stream_calls.append(values)

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

    def __init__(self, reader, profile_id: str, dct_coeffs: str) -> None:
        self.reader = reader
        self.profile_id = profile_id
        self.dct_coeffs = dct_coeffs
        self.batches: list[list[int]] = []
        self.transforms = None
        self.offset = 0
        self.close_count = 0

    def reset(self, batches, *, transforms_by_batch):
        self.batches = [list(batch) for batch in batches]
        self.transforms = transforms_by_batch
        self.offset = 0
        self.prefetched_batch_count = min(2, len(self.batches))
        self.reader.calls.append(
            ("pipeline", self.batches, self.profile_id, self.dct_coeffs, transforms_by_batch)
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.offset >= len(self.batches):
            raise StopIteration
        batch_index = self.offset
        self.offset += 1
        self.prefetched_batch_count = min(len(self.batches), self.offset + 2)
        transforms = None if self.transforms is None else self.transforms[batch_index]
        return _NativeBatch(
            self.batches[batch_index],
            dct_coeffs=self.dct_coeffs,
            transforms=transforms,
        )

    @property
    def metrics(self):
        return _NativeBatch.metrics

    def close(self) -> int:
        self.close_count += 1
        return 0


class _NativeReader:
    def __init__(self, manifest: str) -> None:
        self.manifest = manifest
        self.image_count = 12
        self.initialization_stats = {"manifest_load_ms": 1.0}
        self.calls: list[tuple[str, list[int], str, object]] = []
        self.pipelines: list[_NativePipeline] = []

    def plan(self, image_ids, profile_id, *, transforms):
        self.calls.append(("plan", image_ids, profile_id, transforms))
        return {"layout": "transformed_dct_grid", "image_count": len(image_ids)}

    def pipeline(self, profile_id, *, dct_coeffs="all"):
        pipeline = _NativePipeline(self, profile_id, dct_coeffs)
        self.pipelines.append(pipeline)
        return pipeline

    def read(self, image_ids, profile_id, *, dct_coeffs="all", transforms):
        self.calls.append(("read", image_ids, profile_id, dct_coeffs, transforms))
        return _NativeBatch(
            image_ids, dct_coeffs=dct_coeffs, transforms=transforms
        )

    def image_metadata(self, image_id):
        return {"image_id": image_id}

    def rowgroup_storage_bytes(self, shard_id, rowgroups):
        return shard_id + sum(rowgroups)


def _profile_info(profile_id: str) -> dict[str, object]:
    runtime = {
        VALIDATION.id: "compact-v3-planless-limited-o512-c512-v1",
        SWINV2_VALIDATION.id: "compact-v3-planless-limited-o512-c512-v1",
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
        DIRECT_DCT_BINDING_SCHEMA="galp-direct-dct-binding-v2",
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
        self.assertEqual(
            batch.tensors,
            (
                _FakeTensor(("y", 4, 7), (2, 1)),
                _FakeTensor(("cbcr", 4, 7), (2, 2)),
            ),
        )
        self.assertEqual(batch.global_image_ids, [4, 7])
        self.assertEqual(batch.sample_ids, batch.global_image_ids)
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
        batch.record_stream()
        stream = SimpleNamespace(cuda_stream=1234, device_index=0)
        batch.record_stream(stream)
        self.assertEqual(batch._native.record_stream_calls, [(), (1234, 0)])
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

    def test_swinv2_profile_is_accepted_by_public_reader_methods(self) -> None:
        reader = DirectDctReader("manifest.bin", native_module=_native_module())
        self.assertEqual(SWINV2_VALIDATION.id, "rgbnomore-swinv2-validation-v1")
        self.assertEqual(
            reader.profile_info(SWINV2_VALIDATION)["runtime_policy_id"],
            "compact-v3-planless-limited-o512-c512-v1",
        )
        with reader.pipeline(SWINV2_VALIDATION) as pipeline:
            pipeline.start([[4, 7]])
            self.assertEqual(next(pipeline).profile_id, SWINV2_VALIDATION.id)
        self.assertEqual(reader.read([4, 7], SWINV2_VALIDATION).profile_id, SWINV2_VALIDATION.id)

    def test_coefficient_selection_defaults_to_all_and_is_forwarded(self) -> None:
        reader = DirectDctReader("manifest.bin", native_module=_native_module())

        reader.pipeline(VALIDATION).start([[4, 7]])
        reader.pipeline(VALIDATION, dct_coeffs="first:16").start([[4, 7]])
        reader.read([4, 7], VALIDATION)
        reader.read([4, 7], VALIDATION, dct_coeffs="list:5,0,2")

        self.assertEqual(reader._native.calls[0][3], "all")
        self.assertEqual(reader._native.calls[1][3], "first:16")
        self.assertEqual(reader._native.calls[2][3], "all")
        self.assertEqual(reader._native.calls[3][3], "list:5,0,2")

    def test_pythonic_coefficients_normalize_to_legacy_binding_contract(self) -> None:
        reader = DirectDctReader("manifest.bin", native_module=_native_module())

        legacy_all = reader.read([4, 7], VALIDATION, dct_coeffs="all")
        pythonic_all = reader.read([4, 7], VALIDATION, coefficients=None)
        legacy_prefix = reader.read(
            [4, 7], VALIDATION, dct_coeffs="first:32"
        )
        pythonic_prefix = reader.read(
            [4, 7], VALIDATION, coefficients=range(32)
        )
        legacy_ordered = reader.read(
            [4, 7], VALIDATION, dct_coeffs="list:5,0,2"
        )
        pythonic_ordered = reader.read(
            [4, 7], VALIDATION, coefficients=[5, 0, 2]
        )

        self.assertEqual(legacy_all.coefficients, pythonic_all.coefficients)
        self.assertEqual(legacy_prefix.coefficients, pythonic_prefix.coefficients)
        self.assertEqual(legacy_ordered.coefficients, pythonic_ordered.coefficients)
        self.assertEqual(
            reader._native.calls[-1][3],
            "list:5,0,2",
            "explicit coefficient order must reach the binding unchanged",
        )

    def test_pythonic_coefficients_reject_invalid_and_conflicting_inputs(self) -> None:
        reader = DirectDctReader("manifest.bin", native_module=_native_module())

        for coefficients in ([], [64], [-1], [1, 1], range(65)):
            with self.subTest(coefficients=coefficients):
                with self.assertRaises(ValueError):
                    reader.read(
                        [4, 7], VALIDATION, coefficients=coefficients
                    )

        with self.assertRaisesRegex(TypeError, "mutually exclusive"):
            reader.read(
                [4, 7],
                VALIDATION,
                coefficients=range(32),
                dct_coeffs="first:32",
            )
        with self.assertRaisesRegex(TypeError, "mutually exclusive"):
            reader.pipeline(
                VALIDATION,
                coefficients=range(32),
                dct_coeffs="first:32",
            )

    def test_image_ids_reject_implicit_conversion_and_out_of_range(self) -> None:
        reader = DirectDctReader("manifest.bin", native_module=_native_module())

        for image_id in (True, 1.9, "3"):
            with self.subTest(image_id=image_id):
                with self.assertRaisesRegex(TypeError, "image ids must be integers"):
                    reader.read([image_id], VALIDATION)
                with self.assertRaisesRegex(TypeError, "image ids must be integers"):
                    reader.pipeline(VALIDATION).start([[image_id]])

        for image_id in (-1, reader.image_count, 1 << 32):
            with self.subTest(image_id=image_id):
                with self.assertRaisesRegex(ValueError, "out of range"):
                    reader.read([image_id], VALIDATION)
                with self.assertRaisesRegex(ValueError, "out of range"):
                    reader.pipeline(VALIDATION).start([[image_id]])

    def test_iter_batches_is_thin_pipeline_wrapper_and_closes(self) -> None:
        logical_batches = [[4, 7], [8, 9]]
        transforms = [
            [{"global_image_id": 4}, {"global_image_id": 7}],
            [{"global_image_id": 8}, {"global_image_id": 9}],
        ]

        old_reader = DirectDctReader(
            "manifest.bin", native_module=_native_module()
        )
        with old_reader.pipeline(
            VALIDATION, dct_coeffs="first:32"
        ) as pipeline:
            pipeline.start(logical_batches, transforms_by_batch=transforms)
            old_batches = list(pipeline)

        new_reader = DirectDctReader(
            "manifest.bin", native_module=_native_module()
        )
        new_batches = list(
            new_reader.iter_batches(
                logical_batches,
                profile=VALIDATION,
                coefficients=range(32),
                transforms_by_batch=transforms,
            )
        )

        self.assertEqual(len(old_batches), len(new_batches))
        for old_batch, new_batch in zip(old_batches, new_batches, strict=True):
            self.assertEqual(old_batch.sample_ids, new_batch.sample_ids)
            self.assertEqual(old_batch.y, new_batch.y)
            self.assertEqual(old_batch.cbcr, new_batch.cbcr)
            self.assertEqual(old_batch.y.shape, new_batch.y.shape)
            self.assertEqual(old_batch.cbcr.shape, new_batch.cbcr.shape)
            self.assertEqual(old_batch.y.dtype, new_batch.y.dtype)
            self.assertEqual(old_batch.cbcr.dtype, new_batch.cbcr.dtype)
            self.assertEqual(old_batch.y.stride(), new_batch.y.stride())
            self.assertEqual(old_batch.cbcr.stride(), new_batch.cbcr.stride())
            self.assertEqual(old_batch.coefficients, new_batch.coefficients)
            self.assertEqual(old_batch.layout, new_batch.layout)
            self.assertEqual(
                old_batch.transform_descriptors,
                new_batch.transform_descriptors,
            )
        self.assertEqual(old_reader._native.calls, new_reader._native.calls)
        self.assertEqual(new_reader._native.pipelines[0].close_count, 1)

        early_reader = DirectDctReader(
            "manifest.bin", native_module=_native_module()
        )
        with early_reader.iter_batches(
            logical_batches,
            profile=VALIDATION,
            coefficients=range(32),
        ) as batches:
            next(batches)
        self.assertEqual(early_reader._native.pipelines[0].close_count, 1)

        error_reader = DirectDctReader(
            "manifest.bin", native_module=_native_module()
        )
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with error_reader.iter_batches(
                (batch for batch in logical_batches),
                profile=VALIDATION,
                coefficients=range(32),
            ) as batches:
                next(batches)
                raise RuntimeError("consumer failed")
        self.assertEqual(error_reader._native.pipelines[0].close_count, 1)

    def test_old_binding_schema_is_rejected(self) -> None:
        module = SimpleNamespace(DirectDctReader=_NativeReader)
        with self.assertRaisesRegex(RuntimeError, "rebuild the binding"):
            DirectDctReader("manifest.bin", native_module=module)

    def test_stale_binding_with_data_schemas_is_rejected(self) -> None:
        module = SimpleNamespace(
            DirectDctReader=_NativeReader,
            DIRECT_DCT_PROFILE_SCHEMA="galp-direct-dct-profile-v1",
            DIRECT_DCT_METRICS_SCHEMA="galp-direct-dct-metrics-v2",
            direct_dct_profile_info=_profile_info,
        )
        with self.assertRaisesRegex(RuntimeError, "rebuild the binding"):
            DirectDctReader("manifest.bin", native_module=module)


if __name__ == "__main__":
    unittest.main()
