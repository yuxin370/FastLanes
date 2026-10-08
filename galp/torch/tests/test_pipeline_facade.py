from __future__ import annotations

from pathlib import Path
import unittest

from galp.profiles import DctModelProfile
from galp.profiles.rgbnomore import TRAINING_PLS, VALIDATION
from galp.torch import DirectDctPipeline, TrainingPolicy
from galp.torch.tests.test_public_direct_dct_api import (
    _NativeBatch, _NativePipeline, _NativeReader, _native_module,
)
from galp.torch.tests.test_experimental_direct_dct_pls_api import (
    _NativePipeline as _PlsPipeline,
)


class _ProjectedNativeBatch(_NativeBatch):
    @property
    def projected(self):
        raise AssertionError("the facade must not wait for the full preparation batch")

    def projected_range(self, first, count):
        return self.global_image_ids[first:first + count]


class _ProjectedPipeline(_NativePipeline):
    def __iter__(self):
        return self

    def __next__(self):
        batch = super().__next__()
        return _ProjectedNativeBatch(batch.global_image_ids)


class _ConfiguredReader(_NativeReader):
    def pipeline_batch_options(self, **options):
        pipeline = _ProjectedPipeline(self, "configured", options["dct_coeffs"])
        pipeline.options = options
        return pipeline


def module():
    result = _native_module()
    result.DirectDctReader = _ConfiguredReader
    result.DirectDctPlsPipeline = _PlsPipeline
    return result


def profile(**kwargs):
    return DctModelProfile(
        output_grid_size=112,
        output_channels=((0, 0), (1, 1), (2, 8)),
        normalization=((-88.0, 260.0), (0.1, 0.5), (-0.2, 0.7)),
        **kwargs,
    )


class PipelineFacadeTest(unittest.TestCase):
    def test_cnn_views_preserve_preparation_order_tail_and_lifetime(self):
        with DirectDctPipeline("manifest.bin", profile=profile(), batch_size=2, native_module=module()) as pipeline:
            transforms = [[{"horizontal_flip": True}] * 3, None]
            pipeline.reset([[7, 2, 7], [9]], transforms_by_batch=transforms)
            self.assertEqual(pipeline._native.batches, [[7, 2, 7], [9]])
            self.assertEqual(pipeline._native.transforms, transforms)
            first = next(pipeline)
            self.assertEqual(first.projected, [7, 2])
            self.assertEqual(first.projected_range(1, 1), [2])
            with self.assertRaises(IndexError):
                first.projected_range(1, 2)
            first.record_stream()
            self.assertEqual(first._native.record_stream_calls, [()])
            self.assertEqual([(b.sample_ids, b.projected) for b in pipeline], [([7], [7]), ([9], [9])])
            pipeline.reset([[3]])
            self.assertEqual(next(pipeline).projected, [3])
        # Returned views retain their native owner after pipeline close/reset.
        self.assertEqual(first.projected, [7, 2])

    def test_model_semantics_reach_existing_native_options(self):
        with DirectDctPipeline("manifest.bin", profile=profile(), native_module=module()) as pipeline:
            options = pipeline._native.options
            self.assertEqual(options["output_batch_images"], 64)
            grid = options["grid_transform"]
            self.assertEqual(grid["y_output_width_blocks"], 112)
            self.assertEqual(grid["output_channels"], [[0, 0, -88., 260.], [1, 1, .1, .5], [2, 8, -.2, .7]])
        with DirectDctPipeline(
            "manifest.bin", profile=profile(source_frequency_policy=(5, 0, 2)), native_module=module(),
        ) as pipeline:
            self.assertEqual(pipeline._native.options["dct_coeffs"], "list:5,0,2")
            self.assertFalse(pipeline._native.options["grid_transform"]["require_all_coefficients"])

    def test_registered_profile_keeps_native_batch_boundaries(self):
        with DirectDctPipeline("manifest.bin", profile=VALIDATION, native_module=module()) as pipeline:
            pipeline.reset([[7, 3], [4]])
            self.assertEqual([b.sample_ids for b in pipeline], [[7, 3], [4]])
            with self.assertRaisesRegex(ValueError, "reset"):
                pipeline.start_epoch(0)

    def test_training_policy_and_cnn_profile_are_forwarded_separately(self):
        policy = TrainingPolicy("mapping.csv", seed=17, expected_mapping_sha256="a" * 64, segments_per_pool=2)
        for selected in (TRAINING_PLS, profile()):
            with self.subTest(profile=selected), DirectDctPipeline(
                "manifest.bin", profile=selected, training=policy, batch_size=2, native_module=module(),
            ) as pipeline:
                pipeline.start_epoch(3)
                native = pipeline._native._native
                self.assertEqual(native.epoch, 3)
                self.assertEqual(native.arguments[:4], (
                    str(Path("manifest.bin").resolve()), str(Path("mapping.csv").resolve()), 17, "a" * 64,
                ))
                options = native.arguments[4]
                self.assertEqual(options["segments_per_pool"], 2)
                self.assertEqual(options["microbatch_images"], 2)
                self.assertEqual(options["profile_id"], TRAINING_PLS.id)
                if isinstance(selected, DctModelProfile):
                    self.assertEqual(options["output_grid_size"], 112)
                    self.assertEqual(options["output_channels"], selected._native_channels())
                batch = next(pipeline)
                self.assertEqual(batch.sample_ids, [9, 3])
                self.assertEqual(batch.targets, "pls-targets")
                with self.assertRaisesRegex(ValueError, "start_epoch"):
                    pipeline.reset([[0]])
            self.assertTrue(native.closed)

    def test_unsupported_training_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "all source frequencies"):
            DirectDctPipeline(
                "manifest.bin", profile=profile(source_frequency_policy=(0, 1)),
                training=TrainingPolicy("mapping.csv", 0, ""), native_module=module(),
            )

    def test_profile_rejects_malformed_model_inputs(self):
        for args in (
            dict(output_grid_size=0, output_channels=((0, 0),)),
            dict(output_grid_size=56, output_channels=((0, 64),)),
            dict(output_grid_size=56, output_channels=((0, 0), (0, 0))),
            dict(output_grid_size=56, output_channels=((0, 0),), normalization=((0., 0.),)),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                DctModelProfile(**args)


if __name__ == "__main__":
    unittest.main()
