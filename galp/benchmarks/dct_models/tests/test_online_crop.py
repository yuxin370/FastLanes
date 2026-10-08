"""Online geometry checks and a small native source512 rounding regression."""
import json
import os
import sys
import unittest
from pathlib import Path

import torch

from galp.benchmarks.dct_models.online_crop import CROP, SourceReference, read_source_jpeg

from galp.benchmarks.common import DEFAULT_RGBNOMORE_ROOT


class OnlineCropTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.upstream = DEFAULT_RGBNOMORE_ROOT

    def inputs(self):
        return ([torch.zeros(h, h, 64, dtype=torch.int16) for h in (64, 32, 32)],
                [torch.ones(64, dtype=torch.int16) for _ in range(3)])

    def test_constant_survives_crop_dequantization_and_resize(self):
        q, tables = self.inputs()
        for plane, table in zip(q, tables):
            plane[..., 0] = 80
            table.fill_(2)
        for grid in (28, 56, 112):
            for actual in SourceReference(self.upstream, grid)(q, tables):
                expected = torch.zeros(grid, grid, 64)
                expected[..., 0] = 160
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)

    def test_crop_discards_border_without_resizing_it_into_output(self):
        q, tables = self.inputs()
        for plane, border in zip(q, (4, 2, 2)):
            plane[:border] = 100
            plane[-border:] = 100
            plane[:, :border] = 100
            plane[:, -border:] = 100
        for grid in (28, 56, 112):
            for actual in SourceReference(self.upstream, grid)(q, tables):
                self.assertEqual(int(torch.count_nonzero(actual)), 0)

    def test_discarded_source_frequency_contributes_to_output_dc(self):
        q, tables = self.inputs()
        # Chroma (0,7) is absent from DCT24's four output chroma channels.
        # It nevertheless contributes to output DC after splitting a block.
        q[1][16, 16, 7] = 100
        for grid in (56, 112):
            actual = SourceReference(self.upstream, grid)(q, tables)[1]
            self.assertGreater(float(actual[..., 0].abs().max()), 1.)

    def test_rejects_precomputed_target_as_full_source(self):
        q, tables = self.inputs()
        q[0] = q[0][4:-4, 4:-4]
        with self.assertRaisesRegex(ValueError, "full source component"):
            SourceReference(self.upstream, 56)(q, tables)

    def test_native_upscale_matches_cpu_at_rounding_boundaries(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device is not available")
        repo = Path(__file__).resolve().parents[4]
        root = repo / "galp/data/compressed/imagenet512_val_block_major"
        if not (root / "manifest.bin").exists():
            self.skipTest("source512 ImageNet fixture is not available")
        torch.set_num_threads(1)  # evaluate route J's worker setting
        sys.path.insert(0, str(repo / "build/galp/torch"))
        import _galp_direct_dct as native
        from unittest.mock import patch
        from galp.benchmarks.dct_models.evaluate_shards import apply_b6_runtime
        from galp.benchmarks.dct_models.online_crop import source_options

        entries = json.loads((root / "samples.json").read_text())
        # Three prefix regressions and the sample whose Top-1 changed. Read one
        # image at a time, without activating a whole shard or loading a model.
        reference = SourceReference(self.upstream, 112)
        with patch.dict(os.environ, GALP_BLOCK_MAJOR_ACCESS_DIR=str(root / "access")):
            reader = native.DirectDctReader(str(root / "manifest.bin"))
            for image_id in (0, 1, 2, 22858):
                with self.subTest(image_id=image_id):
                    position = next(i for i, s in enumerate(entries) if s["galp_image_id"] == image_id)
                    q, tables = read_source_jpeg(entries[position]["path"])
                    expected = torch.stack(reference(q, tables)).round().clamp(-32768, 32767)
                    options = apply_b6_runtime(source_options(dict(
                        layout="transformed_dct_grid", enable_planless_execution=True,
                        cache_capacity_mib=0, plan_cache_capacity=0,
                        grid_transform=dict(y_output_width_blocks=112, y_output_height_blocks=112,
                                            cbcr_output_width_blocks=112, cbcr_output_height_blocks=112,
                                            clamp_min=-32768, clamp_max=32767, output_dtype="float32",
                                            dequantize=True)), "vector-range-read-selected-decode"))
                    batch = reader.read_prefetched(reader.prefetch_batch(
                        [position], transforms=[dict(crop=CROP, horizontal_flip=False)], **options))
                    actual = torch.cat((batch.y[0], batch.cbcr[0])).flatten(-2).cpu()
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    del batch
                    # Exercise direct projection independently of grid materialization.
                    # Interleave components and select multiple frequencies so
                    # projected stores exercise both phase and channel ordering.
                    options["grid_transform"]["output_channels"] = [
                        [0, 8, .5, 4.], [1, 0, -1., 2.], [0, 63, 2., 8.], [2, 17, -2., 4.]]
                    batch = reader.read_prefetched(reader.prefetch_batch(
                        [position], transforms=[dict(crop=CROP, horizontal_flip=False)], **options))
                    projected = torch.stack(((expected[0, ..., 8] - .5) / 4.,
                                             (expected[1, ..., 0] + 1.) / 2.,
                                             (expected[0, ..., 63] - 2.) / 8.,
                                             (expected[2, ..., 17] + 2.) / 4.))
                    torch.testing.assert_close(batch.projected[0].cpu(), projected, atol=0, rtol=0)
                    del batch

    def test_bounded_pipeline_delivers_projected_ranges_on_consumer_stream(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device is not available")
        repo = Path(__file__).resolve().parents[4]
        root = repo / "galp/data/compressed/imagenet512_val_block_major"
        if not (root / "manifest.bin").exists():
            self.skipTest("source512 ImageNet fixture is not available")
        sys.path.insert(0, str(repo / "build/galp/torch"))
        import _galp_direct_dct as native
        from contextlib import closing
        from unittest.mock import patch
        from galp.benchmarks.dct_models.evaluate_shards import apply_b6_runtime, native_options
        from galp.benchmarks.dct_models.online_crop import source_options

        options = apply_b6_runtime(source_options(native_options(False, True), "vector-range-read-selected-decode"))
        # Nonconsecutive/duplicate images and a tail exercise delivery ownership.
        ids = [2, 0, 2]
        transforms = [dict(crop=CROP, horizontal_flip=(i == 1)) for i in range(len(ids))]
        with patch.dict(os.environ, GALP_BLOCK_MAJOR_ACCESS_DIR=str(root / "access")):
            reader = native.DirectDctReader(str(root / "manifest.bin"))
            reference = reader.read_prefetched(reader.prefetch_batch(ids, transforms=transforms, **options))
            expected = reference.projected.cpu()
            del reference
            with closing(reader.pipeline_batch_options(**options, output_batch_images=2)) as pipeline:
                pipeline.reset([ids] * 3, transforms_by_batch=[transforms] * 3)
                stream = torch.cuda.Stream()
                for _ in range(3):
                    batch = next(pipeline)
                    with torch.cuda.stream(stream):
                        first = batch.projected_range(0, 2).clone()
                        tail = batch.projected_range(2, 1).clone()
                    stream.synchronize()
                    torch.testing.assert_close(torch.cat((first, tail)).cpu(), expected, atol=0, rtol=0)
                    with self.assertRaises(IndexError):
                        batch.projected_range(2, 2)
                    del batch, first, tail
                self.assertLessEqual(pipeline._native_state_for_test["peak_live_output_slots"], 2)

    def test_multi_shard_activation_preserves_noncontiguous_ids_and_tail(self):
        from galp.benchmarks.dct_models.evaluate_shards import activation_groups
        shards = [dict(shard_id=i, first_global_image_index=i*1024,
                       image_count=1024, rowgroup_count=48) for i in (0, 3, 4, 9, 11)]
        groups = activation_groups(shards, 4)
        self.assertEqual(groups[0]["shard_ids"], [0, 3, 4, 9])
        self.assertEqual(groups[0]["image_ids"][1024], 3072)
        self.assertEqual(groups[0]["image_count"], 4096)
        self.assertEqual(groups[1]["image_ids"], list(range(11264, 12288)))
        self.assertEqual(sum(g["rowgroup_count"] for g in groups), 240)


if __name__ == "__main__":
    unittest.main()
