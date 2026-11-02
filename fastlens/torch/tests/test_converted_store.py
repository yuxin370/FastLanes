"""Integration between the converter and the PyTorch Direct-DCT readers."""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest

from fastlens import cli
from fastlens.profiles.rgbnomore import VALIDATION, VALIDATION_CENTER_CROP_512
from fastlens.torch import DirectDctPipeline, DirectDctReader


class ConvertedStoreTests(unittest.TestCase):
    def test_cuda_reader_consumes_store(self):
        import torch
        import _fastlens_direct_dct as native

        if not torch.cuda.is_available():
            if os.environ.get("FASTLENS_REQUIRE_CUDA_SMOKE") == "1":
                self.fail("CUDA device/driver unavailable but FASTLENS_REQUIRE_CUDA_SMOKE=1")
            self.skipTest("CUDA device/driver unavailable")
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="fastlens-torch-store-") as directory:
            root = Path(directory)
            source = root / "source images"
            source.mkdir()
            paths = [source / name for name in ("z space.jpg", "A.JPEG", "middle.jpe")]
            reference_y, reference_cbcr = [], []
            for image_id, (path, values) in enumerate(zip(
                paths, ((48, 96, 160), (128, 160, 96), (208, 64, 192)), strict=True
            )):
                Image.new("YCbCr", (512, 512), values).save(
                    path, format="JPEG", subsampling=2,
                    qtables=[[2 ** (image_id + 1)] * 64, [2 ** (image_id + 2)] * 64],
                )
                # Independent analytic JPEG-DCT reference: a constant component
                # has DC=8*(value-128), AC=0. Read the JPEG's own quantizers, not
                # the converted store. Values are exactly divisible, avoiding
                # encoder rounding differences. Constant planes are invariant
                # under the validation profiles' resize and center crop.
                components = []
                with Image.open(path) as jpeg:
                    for component, value in enumerate(values):
                        quantizer = jpeg.quantization[jpeg.layer[component][3]][0]
                        dc = 8 * (value - 128)
                        self.assertEqual(dc % quantizer, 0)
                        coefficients = torch.zeros((8, 8), dtype=torch.float32)
                        coefficients[0, 0] = (dc // quantizer) * quantizer
                        normalized = ((coefficients.clamp(-1024, 1016) + 1024) / 2040) * 2 - 1
                        grid = 28 if component == 0 else 14
                        components.append(normalized.expand(grid, grid, 8, 8))
                reference_y.append(components[0].unsqueeze(0))
                reference_cbcr.append(torch.stack(components[1:]))
            reference_y = torch.stack(reference_y)
            reference_cbcr = torch.stack(reference_cbcr)
            listing = root / "ordered.txt"
            listing.write_text("\n".join(str(path) for path in paths) + "\n")
            output = root / "store"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main([
                    "convert", "--input-list", str(listing), "--output", str(output),
                    "--partition-images", "2", "--workers", "2",
                ]), 0)

            reader = DirectDctReader(output / "manifest.bin", native_module=native)

            def check_batch(batch, expected_ids):
                self.assertEqual(list(batch.sample_ids), expected_ids)
                self.assertTrue(batch.y.is_cuda)
                self.assertTrue(batch.cbcr.is_cuda)
                torch.testing.assert_close(batch.y.cpu(), reference_y[expected_ids], rtol=0, atol=1e-6)
                torch.testing.assert_close(batch.cbcr.cpu(), reference_cbcr[expected_ids], rtol=0, atol=1e-6)

            # Center-crop requires complete shards in canonical order; the
            # general validation profile accepts arbitrary image IDs.
            for profile, schedule in (
                (VALIDATION, [[2, 0], [1]]),
                (VALIDATION_CENTER_CROP_512, [[0, 1], [2]]),
            ):
                with self.subTest(profile=profile.id), reader.iter_batches(
                    schedule, profile=profile, coefficients=None
                ) as batches:
                    for expected, batch in zip(schedule, batches, strict=True):
                        check_batch(batch, expected)
            with DirectDctPipeline(output / "manifest.bin", profile=VALIDATION, native_module=native) as pipeline:
                schedule = [[2, 0], [1]]  # The first batch crosses the 2-image shard boundary.
                for expected, batch in zip(schedule, pipeline.start(schedule), strict=True):
                    check_batch(batch, expected)
            torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
