"""Integration between the converter and the PyTorch Direct-DCT readers."""

import contextlib
import io
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
            self.skipTest("CUDA device/driver unavailable")
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="fastlens-torch-store-") as directory:
            root = Path(directory)
            source = root / "source images"
            source.mkdir()
            paths = [source / name for name in ("z space.jpg", "A.JPEG", "middle.jpe")]
            for path in paths:
                Image.new("RGB", (512, 512), (70, 100, 140)).save(path, format="JPEG", subsampling=2)
            listing = root / "ordered.txt"
            listing.write_text("\n".join(str(path) for path in paths) + "\n")
            output = root / "store"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main([
                    "convert", "--input-list", str(listing), "--output", str(output),
                    "--partition-images", "2", "--workers", "2",
                ]), 0)

            reader = DirectDctReader(output / "manifest.bin", native_module=native)
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
                        self.assertEqual(list(batch.sample_ids), expected)
                        self.assertTrue(batch.y.is_cuda)
                        self.assertTrue(batch.cbcr.is_cuda)
                        self.assertEqual(batch.y.shape[0], len(expected))
            with DirectDctPipeline(output / "manifest.bin", profile=VALIDATION, native_module=native) as pipeline:
                batches = list(pipeline.start([[1, 0], [2]]))
                self.assertEqual([list(batch.sample_ids) for batch in batches], [[1, 0], [2]])
                self.assertTrue(all(batch.y.is_cuda for batch in batches))
            torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
