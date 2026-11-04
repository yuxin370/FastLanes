"""Integration between the converter and the PyTorch Direct-DCT readers."""

import contextlib
import io
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fastlens import annotations, cli
from fastlens.profiles.rgbnomore import VALIDATION, VALIDATION_CENTER_CROP_512
from fastlens.torch import DirectDctPipeline, DirectDctReader


class ConvertedStoreTests(unittest.TestCase):
    def test_documented_reader_examples(self):
        import torch
        import _fastlens_direct_dct as native

        if not torch.cuda.is_available():
            if os.environ.get("FASTLENS_REQUIRE_CUDA_SMOKE") == "1":
                self.fail("CUDA device/driver unavailable but FASTLENS_REQUIRE_CUDA_SMOKE=1")
            self.skipTest("CUDA device/driver unavailable")
        from PIL import Image
        from fastlens.tests.test_documentation import introductory_examples

        with tempfile.TemporaryDirectory(prefix="fastlens-doc-examples-") as directory:
            root = Path(directory)
            source = root / "images"
            source.mkdir()
            for index in range(4):
                Image.new("RGB", (512, 512), (index * 64, 96, 160)).save(
                    source / f"{index}.jpg", subsampling=2,
                )
            output = root / "store"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main([
                    "convert", str(source), "--output", str(output),
                    "--partition-images", "4", "--workers", "2", "--verify", "full",
                ]), 0)
            for location, code in introductory_examples():
                # Run the documented code with fixture data and the built extension.
                # Four images in one partition catch scheduled-profile/subset misuse.
                for manifest in ("./store/manifest.bin", "fastlens/data/example-fastlens/manifest.bin"):
                    code = code.replace(f'"{manifest}"', repr(str(output / "manifest.bin")))
                for module_path in ("./build/fastlens/torch", "build/fastlens/torch"):
                    code = code.replace(f'"{module_path}"', repr(str(Path(native.__file__).parent)))
                with self.subTest(location=location), contextlib.redirect_stdout(io.StringIO()):
                    namespace = {}
                    exec(compile(code, location, "exec"), namespace)
                    batch = namespace["batch"]
                    self.assertTrue(batch.y.is_cuda)
                    self.assertTrue(batch.cbcr.is_cuda)
                    self.assertEqual(tuple(batch.y.shape), (2, 1, 28, 28, 8, 8))
                    self.assertEqual(tuple(batch.cbcr.shape), (2, 2, 14, 14, 8, 8))
            torch.cuda.synchronize()

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
            labels = root / "labels.csv"
            labels.write_text("Filepath,Label\n" + "".join(f"{path.name},{index}\n" for index, path in enumerate(paths)))
            output = root / "store"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main([
                    "convert", "--input-list", str(listing), "--output", str(output),
                    "--partition-images", "2", "--workers", "2",
                    "--image-root", str(source), "--labels-csv", str(labels), "--shuffle-seed", "42",
                ]), 0)
            physical_to_source = [paths.index(Path(path)) for path in (output / "conversion/sources.txt").read_text().splitlines()]
            self.assertNotEqual(physical_to_source, [0, 1, 2])
            shutil.rmtree(output / "conversion")
            samples, _ = annotations.load_annotations(output)

            reader = DirectDctReader(output / "manifest.bin", native_module=native)

            def check_batch(batch, expected_ids):
                self.assertEqual(list(batch.sample_ids), expected_ids)
                self.assertTrue(batch.y.is_cuda)
                self.assertTrue(batch.cbcr.is_cuda)
                source_ids = [physical_to_source[index] for index in expected_ids]
                torch.testing.assert_close(batch.y.cpu(), reference_y[source_ids], rtol=0, atol=1e-6)
                torch.testing.assert_close(batch.cbcr.cpu(), reference_cbcr[source_ids], rtol=0, atol=1e-6)
                self.assertEqual([samples[index]["label"] for index in batch.sample_ids], source_ids)

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
            # Exercise the documented classification example with real targets,
            # both no-grad evaluation and a CrossEntropy backward/optimizer step.
            for training in (False, True):
                result = subprocess.run([
                    sys.executable, "-B", str(cli.REPO_ROOT / "fastlens/examples/direct_dct_pipeline_demo.py"),
                    str(output / "manifest.bin"), "--module-path", str(Path(native.__file__).parent),
                    "--annotations", "--num-classes", "3", "--batch-size", "2", "--steps", "2", "--warmup", "0",
                    *(["--train-smoke"] if training else []),
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads(next(line.removeprefix("RESULT_JSON ") for line in result.stdout.splitlines()
                                         if line.startswith("RESULT_JSON ")))
                self.assertTrue(report["annotations"])
                self.assertEqual(report["train_smoke"], training)
                self.assertTrue(math.isfinite(report["final_loss"]))

            for name in ("index.bin", "000000.bin"):
                path = output / "geometry" / name
                original = path.read_bytes()
                for damaged in (None, original[:-1] + bytes([original[-1] ^ 1])):
                    with self.subTest(geometry=name, missing=damaged is None):
                        if damaged is None:
                            path.unlink()
                        else:
                            path.write_bytes(damaged)
                        with self.assertRaises(RuntimeError):
                            damaged_reader = DirectDctReader(output / "manifest.bin", native_module=native)
                            with damaged_reader.iter_batches([[0]], profile=VALIDATION) as batches:
                                next(batches)
                        path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
