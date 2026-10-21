"""Behavioral checks for scan truncation, raw records and aggregate pacing."""
import io
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import time
import unittest

import numpy as np
import torch
from PIL import Image

from .prepare import convert
from .run import PacedJPEGInputs, ReadLimiter
from .storage import EOI, read_record, read_size, split_scans, write_record


def complete_read(limiter):
    limiter.finish(limiter.reserve(100_000))


class PCRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home() / "tmp", prefix="pcr-test-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_conversion_and_partial_record(self):
        images, source_pixels, originals = [], [], []
        jpegtran = shutil.which("jpegtran")
        self.assertIsNotNone(jpegtran, "Put jpegtran on PATH for conversion tests")
        for i, shape in enumerate(((64, 64, 3), (48, 72))):
            pixels = np.random.default_rng(i).integers(0, 256, shape, dtype=np.uint8)
            path = self.root / f"{i}.jpg"
            Image.fromarray(pixels).save(path)
            original = path.read_bytes()
            source, scans = convert({"path": str(path)}, jpegtran)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(source, original)
            originals.append(original)
            images.append(scans)
            with Image.open(path) as image:
                source_pixels.append(np.asarray(image.convert("RGB")))
        record = write_record(self.root / "test.pcr", images, [12, 51])
        levels = len(record["scan_lengths"])
        blobs, labels, size = read_record(self.root, record, levels)
        self.assertEqual(labels, [12, 51])
        self.assertEqual(size, (self.root / "test.pcr").stat().st_size)
        for expected, blob in zip(source_pixels, blobs):
            with Image.open(io.BytesIO(blob)) as image:
                np.testing.assert_array_equal(np.asarray(image.convert("RGB")), expected)
        partial, _, partial_size = read_record(self.root, record, 2)
        self.assertLess(partial_size, size)
        for blob in partial:
            with Image.open(io.BytesIO(blob)) as image:
                image.load()
        # Unread high scans may be physically absent: prefix reading still works.
        with (self.root / "test.pcr").open("r+b") as handle:
            handle.truncate(read_size(record, 2))
        self.assertEqual(read_record(self.root, record, 2)[0], partial)
        with self.assertRaisesRegex(ValueError, "Truncated"):
            read_record(self.root, record, levels)
        packed = write_record(self.root / "jpeg.pcr", [[blob[:-2]] for blob in originals], [12, 51])
        self.assertEqual(read_record(self.root, packed, 1)[0], originals)
        # An all-grayscale record has fewer scans than the dataset-wide budget.
        gray = write_record(self.root / "gray.pcr", [images[1]], [51])
        full_gray, _, _ = read_record(self.root, gray, 10)
        with Image.open(io.BytesIO(full_gray[0])) as image:
            np.testing.assert_array_equal(np.asarray(image.convert("RGB")), source_pixels[1])

    def test_marker_lengths_and_truncation(self):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (10, 20, 30)).save(buffer, format="JPEG", progressive=True)
        blob = buffer.getvalue()
        # An EOI-looking byte sequence inside an APP segment is not a boundary.
        blob = blob[:2] + b"\xff\xef\x00\x06\xff\xd9xx" + blob[2:]
        self.assertEqual(b"".join(split_scans(blob)) + EOI, blob)
        with self.assertRaises(ValueError):
            split_scans(blob[:-2])
        with self.assertRaises(ValueError):
            split_scans(b"not a JPEG")

    def test_bandwidth_is_shared_across_workers(self):
        limiter = ReadLimiter(1)
        workers = [multiprocessing.Process(target=complete_read, args=(limiter,)) for _ in range(3)]
        started = time.monotonic()
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
            self.assertEqual(worker.exitcode, 0)
        self.assertGreaterEqual(time.monotonic() - started, .29)

    def test_paced_jpeg_matches_deployment_reader(self):
        from fastlens.benchmarks.dct_models import rgb

        path = self.root / "sample.jpg"
        Image.fromarray(np.random.default_rng(7).integers(0, 256, (48, 72, 3), dtype=np.uint8)).save(path)
        dataset = rgb.Inputs([dict(path=str(path), model_label=7, ordinal=0)])
        expected = dataset[0]
        actual = PacedJPEGInputs(dataset, ReadLimiter(1000), [path.stat().st_size])[0]
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        self.assertEqual(actual[1:3], expected[1:3])
        self.assertEqual(actual[-1], expected[-1])


if __name__ == "__main__":
    unittest.main()
