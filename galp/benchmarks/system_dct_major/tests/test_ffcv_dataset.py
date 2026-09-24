from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from PIL import Image

from galp.benchmarks.system_dct_major import ffcv_dataset
from galp.benchmarks.system_dct_major.ffcv_dataset import write_beton


class FfcvDatasetTest(unittest.TestCase):
    def test_data_root_cli_uses_manifest_image_count(self) -> None:
        argv = [
            "ffcv_dataset", "--data-root", "images", "--output", "images.beton",
            "--dct-major-label-map", "labels.json", "--dct-major-manifest", "manifest.bin",
            "--sample-count", "100", "--workers", "2",
        ]
        samples = [{"path": "image.jpg", "ordinal": 0}]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(ffcv_dataset, "parse_manifest", return_value={"image_count": 200}),
            mock.patch.object(
                ffcv_dataset, "collect_sequential_samples", autospec=True, return_value=(samples, {})
            ) as collect,
            mock.patch.object(ffcv_dataset, "write_beton") as write,
        ):
            ffcv_dataset.main()
        collect.assert_called_once_with(
            data_root=Path("images"), split="val", label_map_json=Path("labels.json"),
            expected_images=200, sample_count=100, hash_samples=False,
        )
        write.assert_called_once_with(samples, Path("images.beton"), 2)

    def test_raw_conversion_keeps_canonical_pixels_and_ordinals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            for ordinal, color in enumerate(((17, 34, 51), (68, 85, 102))):
                path = root / f"{ordinal}.png"
                Image.new("RGB", (512, 512), color).save(path)
                samples.append({"path": str(path), "ordinal": ordinal})

            fields = ModuleType("ffcv.fields")
            writer_module = ModuleType("ffcv.writer")
            fields.RGBImageField = mock.Mock(return_value="raw_image")
            fields.IntField = mock.Mock(return_value="ordinal")
            writer_module.DatasetWriter = mock.Mock()
            modules = {"ffcv": ModuleType("ffcv"), "ffcv.fields": fields, "ffcv.writer": writer_module}
            with mock.patch.dict(sys.modules, modules):
                write_beton(samples, root / "images.beton", workers=2)

            fields.RGBImageField.assert_called_once_with(write_mode="raw")
            dataset = writer_module.DatasetWriter.return_value.from_indexed_dataset.call_args.args[0]
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0][0].getpixel((0, 0)), (17, 34, 51))
            self.assertEqual(dataset[1][1], 1)


if __name__ == "__main__":
    unittest.main()
