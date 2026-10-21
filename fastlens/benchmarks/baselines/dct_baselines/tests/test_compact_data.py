import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from fastlens.benchmarks.baselines.dct_baselines.compact_data import CompactDataset, prepare
from fastlens.benchmarks.baselines.dct_baselines.native import NativeSpec, encode_ic, fit_statistics, normalize_mosaic


class CompactDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path.home()/"tmp", prefix="compact-data-test-")
        self.root = Path(self.directory.name)
        rng = np.random.default_rng(702)
        rows = []
        self.rgb = []
        for index in range(5):
            path = self.root/f"train{index}.jpg"
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path, quality=80)
            with Image.open(path) as image:
                self.rgb.append(np.array(image.convert("RGB")))
            rows.append(dict(logical_sample_id=f"train/{index}", label=index, path=str(path)))
        self.selection = self.root/"selection.json"
        self.selection.write_text(json.dumps(dict(split="train", samples=rows)))

    def tearDown(self):
        self.directory.cleanup()

    def test_parallel_budgets_share_fs_and_match_scalar_mosaic_order_repeats_and_tail(self):
        manifests = prepare(self.selection, self.root/"prepared", image_size=64, ks=(3, 8), group_size=2, workers=2)
        shared_fs = set()
        for k, path in manifests.items():
            expected = [encode_ic(rgb, NativeSpec(k=k)) for rgb in self.rgb]
            statistics = fit_statistics(expected, split="train")
            metadata = json.loads(Path(path).read_text())
            self.assertEqual(metadata["normalization"], statistics)
            self.assertEqual(metadata["input_shape"], [3, 2*k, 2*k])
            shared_fs.add(metadata["stores"]["fs"]["root"])
            for method in ("ic_shared", "ic_batched", "fs"):
                with self.subTest(k=k, method=method), CompactDataset(path, method, pool_size=2, workers=2) as dataset:
                    indices = [4, 0, 2, 0]
                    actual, labels, counters = dataset.read(indices)
                    oracle = np.stack([normalize_mosaic(expected[index], statistics) for index in indices])
                    np.testing.assert_array_equal(actual.numpy(), oracle)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertEqual(labels.tolist(), indices)
                    self.assertEqual(counters["requested_q_values"], 4*3*4*k*(k+1)//2)
                    self.assertGreater(counters["application_read_bytes"], 0)
                    if method == "fs":
                        self.assertEqual(counters["file_opens"], 3*(k+1))
                        self.assertEqual(counters["decoded_q_values"], 5*3*4*k*(k+1)//2)
                    else:
                        self.assertEqual(counters["file_opens"], 3)
                    pools = list(dataset.iter_pools())
                    self.assertEqual([len(batch[1]) for batch in pools], [2, 2, 1])
                    self.assertEqual(torch.cat([batch[1] for batch in pools]).tolist(), list(range(5)))
                with CompactDataset(path, method, memory_budget_bytes=16) as constrained:
                    with self.assertRaisesRegex(MemoryError, "requested float32 outputs"):
                        constrained.read([0])
        self.assertEqual(len(shared_fs), 1)
        self.assertEqual(len(list((self.root/"prepared"/"fs_native").glob("*_d*.npz"))), 3*63)

    def test_reject_validation_population_and_implicit_resize(self):
        value = json.loads(self.selection.read_text())
        value["split"] = "val"
        self.selection.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "training split"):
            prepare(self.selection, self.root/"val", image_size=64, ks=(3,))
        value["split"] = "train"
        self.selection.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "resize explicitly"):
            prepare(self.selection, self.root/"resize", image_size=256, ks=(3,))


if __name__ == "__main__":
    unittest.main()
