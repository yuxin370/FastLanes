"""JPEG B8 compact adaptation reuses existing stores and real training updates."""
import copy
import json
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch

from fastlens.benchmarks.baselines.dct_baselines.b8_compact import B8CompactDataset, fit, make_contract
from fastlens.benchmarks.baselines.dct_baselines.compact_training import benchmark
from fastlens.benchmarks.baselines.dct_baselines.contracts import artifact
from fastlens.benchmarks.baselines.dct_baselines.materialize import Materializer
from fastlens.benchmarks.baselines.dct_baselines.source import freeze_source, source_images
from fastlens.benchmarks.baselines.dct_baselines.store import build_store
from fastlens.benchmarks.common import DEFAULT_RGBNOMORE_ROOT
from .test_baselines import TemporaryTest

UPSTREAM = str(DEFAULT_RGBNOMORE_ROOT)


class B8CompactTests(TemporaryTest):
    def test_source_mask_compact_layout_real_updates_and_train_only_normalization(self):
        rng = np.random.default_rng(6)
        samples = []
        for index in range(8):
            path = self.root/f"{index}.JPEG"
            Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)).save(path, quality=83, subsampling=2)
            samples.append(dict(path=str(path), logical_sample_id=f"train/{index}", label=index % 2))
        selection = self.root/"selection.json"
        selection.write_text(json.dumps(dict(split="train", samples=samples)))
        source_path = self.root/"source.json"
        freeze_source(selection, source_path, dict(resize="none", encoder="Pillow", quality=83, subsampling=2))
        source = json.loads(source_path.read_text())
        contract = fit(source_path, self.root/"d3.json", diagonals=3, upstream=UPSTREAM, pool_size=4, workers=2)
        self.assertEqual(contract["input_shape"], [3, 84, 84])
        self.assertEqual(make_contract(source_path, 15, UPSTREAM)["input_shape"], [3, 224, 224])
        for method in ("ic_matched_shared", "ic_matched_batched", "fs_matched_b8"):
            build_store(source_images(source), self.root/method, method, source_artifact=artifact(source_path), group_size=4)

        # Independent q-mask and spatial/block-frequency mosaic oracle. In
        # particular, source masking must happen before the spectral resize.
        image = next(source_images(source))
        q = tuple(value.copy() for value in image.coefficients)
        for value in q:
            for u in range(8):
                for v in range(8):
                    if u+v >= 3:
                        value[..., u, v] = 0
        full = make_contract(source_path, 15, UPSTREAM)["profile"]
        raw = Materializer(full)(q, image.tables, dict(crop_box=[32, 32, 448, 448], flip=False))[0]
        oracle = np.zeros((3, 84, 84), dtype=np.float32)
        for component in range(3):
            for u in range(3):
                for v in range(3):
                    if u+v < 3:
                        oracle[component, u::3, v::3] = raw[component*64+u*8+v].numpy()
        with B8CompactDataset(contract, "cpu_dct", batch_size=2, workers=2, normalize=False) as data:
            values, _, _ = data.read([0])
            np.testing.assert_array_equal(values[0].numpy(), oracle)
        reference = None
        final_states = []
        for name, backend in ((None, "cpu_dct"), ("ic_matched_shared", "ic"), ("ic_matched_batched", "ic"), ("fs_matched_b8", "fs")):
            with B8CompactDataset(contract, backend, None if name is None else self.root/name,
                                  batch_size=2, pool_size=4, workers=2) as data:
                if name is not None:
                    with patch("fastlens.benchmarks.baselines.dct_baselines.store.StoreReader._check_content", side_effect=AssertionError("repeated full hash")):
                        with B8CompactDataset(contract, backend, self.root/name, batch_size=2, pool_size=4, workers=2,
                                              content_evidence=data.producer.content_verification) as reused:
                            self.assertTrue(reused.producer.content_verification["reused"])
                            reused_values, _, _ = reused.read([7, 0, 2, 0])
                values, labels, counters = data.read([7, 0, 2, 0])
                if name is not None:
                    torch.testing.assert_close(values,reused_values,atol=0,rtol=0)
                self.assertFalse(values.is_inference())
                self.assertEqual(labels.tolist(), [1, 0, 0, 0])
                if reference is None:
                    reference = values
                    self.assertIsNone(counters)
                torch.testing.assert_close(values, reference, atol=0, rtol=0)
                result = benchmark(data, device="cpu", batch_size=2, pool_size=4, epochs=1,
                                   warmup_seconds=0, repeats=1,
                                   model_factory=lambda: torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2)))
                final_states.append(result["repeats"][0]["final_model_sha256"])
                self.assertEqual(result["repeats"][0]["optimizer_updates"], 4)
                if name is None:
                    self.assertIsNone(result["repeats"][0]["reader_counters"])
                else:
                    self.assertGreater(result["repeats"][0]["reader_counters"]["decoded_q_values"], 0)
        self.assertEqual(len(set(final_states)), 1)
        invalid = copy.deepcopy(source)
        invalid["images"][0]["split"] = "val"
        path = self.root/"val.json"
        path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, "training population"):
            make_contract(path, 3, UPSTREAM)
