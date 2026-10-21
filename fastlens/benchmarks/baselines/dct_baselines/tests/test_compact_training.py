import json
import unittest

import numpy as np
from PIL import Image
from unittest.mock import patch

import torch

from fastlens.benchmarks.baselines.dct_baselines.compact_training import batch_plan, benchmark, _window
from .test_baselines import TemporaryTest
from fastlens.benchmarks.baselines.system_rgbnomore.training.model_factory import capture_rng_state, rng_state_sha256, seed_everything


class TensorDataset:
    def __init__(self, count=7, size=2):
        self.values = torch.linspace(-.7, .9, count*3*size*size).reshape(count, 3, size, size)
        self.targets = torch.arange(count) % 2
        self.image_count, self.input_shape = count, (3, size, size)
        self.manifest = {"split": "train", "fixture": True}
        self.requests = []

    def read(self, indices):
        self.requests.append(indices)
        return self.values[indices], self.targets[indices], {"application_read_bytes": len(indices)*12}


class CompactTrainingTests(unittest.TestCase):
    def test_weighted_accumulation_retains_tails_and_restores_initial_state_after_warmup(self):
        data = TensorDataset(9)
        created = []

        def factory():
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
            created.append(model)
            return model

        rng_before = rng_state_sha256(capture_rng_state())
        result = benchmark(data, device="cpu", batch_size=2, pool_size=5, accumulation=2,
                           warmup_seconds=1e-9, epochs=2, repeats=2, seed=91, learning_rate=.03,
                           momentum=.9, weight_decay=0., model_factory=factory)
        self.assertEqual(rng_state_sha256(capture_rng_state()), rng_before)
        self.assertEqual(result["microbatch_sample_histogram_per_pass"], {2: 3, 3: 1})
        self.assertEqual(result["optimizer_sample_histogram_per_pass"], {4: 1, 5: 1})
        a, b = result["repeats"]
        self.assertEqual(a["final_model_sha256"], b["final_model_sha256"])
        self.assertNotEqual(a["final_model_sha256"], result["initial_model_sha256"])
        self.assertEqual((a["samples"], a["microbatches"], a["optimizer_updates"], a["complete_passes"]), (18, 8, 4, 2))
        self.assertEqual(a["warmup"]["samples"], 9)
        self.assertEqual(a["reader_counters"]["application_read_bytes"], 18*12)
        self.assertEqual(data.requests, [list(range(5)), list(range(5, 9))]*6)

        # Independent full-window oracle: two microbatches must weight samples,
        # and the short final window must not be dropped or carried into a pass.
        seed_everything(91)
        oracle = factory()
        optimizer = torch.optim.SGD(oracle.parameters(), lr=.03, momentum=.9, weight_decay=0.)
        for _ in range(2):
            for start, stop in ((0, 5), (5, 9)):
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(oracle(data.values[start:stop]), data.targets[start:stop]).backward()
                optimizer.step()
        for actual, expected in zip(created[0].parameters(), oracle.parameters()):
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)

    def test_singleton_is_merged_without_switching_batchnorm_to_eval_or_dropping_images(self):
        created = []

        def factory():
            model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.BatchNorm2d(3),
                                        torch.nn.Flatten(), torch.nn.Linear(3, 2))
            created.append(model)
            return model

        for count, expected_batches in ((2, {2: 1}), (3, {3: 1}), (5, {2: 1, 3: 1})):
            result = benchmark(TensorDataset(count), device="cpu", batch_size=2, pool_size=4,
                               warmup_seconds=0., epochs=1, repeats=1, model_factory=factory)
            self.assertEqual(result["microbatch_sample_histogram_per_pass"], expected_batches)
            self.assertEqual(result["repeats"][0]["samples"], count)
            self.assertTrue(created[-1][1].training)
            self.assertEqual(int(created[-1][1].num_batches_tracked), sum(expected_batches.values()))
        with self.assertRaisesRegex(ValueError, ">=2 images"):
            batch_plan(1, 2, 4, 1)

    def test_time_window_contains_reads_and_updates_and_completes_full_passes(self):
        data = TensorDataset(2)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        clock = [0.]
        read, step = data.read, optimizer.step

        def measured_read(indices):
            clock[0] += .25
            return read(indices)

        def measured_step():
            clock[0] += .5
            return step()

        with patch.object(data, "read", measured_read), patch.object(optimizer, "step", measured_step), \
                patch("fastlens.benchmarks.baselines.dct_baselines.compact_training.time.perf_counter", lambda: clock[0]):
            result = _window(data, model, optimizer, batch_plan(2, 2, 2, 1), torch.device("cpu"), seconds=1.1, passes=None)
        self.assertEqual(result["wall_seconds"], 1.5)
        self.assertEqual(result["first_pass_seconds"], .75)
        self.assertEqual(result["complete_passes"], 2)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["optimizer_updates"], 2)

    def test_untrained_standard_resnet50_updates_on_actual_24_square_inputs(self):
        result = benchmark(TensorDataset(4, 24), device="cpu", batch_size=2, pool_size=4,
                           warmup_seconds=0., epochs=1, repeats=1, learning_rate=.001)
        self.assertEqual(result["model"], "torchvision.resnet50(weights=None)")
        self.assertEqual(result["input_shape"], [3, 24, 24])
        self.assertIsNone(result["checkpoint"])
        self.assertEqual(result["repeats"][0]["optimizer_updates"], 2)
        self.assertNotEqual(result["repeats"][0]["final_model_sha256"], result["initial_model_sha256"])
        self.assertFalse(result["headline_eligible"])


class CompactStoreTrainingTests(TemporaryTest):
    def test_real_jpeg_stores_reach_the_same_updates_from_the_same_random_state(self):
        from fastlens.benchmarks.baselines.dct_baselines.compact_data import prepare, CompactDataset
        rng = np.random.default_rng(7)
        samples = []
        for i in range(5):
            path = self.root/f"{i}.JPEG"
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
            samples.append(dict(logical_sample_id=f"train/{i}", path=str(path), label=i % 2))
        selection = self.root/"train.json"
        selection.write_text(json.dumps(dict(split="train", samples=samples)))
        manifest = prepare(selection, self.root/"stores", image_size=64, ks=(3,), group_size=2, workers=2)[3]
        results = []
        for method in ("ic_shared", "ic_batched", "fs"):
            with CompactDataset(manifest, method, workers=2) as dataset:
                result = benchmark(dataset, device="cpu", batch_size=2, pool_size=4, accumulation=2,
                                   warmup_seconds=0, epochs=2, repeats=1,
                                   model_factory=lambda: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(108, 2)))
            results.append(result)
        self.assertEqual(len({row["initial_model_sha256"] for row in results}), 1)
        self.assertEqual(len({row["repeats"][0]["final_model_sha256"] for row in results}), 1)
        self.assertEqual(len({row["dataset"]["data_identity"] for row in results}), 1)
        for row in results:
            self.assertEqual(row["repeats"][0]["samples"], 10)
            self.assertEqual(row["repeats"][0]["optimizer_updates"], 2)
            self.assertEqual(row["input_shape"], [3, 6, 6])
            self.assertGreater(row["repeats"][0]["reader_counters"]["application_read_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
