import functools
import json
import multiprocessing
import os
from pathlib import Path
import unittest

import torch

from fastlens.benchmarks.baselines.dct_baselines.compact_training import benchmark
from fastlens.benchmarks.baselines.dct_baselines.prefetched_compact import PrefetchedCompactDataset
from .test_baselines import TemporaryTest


class ReaderFixture:
    def __init__(self, root, count=10, fail_at=None):
        self.root, self.image_count, self.fail_at = Path(root), count, fail_at
        self.input_shape = (3, 2, 2)
        self.manifest = dict(split="train", representation="prefetch_test")

    def read(self, indices):
        with (self.root/f"reads-{os.getpid()}.jsonl").open("a") as log:
            log.write(json.dumps(indices)+"\n")
        if self.fail_at is not None and self.fail_at in indices:
            raise RuntimeError("fixture read failed")
        values = torch.tensor(indices, dtype=torch.float32)[:, None, None, None].expand(-1, 3, 2, 2).clone()/10
        labels = torch.tensor(indices, dtype=torch.long) % 2
        return values, labels, dict(file_opens=1, requested_q_values=len(indices)*12)

    def close(self):
        pass


def tiny_model():
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))


class PrefetchedCompactTests(TemporaryTest):
    def test_spawn_workers_preserve_tail_order_rng_and_reuse_without_cross_pass_reads(self):
        parent = ReaderFixture(self.root)
        factory = functools.partial(ReaderFixture, self.root)
        before = torch.get_rng_state().clone()
        with PrefetchedCompactDataset(parent, factory, batch_size=2, pool_size=4, workers=2, pin_memory=False) as dataset:
            self.assertFalse(list(self.root.glob("reads-*.jsonl")))
            pids = None
            for epoch in range(2):
                for indices in (list(range(4)), list(range(4, 8)), [8, 9]):
                    values, labels, counters = dataset.read(indices)
                    torch.testing.assert_close(values[:, 0, 0, 0], torch.tensor(indices)/10, atol=0, rtol=0)
                    self.assertEqual(labels.tolist(), [i % 2 for i in indices])
                    self.assertEqual(counters["requested_q_values"], len(indices)*12)
                paths = list(self.root.glob("reads-*.jsonl"))
                current = {int(path.stem.split("-")[1]) for path in paths}
                self.assertEqual(len(current), 2)
                self.assertNotIn(os.getpid(), current)
                if pids is None:
                    pids = current
                self.assertEqual(current, pids)
                calls = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
                self.assertEqual(sorted(calls), sorted([list(range(4)), list(range(4, 8)), [8, 9]]*(epoch+1)))
            torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
        self.assertFalse(pids & {child.pid for child in multiprocessing.active_children()})

    def test_prefetch_keeps_real_training_updates_and_singleton_merge(self):
        parent = ReaderFixture(self.root, count=9)
        factory = functools.partial(ReaderFixture, self.root, count=9)
        options = dict(device="cpu", batch_size=2, pool_size=4, accumulation=2,
                       warmup_seconds=0, epochs=2, repeats=1, model_factory=tiny_model)
        expected = benchmark(parent, **options)
        with PrefetchedCompactDataset(parent, factory, batch_size=2, pool_size=4, workers=2, pin_memory=False) as dataset:
            actual = benchmark(dataset, **options)
        self.assertEqual(expected["initial_model_sha256"], actual["initial_model_sha256"])
        self.assertEqual(expected["repeats"][0]["final_model_sha256"], actual["repeats"][0]["final_model_sha256"])
        self.assertEqual(actual["repeats"][0]["samples"], 18)
        self.assertEqual(expected["microbatch_sample_histogram_per_pass"], actual["microbatch_sample_histogram_per_pass"])
        self.assertEqual(expected["repeats"][0]["reader_counters"], actual["repeats"][0]["reader_counters"])

    def test_wrong_schedule_and_worker_failure_propagate_and_workers_close(self):
        parent = ReaderFixture(self.root, fail_at=4)
        factory = functools.partial(ReaderFixture, self.root, fail_at=4)
        with PrefetchedCompactDataset(parent, factory, batch_size=2, pool_size=4, workers=2, pin_memory=False) as dataset:
            with self.assertRaisesRegex(ValueError, "frozen preparation-pool order"):
                dataset.read([4, 5, 6, 7])
            dataset.read([0, 1, 2, 3])
            with self.assertRaisesRegex(RuntimeError, "fixture read failed"):
                dataset.read([4, 5, 6, 7])
            pids = {int(path.stem.split("-")[1]) for path in self.root.glob("reads-*.jsonl")}
        self.assertFalse(pids & {child.pid for child in multiprocessing.active_children()})
