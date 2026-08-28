#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "galp/benchmarks/system_rgbnomore"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from training.pipeline import GalpTrainingAdapter, TrainingSample  # noqa: E402
from training.pls_experiment import (  # noqa: E402
    augmentation_batches,
    build_execution_plan,
)


class PlsTrainingBridgeTest(unittest.TestCase):
    @staticmethod
    def _samples() -> list[TrainingSample]:
        return [
            TrainingSample(
                logical_sample_id=f"sample-{index}",
                path=Path(f"sample-{index}.jpg"),
                label=index // 4,
                width=512,
                height=512,
                galp_image_id=index,
            )
            for index in range(16)
        ]

    def test_closed_pool_batches_shared_crop_and_resume(self) -> None:
        train_samples = self._samples()
        plan = build_execution_plan(
            train_samples,
            seed=13,
            batch_count=4,
            batch_size=4,
            drop_last=True,
            start_cursor=None,
            organization="current",
            crop_policy="per-shard",
            order_policy="pls-wave",
            segment_images=4,
            segments_per_pool=2,
            distributed_rank=0,
            distributed_world_size=1,
        )
        self.assertEqual([len(batch) for batch in plan.batches], [4, 4, 4, 4])
        for pool, left, right in zip(
            plan.result.pools,
            plan.scheduled_batches[::2],
            plan.scheduled_batches[1::2],
        ):
            self.assertEqual(
                {value.segment_id for value in left + right},
                set(pool.segment_ids),
            )
        decisions = augmentation_batches(
            plan,
            {sample.logical_sample_id: sample for sample in train_samples},
            seed=13,
            domain="dct",
        )
        by_shard: dict[int, set[tuple[int, int, int, int]]] = {}
        for batch, batch_decisions in zip(plan.scheduled_batches, decisions):
            for value, decision in zip(batch, batch_decisions):
                by_shard.setdefault(value.source_shard_id, set()).add(
                    (
                        decision.crop_x,
                        decision.crop_y,
                        decision.crop_width,
                        decision.crop_height,
                    )
                )
        self.assertTrue(all(len(crops) == 1 for crops in by_shard.values()))

        resumed = build_execution_plan(
            train_samples,
            seed=13,
            batch_count=3,
            batch_size=4,
            drop_last=True,
            start_cursor=plan.batches[0][-1].as_dict(),
            organization="current",
            crop_policy="per-shard",
            order_policy="pls-wave",
            segment_images=4,
            segments_per_pool=2,
            distributed_rank=0,
            distributed_world_size=1,
        )
        self.assertEqual(resumed.batches, plan.batches[1:])

    def test_galp_adapter_materializes_one_complete_pool_at_a_time(self) -> None:
        class NativeBatch:
            def __init__(self, image_ids, transforms):
                self.global_image_ids = list(image_ids)
                self.transform_descriptors = [
                    {**transform, "global_image_id": image_id}
                    for image_id, transform in zip(image_ids, transforms)
                ]
                values = torch.arange(len(image_ids), dtype=torch.float32).reshape(-1, 1)
                self.tensors = (values, values.clone())

            def native_execution_stats(self):
                return {}

        class Handle:
            ready = True
            started = True

            def __init__(self, value):
                self.value = value

            def read(self):
                return self.value

        class Reader:
            def __init__(self):
                self.calls = []
                self.pending = []
                self.prefetched = 0

            def start(self, image_id_batches, *, transforms_by_batch):
                self.pending = []
                self.prefetched = len(image_id_batches)
                for image_ids, transforms in zip(image_id_batches, transforms_by_batch):
                    self.calls.append(list(image_ids))
                    self.pending.append(Handle(NativeBatch(image_ids, transforms)))

            def next_batch(self):
                if not self.pending:
                    raise StopIteration
                return self.pending.pop(0).read()

            def prefetched_batch_count(self):
                return self.prefetched

            def metrics(self):
                return {
                    "complete": True,
                    "peak_transient_bytes": 0,
                    "producer_ms": 0.0,
                    "planning_ms": 0.0,
                    "io_ms": 0.0,
                    "decode_ms": 0.0,
                    "transform_ms": 0.0,
                    "consumer_wait_ms": 0.0,
                    "logical_bytes": 0,
                    "physical_bytes": 0,
                }

            def close(self):
                self.pending = []

        train_samples = self._samples()
        plan = build_execution_plan(
            train_samples,
            seed=13,
            batch_count=4,
            batch_size=4,
            drop_last=True,
            start_cursor=None,
            organization="current",
            crop_policy="per-shard",
            order_policy="pls-wave",
            segment_images=4,
            segments_per_pool=2,
            distributed_rank=0,
            distributed_world_size=1,
        )
        decisions = augmentation_batches(
            plan,
            {sample.logical_sample_id: sample for sample in train_samples},
            seed=13,
            domain="dct",
        )
        reader = Reader()
        adapter = GalpTrainingAdapter(
            train_samples,
            batch_size=4,
            workers=1,
            device=torch.device("cpu"),
            config={
                "_direct_dct_training_reader": reader,
                "execution_mode": "runtime",
                "prefetch_depth": 3,
                "pls_gpu_pool": {
                    "enabled": True,
                    "closed_pool_batches": plan.selected_pool_batches(),
                },
            },
        )
        adapter.begin(
            [identity for batch in plan.batches for identity in batch],
            [decision for batch in decisions for decision in batch],
            [4, 4, 4, 4],
        )
        self.assertEqual([len(call) for call in reader.calls], [8])
        first = adapter.next_batch()
        second = adapter.next_batch()
        self.assertEqual(first.identities, plan.batches[0])
        self.assertEqual(second.identities, plan.batches[1])
        self.assertEqual([len(call) for call in reader.calls], [8])
        del first, second
        third = adapter.next_batch()
        self.assertEqual(third.identities, plan.batches[2])
        self.assertEqual([len(call) for call in reader.calls], [8, 8])
        metrics = adapter.loader_metrics()["physical_load_segment_gpu_pool"]
        self.assertEqual(metrics["materialized_pool_count"], 2)
        self.assertEqual(metrics["fully_emitted_release_eligible_pool_count"], 1)
        adapter.close()


if __name__ == "__main__":
    unittest.main()
