#!/usr/bin/env python3

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "examples/image_order_benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from analysis import compare_semantic_artifacts, summarize_training_probe  # noqa: E402
from selection import build_condition_orders, parse_galp_layout  # noqa: E402
from training_order import analyze_training_index  # noqa: E402


def _write_manifest(path: Path, image_count: int = 32, rowgroup_vectors: int = 4) -> None:
    shards = ((0, 0, 16), (1, 16, 16))
    data = bytearray(b"GJDCTSH1")
    data.extend(struct.pack("<IHIIQ", 1, 2, rowgroup_vectors, 256, image_count))
    data.extend(struct.pack("<I", len(shards)))
    for shard_id, first, count in shards:
        fls = f"shard_{shard_id:06d}.fls".encode()
        metadata = f"shard_{shard_id:06d}.meta.bin".encode()
        data.extend(struct.pack("<IQIQQQII", shard_id, first, count, count, 0, count, 4, 1))
        data.extend(struct.pack("<QQ", 1, 1))
        data.extend(struct.pack("<I", len(fls)))
        data.extend(fls)
        data.extend(struct.pack("<I", len(metadata)))
        data.extend(metadata)
    path.write_bytes(data)


class ImageOrderBenchmarkTest(unittest.TestCase):
    def test_paired_selection_preserves_cohorts_and_aligns_rowgroups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.bin"
            _write_manifest(manifest)
            layout = parse_galp_layout(manifest)
            labels = [image_id // 4 for image_id in range(layout.image_count)]
            result = build_condition_orders(
                layout=layout,
                eligible_ids=list(range(layout.image_count)),
                labels=labels,
                batch_size=4,
                warmup_batches=1,
                measurement_batches=3,
                seed=7,
            )
        scattered = result["conditions"]["paired_scattered"]["flat_image_ids"]
        contiguous = result["conditions"]["contiguous"]["flat_image_ids"]
        self.assertEqual(set(scattered[:4]), set(contiguous[:4]))
        self.assertEqual(set(scattered[4:]), set(contiguous[4:]))
        self.assertEqual(result["conditions"]["contiguous"]["access"]["rowgroups_per_batch"]["mean"], 1.0)
        self.assertGreaterEqual(
            result["conditions"]["paired_scattered"]["access"]["rowgroups_per_batch"]["mean"],
            1.0,
        )
        self.assertEqual(result["training_probe"]["train_evaluation_overlap"], 0)

    def test_semantic_comparison_aligns_by_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left.npz"
            right = root / "right.npz"
            image_ids = np.asarray([9, 3, 7], dtype=np.int64)
            labels = np.asarray([1, 0, 2], dtype=np.int64)
            logits = np.asarray(
                [[0.0, 4.0, 1.0, -1.0, -2.0], [3.0, 0.0, -1.0, -2.0, -3.0], [0.0, 1.0, 5.0, -1.0, -2.0]],
                dtype=np.float32,
            )
            np.savez_compressed(left, image_ids=image_ids, labels=labels, logits=logits)
            order = np.asarray([2, 0, 1])
            np.savez_compressed(
                right,
                image_ids=image_ids[order],
                labels=labels[order],
                logits=logits[order],
            )
            result = compare_semantic_artifacts(left, right)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["logit_max_abs"], 0.0)
        self.assertEqual(result["top1_prediction_agreement"], 1.0)
        self.assertEqual(result["accuracy_top1_delta_percentage_points"], 0.0)

    def test_training_summary_is_paired_by_seed(self) -> None:
        runs = []
        for seed, scattered, contiguous in ((1, 0.70, 0.71), (2, 0.72, 0.73), (3, 0.71, 0.72)):
            for condition, accuracy in (("paired_scattered", scattered), ("contiguous", contiguous)):
                runs.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "final_evaluation": {
                            "accuracy_top1": accuracy,
                            "accuracy_top5": accuracy + 0.2,
                            "loss": 1.0 - accuracy,
                        },
                    }
                )
        result = summarize_training_probe(
            {
                "seed": 7,
                "runs": runs,
                "practical_significance_threshold_percentage_points": 0.5,
                "scope": "unit test",
            }
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["paired_seed_count"], 3)
        self.assertAlmostEqual(
            result["contiguous_minus_scattered_top1_percentage_points"]["mean"], 1.0
        )

    def test_training_index_exposes_path_sorted_label_clustering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_csv = Path(temporary) / "train.csv"
            index_csv.write_text(
                "Filepath,Label\n"
                "train/a/3.jpg,0\n"
                "train/b/2.jpg,1\n"
                "train/a/1.jpg,0\n"
                "train/b/4.jpg,1\n",
                encoding="utf-8",
            )
            result = analyze_training_index(index_csv, batch_size=2, seed=7)
        self.assertEqual(result["image_count"], 4)
        self.assertEqual(result["contiguous"]["single_class_batch_fraction"], 1.0)
        self.assertLessEqual(result["random"]["single_class_batch_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
