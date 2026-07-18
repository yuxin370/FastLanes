#!/usr/bin/env python3
"""CPU-only unit tests for benchmark manifests and summary helpers."""

from __future__ import annotations

import csv
import json
import os
import struct
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks/system_rgbnomore"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from common import (  # noqa: E402
    cached_file_fingerprints,
    distribution,
    galp_manifest_payloads,
    load_sample_manifest,
    sample_trace,
    verify_file_fingerprint,
)
from manifest import build_manifest, collect_dataset, validate_galp_label_map  # noqa: E402
from pipeline import GalpAdapter  # noqa: E402
from prepare_dataset import _collect_jpegs, _materialize_selected_data_root  # noqa: E402
from run import GALP_E2E_MIN_THROUGHPUT_IMAGES_PER_S, PRESETS  # noqa: E402
from validate import _evaluate_performance_gates, _semantic_compare  # noqa: E402


class SystemBenchmarkTest(unittest.TestCase):
    def test_selected_dataset_materialization_removes_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "imagenet"
            selected_root = root / "selected"
            paths = [
                source_root / "val/n00000001/a.JPEG",
                source_root / "val/n00000001/b.JPEG",
                source_root / "val/n00000002/c.JPEG",
            ]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"jpeg-{index}".encode("ascii"))

            initial = _materialize_selected_data_root(paths, selected_root)
            self.assertEqual(initial["selected_data_count"], 3)
            self.assertEqual(len(_collect_jpegs(selected_root)), 3)

            stale = selected_root / "val/n00000001/b.JPEG"
            self.assertTrue(stale.exists())
            paths[2].unlink()
            paths[2].write_bytes(b"replacement-c")
            updated = _materialize_selected_data_root([paths[0], paths[2]], selected_root)

            self.assertEqual(updated["selected_data_count"], 2)
            self.assertFalse(stale.exists())
            self.assertEqual(
                {path.relative_to(selected_root) for path in _collect_jpegs(selected_root)},
                {Path("val/n00000001/a.JPEG"), Path("val/n00000002/c.JPEG")},
            )
            self.assertTrue((selected_root / "val/n00000002/c.JPEG").samefile(paths[2]))

    def test_only_smoke_and_canonical_e2e_presets_exist(self) -> None:
        self.assertEqual(set(PRESETS), {"smoke", "e2e"})
        self.assertEqual(PRESETS["e2e"]["batch_size"], 64)
        self.assertEqual(PRESETS["e2e"]["repeats"], 5)
        self.assertGreater(PRESETS["e2e"]["measurement_batches"], PRESETS["smoke"]["measurement_batches"])
        self.assertEqual(GALP_E2E_MIN_THROUGHPUT_IMAGES_PER_S, 3000.0)

    def test_galp_e2e_throughput_gate_is_hard(self) -> None:
        contract = {
            "pipelines": {"enabled": ["galp"]},
            "performance_gates": {"galp": {"minimum_median_throughput_images_per_s": 2500.0}},
        }
        aggregates = [{"pipeline": "galp", "throughput_images_per_s": {"p50": 2499.0}}]
        failures: list[str] = []
        gates = _evaluate_performance_gates(contract, aggregates, failures)
        self.assertFalse(gates[0]["ok"])
        self.assertTrue(any("below required" in failure for failure in failures))

    def test_legacy_duplicate_benchmark_entrypoints_are_removed(self) -> None:
        galp_root = Path(__file__).resolve().parents[1]
        examples = galp_root / "examples"
        for filename in (
            "run_rgbnomore_comparison.py",
            "summarize_rgbnomore_benchmarks.py",
            "validate_rgbnomore_comparison.py",
            "rgbnomore_dali_rgb_baseline_benchmark.py",
            "rgbnomore_dct_baseline_benchmark.py",
            "rgbnomore_rgb_baseline_benchmark.py",
        ):
            self.assertFalse((examples / filename).exists(), filename)
        for filename in (
            "run_system_benchmark.py",
            "prepare_rgbnomore_direct_dct_manifest.py",
            "direct_dct_rgbnomore_benchmark.py",
            "validate_direct_dct_rgbnomore_pushdown.py",
            "scan_rgbnomore_manifests.py",
            "rgbnomore_dct_profile.py",
        ):
            self.assertFalse((examples / filename).exists(), filename)
        self.assertTrue((galp_root / "benchmarks/system_rgbnomore/run.py").is_file())
        self.assertTrue((galp_root / "benchmarks/system_rgbnomore/prepare_dataset.py").is_file())
        self.assertTrue((galp_root / "benchmarks/system_rgbnomore/diagnostics/direct_dct.py").is_file())
        self.assertTrue((galp_root / "benchmarks/system_rgbnomore/diagnostics/validate_pushdown.py").is_file())
        self.assertTrue((galp_root / "benchmarks/system_rgbnomore/diagnostics/scan_manifests.py").is_file())
        self.assertTrue((galp_root / "torch/rgbnomore_dct_profile.py").is_file())

    def test_distribution_and_trace_are_deterministic(self) -> None:
        self.assertEqual(distribution([1.0, 2.0, 3.0])["p50"], 2.0)
        rows = [
            {"ordinal": 0, "sample_id": "val/a.JPEG", "label": 3},
            {"ordinal": 1, "sample_id": "val/b.JPEG", "label": 4},
        ]
        self.assertEqual(sample_trace(rows), sample_trace(list(rows)))

    def test_manifest_fixes_labels_order_and_content_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "imagenet"
            paths = [
                data_root / "val/n00000001/a.JPEG",
                data_root / "val/n00000001/b.JPEG",
                data_root / "val/n00000002/c.JPEG",
            ]
            # Minimal JPEGs with a three-component 4:4:4 SOF marker. The
            # manifest parser needs headers only; decode is outside this test.
            jpeg_header = bytes.fromhex("ffd8ffc00011080001000103011100021100031100ffd9")
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(jpeg_header + bytes([index]))

            index_csv = root / "index.csv"
            with index_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("Filepath", "Label"))
                writer.writeheader()
                writer.writerows(
                    [
                        {"Filepath": "val/n00000002/c.JPEG", "Label": 20},
                        {"Filepath": "val/n00000001/b.JPEG", "Label": 10},
                        {"Filepath": "val/n00000001/a.JPEG", "Label": 10},
                    ]
                )
            label_map = root / "labels.json"
            label_map.write_text(
                json.dumps(
                    {
                        "format": "galp_rgbnomore_label_map_v1",
                        "image_count": 3,
                        "labels": [10, 10, 20],
                        "sample_ids": [
                            "val/n00000001/a.JPEG",
                            "val/n00000001/b.JPEG",
                            "val/n00000002/c.JPEG",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "manifest.json"
            payload, digest = build_manifest(
                data_root=data_root,
                split="val",
                index_csv=index_csv,
                galp_label_map_json=label_map,
                sample_count=3,
                seed=7,
                output=output,
            )
            loaded, samples = load_sample_manifest(output, digest)
            self.assertEqual(loaded["full_dataset_size"], 3)
            self.assertEqual({sample["galp_image_id"] for sample in samples}, {0, 1, 2})
            self.assertTrue(all(len(sample["sha256"]) == 64 for sample in samples))
            self.assertEqual({sample["label"] for sample in samples}, {10, 20})
            self.assertEqual(payload, loaded)

            selected_path = Path(samples[0]["path"])
            original_stat = selected_path.stat()
            content = selected_path.read_bytes()
            selected_path.write_bytes(content[:-1] + bytes([content[-1] ^ 0xFF]))
            os.utime(selected_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                load_sample_manifest(output, digest)

    def test_galp_identity_validation_rejects_same_label_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "imagenet"
            jpeg_header = bytes.fromhex("ffd8ffc00011080001000103011100021100031100ffd9")
            for name in ("a.JPEG", "b.JPEG"):
                path = data_root / "val/n00000001" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(jpeg_header)
            index_csv = root / "index.csv"
            with index_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("Filepath", "Label"))
                writer.writeheader()
                writer.writerows(
                    [
                        {"Filepath": "val/n00000001/a.JPEG", "Label": 10},
                        {"Filepath": "val/n00000001/b.JPEG", "Label": 10},
                    ]
                )
            entries = collect_dataset(data_root, "val", index_csv)
            label_map = root / "labels.json"
            label_map.write_text(
                json.dumps(
                    {
                        "format": "galp_rgbnomore_label_map_v1",
                        "image_count": 2,
                        "labels": [10, 10],
                        "sample_ids": ["val/n00000001/b.JPEG", "val/n00000001/a.JPEG"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sample identity/order mismatch"):
                validate_galp_label_map(label_map, entries)

    def test_galp_payload_fingerprints_cover_every_manifest_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fls = root / "shard.fls"
            metadata = root / "shard.meta.bin"
            fls.write_bytes(b"fls-payload")
            metadata.write_bytes(b"metadata-payload")

            def encoded_string(value: str) -> bytes:
                raw = value.encode("utf-8")
                return struct.pack("<I", len(raw)) + raw

            manifest = root / "manifest.bin"
            manifest.write_bytes(
                b"GJDCTSH1"
                + struct.pack("<IHIIQI", 1, 2, 128, 256, 1, 1)
                + struct.pack("<IQIQQQIIQQ", 0, 0, 1, 1, 0, 1, 1, 1, fls.stat().st_size, metadata.stat().st_size)
                + encoded_string(fls.name)
                + encoded_string(metadata.name)
            )
            payloads = galp_manifest_payloads(manifest)
            self.assertEqual({item["path"] for item in payloads}, {fls.resolve(), metadata.resolve()})
            cache = root / "fingerprints.json"
            fingerprints = cached_file_fingerprints(
                payloads, cache, cache_format="galp_shard_payload_fingerprints_v1"
            )
            self.assertEqual(len(fingerprints), 2)
            for fingerprint in fingerprints:
                verify_file_fingerprint(Path(fingerprint["path"]), fingerprint, "payload")

    def test_payload_hashing_requires_explicit_refresh_and_cached_reads_do_not_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "shard.fls"
            payload.write_bytes(b"large-payload-placeholder")
            files = [
                {
                    "kind": "fls",
                    "relative_path": payload.name,
                    "path": payload,
                    "expected_size": payload.stat().st_size,
                }
            ]
            cache = root / "fingerprints.json"
            kwargs = {"cache_format": "galp_shard_payload_fingerprints_v1"}

            with self.assertRaisesRegex(ValueError, "missing or stale"):
                cached_file_fingerprints(files, cache, allow_hash_misses=False, **kwargs)

            cached_file_fingerprints(files, cache, allow_hash_misses=True, **kwargs)
            with mock.patch("common.fingerprint_file", side_effect=AssertionError("unexpected rehash")):
                fingerprints = cached_file_fingerprints(files, cache, allow_hash_misses=False, **kwargs)
            self.assertEqual(len(fingerprints), 1)

    def test_semantic_gate_requires_exact_top1_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left.npz"
            right = root / "right.npz"
            identity = {
                "ordinals": np.asarray([0], dtype=np.int64),
                "labels": np.asarray([7], dtype=np.int64),
                "input_0": np.zeros((1, 1), dtype=np.float32),
            }
            np.savez(left, **identity, logits=np.asarray([[1.0, 0.999]], dtype=np.float32))
            np.savez(right, **identity, logits=np.asarray([[0.999, 1.0]], dtype=np.float32))
            failures: list[str] = []
            result = _semantic_compare(
                "galp",
                "rgbnomore",
                left,
                right,
                {
                    "input_max_abs": 0.001,
                    "input_mean_abs": 0.0001,
                    "logit_max_abs": 0.25,
                    "logit_cosine_min": 0.999,
                    "logit_top1_agreement_min": 1.0,
                },
                "strict",
                failures,
            )
            self.assertGreater(result["logits"]["cosine_mean"], 0.999)
            self.assertEqual(result["logits"]["top1_agreement"], 0.0)
            self.assertFalse(result["logits"]["within_tolerance"])
            self.assertTrue(any("logits exceed tolerance" in failure for failure in failures))

    def test_galp_adapter_prefetches_two_batches_ahead_in_order(self) -> None:
        prefetch_calls: list[list[int]] = []

        class Pending:
            def __init__(self, image_ids: list[int]) -> None:
                self.image_ids = image_ids

        class SourceBatch:
            execution_stats = {
                "fixed_transform_item_count": 1,
                "projection_item_count": 0,
                "decoded_projection_item_count": 0,
                "project_decoded_ycbcr_grid_launch_count": 0,
            }

        class Module:
            @staticmethod
            def _prefetch_pushdown_batch(reader, args, image_ids):
                del reader, args
                prefetch_calls.append(list(image_ids))
                return Pending(list(image_ids))

            @staticmethod
            def _adapt_prefetched_pushdown_batch(reader, args, image_ids, pending):
                del reader, args
                self.assertEqual(pending.image_ids, image_ids)
                count = len(image_ids)
                return torch.zeros((count, 1)), torch.zeros((count, 2)), [SourceBatch()]

            @staticmethod
            def _empty_totals():
                return {"fixed_transform_items": 0, "projection_items": 0}

            @staticmethod
            def _accumulate_many_stats(totals, batches):
                totals["fixed_transform_items"] += len(batches)

        adapter = object.__new__(GalpAdapter)
        adapter.module = Module()
        adapter.reader = object()
        adapter.args = SimpleNamespace(preprocess="rgbnomore-val-pushdown")
        adapter.device = torch.device("cpu")
        adapter.transform = None
        adapter.batch_size = 2
        adapter.batch_prefetch_depth = 2
        adapter.pending_batches = deque()
        adapter.next_prefetch_batch_index = 0

        first = [
            {"galp_image_id": 10, "label": 3, "ordinal": 0},
            {"galp_image_id": 11, "label": 4, "ordinal": 1},
        ]
        second = [
            {"galp_image_id": 12, "label": 5, "ordinal": 2},
            {"galp_image_id": 13, "label": 6, "ordinal": 3},
        ]
        third = [
            {"galp_image_id": 14, "label": 7, "ordinal": 4},
            {"galp_image_id": 15, "label": 8, "ordinal": 5},
        ]
        adapter.samples = first + second + third
        adapter.total_batches = 3
        adapter.begin_repeat()
        first_batch = adapter.load(first, second)
        second_batch = adapter.load(second, third)
        third_batch = adapter.load(third, None)

        self.assertEqual(prefetch_calls, [[10, 11], [12, 13], [14, 15]])
        self.assertEqual(first_batch.ordinals, [0, 1])
        self.assertEqual(second_batch.ordinals, [2, 3])
        self.assertEqual(third_batch.ordinals, [4, 5])
        self.assertEqual(list(adapter.pending_batches), [])


if __name__ == "__main__":
    unittest.main()
