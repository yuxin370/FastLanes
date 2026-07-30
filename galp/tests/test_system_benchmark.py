#!/usr/bin/env python3
"""CPU-only unit tests for benchmark manifests and summary helpers."""

from __future__ import annotations

import csv
import json
import os
import subprocess
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

from shared.common import (  # noqa: E402
    cached_file_fingerprints,
    distribution,
    galp_manifest_payloads,
    load_sample_manifest,
    sample_trace,
    sha256_json,
    source_tree_metadata,
    verify_file_fingerprint,
)
from diagnostics.audit_planless_storage_io import _counter_values  # noqa: E402
from diagnostics.direct_dct import (  # noqa: E402
    _make_benchmark_image_ids,
    _scale_to_rgbnomore_dct_range,
    adapt_galp_batch_to_rgbnomore,
)
from dataset.manifest import build_manifest, collect_dataset, validate_galp_label_map  # noqa: E402
from inference.pipeline import (  # noqa: E402
    GalpAdapter,
    GalpLegacyAdapter,
    _process_memory_snapshot,
    _resolve_model_stream_priority,
)
from dataset.prepare_dataset import _collect_jpegs, _materialize_selected_data_root  # noqa: E402
from inference.run import (  # noqa: E402
    E2E_MAX_HOT_THROUGHPUT_CV,
    E2E_PIPELINES,
    GALP_E2E_MIN_DALI_HOT_MEDIAN_RATIO,
    PRESETS,
    _parse_args as _parse_run_args,
    _source_revision_policy,
)
from diagnostics.scheduler_matrix import (  # noqa: E402
    _invariant_counter,
    _normalize_limited_candidates,
    _normalize_transform_blocks,
    _pareto_frontier,
    _policy_specs,
    _residency_bounds,
)
from inference.validate import (  # noqa: E402
    _aggregate_pipeline,
    _evaluate_performance_gates,
    _semantic_compare,
    _validate_crop_pushdown_accounting,
)
from inference.crop_io_ab import validate_crop_io_ab_results  # noqa: E402


class SystemBenchmarkTest(unittest.TestCase):
    def test_crop_io_ab_accepts_same_outputs_and_real_physical_byte_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_contract = {
                "pipelines": {
                    "galp": {
                        "manifest": "fixture",
                        "preprocess": "rgbnomore-val-pushdown",
                        "cache_capacity_mib": 0,
                        "plan_cache_capacity": 0,
                    }
                },
                "semantic_validation": {"prediction_agreement_sample_count": 50_000},
            }
            modes = {
                "full_decode": ("full-rowgroup-decode", 8, 100, "rowgroup", "rowgroup"),
                "crop_rowgroup": (
                    "rowgroup-read-selected-decode",
                    4,
                    100,
                    "rowgroup",
                    "selected-vector",
                ),
                "crop_vector": (
                    "vector-range-read-selected-decode",
                    4,
                    50,
                    "selected-vector-range",
                    "selected-vector",
                ),
            }
            results = {}
            for name, (execution_mode, actual_vectors, physical_bytes, storage, decode) in modes.items():
                artifact = root / f"{name}.npz"
                np.savez_compressed(
                    artifact,
                    input_0=np.arange(8, dtype=np.float32).reshape(2, 4),
                    input_1=np.arange(4, dtype=np.float32).reshape(2, 2),
                    logits=np.arange(12, dtype=np.float32).reshape(2, 6),
                    top1_predictions=np.arange(50_000, dtype=np.int64) % 1000,
                )
                results[name] = {
                    "pipeline": "galp",
                    "pipeline_config": {
                        **base_contract["pipelines"]["galp"],
                        "crop_execution_mode": execution_mode,
                    },
                    "semantic_artifact": str(artifact),
                    "repeats": [
                        {
                            "images": 4,
                            "seconds": 1.0,
                            "throughput_images_per_s": 4.0,
                            "correct_top1": 3,
                            "correct_top5": 4,
                            "native_counters": {
                                "planned_vector_count": 4,
                                "actual_vector_count": actual_vectors,
                                "full_vector_count": 8,
                                "compressed_payload_bytes_read": physical_bytes,
                                "full_compressed_payload_bytes": 100,
                                "pread_count": 3,
                                "vector_bundle_rowgroup_count": 0,
                                "vector_bundle_envelope_rowgroup_count": 0,
                                "vector_bundle_pread_count": 0,
                                "requested_source_block_count": 12,
                                "source_blocks_transformed": 12,
                                "rowgroups": 4,
                                "sparse_read_fallback_rowgroup_count": 0,
                            },
                            "native_properties": {
                                "storage_read_granularity": storage,
                                "decode_granularity": decode,
                                "read_amplification": physical_bytes / 100.0,
                                "sparse_read_supported": storage == "selected-vector-range",
                                "sparse_read_fallback_reason": "",
                            },
                            "stage_breakdown_ms": {
                                "native_totals_seconds": {
                                    "decode_seconds": 0.1,
                                    "fixed_transform_kernel_seconds": 0.2,
                                }
                            },
                        }
                    ],
                }
            summary = validate_crop_io_ab_results(base_contract, results)
            self.assertTrue(summary["ok"], summary["failures"])
            self.assertEqual(summary["top1_agreement"]["crop_vector"], 1.0)
            self.assertLess(
                summary["modes"]["crop_vector"]["compressed_payload_bytes_read"],
                summary["modes"]["crop_rowgroup"]["compressed_payload_bytes_read"],
            )

    def test_crop_accounting_rejects_full_vector_pushdown_claim(self) -> None:
        failures: list[str] = []
        _validate_crop_pushdown_accounting(
            {
                "planned_vector_count": 8,
                "actual_vector_count": 8,
                "full_vector_count": 8,
                "compressed_payload_bytes_read": 100,
                "full_compressed_payload_bytes": 100,
                "pread_count": 1,
                "source_blocks_transformed": 10,
            },
            {
                "storage_read_granularity": "rowgroup",
                "decode_granularity": "selected-vector",
                "read_amplification": 1.0,
            },
            "test",
            failures,
        )
        self.assertTrue(any("vector pushdown claimed" in failure for failure in failures))

    def test_crop_accounting_rejects_false_physical_io_reduction(self) -> None:
        failures: list[str] = []
        _validate_crop_pushdown_accounting(
            {
                "planned_vector_count": 4,
                "actual_vector_count": 4,
                "full_vector_count": 8,
                "compressed_payload_bytes_read": 100,
                "full_compressed_payload_bytes": 100,
                "pread_count": 12,
                "source_blocks_transformed": 10,
            },
            {
                "storage_read_granularity": "selected-vector-range",
                "decode_granularity": "selected-vector",
                "read_amplification": 1.0,
            },
            "test",
            failures,
        )
        self.assertTrue(any("lower I/O claimed" in failure for failure in failures))
        self.assertTrue(any("transform block reduction" in failure for failure in failures))

    def test_crop_accounting_accepts_honest_rowgroup_crop(self) -> None:
        failures: list[str] = []
        _validate_crop_pushdown_accounting(
            {
                "planned_vector_count": 8,
                "actual_vector_count": 8,
                "full_vector_count": 8,
                "compressed_payload_bytes_read": 100,
                "full_compressed_payload_bytes": 100,
                "pread_count": 1,
                "source_blocks_transformed": 10,
            },
            {
                "storage_read_granularity": "rowgroup",
                "decode_granularity": "rowgroup",
                "read_amplification": 1.0,
            },
            "test",
            failures,
        )
        self.assertEqual(failures, [])

    def test_measured_limited_overlap_is_the_production_default(self) -> None:
        with mock.patch.object(
            sys, "argv", ["run.py", "--output-dir", "/tmp/galp-default-contract-test"]
        ):
            args = _parse_run_args()
        self.assertEqual(args.galp_scheduling_policy, "limited-overlap")
        self.assertEqual(args.galp_transform_blocks_per_launch, 512)
        self.assertEqual(args.galp_transform_ctas_per_launch, 512)

    def test_model_stream_priority_resolves_framework_range(self) -> None:
        self.assertEqual(_resolve_model_stream_priority("greatest", (0, -3)), -3)
        self.assertEqual(_resolve_model_stream_priority("least", (0, -3)), 0)
        self.assertEqual(_resolve_model_stream_priority(-1, (0, -3)), -1)
        with self.assertRaisesRegex(ValueError, "invalid model stream priority"):
            _resolve_model_stream_priority("high", (0, -3))

    def test_scheduler_matrix_expands_limited_overlap_sweep(self) -> None:
        self.assertEqual(_normalize_transform_blocks([1024, 256, 1024]), [256, 1024])
        candidates = _normalize_limited_candidates(
            [1024, 256, 1024], [128, 64, 128]
        )
        self.assertEqual(candidates, [(256, 64), (1024, 128)])
        self.assertEqual(
            _policy_specs(candidates),
            [
                ("fully-overlapped", "fully-overlapped", 0, 0),
                ("limited-overlap-o256-c64", "limited-overlap", 256, 64),
                ("limited-overlap-o1024-c128", "limited-overlap", 1024, 128),
                ("serial", "serial", 0, 0),
            ],
        )
        self.assertEqual(
            _policy_specs(_normalize_limited_candidates(64, None))[1],
            ("limited-overlap", "limited-overlap", 64, 64),
        )
        with self.assertRaisesRegex(ValueError, "positive integers"):
            _normalize_transform_blocks([0])
        with self.assertRaisesRegex(ValueError, "equal counts"):
            _normalize_limited_candidates([256, 512], [64, 128, 256])

    def test_scheduler_matrix_computes_limited_pareto_frontier(self) -> None:
        summaries = {
            "a": {
                "throughput_images_per_s_median": 4500.0,
                "model_extra_p50_ms_vs_serial": 0.30,
            },
            "b": {
                "throughput_images_per_s_median": 4400.0,
                "model_extra_p50_ms_vs_serial": 0.20,
            },
            "dominated": {
                "throughput_images_per_s_median": 4300.0,
                "model_extra_p50_ms_vs_serial": 0.35,
            },
        }
        self.assertEqual(_pareto_frontier(summaries, ["a", "b", "dominated"]), ["a", "b"])

    def test_scheduler_matrix_computes_resource_limited_residency(self) -> None:
        one_cta_per_sm = _residency_bounds(
            submitted_ctas=128,
            sm_count=128,
            max_active_ctas_per_sm=10,
            threads_per_cta=64,
            max_threads_per_sm=1536,
        )
        self.assertEqual(one_cta_per_sm["max_resident_ctas_per_launch"], 128)
        self.assertEqual(one_cta_per_sm["average_resident_ctas_per_sm_upper_bound"], 1.0)
        self.assertEqual(one_cta_per_sm["sm_coverage_fraction_upper_bound"], 1.0)
        self.assertAlmostEqual(
            one_cta_per_sm["thread_occupancy_fraction_upper_bound"], 1.0 / 24.0
        )
        saturated = _residency_bounds(
            submitted_ctas=2048,
            sm_count=128,
            max_active_ctas_per_sm=10,
            threads_per_cta=64,
            max_threads_per_sm=1536,
        )
        self.assertEqual(saturated["max_resident_ctas_per_launch"], 1280)
        self.assertEqual(saturated["resident_cta_capacity_fraction_upper_bound"], 1.0)
        self.assertAlmostEqual(
            saturated["thread_occupancy_fraction_upper_bound"], 10.0 / 24.0
        )

    def test_scheduler_matrix_requires_invariant_native_priority_counters(self) -> None:
        repeats = [
            {"native_counters": {"direct_dct_stream_priority": 0}},
            {"native_counters": {"direct_dct_stream_priority": 0}},
        ]
        self.assertEqual(_invariant_counter(repeats, "direct_dct_stream_priority"), 0)
        repeats[1]["native_counters"]["direct_dct_stream_priority"] = -1
        with self.assertRaisesRegex(RuntimeError, "not invariant"):
            _invariant_counter(repeats, "direct_dct_stream_priority")

    def test_in_place_rgbnomore_range_scale_matches_reference(self) -> None:
        values = torch.arange(-1024, 1017, dtype=torch.float32)
        reference = (values + 1024.0) / 2040.0 * 2.0 - 1.0
        actual = _scale_to_rgbnomore_dct_range(values.clone())
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=torch.finfo(torch.float32).eps)
        self.assertEqual(float(actual[0]), -1.0)
        self.assertEqual(float(actual[-1]), 1.0)

    def test_native_float32_direct_dct_adapter_is_zero_copy(self) -> None:
        y = torch.randn((2, 1, 28, 28, 8, 8), dtype=torch.float32)
        cbcr = torch.randn((2, 2, 14, 14, 8, 8), dtype=torch.float32)
        batch = SimpleNamespace(
            layout="transformed_dct_grid",
            selected_coefficients=list(range(64)),
            y=y,
            cbcr=cbcr,
        )
        actual_y, actual_cbcr = adapt_galp_batch_to_rgbnomore(
            object(),
            batch,
            [0, 1],
            dequantize=True,
            scale=True,
            preprocess="rgbnomore-val-pushdown",
            rgbnomore_dct_val_transform=None,
        )
        self.assertIs(actual_y, y)
        self.assertIs(actual_cbcr, cbcr)
        with self.assertRaisesRegex(RuntimeError, "already dequantized"):
            adapt_galp_batch_to_rgbnomore(
                object(),
                batch,
                [0, 1],
                dequantize=True,
                scale=False,
                preprocess="rgbnomore-val-pushdown",
                rgbnomore_dct_val_transform=None,
            )

    def test_source_tree_cleanliness_is_scoped_to_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Benchmark Test"], check=True)
            runtime = root / "runtime.py"
            runtime.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "runtime.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)

            clean = source_tree_metadata(root, ["runtime.py"])
            self.assertTrue(clean["benchmark_source_clean"])
            self.assertFalse(clean["git_dirty"])

            (root / "unrelated.txt").write_text("user data\n", encoding="utf-8")
            unrelated = source_tree_metadata(root, ["runtime.py"])
            self.assertTrue(unrelated["benchmark_source_clean"])
            self.assertTrue(unrelated["git_dirty"])

            runtime.write_text("VALUE = 2\n", encoding="utf-8")
            dirty = source_tree_metadata(root, ["runtime.py"])
            self.assertFalse(dirty["benchmark_source_clean"])
            self.assertTrue(dirty["runtime_git_status"])

    def test_dirty_runtime_sources_are_recorded_as_warning_policy(self) -> None:
        policy = _source_revision_policy(
            {
                "fastlanes": {
                    "benchmark_source_clean": False,
                    "runtime_git_status": [" M runtime.py"],
                },
                "rgbnomore": {
                    "benchmark_source_clean": True,
                    "runtime_git_status": [],
                },
            }
        )
        self.assertEqual(policy["cleanliness_enforcement"], "warning")
        self.assertEqual(
            policy["dirty_sources_at_contract_creation"],
            {"fastlanes": [" M runtime.py"]},
        )
        self.assertTrue(policy["runtime_file_hashes_recorded"])
        self.assertTrue(policy["runtime_file_changes_during_benchmark_are_errors"])

    def test_process_memory_snapshot_reports_linux_rss_and_peak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status = Path(temporary) / "status"
            status.write_text("Name:\ttest\nVmHWM:\t2048 kB\nVmRSS:\t1024 kB\n", encoding="utf-8")
            self.assertEqual(
                _process_memory_snapshot(status),
                {"rss_bytes": 1024 * 1024, "peak_rss_bytes": 2048 * 1024},
            )

    def test_pipeline_aggregate_reports_host_peak_rss(self) -> None:
        payload = {
            "pipeline": "galp",
            "domain": "dct",
            "execution": {"aggregate_exclude_first_repeat": True},
            "repeats": [
                {
                    "repeat": 0,
                    "throughput_images_per_s": 1.0,
                    "end_to_end_latency_ms": {"mean": 1.0, "p95": 1.0},
                    "accuracy_top1": 1.0,
                    "accuracy_top5": 1.0,
                    "peak_gpu_memory_allocated_bytes": 1,
                    "peak_gpu_memory_reserved_bytes": 2,
                    "peak_gpu_memory_scope": "torch_allocator",
                    "host_process_rss_after_measurement_bytes": 10,
                    "host_process_peak_rss_bytes": 20,
                    "host_process_memory_scope": "main_process",
                },
                {
                    "repeat": 1,
                    "throughput_images_per_s": 2.0,
                    "end_to_end_latency_ms": {"mean": 2.0, "p95": 2.0},
                    "accuracy_top1": 1.0,
                    "accuracy_top5": 1.0,
                    "peak_gpu_memory_allocated_bytes": 3,
                    "peak_gpu_memory_reserved_bytes": 4,
                    "peak_gpu_memory_scope": "torch_allocator",
                    "host_process_rss_after_measurement_bytes": 30,
                    "host_process_peak_rss_bytes": 40,
                    "host_process_memory_scope": "main_process",
                },
            ],
        }
        aggregate = _aggregate_pipeline(payload)
        self.assertEqual(aggregate["host_process_rss_after_measurement_bytes"]["p50"], 30.0)
        self.assertEqual(aggregate["host_process_peak_rss_bytes"]["p50"], 40.0)
        self.assertEqual(aggregate["host_process_memory_scope"], "main_process")

    def test_deterministic_shuffled_direct_dct_trace_covers_population_once(self) -> None:
        reader = SimpleNamespace(image_count=10)
        args = SimpleNamespace(
            sampling_policy="all",
            preprocess="rgbnomore-val-pushdown",
            image_order="shuffled",
            shuffle_seed=17,
            no_wrap_image_ids=True,
            batch_size=3,
        )
        actual = []
        for step in range(4):
            image_ids, skipped = _make_benchmark_image_ids(reader, args, step)
            self.assertEqual(skipped, [])
            actual.extend(image_ids)
        expected = list(range(10))
        import random

        random.Random(17).shuffle(expected)
        self.assertEqual(actual, expected)

    def test_storage_counter_reader_accepts_pipeline_repeat_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "pipeline.json"
            result.write_text(
                json.dumps(
                    {
                        "repeats": [
                            {"native_counters": {"rowgroup_storage_bytes_read": 11}},
                            {"native_counters": {"rowgroup_storage_bytes_read": 13}},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(_counter_values(result), [11, 13])

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
        self.assertEqual(PRESETS["e2e"]["batch_size"], 50)
        self.assertEqual(PRESETS["e2e"]["warmup_batches"], 0)
        self.assertEqual(PRESETS["e2e"]["measurement_batches"], 1000)
        self.assertEqual(PRESETS["e2e"]["repeats"], 5)
        self.assertEqual(E2E_PIPELINES, ("galp", "galp_legacy", "rgbnomore", "dali"))
        self.assertGreater(PRESETS["e2e"]["measurement_batches"], PRESETS["smoke"]["measurement_batches"])
        self.assertEqual(GALP_E2E_MIN_DALI_HOT_MEDIAN_RATIO, 1.10)
        self.assertEqual(E2E_MAX_HOT_THROUGHPUT_CV, 0.05)

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

    def test_galp_e2e_relative_dali_and_stability_gates_are_hard(self) -> None:
        contract = {
            "pipelines": {"enabled": ["galp", "dali"]},
            "performance_gates": {
                "galp": {
                    "minimum_hot_median_to_dali_hot_median_ratio": 1.10,
                    "require_hot_min_above_dali_hot_median": True,
                    "maximum_hot_throughput_cv": 0.05,
                }
            },
        }
        aggregates = [
            {
                "pipeline": "galp",
                "throughput_images_per_s": {
                    "p50": 1890.0,
                    "min": 1750.0,
                    "cv_population": 0.04,
                },
            },
            {
                "pipeline": "dali",
                "throughput_images_per_s": {
                    "p50": 1720.0,
                    "min": 1700.0,
                    "cv_population": 0.03,
                },
            },
        ]
        failures: list[str] = []
        gates = _evaluate_performance_gates(contract, aggregates, failures)
        by_metric = {gate["metric"]: gate for gate in gates}
        self.assertFalse(by_metric["hot_median_to_dali_hot_median_ratio"]["ok"])
        self.assertTrue(by_metric["hot_min_throughput_above_dali_hot_median"]["ok"])
        self.assertTrue(by_metric["galp_hot_throughput_cv"]["ok"])
        self.assertTrue(by_metric["dali_hot_throughput_cv"]["ok"])
        self.assertTrue(failures)

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
        self.assertTrue((galp_root / "benchmarks/system_rgbnomore/inference/run.py").is_file())
        self.assertTrue((galp_root / "benchmarks/system_rgbnomore/dataset/prepare_dataset.py").is_file())
        self.assertTrue((galp_root / "benchmarks/system_rgbnomore/diagnostics/direct_dct.py").is_file())
        self.assertTrue((galp_root / "benchmarks/system_rgbnomore/diagnostics/validate_pushdown.py").is_file())
        self.assertTrue((galp_root / "benchmarks/system_rgbnomore/diagnostics/scan_manifests.py").is_file())
        self.assertTrue((galp_root / "torch/rgbnomore_dct_profile.py").is_file())

    def test_distribution_and_trace_are_deterministic(self) -> None:
        self.assertEqual(distribution([1.0, 2.0, 3.0])["p50"], 2.0)
        self.assertAlmostEqual(distribution([1.0, 2.0, 3.0])["cv_population"], (2.0 / 3.0) ** 0.5 / 2.0)
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
            with mock.patch("shared.common.fingerprint_file", side_effect=AssertionError("unexpected rehash")):
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
                "prediction_ordinals": np.asarray([0], dtype=np.int64),
                "prediction_labels": np.asarray([7], dtype=np.int64),
                "top1_predictions": np.asarray([7], dtype=np.int64),
                "top5_predictions": np.asarray([[7, 1, 2, 3, 4]], dtype=np.int64),
                "metadata_json": np.asarray(json.dumps({"prediction_agreement_sample_count": 1})),
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
                    "full_prediction_top1_agreement_min": 1.0,
                    "full_prediction_sample_count": 1,
                },
                "strict",
                failures,
            )
            self.assertGreater(result["logits"]["cosine_mean"], 0.999)
            self.assertEqual(result["logits"]["top1_agreement"], 0.0)
            self.assertFalse(result["logits"]["within_tolerance"])
            self.assertTrue(result["full_prediction"]["within_tolerance"])
            self.assertTrue(any("logits exceed tolerance" in failure for failure in failures))

    def test_galp_adapter_prefetches_two_batches_ahead_in_order(self) -> None:
        prefetch_calls: list[list[int]] = []

        class Pending:
            def __init__(self, image_ids: list[int]) -> None:
                self.image_ids = image_ids

        class SourceBatch:
            execution_stats = {
                "fixed_transform_item_count": 0,
                "planless_image_descriptor_count": 2,
                "host_expanded_transform_items_created": 0,
                "host_output_block_source_lists_created": 0,
                "host_global_transform_sort_items": 0,
                "device_mapping_fused": True,
                "projection_item_count": 0,
                "decoded_projection_item_count": 0,
                "project_decoded_ycbcr_grid_launch_count": 0,
                "fixed_grid_output_float32": True,
                "fixed_grid_output_affine_applied": True,
                "fixed_grid_finalize_kernel_launch_count": 1,
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

    def test_galp_serial_policy_defers_next_prefetch_until_next_load(self) -> None:
        prefetch_calls: list[list[int]] = []

        class Pending:
            def __init__(self, image_ids: list[int]) -> None:
                self.image_ids = image_ids

        class SourceBatch:
            execution_stats = {
                "fixed_transform_item_count": 0,
                "planless_image_descriptor_count": 2,
                "host_expanded_transform_items_created": 0,
                "host_output_block_source_lists_created": 0,
                "host_global_transform_sort_items": 0,
                "device_mapping_fused": True,
                "projection_item_count": 0,
                "decoded_projection_item_count": 0,
                "project_decoded_ycbcr_grid_launch_count": 0,
                "fixed_grid_output_float32": True,
                "fixed_grid_output_affine_applied": True,
                "fixed_grid_finalize_kernel_launch_count": 1,
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
        adapter.scheduling_policy = "serial"
        adapter.pending_batches = deque()
        adapter.next_prefetch_batch_index = 0
        first = [
            {"galp_image_id": 30, "label": 1, "ordinal": 0},
            {"galp_image_id": 31, "label": 2, "ordinal": 1},
        ]
        second = [
            {"galp_image_id": 32, "label": 3, "ordinal": 2},
            {"galp_image_id": 33, "label": 4, "ordinal": 3},
        ]
        adapter.samples = first + second
        adapter.total_batches = 2

        adapter.begin_repeat()
        self.assertEqual(prefetch_calls, [[30, 31]])
        adapter.load(first, second)
        self.assertEqual(prefetch_calls, [[30, 31]])
        adapter.load(second, None)
        self.assertEqual(prefetch_calls, [[30, 31], [32, 33]])
        self.assertEqual(list(adapter.pending_batches), [])

    def test_galp_legacy_adapter_uses_same_prefetch_path_and_expanded_graph(self) -> None:
        prefetch_calls: list[list[int]] = []

        class Pending:
            def __init__(self, image_ids: list[int]) -> None:
                self.image_ids = image_ids

        class SourceBatch:
            execution_stats = {
                "fixed_transform_item_count": 8,
                "planless_image_descriptor_count": 0,
                "host_expanded_transform_items_created": 8,
                "host_global_transform_sort_items": 8,
                "exact_batch_plan_cache_enabled": False,
                "cache_enabled": False,
                "fixed_grid_output_float32": True,
                "fixed_grid_output_affine_applied": True,
                "fixed_grid_finalize_kernel_launch_count": 1,
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
            def read_and_adapt_batch(*args, **kwargs):
                raise AssertionError("legacy A/B must use the same prefetch path")

            @staticmethod
            def _empty_totals():
                return {"fixed_transform_items": 0, "projection_items": 0}

            @staticmethod
            def _accumulate_many_stats(totals, batches):
                totals["fixed_transform_items"] += len(batches)

        adapter = object.__new__(GalpLegacyAdapter)
        adapter.module = Module()
        adapter.reader = object()
        adapter.args = SimpleNamespace(preprocess="rgbnomore-val-pushdown")
        adapter.device = torch.device("cpu")
        adapter.transform = None
        adapter.batch_size = 2
        adapter.batch_prefetch_depth = 1
        adapter.pending_batches = deque()
        adapter.next_prefetch_batch_index = 0
        adapter.samples = [
            {"galp_image_id": 20, "label": 1, "ordinal": 0},
            {"galp_image_id": 21, "label": 2, "ordinal": 1},
        ]
        adapter.total_batches = 1

        adapter.begin_repeat()
        loaded = adapter.load(adapter.samples, None)

        self.assertEqual(prefetch_calls, [[20, 21]])
        self.assertEqual(loaded.ordinals, [0, 1])
        self.assertEqual(list(adapter.pending_batches), [])


if __name__ == "__main__":
    unittest.main()
