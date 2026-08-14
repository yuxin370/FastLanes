from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from diagnostics import run_controlled_cold
from diagnostics import select_manifest_shard_tuning
from diagnostics import evaluate_bounded_candidate


class ControlledColdDiagnosticsTest(unittest.TestCase):
    def test_failed_validation_round_remains_available_for_candidate_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            (round_dir / "results.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "failures": ["duplicate physical read"],
                        "aggregates": [{"pipeline": "dct_major_pushdown"}],
                    }
                ),
                encoding="utf-8",
            )
            (round_dir / "pipeline_dct_major_pushdown.json").write_text(
                json.dumps(
                    {
                        "repeats": [{"native_totals": {"duplicate_physical_read_count": 1}}],
                        "semantic_artifact": "semantic.npz",
                    }
                ),
                encoding="utf-8",
            )

            loaded = run_controlled_cold._load_round(round_dir)

        self.assertFalse(loaded["summary"]["ok"])
        self.assertEqual(loaded["summary"]["failures"], ["duplicate physical read"])
        self.assertIn("dct_major_pushdown", loaded["pipelines"])

    def test_aggregate_preserves_direct_input_ready_stall_ratio(self) -> None:
        def round_record(
            stall_percent: float | None,
            storage_read_bytes: int | None = 8192,
        ) -> dict:
            cold_start = {
                "throughput_images_per_s": 100.0,
                "time_to_first_batch_ms": 10.0,
                "first_shard_ready_ms": 9.0,
            }
            if stall_percent is not None:
                cold_start["input_ready_stall_percent"] = stall_percent
            if storage_read_bytes is not None:
                cold_start["process_io"] = {"storage_read_bytes": storage_read_bytes}
            return {
                "pipelines": {
                    "dct_major_pushdown": {
                        "aggregate": {"cold_start": cold_start},
                        "repeat": {
                            "latency_ms": {"p50": 1.0, "p95": 2.0},
                            "loader_submit_ms": {"mean": 0.1},
                            "top_level_h2d_ms": {"mean": 0.01},
                            "model_ms": {"mean": 0.8},
                            "gpu_utilization_percent": {"count": 0},
                            "host_peak_rss_bytes": 1000,
                            "peak_torch_gpu_allocated_bytes": 2000,
                            "peak_torch_gpu_reserved_bytes": 3000,
                            "native_totals": {},
                        },
                    }
                }
            }

        aggregate = run_controlled_cold._aggregate(
            [round_record(1.0, 4096), round_record(2.0, 8192), round_record(3.0, 12288)]
        )
        self.assertEqual(
            aggregate["dct_major_pushdown"]["input_ready_stall_percent"]["median"],
            2.0,
        )
        self.assertEqual(
            aggregate["dct_major_pushdown"]["process_io_storage_read_bytes"]["median"],
            8192.0,
        )

        missing = run_controlled_cold._aggregate(
            [round_record(1.0), round_record(None, None)]
        )
        self.assertIsNone(missing["dct_major_pushdown"]["input_ready_stall_percent"])
        self.assertIsNone(missing["dct_major_pushdown"]["process_io_storage_read_bytes"])


class ManifestShardSelectionTest(unittest.TestCase):
    @staticmethod
    def _candidate(
        throughput: float,
        memory: float,
        storage: int,
        ttfb: float,
        *,
        eligible: bool = True,
    ) -> dict[str, object]:
        return {
            "cold_throughput_images_per_s": throughput,
            "selection_peak_bytes": memory,
            "total_storage_bytes": storage,
            "process_scope_ttfb_ms": ttfb,
            "eligible": eligible,
        }

    def test_selection_excludes_invalid_and_applies_three_percent_tiebreaks(self) -> None:
        candidates = [
            self._candidate(110.0, 1.0, 1, 1.0, eligible=False),
            self._candidate(100.0, 100.0, 100, 10.0),
            self._candidate(98.0, 90.0, 100, 9.0),
            self._candidate(99.0, 90.0, 80, 12.0),
        ]
        with mock.patch.object(
            select_manifest_shard_tuning,
            "_load_candidate",
            side_effect=candidates,
        ):
            result = select_manifest_shard_tuning.select(
                [Path(str(index)) for index in range(len(candidates))]
            )

        self.assertEqual(result["best_observed_throughput_images_per_s"], 100.0)
        self.assertIs(result["selected"], candidates[3])

    def test_selection_rejects_an_entirely_ineligible_sweep(self) -> None:
        candidate = self._candidate(100.0, 1.0, 1, 1.0, eligible=False)
        with mock.patch.object(
            select_manifest_shard_tuning,
            "_load_candidate",
            return_value=candidate,
        ):
            with self.assertRaisesRegex(RuntimeError, "no tuning candidate"):
                select_manifest_shard_tuning.select([Path("excluded")])


class BoundedCandidateEvaluationTest(unittest.TestCase):
    @staticmethod
    def _payload(*, throughput: float, stall: float, physical: int = 841_123_964) -> dict:
        native = {
            "bounded_read_amplification_ppm": 1_020_000,
            "bounded_read_local_amplification_ppm": 0,
            "bounded_read_max_run_bytes": 0,
            "bounded_exact_storage_bytes": 824_678_664,
            "bounded_physical_storage_bytes": physical,
            "bounded_merged_gap_bytes": physical - 824_678_664,
            "bounded_physical_run_count": 247_761,
            "read_amplification": physical / 824_678_664,
            "compressed_payload_bytes_read": physical,
            "selected_compressed_payload_bytes": 824_678_664,
            "planned_vector_count": 57_534,
            "actual_vector_count": 57_534,
            "rowgroup_count": 788,
            "run_interval_bounded_rowgroup_count": 788,
            "sparse_read_fallback_rowgroup_count": 0,
            "actual_transient_total_used_high_water_bytes": 96 * 1024 * 1024,
            "actual_transient_total_allocated_high_water_bytes": 128 * 1024 * 1024,
            "actual_transient_memory_gate_passed": True,
        }
        native.update({name: 0 for name in evaluate_bounded_candidate.ZERO_COUNTERS})
        return {
            "ok": True,
            "failures": [],
            "process_repeats": 1,
            "round_validation": [{"ok": True, "failures": []}],
            "pipelines": {
                "dct_major_pushdown": {
                    "throughput_images_per_s": {"median": throughput},
                    "input_ready_stall_percent": {"median": stall},
                    "process_io_storage_read_bytes": {"median": 900_000_000},
                    "native_totals": [native],
                }
            },
        }

    @staticmethod
    def _hot_payload(*, throughput: float, bounded: bool = True, cap: float = 1.02) -> dict:
        return {
            "ok": True,
            "failures": [],
            "source_changes": [],
            "aggregates": [
                {
                    "pipeline": "dct_major_pushdown",
                    "throughput_images_per_s": {"median": throughput},
                }
            ],
            "_bounded": bounded,
            "_cap": cap,
        }

    def _evaluate(self, payload: dict, hot: dict | None = None) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "controlled_cold_results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            hot_path = None
            if hot is not None:
                contract_path = root / "hot_contract.json"
                contract_path.write_text(
                    json.dumps(
                        {
                            "pipelines": {
                                "dct_major_pushdown": {
                                    "crop_execution_mode": (
                                        "bounded-range-read-selected-decode"
                                        if hot.pop("_bounded")
                                        else "vector-range-read-selected-decode"
                                    ),
                                    "bounded_read_amplification_cap": hot.pop("_cap"),
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                hot["contract"] = str(contract_path)
                hot_path = root / "hot_results.json"
                hot_path.write_text(json.dumps(hot), encoding="utf-8")
            return evaluate_bounded_candidate.evaluate(path, 1.02, hot_summary_path=hot_path)

    def test_cold_success_requires_hot_at_the_same_cap(self) -> None:
        result = self._evaluate(self._payload(throughput=4_720.0, stall=1.5))
        self.assertTrue(result["correctness_passed"])
        self.assertTrue(result["cold_performance_passed"])
        self.assertEqual(result["decision"], "run-hot-at-same-cap")
        self.assertIsNone(result["next_cap"])

        success = self._evaluate(
            self._payload(throughput=4_720.0, stall=1.5),
            self._hot_payload(throughput=4_880.0),
        )
        self.assertTrue(success["hot_evidence_passed"])
        self.assertEqual(success["decision"], "stop-success")

    def test_hot_must_use_the_same_bounded_cap(self) -> None:
        exact_hot = self._evaluate(
            self._payload(throughput=4_720.0, stall=1.5),
            self._hot_payload(throughput=4_900.0, bounded=False),
        )
        self.assertFalse(exact_hot["hot_evidence_passed"])
        self.assertEqual(exact_hot["decision"], "reject-hot-hard-gate")

        wrong_cap = self._evaluate(
            self._payload(throughput=4_720.0, stall=1.5),
            self._hot_payload(throughput=4_900.0, cap=1.05),
        )
        self.assertFalse(wrong_cap["hot_evidence_passed"])
        self.assertEqual(wrong_cap["decision"], "reject-hot-hard-gate")

    def test_cold_miss_advances_only_to_the_next_frozen_cap(self) -> None:
        result = self._evaluate(self._payload(throughput=4_700.0, stall=1.5))
        self.assertEqual(result["decision"], "try-next-cap")
        self.assertEqual(result["next_cap"], 1.05)

        invalid = self._evaluate(self._payload(throughput=4_720.0, stall=1.5, physical=900_000_000))
        self.assertFalse(invalid["correctness_passed"])
        self.assertEqual(invalid["decision"], "reject-correctness-or-hard-gate")

        replay_mismatch = self._payload(throughput=4_720.0, stall=1.5)
        replay_mismatch["pipelines"]["dct_major_pushdown"]["native_totals"][0][
            "bounded_physical_run_count"
        ] -= 1
        mismatch_result = self._evaluate(replay_mismatch)
        self.assertFalse(mismatch_result["correctness_passed"])
        self.assertEqual(mismatch_result["decision"], "reject-correctness-or-hard-gate")

        ratio_mismatch = self._payload(throughput=4_720.0, stall=1.5)
        ratio_mismatch["pipelines"]["dct_major_pushdown"]["native_totals"][0][
            "read_amplification"
        ] = 49.0
        ratio_result = self._evaluate(ratio_mismatch)
        self.assertFalse(ratio_result["correctness_passed"])
        self.assertEqual(ratio_result["decision"], "reject-correctness-or-hard-gate")

    def test_missing_stall_evidence_is_rejected_instead_of_widening_cap(self) -> None:
        payload = self._payload(throughput=4_720.0, stall=1.5)
        payload["pipelines"]["dct_major_pushdown"]["input_ready_stall_percent"] = None
        result = self._evaluate(payload)
        self.assertFalse(result["correctness_passed"])
        self.assertEqual(result["decision"], "reject-correctness-or-hard-gate")

        payload = self._payload(throughput=4_720.0, stall=1.5)
        payload["pipelines"]["dct_major_pushdown"]["process_io_storage_read_bytes"] = None
        result = self._evaluate(payload)
        self.assertFalse(result["correctness_passed"])
        self.assertEqual(result["decision"], "reject-correctness-or-hard-gate")

        payload = self._payload(throughput=4_720.0, stall=1.5)
        payload["pipelines"]["dct_major_pushdown"]["process_io_storage_read_bytes"] = {
            "median": 841_123_963
        }
        result = self._evaluate(payload)
        self.assertFalse(result["correctness_passed"])
        self.assertEqual(result["decision"], "reject-correctness-or-hard-gate")


if __name__ == "__main__":
    unittest.main()
