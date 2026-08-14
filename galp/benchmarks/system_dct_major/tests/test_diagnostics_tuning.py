from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from diagnostics import run_controlled_cold


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


if __name__ == "__main__":
    unittest.main()
