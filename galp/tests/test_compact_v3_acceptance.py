from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "tools/jpeg_dct/compact_v3_acceptance.py"
SPEC = importlib.util.spec_from_file_location("compact_v3_acceptance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
acceptance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = acceptance
SPEC.loader.exec_module(acceptance)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _compact_descriptor_bytes(payload_size: int = 16) -> bytes:
    descriptor = bytearray(649)
    descriptor[:8] = b"GALPCV3\0"
    struct.pack_into("<HHI", descriptor, 8, 3, 256, 0x3F)
    struct.pack_into("<QQQQIIIIII", descriptor, 16, 649, payload_size, 0, 1, 64, 1024, 3, 1, 1, 1)
    sections = ((256, 24), (280, 32), (312, 32), (344, 48), (392, 256), (648, 1))
    for index, section in enumerate(sections):
        struct.pack_into("<QQ", descriptor, 72 + index * 16, *section)
    struct.pack_into("<QQ", descriptor, 256, 16, 24)
    descriptor[272:280] = b"schema00"
    struct.pack_into("<IIII", descriptor, 280, 0, 1, 1, 0)
    struct.pack_into("<H", descriptor, 296, 1)
    struct.pack_into("<I", descriptor, 300, 3)
    struct.pack_into("<Q", descriptor, 304, 0)
    struct.pack_into("<IIIIIII", descriptor, 312, 0, 1, 1, 1, 1, 0, 0)
    struct.pack_into("<QIIQI", descriptor, 344, 24, payload_size, 1, 0, 1)
    struct.pack_into("<II", descriptor, 372, 0, 0)
    struct.pack_into("<I", descriptor, 392, payload_size)
    descriptor[648] = 0
    struct.pack_into("<Q", descriptor, 168, acceptance._crc64_ecma_bytes(descriptor))
    return bytes(descriptor)


def _fls_bytes(payload_size: int = 16) -> bytes:
    return bytes(24) + bytes(payload_size) + _compact_descriptor_bytes(payload_size) + bytes(24)


def _manifest_bytes(
    fls_name: str = "generation/shard_000000.fls", payload_size: int = 16
) -> bytes:
    payload = bytearray(b"GJDCTSH1")
    payload += struct.pack("<IHIIQI", 3, 2, 1, 8192, 1, 1)
    payload += struct.pack("<IQIQQQII", 0, 0, 1, 1, 0, 1, 1, 0)
    payload += struct.pack("<QQ", 697 + payload_size, 3)
    payload += _string(fls_name)
    payload += _string("generation/shard_000000.meta.bin")
    payload += b"GJDCCV31"
    payload += _string("image-major-vector-rowgroups")
    payload += _string("galp-compact-v1")
    payload += struct.pack("<I", 1024)
    payload += _string("tiled-z32")
    payload += struct.pack("<HI", 3, 1)
    payload += struct.pack("<IQQQQ", 0, payload_size, 0, 649, 4000)
    return bytes(payload)


class CompactV3AcceptanceTest(unittest.TestCase):
    def _dataset(
        self,
        root: Path,
        manifest_bytes: bytes | None = None,
        payload_size: int = 16,
    ) -> Path:
        generation = root / "generation"
        generation.mkdir()
        (generation / "shard_000000.fls").write_bytes(_fls_bytes(payload_size))
        (generation / "shard_000000.meta.bin").write_bytes(b"abc")
        manifest = root / "manifest.bin"
        manifest.write_bytes(
            manifest_bytes
            if manifest_bytes is not None
            else _manifest_bytes(payload_size=payload_size)
        )
        return manifest

    def test_strict_manifest_contract_and_storage_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._dataset(Path(temporary))
            parsed = acceptance.parse_manifest(manifest)
            self.assertEqual(parsed["image_count"], 1)
            self.assertEqual(parsed["rowgroup_vectors"], 1)
            self.assertEqual(parsed["compact"]["vector_size"], 1024)
            self.assertEqual(parsed["compact"]["spatial_order"], "tiled-z32")
            self.assertEqual(parsed["compressed_payload_bytes"], 16)
            self.assertEqual(parsed["compact_descriptor_bytes"], 649)
            self.assertEqual(parsed["source_descriptor_bytes"], 4000)
            self.assertAlmostEqual(parsed["descriptor_reduction"], 1.0 - 649 / 4000)
            self.assertTrue(parsed["optimized_runtime_schema"])
            self.assertEqual(parsed["runtime_total_bytes"], manifest.stat().st_size + 716)

    def test_accepts_an_all_metadata_shard_without_dividing_by_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._dataset(Path(temporary), payload_size=0)
            parsed = acceptance.parse_manifest(manifest)
            self.assertEqual(parsed["compressed_payload_bytes"], 0)
            self.assertEqual(
                parsed["shards"][0]["compact_descriptor"]["zero_payload_rowgroup_count"],
                1,
            )
            self.assertIsNone(parsed["compact_metadata_to_payload_ratio"])
            self.assertIsNone(
                parsed["storage_gates"]["recommended_metadata_at_most_10pct_payload"]
            )

    def test_rejects_unknown_trailer_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._dataset(Path(temporary), _manifest_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "trailing bytes"):
                acceptance.parse_manifest(manifest)

    def test_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.bin"
            manifest.write_bytes(_manifest_bytes("../shard_000000.fls"))
            with self.assertRaisesRegex(ValueError, "parent traversal"):
                acceptance.parse_manifest(manifest)

    def test_rejects_sparse_bundle_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._dataset(root)
            (root / "forbidden.svb").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "forbidden .svb"):
                acceptance.parse_manifest(manifest)

    def test_rejects_payload_crc_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._dataset(root)
            fls = root / "generation/shard_000000.fls"
            corrupted = bytearray(fls.read_bytes())
            corrupted[24] = 1
            fls.write_bytes(corrupted)
            with self.assertRaisesRegex(ValueError, "payload CRC64 mismatch"):
                acceptance.parse_manifest(manifest)

    def test_rejects_descriptor_crc_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._dataset(root)
            fls = root / "generation/shard_000000.fls"
            corrupted = bytearray(fls.read_bytes())
            corrupted[24 + 16 + 200] = 1
            fls.write_bytes(corrupted)
            with self.assertRaisesRegex(ValueError, "descriptor CRC64 mismatch"):
                acceptance.parse_manifest(manifest)

    def test_rejects_rowgroup_partition_with_valid_descriptor_crc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._dataset(root)
            fls = root / "generation/shard_000000.fls"
            file_bytes = bytearray(fls.read_bytes())
            descriptor_offset = 24 + 16
            descriptor = bytearray(file_bytes[descriptor_offset : descriptor_offset + 649])
            struct.pack_into("<I", descriptor, 344 + 8, 15)
            descriptor[168:176] = bytes(8)
            struct.pack_into("<Q", descriptor, 168, acceptance._crc64_ecma_bytes(descriptor))
            file_bytes[descriptor_offset : descriptor_offset + 649] = descriptor
            fls.write_bytes(file_bytes)
            with self.assertRaisesRegex(ValueError, "partition"):
                acceptance.parse_manifest(manifest)

    def test_io_matrix_summary_enforces_two_dimensional_pushdown(self) -> None:
        def record(
            workload: str,
            coefficient_count: int,
            pages: int,
            payload: int,
            full_payload: int,
            vector_ratio: float,
        ) -> dict[str, object]:
            return {
                "workload": workload,
                "coefficient_count": coefficient_count,
                "physical_page_bytes_covered": pages,
                "physical_range_bytes_read": payload,
                "full_compressed_payload_bytes": full_payload,
                "logical_compressed_bytes": payload,
                "actual_block_device_read_bytes": 0,
                "native_pread_count": 3,
                "native_preadv_count": 1 if workload == "full-all" else 0,
                "coalesced_run_count": 3,
                "selected_vector_count": 100 if not workload.startswith("crop") else 50,
                "full_vector_count": 100,
                "selected_vector_ratio": vector_ratio,
                "selected_coefficient_ratio": coefficient_count / 64.0,
            }

        counts = (1, 4, 8, 16, 32, 64)
        full_pages = (100, 160, 260, 420, 700, 1000)
        full_payloads = (80, 140, 230, 390, 670, 1000)
        crop_pages = (40, 70, 120, 200, 350, 500)
        crop_payloads = (30, 60, 100, 180, 330, 500)
        records = [
            record("full-all", 64, 1000, 1000, 1000, 1.0),
            record("crop-all", 64, 500, 500, 500, 0.5),
        ]
        records.extend(
            record("full-prefix", count, pages, payload, 1000, 1.0)
            for count, pages, payload in zip(counts, full_pages, full_payloads)
        )
        records.extend(
            record("crop-prefix", count, pages, payload, 500, 0.5)
            for count, pages, payload in zip(counts, crop_pages, crop_payloads)
        )
        summary = acceptance.summarize_io_records(records)
        gates = summary["gates"]
        self.assertTrue(gates["crop_all_vector_ratio_below_one"])
        self.assertTrue(gates["crop_all_page_ratio_below_one"])
        self.assertTrue(gates["full_prefix_pages_monotonic_with_k"])
        self.assertTrue(gates["crop_prefix_below_crop_all_and_full_prefix_for_k_below_64"])
        self.assertTrue(gates["coefficient_ratios_equal_k_over_64"])
        self.assertTrue(gates["native_range_accounting_consistent"])
        self.assertTrue(gates["full_all_preadv_coalescing_avoids_per_rowgroup_syscalls"])

    def test_io_matrix_summary_rejects_incomplete_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete"):
            acceptance.summarize_io_records([])

    def test_gate_enforcement_rejects_false_without_treating_manual_evidence_as_passed(self) -> None:
        with self.assertRaisesRegex(ValueError, "hard_failure"):
            acceptance._require_passing_boolean_gates(
                {"passing": True, "hard_failure": False, "manual_evidence": "unverified"},
                "fixture",
            )
        acceptance._require_passing_boolean_gates(
            {"passing": True, "manual_evidence": "unverified"},
            "fixture",
        )

    def test_validation_evidence_is_bound_to_the_ab_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = {"execution": {"warmup_batches": 0}, "identity": "same"}
            contract_path = root / "candidate.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            evidence_dir = root / "validated"
            evidence_dir.mkdir()
            (evidence_dir / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
            validation_path = evidence_dir / "validation.json"
            validation_path.write_text(json.dumps({"ok": True, "failures": []}), encoding="utf-8")
            evidence = acceptance._load_validation_evidence(validation_path, contract_path)
            self.assertTrue(evidence["ok"])
            contract_path.write_text(json.dumps({"identity": "different"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                acceptance._load_validation_evidence(validation_path, contract_path)

    def test_contract_adapter_writes_canonical_pipeline_names_from_legacy_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = self._dataset(dataset)
            binding_dir = root / "binding"
            binding_dir.mkdir()
            (binding_dir / "_galp_direct_dct.fixture.so").write_bytes(b"binding")
            sample_manifest = root / "samples.json"
            sample_manifest.write_text(
                json.dumps({"samples": [{"ordinal": 0}]}), encoding="utf-8"
            )
            template = root / "template.json"
            template.write_text(
                json.dumps(
                    {
                        "execution": {
                            "batch_size": 1,
                            "warmup_batches": 0,
                            "measurement_batches": 1,
                            "repeats": 1,
                            "aggregate_exclude_first_repeat": False,
                        },
                        "dataset": {"sample_manifest": str(sample_manifest)},
                        "pipelines": {
                            "enabled": ["galp"],
                            "galp": {
                                "torch_binding_dir": str(binding_dir),
                                "enable_planless_execution": True,
                            },
                        },
                        "performance_gates": {"galp": {"minimum": 1}},
                        "semantic_validation": {
                            "comparison_groups": [
                                {"pipelines": ["galp", "rgbnomore"]}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "adapted" / "contract.json"
            acceptance.adapt_contract(
                SimpleNamespace(
                    manifest=manifest,
                    template=template,
                    output=output,
                    fingerprint_cache=None,
                    enabled_pipelines=("galp",),
                    repeats=2,
                    measurement_batches=1,
                )
            )
            adapted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(adapted["pipelines"]["enabled"], ["galp"])
            self.assertNotIn("transform_execution_mode", adapted["pipelines"]["galp"])
            self.assertNotIn("enable_planless_execution", adapted["pipelines"]["galp"])
            self.assertEqual(
                adapted["pipelines"]["galp"]["runtime_profile"],
                "compact-v3-planless-limited-o512-c512-v1",
            )
            self.assertIn("galp", adapted["performance_gates"])
            self.assertEqual(
                adapted["semantic_validation"]["comparison_groups"][0]["pipelines"],
                ["galp", "rgbnomore"],
            )

    def test_acceptance_cli_defaults_to_canonical_planless_pipeline(self) -> None:
        args = acceptance._parser().parse_args(
            [
                "run-ab",
                "--contract-a",
                "a.json",
                "--contract-b",
                "b.json",
                "--validation-a",
                "a-validation.json",
                "--validation-b",
                "b-validation.json",
                "--output-dir",
                "out",
            ]
        )
        self.assertEqual(args.pipeline, "galp")

    def test_ab_summary_accepts_single_batch_ordered_latency(self) -> None:
        legs = []
        for pair in range(5):
            for variant in ("A", "B"):
                candidate = variant == "B"
                legs.append(
                    {
                        "variant": variant,
                        "cold_throughput_images_per_s": 101.0 if candidate else 100.0,
                        "hot_throughput_images_per_s": 202.0 if candidate else 200.0,
                        "ordered_first_batch_latency_ms": 9.0 if candidate else 10.0,
                        "torch_peak_gpu_allocated_bytes": 100,
                        "torch_peak_gpu_reserved_bytes": 120,
                        "galp_native_device_peak_in_use_bytes": 20,
                        "galp_native_pinned_peak_in_use_bytes": 30,
                        "peak_process_tree_pss_bytes": 1000,
                        "peak_fls_mmap_pss_bytes": 10,
                        "contract_sources_clean_and_hashed": True,
                        "comparison_signature": "same-fixture",
                        "cold_sample_trace_sha256": "same-order",
                    }
                )
        summary = acceptance._summarize_ab(legs, seed=7)
        gates = summary["gates"]
        self.assertTrue(gates["at_least_five_independent_processes_per_variant"])
        self.assertTrue(gates["cold_B_median_at_least_A"])
        self.assertTrue(gates["hot_B_median_at_least_A"])
        self.assertTrue(gates["ordered_first_batch_latency_at_most_A"])
        self.assertTrue(gates["total_gpu_peak_within_1_05x_A"])
        self.assertTrue(gates["clean_hashed_baseline_contract"])
        self.assertTrue(gates["same_hardware_data_model_and_execution"])
        self.assertTrue(gates["same_sample_order"])
        self.assertEqual(summary["variant_B"]["peak_total_gpu_bytes"], 140)


if __name__ == "__main__":
    unittest.main()
