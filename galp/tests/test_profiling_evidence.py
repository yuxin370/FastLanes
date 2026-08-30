#!/usr/bin/env python3
"""Contract tests for bounded profiling evidence helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from galp.benchmarks.profiling.prepare_equal_image_profile_replay import (
    _parse_args as _parse_replay_args,
    _validate_replay_contracts,
)
from galp.benchmarks.profiling.profile_native_pls_nsys import (
    _profiling_contract_validation,
)
from galp.benchmarks.profiling.summarize_equal_image_nsys import (
    _capture_cardinality,
    _parse_args as _parse_summary_args,
)
from galp.benchmarks.system_dct_major.training_pls import train


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(*, contract_hash: str, profiling_enabled: bool) -> dict:
    return {
        "benchmark": "equal-image-epoch-aware-rgb-training-v2",
        "contract_hash": contract_hash,
        "pipelines": ["d2", "d3", "pytorch"],
        "dataset": {"train_manifest": "/data/train.csv"},
        "dali_variants": {"d2": {"num_threads": 8}},
        "profiling": {
            "enabled": profiling_enabled,
            "epoch": 2 if profiling_enabled else None,
            "skip_profiled_epoch_validation": profiling_enabled,
        },
        "validation": {
            "epochs": [0, 1] if profiling_enabled else [0, 1, 2],
            "transform": "center crop",
        },
    }


class ProfileReplayContractTest(unittest.TestCase):
    def test_default_pipeline_is_registered_d2(self) -> None:
        args = _parse_replay_args(
            ["--source-root", "source", "--target-root", "target"]
        )
        self.assertEqual(args.pipeline, "d2")

    def test_only_profiling_controls_may_differ(self) -> None:
        source = _contract(contract_hash="source", profiling_enabled=False)
        target = _contract(contract_hash="target", profiling_enabled=True)
        _validate_replay_contracts(source, target, pipeline="d2")

        target["dataset"]["train_manifest"] = "/data/other.csv"
        with self.assertRaisesRegex(ValueError, "workload contracts differ"):
            _validate_replay_contracts(source, target, pipeline="d2")

        target = _contract(contract_hash="target", profiling_enabled=True)
        target["dali_variants"]["d2"]["num_threads"] = 4
        with self.assertRaisesRegex(ValueError, "workload contracts differ"):
            _validate_replay_contracts(source, target, pipeline="d2")

    def test_pipeline_must_be_registered_in_both_contracts(self) -> None:
        source = _contract(contract_hash="source", profiling_enabled=False)
        target = _contract(contract_hash="target", profiling_enabled=True)
        target["pipelines"] = ["d3", "pytorch"]
        with self.assertRaisesRegex(ValueError, "not registered"):
            _validate_replay_contracts(source, target, pipeline="d2")


class NativeProfileIdentityTest(unittest.TestCase):
    def test_physical_manifest_content_is_part_of_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.bin"
            mapping = root / "mapping.csv"
            run_manifest = root / "run_manifest.json"
            manifest.write_bytes(b"physical-v1")
            mapping.write_text("image_id,label\n0,0\n", encoding="utf-8")
            run_manifest.write_text(
                json.dumps(
                    {
                        "condition_id": "B6",
                        "training_seed": 11997733,
                        "recipe_hash": "recipe",
                        "layout_hash": "layout",
                        "execution_mode": "native_physical_pls",
                        "physical_execution": {
                            "physical_galp_manifest": str(manifest.resolve()),
                            "physical_galp_manifest_sha256": _sha256(manifest),
                            "premixed_mapping_csv": str(mapping.resolve()),
                            "premixed_mapping_sha256": _sha256(mapping),
                        },
                    }
                ),
                encoding="utf-8",
            )
            _profiling_contract_validation(
                run_manifest,
                condition_id="B6",
                seed=11997733,
                recipe_hash="recipe",
                layout_hash="layout",
                execution_backend=train.NATIVE_PHYSICAL_BACKEND,
                physical_galp_manifest=manifest,
                premixed_mapping_csv=mapping,
                expected_mapping_sha256=_sha256(mapping),
            )

            manifest.write_bytes(b"physical-v2")
            with self.assertRaisesRegex(ValueError, "physical-layout identity differs"):
                _profiling_contract_validation(
                    run_manifest,
                    condition_id="B6",
                    seed=11997733,
                    recipe_hash="recipe",
                    layout_hash="layout",
                    execution_backend=train.NATIVE_PHYSICAL_BACKEND,
                    physical_galp_manifest=manifest,
                    premixed_mapping_csv=mapping,
                    expected_mapping_sha256=_sha256(mapping),
                )


class NsysSummaryContractTest(unittest.TestCase):
    def test_default_window_uses_registered_d2_pipeline(self) -> None:
        args = _parse_summary_args(["capture.sqlite", "--output-json", "out.json"])
        self.assertEqual(
            args.window_name, "profile-rgb-d2-microbatches_512_1535"
        )

    def test_capture_counts_come_from_nvtx_ranges(self) -> None:
        microbatches, updates = _capture_cardinality(
            {
                "training.loader.next_batch": [1] * 32,
                "training.optimizer": [1] * 2,
            }
        )
        self.assertEqual(microbatches, 32)
        self.assertEqual(updates, 2)


if __name__ == "__main__":
    unittest.main()
