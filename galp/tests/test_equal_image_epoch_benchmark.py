#!/usr/bin/env python3
"""Tests for the equal-image RGB/native performance benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch


from galp.benchmarks.system_dct_major.training_pls.published_optimizer import (
    build_published_optimizer,
)
from galp.benchmarks.system_dct_major.training_pls.report_equal_image_performance import (
    build_report,
)
from galp.benchmarks.system_rgbnomore.training.equal_image_epoch_benchmark import (
    GRADIENT_ACCUMULATION,
    MICROBATCH_IMAGES,
    MetricsWriter,
    _restore_checkpoint,
    _save_checkpoint,
    batch_lengths,
    schedule_summary,
)


class EqualImageScheduleTest(unittest.TestCase):
    def test_official_imagenet_two_epoch_schedule(self) -> None:
        schedule = schedule_summary(1_281_167)
        self.assertEqual(schedule["microbatch_images"], 64)
        self.assertEqual(schedule["gradient_accumulation"], 16)
        self.assertEqual(schedule["microbatches_per_epoch"], 20_019)
        self.assertEqual(schedule["optimizer_updates_per_epoch"], 1_252)
        self.assertEqual(schedule["processed_images"], 2_562_334)
        self.assertEqual(schedule["total_microbatches"], 40_038)
        self.assertEqual(schedule["total_optimizer_updates"], 2_504)
        self.assertEqual(schedule["tail_microbatch_images"], 15)
        self.assertEqual(schedule["tail_accumulation_microbatches"], 3)

    def test_batch_lengths_preserve_tail(self) -> None:
        lengths = batch_lengths(1_281_167)
        self.assertEqual(len(lengths), 20_019)
        self.assertTrue(all(value == MICROBATCH_IMAGES for value in lengths[:-1]))
        self.assertEqual(lengths[-1], 15)
        self.assertEqual(sum(lengths), 1_281_167)
        self.assertEqual(
            len(range(0, len(lengths), GRADIENT_ACCUMULATION)), 1_252
        )


class EqualImageResumeTest(unittest.TestCase):
    def test_epoch_boundary_checkpoint_restores_and_truncates_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            metrics = MetricsWriter(run_dir / "metrics.jsonl")
            metrics.append({"record_type": "validation", "epoch": 0})
            model = torch.nn.Sequential(torch.nn.Linear(3, 2))
            optimizer, decayer, scheduler = build_published_optimizer(
                model,
                learning_rate=3.0e-3,
                weight_decay=1.0e-4,
                warmup_updates=2,
                total_updates=20,
            )
            initial_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            _save_checkpoint(
                run_dir,
                permanent_epoch=0,
                pipeline="pytorch",
                contract_hash="contract",
                initial_model_hash="initial",
                model=model,
                optimizer=optimizer,
                weight_decayer=decayer,
                scheduler=scheduler,
                completed_epoch=0,
                global_update=0,
                processed_images=0,
                metrics=metrics,
                pending_validation_epoch=None,
            )
            metrics.append({"record_type": "orphan", "epoch": 1})
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10.0)
            restored = _restore_checkpoint(
                run_dir / "latest.pt",
                pipeline="pytorch",
                contract_hash="contract",
                initial_model_hash="initial",
                model=model,
                optimizer=optimizer,
                weight_decayer=decayer,
                scheduler=scheduler,
                metrics=metrics,
            )
            self.assertEqual(restored, (0, 0, 0, None))
            self.assertEqual(metrics.lines, 1)
            self.assertEqual(metrics.records(), [{"epoch": 0, "record_type": "validation"}])
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, initial_state[name]))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _epoch(pipeline: str, epoch: int, images_per_second: float) -> dict:
    return {
        "record_type": "train",
        "scope": "epoch",
        "pipeline": pipeline,
        "epoch": epoch,
        "epoch_samples": 1_281_167,
        "epoch_microbatches": 20_019,
        "epoch_optimizer_updates": 1_252,
        "epoch_seconds": 1_281_167 / images_per_second,
        "images_per_second": images_per_second,
    }


class EqualImageReportTest(unittest.TestCase):
    def _fixture(self, root: Path, native_gpu_uuid: str = "gpu-1") -> tuple[Path, Path]:
        standard = root / "standard"
        native = root / "native"
        _write_json(
            standard / "contract.json",
            {
                "benchmark": "equal-image-epoch-aware-rgb-training-v1",
                "contract_hash": "contract",
                "seed": 11997733,
                "prefix_schedule": {
                    "processed_images": 2_562_334,
                    "total_microbatches": 40_038,
                    "total_optimizer_updates": 2_504,
                },
            },
        )
        environment = {
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_uuid": "gpu-1",
            "hostname": "host",
            "torch": "test",
            "torch_cuda_build": "test",
        }
        _write_json(standard / "environment.json", environment)
        _write_json(
            native / "environment.json",
            {
                **environment,
                "gpu_uuid": native_gpu_uuid,
                "execution_mode": "native_physical_pls",
                "physical_fls_observed": True,
                "physical_gpu_pool": True,
            },
        )
        _write_json(native / "condition_contract.json", {"condition_id": "B6"})
        for pipeline, speeds in (("dali", (1000.0, 1100.0)), ("pytorch", (800.0, 900.0))):
            _write_json(
                standard / "runs" / pipeline / "final_result.json",
                {
                    "state": "completed",
                    "completed_epoch": 2,
                    "epoch_records": [
                        _epoch(pipeline, 1, speeds[0]),
                        _epoch(pipeline, 2, speeds[1]),
                    ],
                },
            )
        _write_json(
            native / "run_status.json",
            {"completed_epoch": 2, "seed": 11997733},
        )
        native.mkdir(parents=True, exist_ok=True)
        with (native / "metrics.jsonl").open("w", encoding="utf-8") as output:
            output.write(json.dumps(_epoch("native_b6", 1, 1500.0)) + "\n")
            output.write(json.dumps(_epoch("native_b6", 2, 1600.0)) + "\n")
        return standard, native

    def test_report_accepts_equal_h100_workloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            standard, native = self._fixture(Path(directory))
            report, rows = build_report(standard_root=standard, native_run=native)
            self.assertEqual(report["hardware"]["uuid"], "gpu-1")
            self.assertEqual(len(rows), 6)
            warm = {row["pipeline"]: row for row in rows if row["epoch"] == 2}
            self.assertEqual(warm["native_b6"]["images_per_second"], 1600.0)
            self.assertAlmostEqual(
                warm["dali"]["warm_throughput_relative_to_native_b6"],
                1100.0 / 1600.0,
            )

    def test_report_rejects_different_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            standard, native = self._fixture(Path(directory), native_gpu_uuid="gpu-2")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                build_report(standard_root=standard, native_run=native)

    def test_interim_report_accepts_dali_before_pytorch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            standard, native = self._fixture(Path(directory))
            report, rows = build_report(
                standard_root=standard,
                native_run=native,
                standard_pipelines=("dali",),
            )
            self.assertEqual(report["included_pipelines"], ["native_b6", "dali"])
            self.assertEqual(len(rows), 4)
            self.assertIn("pending", report["claims"]["dali_vs_pytorch"])


if __name__ == "__main__":
    unittest.main()
