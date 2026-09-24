#!/usr/bin/env python3
"""Tests for the equal-image RGB/native performance benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch


from galp.benchmarks.training_pls.published_optimizer import (
    build_published_optimizer,
)
from galp.benchmarks.training_pls.report_equal_image_performance import (
    build_report,
)
from galp.benchmarks.training_pls.contracts import (
    condition_identity_hash,
)
from galp.benchmarks.training_pls.recipe import sha256_json
from galp.benchmarks.system_rgbnomore.training.equal_image_epoch_benchmark import (
    DALI_NATIVE_DECISION_DIGEST,
    GRADIENT_ACCUMULATION,
    MICROBATCH_IMAGES,
    MetricsWriter,
    _build_training_augmentation_plan,
    _comparison_scope,
    _evaluate,
    _parse_pipelines,
    _profiling_skips_validation,
    _record_pipeline_failure,
    _restore_checkpoint,
    _save_checkpoint,
    _train_epoch,
    _validation_skipped_record,
    batch_lengths,
    schedule_summary,
)
from galp.benchmarks.system_rgbnomore.training.pipeline import (
    TrainingBatch,
    TrainingSample,
    resolve_dali_variant,
)
from galp.benchmarks.system_rgbnomore.training.sample_order import SampleIdentity
from galp.benchmarks.training_audit_policy import (
    TrainingAuditState,
    audit_cursor,
    build_audit_policy,
    decision_for_next_update,
    epoch_audit_summary,
    validate_audit_cursor,
)
from galp.benchmarks.model_only_training_calibration import (
    run_model_only_calibration,
)


class EqualImageScheduleTest(unittest.TestCase):
    @staticmethod
    def _samples() -> list[TrainingSample]:
        return [
            TrainingSample(
                logical_sample_id=f"sample-{index}",
                path=Path(f"/not-read-by-this-test/sample-{index}.jpg"),
                label=index,
                width=320 + index,
                height=240 + index,
            )
            for index in range(2)
        ]

    @staticmethod
    def _augmentation_contract() -> dict:
        return {
            "benchmark": "unit-test",
            "contract_hash": "contract",
            "training": {"precision": "fp32"},
            "audit_policy": build_audit_policy(),
            "dali_variants": {
                variant: resolve_dali_variant(variant)
                for variant in ("d2", "d3")
            },
        }

    def test_training_augmentation_plan_is_shared_and_streaming_digest_compatible(
        self,
    ) -> None:
        samples = self._samples()
        by_id = {sample.logical_sample_id: sample for sample in samples}
        identities = [
            SampleIdentity(0, index, sample.logical_sample_id)
            for index, sample in enumerate(samples)
        ]
        contract = self._augmentation_contract()

        d2 = _build_training_augmentation_plan(
            pipeline="d2",
            contract=contract,
            identities=identities,
            samples_by_id=by_id,
            seed=17,
            epoch=0,
        )
        pytorch = _build_training_augmentation_plan(
            pipeline="pytorch",
            contract=contract,
            identities=identities,
            samples_by_id=by_id,
            seed=17,
            epoch=0,
        )
        d3 = _build_training_augmentation_plan(
            pipeline="d3",
            contract=contract,
            identities=identities,
            samples_by_id=by_id,
            seed=17,
            epoch=0,
        )

        self.assertEqual(d2.mode, "planned")
        self.assertEqual(
            [decision.as_dict() for decision in d2.decisions],
            [decision.as_dict() for decision in pytorch.decisions],
        )
        self.assertEqual(d2.decision_digest, pytorch.decision_digest)
        self.assertEqual(
            d2.decision_digest,
            sha256_json([decision.as_dict() for decision in d2.decisions]),
        )
        self.assertEqual(d3.mode, "dali-native")
        self.assertEqual(d3.decisions, ())
        self.assertEqual(d3.decision_digest, DALI_NATIVE_DECISION_DIGEST)

    def test_training_augmentation_plan_rejects_variant_semantic_drift(self) -> None:
        samples = self._samples()
        identity = SampleIdentity(0, 0, samples[0].logical_sample_id)
        contract = self._augmentation_contract()
        contract["dali_variants"]["d2"]["augmentation_mode"] = "native"
        with self.assertRaisesRegex(ValueError, "augmentation_mode"):
            _build_training_augmentation_plan(
                pipeline="d2",
                contract=contract,
                identities=[identity],
                samples_by_id={samples[0].logical_sample_id: samples[0]},
                seed=17,
                epoch=0,
            )

    def test_validation_runs_before_training_for_every_rgb_pipeline(self) -> None:
        class ValidationAdapter:
            def begin(self, identities, decisions, lengths) -> None:
                self.identities = list(identities)
                self.decisions = list(decisions)
                self.lengths = list(lengths)
                self.cursor = 0

            def next_batch(self) -> TrainingBatch:
                length = self.lengths.pop(0)
                identities = self.identities[self.cursor : self.cursor + length]
                self.cursor += length
                return TrainingBatch(
                    inputs=(torch.ones(length, 2),),
                    labels=torch.zeros(length, dtype=torch.long),
                    identities=identities,
                    augmentations=[],
                    on_device=True,
                )

            def close(self) -> None:
                pass

        samples = self._samples()
        contract = self._augmentation_contract()
        for pipeline in ("d2", "d3", "pytorch"):
            adapter = ValidationAdapter()
            model = torch.nn.Linear(2, 5)
            with self.subTest(pipeline=pipeline), mock.patch(
                "galp.benchmarks.system_rgbnomore.training."
                "equal_image_epoch_benchmark.build_training_adapter",
                return_value=adapter,
            ):
                record = _evaluate(
                    pipeline=pipeline,
                    execution_model=model,
                    samples=samples,
                    workers=0,
                    device=torch.device("cpu"),
                    contract=contract,
                    seed=17,
                    epoch=0,
                    optimizer_update=0,
                    processed_images=0,
                )
            self.assertEqual(record["validation_samples"], len(samples))
            self.assertEqual(len(adapter.decisions), len(samples))

    def test_training_epoch_consumes_plan_for_every_rgb_pipeline(self) -> None:
        class TrainingAdapter:
            def __init__(self, preserves_order: bool) -> None:
                self._preserves_order = preserves_order

            def begin(self, identities, decisions, lengths) -> None:
                self.identities = list(identities)
                self.decisions = list(decisions)
                self.lengths = list(lengths)
                self.cursor = 0

            def preserves_canonical_order(self) -> bool:
                return self._preserves_order

            def next_batch(self) -> TrainingBatch:
                length = self.lengths.pop(0)
                identities = self.identities[self.cursor : self.cursor + length]
                self.cursor += length
                return TrainingBatch(
                    inputs=(torch.ones(length, 2),),
                    labels=torch.zeros(length, dtype=torch.long),
                    identities=identities,
                    augmentations=[],
                    on_device=True,
                )

            def loader_metrics(self) -> dict:
                return {}

            def close(self) -> None:
                pass

        samples = self._samples()
        contract = {
            **self._augmentation_contract(),
            "training": {"precision": "fp32", "model_compile": {"mode": "default"}},
            "optimizer": {"gradient_clipping_norm": 1.0},
            "profiling": {"enabled": False},
        }
        records = {}
        for pipeline in ("d2", "d3", "pytorch"):
            model = torch.nn.Sequential(torch.nn.Linear(2, 5))
            optimizer, weight_decayer, scheduler = build_published_optimizer(
                model, warmup_updates=2, total_updates=20
            )
            adapter = TrainingAdapter(preserves_order=pipeline != "d3")
            with self.subTest(pipeline=pipeline), mock.patch(
                "galp.benchmarks.system_rgbnomore.training."
                "equal_image_epoch_benchmark.build_training_adapter",
                return_value=adapter,
            ):
                record, global_update, processed_images = _train_epoch(
                    pipeline=pipeline,
                    epoch=0,
                    seed=17,
                    execution_model=model,
                    model=model,
                    optimizer=optimizer,
                    weight_decayer=weight_decayer,
                    scheduler=scheduler,
                    samples=samples,
                    workers=0,
                    device=torch.device("cpu"),
                    contract=contract,
                    global_update=0,
                    processed_images=0,
                )
            records[pipeline] = record
            self.assertEqual(global_update, 1)
            self.assertEqual(processed_images, len(samples))
            self.assertEqual(record["epoch_samples"], len(samples))

        self.assertEqual(
            records["d2"]["augmentation_decision_sha256"],
            records["pytorch"]["augmentation_decision_sha256"],
        )
        self.assertEqual(records["d2"]["augmentation_decision_mode"], "planned")
        self.assertEqual(
            records["d3"]["augmentation_decision_sha256"],
            DALI_NATIVE_DECISION_DIGEST,
        )
        self.assertEqual(
            records["d3"]["augmentation_decision_mode"], "dali-native"
        )

    def test_shared_audit_boundary_is_identical_for_all_four_pipelines(self) -> None:
        policy = build_audit_policy("runtime-first-100")
        for pipeline in ("native_b6", "d2", "d3", "pytorch"):
            decisions = [
                decision_for_next_update(policy, completed)
                for completed in (0, 99, 100)
            ]
            self.assertEqual([value.update for value in decisions], [1, 100, 101])
            self.assertEqual([value.strict for value in decisions], [True, True, False])
            self.assertFalse(decisions[-1].per_microbatch_loss_readback, pipeline)

    def test_legacy_runtime_adapter_accepts_the_declarative_policy(self) -> None:
        state = TrainingAuditState(
            build_audit_policy(),
            completed_updates=100,
            device=torch.device("cpu"),
        )
        self.assertFalse(state.strict_update)
        self.assertEqual(
            state.policy.as_contract()["audit_mode"], "runtime-first-100"
        )
        model = torch.nn.Linear(2, 2)
        logits = model(torch.ones(1, 2))
        loss = logits.sum()
        self.assertEqual(state.observe(loss, logits, 1), 0.0)
        loss.backward()
        state.check_gradients_and_clip(model.parameters(), max_norm=1.0)
        self.assertEqual(state.counters.finite_host_reads, 0)
        self.assertEqual(state.counters.loss_host_reads, 0)

    def test_strict_audit_never_switches_to_deferred(self) -> None:
        policy = build_audit_policy("strict-audit")
        decision = decision_for_next_update(policy, 100_000)
        self.assertTrue(decision.strict)
        self.assertTrue(decision.per_microbatch_loss_readback)
        summary = epoch_audit_summary(policy, start_update=100, end_update=1252)
        self.assertEqual(summary["checked_updates"], 1152)
        self.assertEqual(summary["deferred_updates"], 0)

    def test_audit_policy_does_not_change_optimizer_or_scheduler_state(self) -> None:
        def run(mode: str) -> tuple[dict, dict]:
            torch.manual_seed(7)
            model = torch.nn.Sequential(torch.nn.Linear(2, 2))
            optimizer, decayer, scheduler = build_published_optimizer(
                model, warmup_updates=2, total_updates=20
            )
            policy = build_audit_policy(mode)
            for completed in range(3):
                decision_for_next_update(policy, completed)
                optimizer.zero_grad(set_to_none=True)
                learning_rate = scheduler.prepare_next_update()
                model(torch.ones(1, 2)).sum().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                decayer.step(learning_rate)
                scheduler.complete_update()
            return model.state_dict(), scheduler.state_dict()

        strict_model, strict_scheduler = run("strict-audit")
        runtime_model, runtime_scheduler = run("runtime-first-100")
        for name in strict_model:
            torch.testing.assert_close(strict_model[name], runtime_model[name])
        self.assertEqual(strict_scheduler, runtime_scheduler)

    def test_model_only_calibration_crosses_shared_runtime_boundary_on_cpu(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        optimizer, decayer, scheduler = build_published_optimizer(
            model, warmup_updates=2, total_updates=130
        )
        result = run_model_only_calibration(
            domain="rgb",
            execution_model=model,
            model=model,
            optimizer=optimizer,
            weight_decayer=decayer,
            scheduler=scheduler,
            inputs=(torch.ones(1, 2),),
            labels=torch.zeros(1, dtype=torch.long),
            device=torch.device("cpu"),
            audit_policy=build_audit_policy(),
            microbatch_images=1,
            gradient_accumulation=1,
            gradient_clipping_norm=1.0,
            warmup_updates=1,
            measured_updates=101,
        )
        self.assertEqual(result["measured_optimizer_updates"], 101)
        self.assertEqual(result["measured_microbatches"], 101)
        self.assertGreater(result["images_per_second"], 0.0)
    def test_equal_image_pipeline_ids_are_d2_d3_and_pytorch(self) -> None:
        self.assertEqual(_parse_pipelines("d2,d3,pytorch"), ["d2", "d3", "pytorch"])
        with self.assertRaisesRegex(ValueError, "subset"):
            _parse_pipelines("dali,pytorch")

    def test_dali_variant_semantics_are_frozen(self) -> None:
        d2 = resolve_dali_variant("D2")
        d3 = resolve_dali_variant("d3")
        self.assertEqual(
            (d2["source_mode"], d2["decoder_mode"], d2["augmentation_mode"]),
            ("reader", "roi", "planned"),
        )
        self.assertEqual(
            (d3["source_mode"], d3["decoder_mode"], d3["augmentation_mode"]),
            ("reader", "roi", "native"),
        )

    def test_comparison_scope_separates_d2_and_d3_claims(self) -> None:
        scope = _comparison_scope()
        self.assertIn("d2_vs_pytorch", scope)
        self.assertIn("d3_performance_ceiling", scope)
        self.assertNotIn("dali_vs_pytorch", scope)

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
    def test_pipeline_failure_updates_run_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            status_path = output_dir / "runs" / "d2" / "run_status.json"
            _write_json(
                status_path,
                {
                    "state": "running",
                    "pipeline": "d2",
                    "seed": 17,
                    "completed_epoch": 1,
                    "optimizer_update": 3,
                    "processed_images": 5,
                },
            )
            failure = _record_pipeline_failure(
                output_dir=output_dir,
                pipeline="d2",
                seed=17,
                error=NameError("missing augmentation plan"),
            )
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["completed_epoch"], 1)
            self.assertEqual(status["error_type"], "NameError")
            self.assertEqual(status["error"], "missing augmentation plan")
            self.assertFalse(status["scientific_result"])
            self.assertEqual(failure["pipeline"], "d2")

    def test_profiling_validation_skip_is_explicit_and_epoch_scoped(self) -> None:
        contract = {
            "profiling": {
                "enabled": True,
                "epoch": 2,
                "skip_profiled_epoch_validation": True,
            }
        }
        self.assertFalse(_profiling_skips_validation(contract, 1))
        self.assertTrue(_profiling_skips_validation(contract, 2))
        self.assertFalse(
            _profiling_skips_validation(
                {"profiling": {**contract["profiling"], "enabled": False}}, 2
            )
        )
        record = _validation_skipped_record(
            pipeline="d2",
            seed=11997733,
            epoch=2,
            optimizer_update=2504,
            processed_images=2_562_334,
        )
        self.assertEqual(record["record_type"], "validation_skipped")
        self.assertFalse(record["scientific_result"])

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
                audit_policy=build_audit_policy(),
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
                audit_policy=build_audit_policy(),
            )
            self.assertEqual(restored, (0, 0, 0, None))
            self.assertEqual(metrics.lines, 1)
            self.assertEqual(metrics.records(), [{"epoch": 0, "record_type": "validation"}])
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, initial_state[name]))

    def test_resume_rejects_audit_policy_or_cursor_change(self) -> None:
        policy = build_audit_policy()
        cursor = audit_cursor(policy, 100)
        self.assertEqual(validate_audit_cursor(cursor, policy, completed_updates=100), cursor)
        with self.assertRaisesRegex(ValueError, "audit policy/update cursor"):
            validate_audit_cursor(cursor, policy, completed_updates=101)
        with self.assertRaisesRegex(ValueError, "audit policy/update cursor"):
            validate_audit_cursor(
                cursor, build_audit_policy("strict-audit"), completed_updates=100
            )


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
        "data_preparation_seconds": 1.0,
        "exposed_input_wait_seconds": 2.0,
        "model_forward_backward_optimizer_seconds": None,
        "audit_seconds": 3.0,
        "boundary_sync_seconds": 4.0,
        "pipeline_internal_work_seconds": None,
        "timing_semantics": {"validation_included": False},
        "sample_order": {"emitted_order_sha256": f"order-{epoch}"},
        "augmentation_decision_sha256": f"augmentation-{epoch}",
        "sample_order_digest": f"dct-order-{epoch}",
        "pool_membership_digest": f"dct-pool-{epoch}",
    }


class EqualImageReportTest(unittest.TestCase):
    def _fixture(self, root: Path, native_gpu_uuid: str = "gpu-1") -> tuple[Path, Path]:
        standard = root / "standard"
        native = root / "native"
        audit_policy = build_audit_policy()
        source_identity = {
            "training_source_scope": "galp-training-runtime-v2",
            "training_runtime_source_tree_sha256": "source-hash",
        }
        optimizer = {"type": "adamw"}
        scheduler = {
            "type": "rgbnomore-warmup-then-cosine",
            "warmup_optimizer_updates": 10_000,
        }
        standard_contract = {
                "schema_version": "galp-equal-image-rgb-epoch-contract-v3",
                "benchmark": "equal-image-epoch-aware-rgb-training-v3",
                "model": {"model_id": "vitti"},
                "seed": 11997733,
                "required_gpu_name_substring": "RTX 4090",
                "profiling": {"enabled": False},
                "prefix_schedule": {
                    "processed_images": 2_562_334,
                    "total_microbatches": 40_038,
                    "total_optimizer_updates": 2_504,
                },
                "training": {
                    "microbatch_images": 64,
                    "gradient_accumulation": 16,
                    "precision": "fp32",
                },
                "optimizer": optimizer,
                "scheduler": scheduler,
                "augmentation": {"mixup": False, "randaugment": False},
                "audit_policy": audit_policy,
                "source_identity": source_identity,
            }
        standard_contract["contract_hash"] = sha256_json(standard_contract)
        standard_contract_hash = standard_contract["contract_hash"]
        _write_json(
            standard / "contract.json",
            standard_contract,
        )
        environment = {
            "gpu_name": "NVIDIA GeForce RTX 4090",
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
        native_contract = {
                "schema_version": "galp-pls-condition-contract-v3",
                "run_manifest_schema": "galp-pls-run-manifest-v2",
                "condition_id": "B6",
                "model_id": "vitti",
                "training_seed": 11997733,
                "initial_model_hash": "dct-initial",
                "required_gpu_name_substring": "RTX 4090",
                "audit_policy": audit_policy,
                "code_version": source_identity,
                "microbatch_size": 64,
                "gradient_accumulation": 16,
                "precision": "fp32",
                "optimizer_configuration": optimizer,
                "scheduler": scheduler,
                "crop_policy": "per-pls",
                "order_policy": "closed-pool",
                "mixup": {"alpha": 0.2},
                "randaugment": {"num_operations": 2},
                "execution_mode": "native_physical_pls",
                "physical_fls_observed": True,
                "physical_gpu_pool": True,
                "physical_execution": {"premixed_mapping_sha256": "a" * 64},
            }
        native_contract["condition_hash"] = condition_identity_hash(native_contract)
        native_contract["run_manifest_hash"] = sha256_json(native_contract)
        _write_json(native / "run_manifest.json", native_contract)
        for pipeline, speeds in (
            ("d2", (1000.0, 1100.0)),
            ("d3", (1050.0, 1150.0)),
            ("pytorch", (800.0, 900.0)),
        ):
            _write_json(
                standard / "runs" / pipeline / "final_result.json",
                {
                    "state": "completed",
                    "completed_epoch": 2,
                    "contract_hash": standard_contract_hash,
                    "audit_policy_hash": audit_policy["audit_policy_hash"],
                    "initial_model_hash": "rgb-initial",
                    "model_domain": "rgb",
                    "model_parameter_count": 1_000,
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
        model_only = {
            "images_per_second": 2000.0,
            "milliseconds_per_microbatch": 32.0,
            "model_parameter_count": 1_000,
            "audit_policy_hash": audit_policy["audit_policy_hash"],
            "microbatch_images": 64,
            "gradient_accumulation": 16,
            "precision": "fp32",
            "warmup_optimizer_updates": 5,
            "measured_optimizer_updates": 120,
        }
        _write_json(
            standard / "model_only_rgb.json",
            {
                **model_only,
                "contract_hash": standard_contract_hash,
                "initial_model_hash": "rgb-initial",
            },
        )
        _write_json(
            native / "model_only_dct.json",
            {
                **model_only,
                "condition_hash": native_contract["condition_hash"],
                "initial_model_hash": "dct-initial",
            },
        )
        return standard, native

    def _rgbnomore_fixture(self, root: Path, native: Path) -> Path:
        rgbnomore = root / "rgbnomore"
        native_contract = json.loads(
            (native / "run_manifest.json").read_text(encoding="utf-8")
        )
        contract = {
            key: value
            for key, value in native_contract.items()
            if key not in {
                "condition_hash",
                "run_manifest_hash",
                "physical_execution",
            }
        }
        contract.update(
            execution_mode="standard_rgbnomore_dct",
            semantic_emulation=False,
            physical_fls_observed=False,
            physical_gpu_pool=False,
            backend_implementation="rgbnomore-jpeg-dct-premixed-reference-v2",
            standard_dct_reference={
                "premixed_mapping_sha256": "a" * 64,
            },
        )
        contract["condition_hash"] = condition_identity_hash(contract)
        contract["run_manifest_hash"] = sha256_json(contract)
        _write_json(rgbnomore / "run_manifest.json", contract)
        native_environment = json.loads(
            (native / "environment.json").read_text(encoding="utf-8")
        )
        _write_json(
            rgbnomore / "environment.json",
            {
                **native_environment,
                "execution_mode": "standard_rgbnomore_dct",
                "physical_fls_observed": False,
                "physical_gpu_pool": False,
            },
        )
        _write_json(
            rgbnomore / "run_status.json",
            {"completed_epoch": 2, "seed": 11997733},
        )
        rgbnomore.mkdir(parents=True, exist_ok=True)
        with (rgbnomore / "metrics.jsonl").open("w", encoding="utf-8") as output:
            output.write(json.dumps(_epoch("rgbnomore_dct", 1, 700.0)) + "\n")
            output.write(json.dumps(_epoch("rgbnomore_dct", 2, 750.0)) + "\n")
        native_model_only = json.loads(
            (native / "model_only_dct.json").read_text(encoding="utf-8")
        )
        _write_json(
            rgbnomore / "model_only_dct.json",
            {
                **native_model_only,
                "condition_hash": contract["condition_hash"],
            },
        )
        return rgbnomore

    def test_report_accepts_equal_h100_workloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            standard, native = self._fixture(Path(directory))
            report, rows = build_report(standard_root=standard, native_run=native)
            self.assertEqual(report["hardware"]["uuid"], "gpu-1")
            self.assertEqual(len(rows), 8)
            warm = {row["pipeline"]: row for row in rows if row["epoch"] == 2}
            self.assertEqual(warm["native_b6"]["images_per_second"], 1600.0)
            self.assertAlmostEqual(
                warm["d2"]["warm_throughput_relative_to_native_b6"],
                1100.0 / 1600.0,
            )
            self.assertEqual(report["requested_claim"], "system-level")
            self.assertIn("fairness_matrix", report)

    def test_report_includes_recipe_matched_rgbnomore_dct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standard, native = self._fixture(root)
            rgbnomore = self._rgbnomore_fixture(root, native)
            report, rows = build_report(
                standard_root=standard,
                native_run=native,
                rgbnomore_run=rgbnomore,
            )
            self.assertEqual(len(rows), 10)
            self.assertEqual(
                report["included_pipelines"],
                ["native_b6", "rgbnomore_dct", "d2", "d3", "pytorch"],
            )
            warm = {row["pipeline"]: row for row in rows if row["epoch"] == 2}
            self.assertAlmostEqual(
                warm["rgbnomore_dct"]["warm_throughput_relative_to_native_b6"],
                750.0 / 1600.0,
            )
            self.assertIn(
                "direct recipe-matched",
                report["claims"]["galp_vs_rgbnomore_dct"],
            )

    def test_report_rejects_cross_domain_pipeline_only_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            standard, native = self._fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "pipeline-only claim is invalid"):
                build_report(
                    standard_root=standard,
                    native_run=native,
                    requested_claim="pipeline-only",
                )

    def test_report_rejects_software_or_audit_policy_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            standard, native = self._fixture(Path(directory))
            environment_path = native / "environment.json"
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            environment["torch"] = "different"
            _write_json(environment_path, environment)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                build_report(standard_root=standard, native_run=native)

            standard, native = self._fixture(Path(directory) / "audit")
            condition_path = native / "run_manifest.json"
            condition = json.loads(condition_path.read_text(encoding="utf-8"))
            condition["audit_policy"] = build_audit_policy("strict-audit")
            condition["condition_hash"] = condition_identity_hash(condition)
            condition["run_manifest_hash"] = sha256_json(
                {
                    key: value
                    for key, value in condition.items()
                    if key != "run_manifest_hash"
                }
            )
            _write_json(condition_path, condition)
            with self.assertRaisesRegex(ValueError, "audit policy differs"):
                build_report(standard_root=standard, native_run=native)

    def test_report_rejects_legacy_v2_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            standard, native = self._fixture(Path(directory))
            contract_path = standard / "contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["schema_version"] = "galp-equal-image-rgb-epoch-contract-v2"
            _write_json(contract_path, contract)
            with self.assertRaisesRegex(ValueError, "legacy/unsupported"):
                build_report(standard_root=standard, native_run=native)

    def test_report_rejects_different_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            standard, native = self._fixture(Path(directory), native_gpu_uuid="gpu-2")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                build_report(standard_root=standard, native_run=native)

    def test_report_rejects_profiling_only_rgb_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            standard, native = self._fixture(Path(directory))
            contract_path = standard / "contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["profiling"]["enabled"] = True
            contract["contract_hash"] = sha256_json(
                {key: value for key, value in contract.items() if key != "contract_hash"}
            )
            _write_json(contract_path, contract)
            with self.assertRaisesRegex(ValueError, "profiling-only"):
                build_report(standard_root=standard, native_run=native)

    def test_interim_report_accepts_d2_before_other_rgb_pipelines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            standard, native = self._fixture(Path(directory))
            report, rows = build_report(
                standard_root=standard,
                native_run=native,
                standard_pipelines=("d2",),
            )
            self.assertEqual(report["included_pipelines"], ["native_b6", "d2"])
            self.assertEqual(len(rows), 4)
            self.assertIn("pending", report["claims"]["d2_vs_pytorch"])

    def test_report_accepts_independent_pipeline_root_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            standard, native = self._fixture(Path(directory))
            report, rows = build_report(
                standard_roots={
                    "d2": standard,
                    "d3": standard,
                    "pytorch": standard,
                },
                native_run=native,
            )
            self.assertEqual(len(rows), 8)
            self.assertEqual(report["included_pipelines"], [
                "native_b6", "d2", "d3", "pytorch"
            ])

    def test_report_surfaces_recorded_pipeline_failure_when_result_is_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            standard, native = self._fixture(Path(directory))
            (standard / "runs" / "d2" / "final_result.json").unlink()
            _write_json(
                standard / "runs" / "d2" / "run_status.json",
                {
                    "state": "failed",
                    "completed_epoch": 0,
                    "error_type": "NameError",
                    "error": "name 'augmentation_plan' is not defined",
                },
            )
            _write_json(
                standard / "results.json",
                {
                    "results": [
                        {
                            "state": "failed",
                            "pipeline": "d2",
                            "error_type": "NameError",
                            "error": "name 'augmentation_plan' is not defined",
                        }
                    ]
                },
            )
            with self.assertRaisesRegex(
                ValueError,
                "d2 equal-image run is incomplete.*recorded failure=NameError",
            ):
                build_report(
                    standard_root=standard,
                    native_run=native,
                    standard_pipelines=("d2",),
                )


if __name__ == "__main__":
    unittest.main()
