#!/usr/bin/env python3
"""CPU audit tests for the formal four-pipeline training benchmark."""

from __future__ import annotations

import copy
import gc
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks/system_rgbnomore"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from training.artifacts import (  # noqa: E402
    nested_state_sha256,
    repository_provenance,
    sha256_file,
    tensor_state_sha256,
    verify_artifact_hashes,
    write_artifact_hashes,
)
from training.augmentation import (  # noqa: E402
    derive_augmentation,
    horizontal_flip_dct,
)
from training.metrics import gradient_summary, tensor_is_finite  # noqa: E402
from training.model_factory import (  # noqa: E402
    EXPECTED_PARAMETER_COUNTS,
    build_model,
    capture_rng_state,
    capture_training_state,
    initialize_model,
    model_configuration,
    reset_training_state,
    seed_everything,
)
from training.optimizer import build_optimizer, build_scheduler  # noqa: E402
from training.pipeline import (  # noqa: E402
    DaliTrainingAdapter,
    GalpTrainingAdapter,
    PyTorchTrainingAdapter,
    OrderedAsyncPrefetchQueue,
    RgbNoMoreTrainingAdapter,
    TrainingBatch,
    TrainingSample,
    _uniform_dali_tensor,
    load_training_manifest,
    validate_dataset_separation,
)
from training.run import (  # noqa: E402
    _aggregate_step_repeats,
    _collect_batches,
    _parse_args,
    _load_resume,
    _first_step_probe_policy,
    _resolve_runtime_record,
    _SyncLedger,
    _sync,
    _semantic_compare,
    _train_one_step,
    _train_one_step_runtime,
    _validate_args,
    run,
)
from training.sample_order import (  # noqa: E402
    SampleIdentity,
    SampleOrderLedger,
    canonical_epoch_order,
)
from training.schema import (  # noqa: E402
    TRAINING_CONTRACT_SCHEMA,
    empty_status,
    validate_required_group_coverage,
    validate_training_document,
)
from training.validate import validate_output  # noqa: E402


RGBNOMORE_ROOT = Path("/home/tangyuxin/RGB-no-more")


def _optimizer_config() -> dict[str, object]:
    return {
        "type": "sgd",
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "momentum": 0.0,
        "nesterov": False,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
    }


def _write_jpeg(path: Path, value: int) -> None:
    pixels = np.full((256, 256, 3), value, dtype=np.uint8)
    pixels[32:224, 48:208, :] = (value + 37) % 255
    Image.fromarray(pixels, mode="RGB").save(path, quality=95, subsampling=2)


def _samples(root: Path, count: int = 4) -> list[TrainingSample]:
    result = []
    for index in range(count):
        path = root / f"sample_{index}.jpg"
        _write_jpeg(path, 20 + index * 30)
        result.append(TrainingSample(f"id-{index}", path, index, 256, 256, index))
    return result


class _OneBatchAdapter:
    def __init__(self, batch: TrainingBatch) -> None:
        self.batch = batch

    def next_batch(self) -> TrainingBatch:
        return self.batch


class _FakeAsyncHandle:
    def __init__(self, value: object = None, *, ready: bool = True, error: Exception | None = None) -> None:
        self.value = value
        self.ready = ready
        self.started = True
        self.producer_active_ms = 2.0
        self.error = error
        self.read_count = 0
        self.cancel_count = 0

    def read(self):
        self.read_count += 1
        if self.error is not None:
            raise self.error
        return self.value

    def cancel(self) -> bool:
        self.cancel_count += 1
        return False


class _FakeDaliTensorList:
    class _Tensor:
        def shape(self):
            return [np.int64(2), np.int32(3), 4, 4]

    def __init__(self) -> None:
        self.tensor = self._Tensor()

    def as_tensor(self):
        return self.tensor


class TrainingBenchmarkTest(unittest.TestCase):
    def test_dali_tensor_list_is_collapsed_before_torch_allocation(self) -> None:
        tensor_list = _FakeDaliTensorList()
        tensor, shape = _uniform_dali_tensor(tensor_list)
        self.assertIs(tensor, tensor_list.tensor)
        self.assertEqual(shape, [np.int64(2), np.int32(3), 4, 4])
        self.assertEqual(tuple(torch.empty(shape).shape), (2, 3, 4, 4))

    def test_execution_mode_cli_default_and_explicit_runtime(self) -> None:
        default = _parse_args([])
        runtime = _parse_args(["--execution-mode", "runtime"])
        self.assertEqual(default.execution_mode, "audit")
        self.assertFalse(default.execution_mode_explicit)
        self.assertEqual(runtime.execution_mode, "runtime")
        self.assertTrue(runtime.execution_mode_explicit)
        self.assertEqual(_first_step_probe_policy("audit"), "audit")
        self.assertEqual(_first_step_probe_policy("runtime"), "audit")

    def test_sync_ledger_records_every_explicit_reason(self) -> None:
        ledger = _SyncLedger()
        device = SimpleNamespace(type="cuda")
        with mock.patch("training.run.torch.cuda.synchronize") as synchronize:
            _sync(device, ledger, "measured_region_start")
            _sync(device, ledger, "measured_region_end")
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(
            ledger.as_dict(),
            {
                "count": 2,
                "reasons": {
                    "measured_region_end": 1,
                    "measured_region_start": 1,
                },
            },
        )

    def test_resume_rejects_cross_execution_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "contract.json").write_text(
                json.dumps({"execution": {"mode": "audit"}}), encoding="utf-8"
            )
            args = _parse_args(
                ["--resume-run", str(root), "--execution-mode", "runtime"]
            )
            with self.assertRaisesRegex(ValueError, "immutable contract mode"):
                _load_resume(args)

    def test_ordered_async_queue_order_backpressure_exception_and_close(self) -> None:
        queue = OrderedAsyncPrefetchQueue(2)
        first = _FakeAsyncHandle("first", ready=True)
        second = _FakeAsyncHandle("second", ready=False)
        queue.submit("meta-1", lambda: first)
        queue.submit("meta-2", lambda: second)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            queue.submit("overflow", lambda: _FakeAsyncHandle())
        self.assertEqual(queue.pop(), ("meta-1", "first"))
        self.assertEqual(queue.pop(), ("meta-2", "second"))
        metrics = queue.metrics()
        self.assertEqual(metrics["queue_hit_batches"], 1)
        self.assertEqual(metrics["queue_miss_batches"], 1)
        self.assertEqual(metrics["backpressure_events"], 1)

        failing = OrderedAsyncPrefetchQueue(2)
        error_handle = _FakeAsyncHandle(error=RuntimeError("producer failed"))
        drain_handle = _FakeAsyncHandle("drain")
        failing.submit("bad", lambda: error_handle)
        failing.submit("pending", lambda: drain_handle)
        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            failing.pop()
        failing.close()
        self.assertEqual(drain_handle.read_count, 1)
        self.assertEqual(drain_handle.cancel_count, 1)
        self.assertTrue(failing.metrics()["closed"])
        self.assertEqual(failing.metrics()["current_queue_depth_batches"], 0)

    def test_runtime_step_avoids_audit_scans_syncs_and_scalar_materialization(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 1000)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        identities = [SampleIdentity(0, 0, "a"), SampleIdentity(0, 1, "b")]
        batch = TrainingBatch(
            inputs=(torch.randn(2, 3, 224, 224),),
            labels=torch.tensor([1, 2]),
            identities=identities,
            augmentations=[{}, {}],
            on_device=False,
        )
        before = model[-1].weight.detach().clone()
        with (
            mock.patch("training.run.gradient_summary") as gradient_scan,
            mock.patch("training.run.parameter_update_summary") as parameter_scan,
            mock.patch("training.run._sync") as explicit_sync,
        ):
            record, emitted = _train_one_step_runtime(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                adapter=_OneBatchAdapter(batch),
                expected=identities,
                domain="rgb",
                device=torch.device("cpu"),
                label_smoothing=0.0,
                gradient_clipping=None,
            )
        gradient_scan.assert_not_called()
        parameter_scan.assert_not_called()
        explicit_sync.assert_not_called()
        self.assertIs(emitted, batch)
        self.assertEqual(record["deep_parameter_scans"], 0)
        self.assertEqual(record["host_scalar_materializations_in_step"], 0)
        self.assertIn("loss_tensor", record)
        _resolve_runtime_record(record)
        self.assertFalse(torch.equal(before, model[-1].weight.detach()))

    def test_audit_and_runtime_use_identical_first_step_probe_outputs(self) -> None:
        torch.manual_seed(37)
        source = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 1000)
        )
        initial = copy.deepcopy(source.state_dict())
        identities = [SampleIdentity(0, 0, "probe")]
        inputs = torch.randn(1, 3, 224, 224)
        outputs = []
        for mode in ("audit", "runtime"):
            self.assertEqual(_first_step_probe_policy(mode), "audit")
            model = copy.deepcopy(source)
            model.load_state_dict(initial)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda _step: 1.0
            )
            batch = TrainingBatch(
                inputs=(inputs.clone(),),
                labels=torch.tensor([7]),
                identities=identities,
                augmentations=[{"augmentation_key": "same"}],
                on_device=False,
            )
            record, *_ = _train_one_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                adapter=_OneBatchAdapter(batch),
                expected=identities,
                domain="rgb",
                device=torch.device("cpu"),
                label_smoothing=0.0,
                gradient_clipping=None,
                collect_numerics=True,
            )
            outputs.append(
                (
                    record["loss"],
                    record["gradients"]["global_l2_norm"],
                    record["parameter_update"]["global_l2_norm"],
                    tensor_state_sha256(model.state_dict()),
                )
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_galp_zero_workers_has_explicit_failure_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.bin"
            manifest.write_bytes(b"fixture")
            args = _parse_args(
                [
                    "--pipeline",
                    "galp",
                    "--train-manifest",
                    str(root / "train.json"),
                    "--val-manifest",
                    str(root / "val.json"),
                    "--galp-manifest",
                    str(manifest),
                    "--output-dir",
                    str(root / "out"),
                    "--workers",
                    "0",
                    "--device",
                    "cpu",
                    "--dry-run-contract",
                ]
            )
            with self.assertRaisesRegex(ValueError, "rowgroup-prefetch workers"):
                _validate_args(args)
    def test_formal_rgb_and_dct_models_execute_real_optimizer_steps(self) -> None:
        if not RGBNOMORE_ROOT.is_dir():
            self.skipTest("external RGB-no-more checkout is unavailable")
        seed_everything(7)
        for domain in ("rgb", "dct"):
            model = build_model(RGBNOMORE_ROOT, domain, torch.device("cpu"))
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                EXPECTED_PARAMETER_COUNTS[domain],
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            before = next(model.parameters()).detach().clone()
            if domain == "rgb":
                logits = model(torch.randn(1, 3, 224, 224))
            else:
                logits = model(
                    torch.randn(1, 1, 28, 28, 8, 8),
                    torch.randn(1, 2, 14, 14, 8, 8),
                )
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([3]))
            loss.backward()
            summary = gradient_summary(model)
            optimizer.step()
            self.assertEqual(tuple(logits.shape), (1, 1000))
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(summary["finite"])
            self.assertTrue(summary["nonzero"])
            self.assertFalse(torch.equal(before, next(model.parameters()).detach()))
            del model, optimizer, logits, loss
            gc.collect()

    def test_tiny_training_step_checks_loss_gradient_and_update(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 1000)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        identities = [SampleIdentity(0, 0, "a"), SampleIdentity(0, 1, "b")]
        batch = TrainingBatch(
            inputs=(torch.randn(2, 3, 224, 224),),
            labels=torch.tensor([1, 2]),
            identities=identities,
            augmentations=[{}, {}],
            on_device=False,
        )
        record, *_rest = _train_one_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            adapter=_OneBatchAdapter(batch),
            expected=identities,
            domain="rgb",
            device=torch.device("cpu"),
            label_smoothing=0.0,
            gradient_clipping=None,
            collect_numerics=True,
        )
        self.assertTrue(record["loss_finite"])
        self.assertTrue(record["gradients"]["finite"])
        self.assertTrue(record["gradients"]["nonzero"])
        self.assertTrue(record["parameter_update"]["changed"])
        self.assertFalse(record["batch_failures"])

    def test_optimizer_scheduler_and_rng_reset_are_exact(self) -> None:
        seed_everything(19)
        model = torch.nn.Linear(4, 2)
        optimizer, groups = build_optimizer(model, _optimizer_config())
        scheduler = build_scheduler(optimizer, {"type": "cosine", "warmup_steps": 1}, total_steps=4)
        initial = capture_training_state(model, optimizer, scheduler)
        initial.update(sample_order_cursor={"epoch": 0, "position": -1}, augmentation_state={"stateless_keyed": True})
        expected_after_reset = torch.rand(4)
        torch.nn.functional.cross_entropy(model(torch.randn(2, 4)), torch.tensor([0, 1])).backward()
        optimizer.step()
        scheduler.step()
        reset_training_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=None, initial=initial)
        self.assertEqual(tensor_state_sha256(model.state_dict()), tensor_state_sha256(initial["model"]))
        self.assertEqual(nested_state_sha256(optimizer.state_dict()), nested_state_sha256(initial["optimizer"]))
        self.assertEqual(nested_state_sha256(scheduler.state_dict()), nested_state_sha256(initial["scheduler"]))
        torch.testing.assert_close(torch.rand(4), expected_after_reset)
        self.assertEqual([group["stable_id"] for group in groups], ["decay", "no_decay"])

    def test_weight_checkpoint_load_is_strict_and_full_checkpoint_does_not_degrade(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.pt"
            source = torch.nn.Linear(4, 2)
            torch.save(
                {
                    "model_architecture": "rgbnomore-vitti-v1",
                    "model_domain": "rgb",
                    "model_state_dict": source.state_dict(),
                },
                path,
            )
            target = torch.nn.Linear(4, 2)
            provenance = initialize_model(
                target, init_mode="weights", checkpoint=path, domain="rgb"
            )
            self.assertTrue(provenance["checkpoint"]["strict_load"])
            self.assertEqual(provenance["checkpoint"]["missing_keys"], [])
            self.assertEqual(tensor_state_sha256(target.state_dict()), tensor_state_sha256(source.state_dict()))
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                initialize_model(
                    torch.nn.Linear(4, 2),
                    init_mode="full-checkpoint",
                    checkpoint=path,
                    domain="rgb",
                )
            optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
            full_path = Path(temporary) / "full.pt"
            torch.save(
                {
                    "model_architecture": "rgbnomore-vitti-v1",
                    "model_domain": "rgb",
                    "model_configuration": model_configuration("rgb"),
                    "optimizer_configuration": {"type": "sgd"},
                    "scheduler_configuration": {"type": "constant"},
                    "scheduler_total_steps": 10,
                    "model_state_dict": source.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": None,
                    "global_step": 8,
                    "epoch": 2,
                    "rng_state": capture_rng_state(),
                    "augmentation_state": {"stateless_keyed": True},
                    "sample_order_cursor": {
                        "epoch": 2,
                        "position": 3,
                        "logical_sample_id": "fixture",
                    },
                },
                full_path,
            )
            full_provenance = initialize_model(
                torch.nn.Linear(4, 2),
                init_mode="full-checkpoint",
                checkpoint=full_path,
                domain="rgb",
            )
            self.assertEqual(full_provenance["convergence_classification"], "resumed_training")
            self.assertEqual(full_provenance["_full_checkpoint_payload"]["global_step"], 8)
            with self.assertRaises(RuntimeError):
                initialize_model(
                    torch.nn.Linear(5, 2), init_mode="weights", checkpoint=path, domain="rgb"
                )

    def test_pytorch_and_rgbnomore_adapters_use_one_planned_interface(self) -> None:
        if not RGBNOMORE_ROOT.is_dir():
            self.skipTest("external RGB-no-more checkout is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            samples = _samples(Path(temporary), 2)
            identities = [SampleIdentity(0, index, sample.logical_sample_id) for index, sample in enumerate(samples)]
            for adapter_class, domain, expected_shapes in (
                (PyTorchTrainingAdapter, "rgb", ((2, 3, 224, 224),)),
                (RgbNoMoreTrainingAdapter, "dct", ((2, 1, 28, 28, 8, 8), (2, 2, 14, 14, 8, 8))),
            ):
                decisions = [
                    derive_augmentation(
                        seed=5,
                        epoch=0,
                        logical_sample_id=sample.logical_sample_id,
                        source_width=sample.width,
                        source_height=sample.height,
                        domain=domain,
                    )
                    for sample in samples
                ]
                adapter = adapter_class(
                    samples,
                    batch_size=2,
                    workers=0,
                    device=torch.device("cpu"),
                    config={"rgbnomore_root": str(RGBNOMORE_ROOT), "prefetch_depth": 0},
                )
                adapter.begin(identities, decisions, [2])
                batch = adapter.next_batch()
                self.assertEqual([tuple(value.shape) for value in batch.inputs], list(expected_shapes))
                self.assertEqual(batch.identities, identities)
                self.assertTrue(all(value.dtype == torch.float32 for value in batch.inputs))
                self.assertTrue(adapter.prefetched_read_identities())
                adapter.close()

    def test_gpu_only_adapters_reject_cpu_as_environment_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            samples = _samples(Path(temporary), 1)
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                GalpTrainingAdapter(
                    samples,
                    batch_size=1,
                    workers=0,
                    device=torch.device("cpu"),
                    config={},
                )
            dali = DaliTrainingAdapter(
                samples,
                batch_size=1,
                workers=0,
                device=torch.device("cpu"),
                config={},
            )
            decision = derive_augmentation(
                seed=1,
                epoch=0,
                logical_sample_id="id-0",
                source_width=256,
                source_height=256,
                domain="rgb",
            )
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                dali.begin([SampleIdentity(0, 0, "id-0")], [decision], [1])

    def test_augmentation_is_keyed_worker_independent_and_dct_aligned(self) -> None:
        first = derive_augmentation(
            seed=9, epoch=2, logical_sample_id="x", source_width=320, source_height=288, domain="dct"
        )
        again = derive_augmentation(
            seed=9, epoch=2, logical_sample_id="x", source_width=320, source_height=288, domain="dct"
        )
        changed = derive_augmentation(
            seed=9, epoch=3, logical_sample_id="x", source_width=320, source_height=288, domain="dct"
        )
        self.assertEqual(first, again)
        self.assertNotEqual(first.augmentation_key, changed.augmentation_key)
        for value in (first.crop_x, first.crop_y, first.crop_width, first.crop_height):
            self.assertEqual(value % 16, 0)
        descriptor = first.native_dct_descriptor()
        self.assertEqual(descriptor["crop"]["unit"], "source_pixels")
        self.assertEqual(descriptor["augmentation_key"], first.augmentation_key)

    def test_dct_augmentation_rejects_sources_smaller_than_alignment(self) -> None:
        for width, height in ((15, 32), (32, 15), (8, 8)):
            with self.assertRaisesRegex(ValueError, "at least one alignment unit"):
                derive_augmentation(
                    seed=9,
                    epoch=0,
                    logical_sample_id=f"{width}x{height}",
                    source_width=width,
                    source_height=height,
                    domain="dct",
                )

    def test_dct_horizontal_flip_uses_block_reverse_and_odd_u_sign(self) -> None:
        y = torch.arange(1 * 1 * 2 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 1, 2, 3, 8, 8)
        cbcr = y.repeat(1, 2, 1, 1, 1, 1)
        actual_y, actual_cbcr = horizontal_flip_dct(y, cbcr)
        expected = y.flip(-3)
        expected[..., 1::2] *= -1
        torch.testing.assert_close(actual_y, expected)
        torch.testing.assert_close(actual_cbcr[:, 0:1], expected)

    def test_order_cursor_drop_last_prefetch_and_duplicate_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            samples = _samples(Path(temporary), 5)
            batches, dropped = _collect_batches(
                samples, seed=11, batch_size=2, batch_count=3, drop_last=True
            )
            self.assertEqual([batch[0].epoch for batch in batches], [0, 0, 1])
            self.assertEqual(dropped, {0: 1})
            cursor = batches[0][-1].as_dict()
            resumed, _ = _collect_batches(
                samples, seed=11, batch_size=2, batch_count=1, drop_last=True, start_cursor=cursor
            )
            self.assertEqual(resumed[0], batches[1])
            ledger = SampleOrderLedger("pytorch")
            expected = batches[0]
            ledger.record_requested(expected + batches[1])
            ledger.record_prefetched(expected + batches[1])
            ledger.record_emitted(expected)
            ledger.record_consumed(expected)
            self.assertTrue(ledger.validate(expected)["ok"])
            self.assertEqual(ledger.validate(expected)["prefetch_overrun"], 2)
            ledger.record_consumed([expected[0]])
            result = ledger.validate(expected)
            self.assertFalse(result["ok"])
            self.assertTrue(result["duplicate_consumed"])
            missing = SampleOrderLedger("pytorch")
            missing.record_requested(expected)
            missing.record_prefetched(expected)
            missing.record_emitted(expected[:1])
            missing.record_consumed(expected[:1])
            missing_result = missing.validate(expected)
            self.assertFalse(missing_result["ok"])
            self.assertTrue(
                any("canonical order" in value for value in missing_result["failures"])
            )
            with self.assertRaisesRegex(ValueError, "cannot produce a batch"):
                _collect_batches(samples[:1], seed=1, batch_size=2, batch_count=1, drop_last=True)

    def test_manifest_labels_hashes_and_train_validation_separation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = _samples(root, 2)
            manifest = root / "train.json"
            manifest.write_text(
                json.dumps(
                    {
                        "split": "train",
                        "samples": [
                            {
                                "logical_sample_id": sample.logical_sample_id,
                                "path": sample.path.name,
                                "label": sample.label,
                                "width": 256,
                                "height": 256,
                                "payload_sha256": sha256_file(sample.path),
                            }
                            for sample in samples
                        ],
                    }
                ),
                encoding="utf-8",
            )
            loaded, metadata = load_training_manifest(manifest, root=None, expected_split="train")
            self.assertEqual(len(loaded), 2)
            self.assertEqual(metadata["sample_count"], 2)
            self.assertFalse(validate_dataset_separation(loaded, loaded)["ok"])
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["samples"][0]["label"] = 1000
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid ImageNet-1K label"):
                load_training_manifest(manifest, root=None, expected_split="train")

    def test_semantic_gates_reject_nan_and_gate_first_update(self) -> None:
        identity = [SampleIdentity(0, 0, "a")]
        base = {
            "initial_hash": "same",
            "initial_optimizer_hash": "same-optimizer",
            "initial_scheduler_hash": "same-scheduler",
            "model_configuration": {"architecture": "fixture"},
            "optimizer_configuration_sha256": "same-optimizer-config",
            "sample_ids": identity,
            "augmentation_decisions": [{"key": "same"}],
            "labels": torch.tensor([0]),
            "inputs": [torch.ones(1, 3)],
            "logits": torch.tensor([[1.0, 0.0]]),
            "loss": 0.5,
            "gradients": {"w": torch.ones(3)},
            "updates": {"w": torch.ones(3)},
        }
        gates = {
            "dct_input_atol": 1e-5,
            "dct_rtol": 1e-4,
            "rgb_input_warning_atol": 0.1,
            "rgb_input_failure_atol": 0.5,
            "gradient_cosine_dct": 0.999,
            "gradient_cosine_rgb": 0.99,
        }
        self.assertEqual(_semantic_compare("rgb", "a", base, "b", copy.deepcopy(base), gates)["status"], "passed")
        invalid = copy.deepcopy(base)
        invalid["inputs"][0][0, 0] = torch.nan
        result = _semantic_compare("rgb", "a", base, "b", invalid, gates)
        self.assertEqual(result["status"], "failed")
        opposite = copy.deepcopy(base)
        opposite["updates"]["w"] *= -1
        result = _semantic_compare("rgb", "a", base, "b", opposite, gates)
        self.assertTrue(any("update cosine" in value for value in result["failures"]))
        nan_loss = copy.deepcopy(base)
        nan_loss["loss"] = float("nan")
        result = _semantic_compare("rgb", "a", base, "b", nan_loss, gates)
        self.assertTrue(any("loss contains NaN" in value for value in result["failures"]))
        self.assertFalse(tensor_is_finite(torch.tensor(float("nan"))))

    def test_numerical_and_performance_statuses_are_independent(self) -> None:
        model = torch.nn.Linear(2, 1)
        model.weight.grad = torch.zeros_like(model.weight)
        model.bias.grad = torch.zeros_like(model.bias)
        summary = gradient_summary(model)
        self.assertTrue(summary["finite"])
        self.assertFalse(summary["nonzero"])
        model.bias.grad = torch.full_like(model.bias, float("inf"))
        self.assertFalse(gradient_summary(model)["finite"])
        repeats = [
            {"repeat": index, "throughput_images_per_s": value}
            for index, value in enumerate((100.0, 100.0, 200.0, 100.0, 200.0))
        ]
        aggregate = _aggregate_step_repeats(repeats, 0.01)
        self.assertEqual(aggregate["included_repeats"], [1, 2, 3, 4])
        self.assertEqual(aggregate["performance_status"], "failed")
        self.assertNotIn("correctness", aggregate)

    def test_schema_group_coverage_and_not_run_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing pipelines"):
            validate_required_group_coverage(["pytorch"], ["rgb"])
        self.assertTrue(all(value == "not_run" for value in empty_status("not_run").values()))
        minimal = {"schema_version": TRAINING_CONTRACT_SCHEMA}
        self.assertIn("contract missing model", validate_training_document(minimal, TRAINING_CONTRACT_SCHEMA))

    def test_artifact_hashes_and_dirty_source_provenance_are_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime.py"
            runtime.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Training Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "runtime.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
            (root / "unrelated.bin").write_bytes(b"user data")
            provenance = repository_provenance(root, [runtime])
            self.assertTrue(provenance["dirty"])
            self.assertFalse(provenance["untracked_runtime_files"])
            output = root / "artifacts"
            output.mkdir()
            (output / "result.json").write_text("{}\n", encoding="utf-8")
            hashes = write_artifact_hashes(output)
            self.assertFalse(verify_artifact_hashes(output, hashes))
            (output / "result.json").write_text("tampered\n", encoding="utf-8")
            self.assertTrue(verify_artifact_hashes(output, hashes))

    def test_cpu_dry_run_emits_single_pipeline_contract_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = _samples(root, 2)
            val_root = root / "val"
            val_root.mkdir()
            val = _samples(val_root, 1)
            manifests = []
            for name, split, values in (("train", "train", train), ("val", "val", val)):
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "split": split,
                            "samples": [
                                {
                                    "logical_sample_id": f"{split}-{sample.logical_sample_id}",
                                    "path": str(sample.path),
                                    "label": sample.label,
                                    "width": 256,
                                    "height": 256,
                                }
                                for sample in values
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                manifests.append(path)
            args = _parse_args(
                [
                    "--pipeline",
                    "pytorch",
                    "--phase",
                    "smoke",
                    "--train-manifest",
                    str(manifests[0]),
                    "--val-manifest",
                    str(manifests[1]),
                    "--output-dir",
                    str(root / "out"),
                    "--device",
                    "cpu",
                    "--batch-size",
                    "2",
                    "--workers",
                    "0",
                    "--execution-mode",
                    "runtime",
                    "--dry-run-contract",
                ]
            )
            contract = run(args)
            self.assertEqual(contract["enabled_pipelines"], ["pytorch"])
            self.assertEqual(contract["model"]["architecture"], "rgbnomore-vitti-v1")
            self.assertEqual(contract["execution"]["mode"], "runtime")
            self.assertEqual(
                contract["execution"]["measurement_policy"]["selected"], "runtime"
            )
            self.assertFalse(contract["claims"]["cross_dct_rgb_tensor_equivalence"])
            self.assertEqual(contract["validation_augmentation"]["horizontal_flip"], False)

    def test_cpu_single_pipeline_smoke_run_writes_validated_artifacts(self) -> None:
        if not RGBNOMORE_ROOT.is_dir():
            self.skipTest("external RGB-no-more checkout is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = _samples(root, 2)
            val_root = root / "val"
            val_root.mkdir()
            val = _samples(val_root, 1)
            manifests = []
            for name, split, values in (("train", "train", train), ("val", "val", val)):
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "split": split,
                            "samples": [
                                {
                                    "logical_sample_id": f"{split}-{sample.logical_sample_id}",
                                    "path": str(sample.path),
                                    "label": sample.label,
                                    "width": 256,
                                    "height": 256,
                                }
                                for sample in values
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                manifests.append(path)
            output = root / "run"
            args = _parse_args(
                [
                    "--pipeline",
                    "pytorch",
                    "--phase",
                    "smoke",
                    "--train-manifest",
                    str(manifests[0]),
                    "--val-manifest",
                    str(manifests[1]),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                    "--batch-size",
                    "2",
                    "--workers",
                    "0",
                    "--prefetch-depth",
                    "0",
                    "--execution-mode",
                    "runtime",
                    "--warmup-steps",
                    "1",
                    "--measured-steps",
                    "1",
                ]
            )
            result = run(args)
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            pipeline = json.loads((output / "pipeline_pytorch.json").read_text(encoding="utf-8"))
            semantic = json.loads((output / "semantic_comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(validation["ok"], validation["failures"])
            self.assertEqual(result["pipeline_status"]["galp"]["overall"], "not_run")
            self.assertEqual(pipeline["status"]["correctness"], "passed")
            self.assertEqual(pipeline["execution_mode"], "runtime")
            self.assertEqual(semantic["rgb"]["status"], "not_run")
            self.assertEqual(semantic["dct"]["status"], "not_run")
            self.assertEqual(pipeline["phase_results"]["smoke"]["repeats"][0]["optimizer_steps"], 2)
            repeat = pipeline["phase_results"]["smoke"]["repeats"][0]
            self.assertEqual(
                repeat["measurement_instrumentation"][
                    "deep_gradient_scans_in_measured_path"
                ],
                0,
            )
            self.assertEqual(
                repeat["synchronization_accounting"]["explicit_host_device"]["count"],
                0,
            )
            self.assertTrue((output / "artifact_hashes.json").is_file())
            contract_before = (output / "contract.json").read_bytes()
            initial_path = output / "initial_state_rgb_seed11997733.pt"
            initial_hash_before = sha256_file(initial_path)
            resumed = run(_parse_args(["--resume-run", str(output)]))
            validation_after = json.loads(
                (output / "validation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(validation_after["ok"], validation_after["failures"])
            self.assertEqual(resumed["contract_sha256"], result["contract_sha256"])
            self.assertEqual((output / "contract.json").read_bytes(), contract_before)
            self.assertEqual(sha256_file(initial_path), initial_hash_before)
            (output / "commands.json").write_text("{}\n", encoding="utf-8")
            tampered = validate_output(output, write_result=False)
            self.assertFalse(tampered["ok"])
            self.assertTrue(
                any("artifact hash mismatch" in value for value in tampered["failures"])
            )

    @unittest.skipUnless(torch.cuda.is_available(), "real GPU training smoke not executed: CUDA unavailable")
    def test_real_gpu_formal_model_step(self) -> None:
        model = build_model(RGBNOMORE_ROOT, "rgb", torch.device("cuda:0"))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        loss = torch.nn.functional.cross_entropy(
            model(torch.randn(1, 3, 224, 224, device="cuda:0")),
            torch.tensor([1], device="cuda:0"),
        )
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
