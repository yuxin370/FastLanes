#!/usr/bin/env python3
"""Run equal-image, epoch-aware D2/D3/PyTorch RGB training on one GPU.

This is the RGB-side companion to the native physical PLS runner.  It fixes
the workload to full ImageNet epochs, microbatch 64, accumulation 16, no
dropped tail, and the published 300-epoch optimizer-update horizon.  Epoch 1
is the cold observation and epoch 2 is the primary warm observation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


HERE = Path(__file__).resolve().parent
FASTLANES_ROOT = HERE.parents[3]

from galp.benchmarks.system_rgbnomore.training.artifacts import (
    file_record,
    sha256_file,
    sha256_json,
    tensor_state_sha256,
)
from galp.benchmarks.system_rgbnomore.training.augmentation import (
    AugmentationDecision,
    derive_augmentation,
)
from galp.benchmarks.system_rgbnomore.training.model_factory import (
    build_model,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)
from galp.benchmarks.system_rgbnomore.training.pipeline import (
    DALI_VARIANTS,
    TrainingBatch,
    TrainingSample,
    build_training_adapter,
    load_training_manifest,
    resolve_dali_variant,
    validate_dataset_separation,
)
from galp.benchmarks.system_rgbnomore.training.sample_order import (
    SampleIdentity,
    canonical_epoch_order,
)

from galp.benchmarks.system_dct_major.training_pls.published_optimizer import (
    build_published_optimizer,
)
from galp.benchmarks.system_dct_major.training_pls.recipe import (
    RECIPE_NAME,
    recipe_contract,
)
from galp.benchmarks.system_dct_major.training_pls.train import (
    compile_published_model,
)


CONTRACT_SCHEMA = "galp-equal-image-rgb-epoch-contract-v2"
CHECKPOINT_SCHEMA = "galp-equal-image-rgb-epoch-checkpoint-v2"
RESULT_SCHEMA = "galp-equal-image-rgb-epoch-result-v2"
PIPELINES = (*DALI_VARIANTS, "pytorch")
MICROBATCH_IMAGES = 64
GRADIENT_ACCUMULATION = 16
REFERENCE_EPOCHS = 300
DEFAULT_PREFIX_EPOCHS = 2
EXPECTED_TRAIN_IMAGES = 1_281_167
EXPECTED_VALIDATION_IMAGES = 50_000


def schedule_summary(
    sample_count: int,
    *,
    epochs: int = DEFAULT_PREFIX_EPOCHS,
    microbatch_images: int = MICROBATCH_IMAGES,
    accumulation: int = GRADIENT_ACCUMULATION,
) -> dict[str, int]:
    if sample_count <= 0 or epochs <= 0:
        raise ValueError("sample_count and epochs must be positive")
    if microbatch_images <= 0 or accumulation <= 0:
        raise ValueError("microbatch and accumulation must be positive")
    microbatches = math.ceil(sample_count / microbatch_images)
    updates = math.ceil(microbatches / accumulation)
    return {
        "sample_count": sample_count,
        "epochs": epochs,
        "microbatch_images": microbatch_images,
        "gradient_accumulation": accumulation,
        "microbatches_per_epoch": microbatches,
        "optimizer_updates_per_epoch": updates,
        "processed_images": sample_count * epochs,
        "total_microbatches": microbatches * epochs,
        "total_optimizer_updates": updates * epochs,
        "tail_microbatch_images": sample_count % microbatch_images
        or microbatch_images,
        "tail_accumulation_microbatches": microbatches % accumulation
        or accumulation,
    }


def batch_lengths(sample_count: int) -> list[int]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return [
        min(MICROBATCH_IMAGES, sample_count - begin)
        for begin in range(0, sample_count, MICROBATCH_IMAGES)
    ]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _retain_checkpoint(latest: Path, permanent: Path) -> None:
    temporary = permanent.with_name(f".{permanent.name}.{os.getpid()}.tmp")
    try:
        try:
            os.link(latest, temporary)
        except OSError:
            shutil.copy2(latest, temporary)
        os.replace(temporary, permanent)
    finally:
        if temporary.exists():
            temporary.unlink()


class MetricsWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        with path.open("rb") as source:
            self.lines = sum(1 for _line in source)

    def append(self, payload: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        with self.path.open("ab") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        self.lines += 1

    def cursor(self) -> dict[str, int]:
        return {"lines": self.lines, "bytes": self.path.stat().st_size}

    def truncate(self, cursor: Mapping[str, Any]) -> None:
        with self.path.open("r+b") as output:
            output.truncate(int(cursor["bytes"]))
        self.lines = int(cursor["lines"])

    def records(self) -> list[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]


def _manifest_header(path: Path, expected_split: str) -> dict[str, Any]:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    aliases = {"train": {"train", "training"}, "val": {"val", "validation"}}
    split = str(payload.get("split", ""))
    if split not in aliases[expected_split]:
        raise ValueError(f"{resolved} split is {split!r}, expected {expected_split}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"manifest contains no samples: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "split": split,
        "sample_count": len(samples),
        "format": payload.get("format"),
    }


def _parse_pipelines(value: str) -> list[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not values or any(item not in PIPELINES for item in values):
        raise ValueError(f"--pipelines must be a subset of {PIPELINES}")
    if len(values) != len(set(values)):
        raise ValueError("--pipelines contains duplicates")
    return values


def _dali_variant_contract(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    resolved = resolve_dali_variant(variant)
    return {
        **resolved,
        "num_threads": int(args.dali_num_threads),
        "prefetch_queue_depth": int(args.dali_prefetch_depth),
        "hybrid_huffman_threshold": int(args.dali_hybrid_huffman_threshold),
        "hw_decoder_load": float(args.dali_hw_decoder_load),
        "reader_initial_fill": int(args.dali_reader_initial_fill),
        "reader_dont_use_mmap": bool(args.dali_reader_dont_use_mmap),
        "reader_read_ahead": bool(args.dali_reader_read_ahead),
        "seed": int(args.seed),
        "strict_order": bool(resolved["preserves_canonical_order"]),
        "tail_policy": (
            "native reader pads its final physical batch and the adapter consumes "
            "only the registered logical tail; over-read is counted"
        ),
        "torch_handoff": "DLPack zero-copy with the DALI dynamic executor",
    }


def _comparison_scope() -> dict[str, Any]:
    return {
        "d2_vs_pytorch": (
            "direct RGB comparison with canonical order and planned crop/flip"
        ),
        "d3_performance_ceiling": (
            "DALI-native shuffle/crop/flip performance ceiling; order and "
            "augmentation decisions differ from D2/PyTorch"
        ),
        "vs_native_dct_b6": (
            "equal-image full-application comparison; not loader-only because "
            "the model input domains differ"
        ),
        "primary_performance_observation": "epoch 2 warm throughput",
        "pipeline_process_isolation": False,
        "epoch_1_cold_compile_order_bias": True,
        "epoch_1_comparison_role": "diagnostic only",
    }


def _runtime_files() -> list[Path]:
    return [
        Path(__file__).resolve(),
        HERE / "pipeline.py",
        HERE / "augmentation.py",
        HERE / "sample_order.py",
        HERE / "model_factory.py",
        FASTLANES_ROOT
        / "galp/benchmarks/system_dct_major/training_pls/published_optimizer.py",
        FASTLANES_ROOT / "galp/benchmarks/system_dct_major/training_pls/recipe.py",
        FASTLANES_ROOT / "galp/benchmarks/system_dct_major/training_pls/train.py",
    ]


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    train = _manifest_header(args.train_manifest, "train")
    validation = _manifest_header(args.val_manifest, "val")
    if train["sample_count"] != int(args.expected_train_images):
        raise ValueError(
            f"train manifest has {train['sample_count']} images; expected "
            f"{args.expected_train_images}"
        )
    if validation["sample_count"] != int(args.expected_validation_images):
        raise ValueError(
            f"validation manifest has {validation['sample_count']} images; expected "
            f"{args.expected_validation_images}"
        )
    prefix = schedule_summary(train["sample_count"], epochs=args.epochs)
    full_schedule = schedule_summary(train["sample_count"], epochs=REFERENCE_EPOCHS)
    published_reference_schedule = schedule_summary(
        EXPECTED_TRAIN_IMAGES, epochs=REFERENCE_EPOCHS
    )
    published = recipe_contract(RECIPE_NAME)
    payload: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA,
        "benchmark": "equal-image-epoch-aware-rgb-training-v2",
        "pipelines": list(args.resolved_pipelines),
        "seed": int(args.seed),
        "device": str(args.device),
        "required_gpu_name_substring": str(args.required_gpu_name_substring),
        "datasets": {"train": train, "validation": validation},
        "prefix_schedule": prefix,
        "reference_schedule": full_schedule,
        "training": {
            "prefix_epochs": int(args.epochs),
            "reference_epochs": REFERENCE_EPOCHS,
            "microbatch_images": MICROBATCH_IMAGES,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_full_update_images": (
                MICROBATCH_IMAGES * GRADIENT_ACCUMULATION
            ),
            "drop_last": False,
            "partial_update_normalization": (
                "actual samples in the accumulation window"
            ),
            "accumulation_crosses_epoch": False,
            "precision": "fp32",
            "model_domain": "rgb",
            "model_compile": published["execution"]["model_compile"],
            "float32_matmul_precision": published["execution"][
                "float32_matmul_precision"
            ],
        },
        "optimizer": published["optimizer"],
        "scheduler": {
            **published["scheduler"],
            "total_optimizer_updates": published_reference_schedule[
                "total_optimizer_updates"
            ],
            "horizon_dataset": "official ImageNet train (1,281,167 images)",
        },
        "augmentation": {
            "policy": "standard RGB/JPEG path",
            "decision_source": {
                "d2": "training.augmentation.derive_augmentation",
                "d3": "DALI native shuffle, image_random_crop, and coin_flip",
                "pytorch": "training.augmentation.derive_augmentation",
            },
            "crop": "per-sample RandomResizedCrop RGB",
            "horizontal_flip": "D2/PyTorch keyed; D3 DALI-native",
            "dali_decode": "JPEG ROI decode before resize/normalize",
            "dali_torch_handoff": "DLPack zero-copy with the DALI dynamic executor",
            "pytorch_decode": "PIL full JPEG decode then crop/resize/normalize",
            "mixup": False,
            "randaugment": False,
        },
        "dali_variants": {
            variant: _dali_variant_contract(args, variant)
            for variant in DALI_VARIANTS
            if variant in args.resolved_pipelines
        },
        "profiling": {
            "enabled": args.profile_epoch is not None,
            "epoch": args.profile_epoch,
            "warmup_microbatches": int(args.profile_warmup_microbatches),
            "capture_microbatches": int(args.profile_microbatches),
            "capture_control": "cudaProfilerApi",
            "nvtx_stage_ranges": True,
            "skip_profiled_epoch_validation": bool(
                args.profile_skip_validation
            ),
        },
        "validation": {
            "epochs": [
                epoch
                for epoch in (0, 1, 2)
                if not (
                    args.profile_skip_validation
                    and args.profile_epoch == epoch
                )
            ],
            "transform": "standard RGB centered square crop resized to 224",
            "timing_excluded_from_training_throughput": True,
        },
        "comparison_scope": _comparison_scope(),
        "runtime_files": [file_record(path) for path in _runtime_files()],
    }
    payload["contract_hash"] = sha256_json(payload)
    return payload


def _write_or_validate_contract(path: Path, contract: Mapping[str, Any]) -> None:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != contract:
            raise ValueError(
                f"existing benchmark contract differs: {path}; use a new output directory"
            )
        return
    _atomic_json(path, contract)


def _environment(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("equal-image formal benchmark requires CUDA")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": "galp-equal-image-environment-v1",
        "generated_at_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu_name": properties.name,
        "gpu_uuid": (
            None
            if getattr(properties, "uuid", None) is None
            else str(properties.uuid)
        ),
        "gpu_total_memory": int(properties.total_memory),
    }


def _canonical_initial_state(
    output_dir: Path, *, rgbnomore_root: Path, seed: int
) -> tuple[dict[str, torch.Tensor], str]:
    path = output_dir / "initial_rgb_state.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["model_state"]
        initial_hash = tensor_state_sha256(state)
        if payload.get("format") != "galp-equal-image-rgb-initial-state-v1":
            raise ValueError("initial RGB state format differs")
        if int(payload["seed"]) != seed or payload["initial_model_hash"] != initial_hash:
            raise ValueError("initial RGB state contract/hash differs")
        return state, initial_hash
    seed_everything(seed)
    canonical = build_model(rgbnomore_root, "rgb", torch.device("cpu"))
    state = {
        name: value.detach().cpu().clone()
        for name, value in canonical.state_dict().items()
    }
    initial_hash = tensor_state_sha256(state)
    _atomic_checkpoint(
        path,
        {
            "format": "galp-equal-image-rgb-initial-state-v1",
            "seed": seed,
            "initial_model_hash": initial_hash,
            "model_state": state,
        },
    )
    del canonical
    return state, initial_hash


def _center_validation_decision(sample: TrainingSample) -> AugmentationDecision:
    side = min(sample.width, sample.height)
    return AugmentationDecision(
        seed=0,
        epoch=0,
        logical_sample_id=sample.logical_sample_id,
        source_width=sample.width,
        source_height=sample.height,
        crop_x=(sample.width - side) // 2,
        crop_y=(sample.height - side) // 2,
        crop_width=side,
        crop_height=side,
        resize_width=224,
        resize_height=224,
        horizontal_flip=False,
        interpolation="bilinear",
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        domain="rgb",
    )


def _move_batch(
    batch: TrainingBatch, device: torch.device
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    if batch.on_device:
        return batch.inputs, batch.labels
    return (
        tuple(value.to(device, non_blocking=True) for value in batch.inputs),
        batch.labels.to(device, non_blocking=True),
    )


def _all_finite(values: Sequence[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in values)


@contextlib.contextmanager
def _nvtx_range(enabled: bool, name: str):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def _validate_emitted(
    batch: TrainingBatch,
    expected: Sequence[SampleIdentity],
    *,
    require_order: bool = True,
) -> None:
    if require_order and batch.identities != list(expected):
        raise RuntimeError("adapter emitted sample order different from canonical order")
    if int(batch.labels.shape[0]) != len(expected):
        raise RuntimeError("adapter batch cardinality differs from schedule")


def _adapter_pipeline(pipeline: str) -> str:
    return "dali" if pipeline in DALI_VARIANTS else pipeline


def _adapter_config(
    contract: Mapping[str, Any], *, pipeline: str, phase: str
) -> dict[str, Any]:
    config = {
        "execution_mode": "runtime",
        "benchmark": contract["benchmark"],
        "equal_image_contract_hash": contract["contract_hash"],
        "phase": phase,
    }
    if pipeline in DALI_VARIANTS:
        config["dali"] = dict(contract["dali_variants"][pipeline])
    return config


def _evaluate(
    *,
    pipeline: str,
    execution_model: torch.nn.Module,
    samples: Sequence[TrainingSample],
    workers: int,
    device: torch.device,
    contract: Mapping[str, Any],
    seed: int,
    epoch: int,
    optimizer_update: int,
    processed_images: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    identities = [
        SampleIdentity(0, index, sample.logical_sample_id)
        for index, sample in enumerate(samples)
    ]
    decisions = [_center_validation_decision(sample) for sample in samples]
    lengths = batch_lengths(len(samples))
    adapter = build_training_adapter(
        _adapter_pipeline(pipeline),
        samples,
        batch_size=MICROBATCH_IMAGES,
        workers=workers,
        device=device,
        config=_adapter_config(contract, pipeline=pipeline, phase="validation"),
    )
    adapter.begin(identities, decisions, lengths)
    execution_model.eval()
    total = 0
    loss_sum = 0.0
    top1 = 0
    top5 = 0
    loader_wait = 0.0
    try:
        with torch.no_grad():
            cursor = 0
            for length in lengths:
                batch = adapter.next_batch()
                expected = identities[cursor : cursor + length]
                _validate_emitted(batch, expected)
                inputs, labels = _move_batch(batch, device)
                logits = execution_model(*inputs)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("validation loss is non-finite")
                predictions = logits.topk(5, dim=1).indices
                top1 += int((predictions[:, 0] == labels).sum().item())
                top5 += int((predictions == labels[:, None]).any(dim=1).sum().item())
                loss_sum += float(loss.item()) * length
                loader_wait += float(batch.stage_seconds.get("loader_data_wait", 0.0))
                total += length
                cursor += length
                del batch, inputs, labels, logits, loss, predictions
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        adapter.close()
        execution_model.train()
    seconds = time.perf_counter() - started
    if total != len(samples):
        raise RuntimeError(f"validation consumed {total}/{len(samples)} samples")
    return {
        "record_type": "validation",
        "pipeline": pipeline,
        "seed": seed,
        "epoch": epoch,
        "optimizer_update": optimizer_update,
        "processed_images": processed_images,
        "validation_samples": total,
        "validation_loss": loss_sum / total,
        "validation_top1": 100.0 * top1 / total,
        "validation_top5": 100.0 * top5 / total,
        "validation_latency_seconds": seconds,
        "loader_wait_seconds": loader_wait,
    }


def _train_epoch(
    *,
    pipeline: str,
    epoch: int,
    seed: int,
    execution_model: torch.nn.Module,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    weight_decayer: Any,
    scheduler: Any,
    samples: Sequence[TrainingSample],
    workers: int,
    device: torch.device,
    contract: Mapping[str, Any],
    global_update: int,
    processed_images: int,
) -> tuple[dict[str, Any], int, int]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    epoch_started = time.perf_counter()
    planning_started = time.perf_counter()
    identities = canonical_epoch_order(
        [sample.logical_sample_id for sample in samples], seed, epoch
    )
    by_id = {sample.logical_sample_id: sample for sample in samples}
    dali_native_augmentation = pipeline in DALI_VARIANTS and str(
        contract["dali_variants"][pipeline]["augmentation_mode"]
    ) == "native"
    decisions = (
        []
        if dali_native_augmentation
        else [
            derive_augmentation(
                seed=seed,
                epoch=epoch,
                logical_sample_id=identity.logical_sample_id,
                source_width=by_id[identity.logical_sample_id].width,
                source_height=by_id[identity.logical_sample_id].height,
                domain="rgb",
            )
            for identity in identities
        ]
    )
    lengths = batch_lengths(len(samples))
    adapter = build_training_adapter(
        _adapter_pipeline(pipeline),
        samples,
        batch_size=MICROBATCH_IMAGES,
        workers=workers,
        device=device,
        config=_adapter_config(contract, pipeline=pipeline, phase="train"),
    )
    adapter.begin(identities, decisions, lengths)
    preparation_seconds = time.perf_counter() - planning_started
    steady_started = time.perf_counter()
    preserves_canonical_order = adapter.preserves_canonical_order()
    coverage = bytearray(len(identities))
    emitted_order_digest = hashlib.sha256()
    epoch_loss_sum = 0.0
    epoch_samples = 0
    epoch_updates = 0
    loader_wait = 0.0
    loader_stage_totals: dict[str, float] = {}
    capture_loader_stage_totals: dict[str, float] = {}
    cursor = 0
    profile = dict(contract.get("profiling", {}))
    profile_this_epoch = bool(profile.get("enabled")) and int(
        profile.get("epoch")
    ) == epoch + 1
    profile_begin = int(profile.get("warmup_microbatches", 0))
    profile_end = profile_begin + int(profile.get("capture_microbatches", 0))
    profiling_active = False
    try:
        for window_begin in range(0, len(lengths), GRADIENT_ACCUMULATION):
            window_lengths = lengths[
                window_begin : window_begin + GRADIENT_ACCUMULATION
            ]
            window_samples = sum(window_lengths)
            optimizer.zero_grad(set_to_none=True)
            learning_rate = scheduler.prepare_next_update()
            if profile_this_epoch and window_begin == profile_begin:
                torch.cuda.synchronize(device)
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push(
                    f"profile-rgb-{pipeline}-microbatches_"
                    f"{profile_begin}_{profile_end - 1}"
                )
                profiling_active = True
            for length in window_lengths:
                stage_nvtx = profile_this_epoch and profiling_active
                with _nvtx_range(stage_nvtx, "training.loader.next_batch"):
                    batch = adapter.next_batch()
                expected = identities[cursor : cursor + length]
                _validate_emitted(
                    batch,
                    expected,
                    require_order=preserves_canonical_order,
                )
                for identity in batch.identities:
                    if identity.epoch != epoch:
                        raise RuntimeError("adapter emitted an identity from another epoch")
                    if not 0 <= identity.position < len(identities):
                        raise RuntimeError("adapter emitted an out-of-range sample position")
                    if identities[identity.position] != identity:
                        raise RuntimeError(
                            "adapter identity position does not match the canonical epoch set"
                        )
                    if coverage[identity.position]:
                        raise RuntimeError("adapter emitted a duplicate sample position")
                    coverage[identity.position] = 1
                    emitted_order_digest.update(
                        f"{identity.epoch}:{identity.position}:{identity.logical_sample_id}\n".encode(
                            "utf-8"
                        )
                    )
                with _nvtx_range(stage_nvtx, "training.input_handoff"):
                    inputs, labels = _move_batch(batch, device)
                with _nvtx_range(stage_nvtx, "training.model.forward"):
                    logits = execution_model(*inputs)
                with _nvtx_range(stage_nvtx, "training.loss"):
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                if not bool(torch.isfinite(loss).item()) or not bool(
                    torch.isfinite(logits).all().item()
                ):
                    raise FloatingPointError(
                        f"non-finite loss/logits in {pipeline} epoch {epoch + 1}"
                    )
                with _nvtx_range(stage_nvtx, "training.model.backward"):
                    (loss * (length / window_samples)).backward()
                loss_value = float(loss.detach().item())
                epoch_loss_sum += loss_value * length
                epoch_samples += length
                cursor += length
                for name, value in batch.stage_seconds.items():
                    loader_stage_totals[name] = (
                        loader_stage_totals.get(name, 0.0) + float(value)
                    )
                    if stage_nvtx:
                        capture_loader_stage_totals[name] = (
                            capture_loader_stage_totals.get(name, 0.0)
                            + float(value)
                        )
                loader_wait += float(batch.stage_seconds.get("loader_data_wait", 0.0))
                del batch, inputs, labels, logits, loss
            stage_nvtx = profile_this_epoch and profiling_active
            with _nvtx_range(stage_nvtx, "training.optimizer"):
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                if not gradients or not _all_finite(gradients):
                    raise FloatingPointError(
                        f"non-finite/empty gradients in {pipeline} update {global_update + 1}"
                    )
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(
                        contract["optimizer"]["gradient_clipping_norm"]
                    ),
                )
                optimizer.step()
                weight_decayer.step(learning_rate)
                scheduler.complete_update()
            global_update += 1
            epoch_updates += 1
            processed_images += window_samples
            if (
                profile_this_epoch
                and profiling_active
                and window_begin + len(window_lengths) == profile_end
            ):
                torch.cuda.synchronize(device)
                torch.cuda.nvtx.range_pop()
                capture_metrics_path = os.environ.get(
                    "GALP_NSYS_CAPTURE_METRICS"
                )
                if capture_metrics_path:
                    capture_metrics = {
                        "schema_version": "galp-training-nsys-capture-metrics-v1",
                        "pipeline": pipeline,
                        "begin_microbatch": profile_begin,
                        "end_microbatch_exclusive": profile_end,
                        "captured_microbatches": profile_end - profile_begin,
                        "captured_images": sum(lengths[profile_begin:profile_end]),
                        "loader_stage_seconds": capture_loader_stage_totals,
                        "loader_stage_semantics": {
                            "loader_data_wait": (
                                "main-thread exposed wall time; comparable to the "
                                "training.loader.next_batch NVTX union"
                            ),
                            "read_decode_augmentation_preprocess": (
                                "summed per-sample worker work; stages overlap across "
                                "DataLoader workers and must not be added to wall time"
                            ),
                        },
                    }
                    capture_metrics_file = Path(capture_metrics_path)
                    capture_metrics_file.parent.mkdir(parents=True, exist_ok=True)
                    capture_metrics_file.write_text(
                        json.dumps(capture_metrics, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                torch.cuda.profiler.stop()
                profiling_active = False
    finally:
        if profiling_active:
            torch.cuda.synchronize(device)
            torch.cuda.nvtx.range_pop()
            torch.cuda.profiler.stop()
            profiling_active = False
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        adapter_metrics = adapter.loader_metrics()
        adapter.close()
    steady_seconds = time.perf_counter() - steady_started
    seconds = time.perf_counter() - epoch_started
    expected_schedule = schedule_summary(len(samples), epochs=1)
    if cursor != len(samples) or epoch_samples != len(samples):
        raise RuntimeError(
            f"{pipeline} epoch consumed {epoch_samples}/{len(samples)} images"
        )
    if epoch_updates != expected_schedule["optimizer_updates_per_epoch"]:
        raise RuntimeError(
            f"{pipeline} epoch performed {epoch_updates} updates; expected "
            f"{expected_schedule['optimizer_updates_per_epoch']}"
        )
    if coverage.count(1) != len(identities):
        raise RuntimeError(
            f"{pipeline} epoch covered {coverage.count(1)}/{len(identities)} unique positions"
        )
    tail_window_microbatches = (
        len(lengths) % GRADIENT_ACCUMULATION or GRADIENT_ACCUMULATION
    )
    return (
        {
            "record_type": "train",
            "scope": "epoch",
            "pipeline": pipeline,
            "seed": seed,
            "epoch": epoch + 1,
            "optimizer_update": global_update,
            "processed_images": processed_images,
            "epoch_samples": epoch_samples,
            "epoch_microbatches": len(lengths),
            "epoch_optimizer_updates": epoch_updates,
            "epoch_seconds": seconds,
            "images_per_second": epoch_samples / seconds,
            "train_loss": epoch_loss_sum / epoch_samples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "data_preparation_seconds": preparation_seconds,
            "steady_training_seconds": steady_seconds,
            "loader_wait_seconds": loader_wait,
            "loader_stage_seconds": loader_stage_totals,
            "adapter_metrics": adapter_metrics,
            "microbatch_images": MICROBATCH_IMAGES,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "tail_microbatch_images": lengths[-1],
            "tail_update_samples": sum(lengths[-tail_window_microbatches:]),
            "sample_order": {
                "preserves_canonical_order": preserves_canonical_order,
                "unique_positions": coverage.count(1),
                "emitted_order_sha256": emitted_order_digest.hexdigest(),
            },
            "profile_capture": (
                {
                    "begin_microbatch": profile_begin,
                    "end_microbatch_exclusive": profile_end,
                    "captured_microbatches": profile_end - profile_begin,
                }
                if profile_this_epoch
                else None
            ),
            "training_timing_includes": (
                "epoch schedule/augmentation construction, adapter setup, JPEG read/decode/"
                "transform, H2D, forward, backward, optimizer, synchronization, adapter close"
            ),
        },
        global_update,
        processed_images,
    )


def _checkpoint_payload(
    *,
    pipeline: str,
    contract_hash: str,
    initial_model_hash: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    weight_decayer: Any,
    scheduler: Any,
    completed_epoch: int,
    global_update: int,
    processed_images: int,
    metrics: MetricsWriter,
    pending_validation_epoch: int | None,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_SCHEMA,
        "pipeline": pipeline,
        "contract_hash": contract_hash,
        "initial_model_hash": initial_model_hash,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "weight_decay_state": weight_decayer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "completed_epoch": completed_epoch,
        "global_optimizer_update": global_update,
        "processed_image_count": processed_images,
        "metrics_cursor": metrics.cursor(),
        "gradient_accumulation_state": {
            "microbatches": 0,
            "samples": 0,
            "boundary": "completed-epoch",
        },
        "pending_validation_epoch": pending_validation_epoch,
    }


def _save_checkpoint(
    run_dir: Path,
    *,
    permanent_epoch: int | None,
    **kwargs: Any,
) -> None:
    latest = run_dir / "latest.pt"
    _atomic_checkpoint(latest, _checkpoint_payload(**kwargs))
    if permanent_epoch is not None:
        _retain_checkpoint(
            latest, run_dir / f"checkpoint_epoch_{permanent_epoch:03d}.pt"
        )


def _restore_checkpoint(
    path: Path,
    *,
    pipeline: str,
    contract_hash: str,
    initial_model_hash: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    weight_decayer: Any,
    scheduler: Any,
    metrics: MetricsWriter,
) -> tuple[int, int, int, int | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "format": CHECKPOINT_SCHEMA,
        "pipeline": pipeline,
        "contract_hash": contract_hash,
        "initial_model_hash": initial_model_hash,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"resume checkpoint contract differs: {observed}")
    accumulation = payload.get("gradient_accumulation_state", {})
    if accumulation.get("microbatches") != 0 or accumulation.get("samples") != 0:
        raise ValueError("resume checkpoint is not at an epoch boundary")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    weight_decayer.load_state_dict(payload["weight_decay_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    restore_rng_state(payload["rng_state"])
    metrics.truncate(payload["metrics_cursor"])
    return (
        int(payload["completed_epoch"]),
        int(payload["global_optimizer_update"]),
        int(payload["processed_image_count"]),
        payload.get("pending_validation_epoch"),
    )


def _profiling_skips_validation(
    contract: Mapping[str, Any], epoch: int
) -> bool:
    profiling = dict(contract.get("profiling", {}))
    return (
        bool(profiling.get("enabled"))
        and bool(profiling.get("skip_profiled_epoch_validation", False))
        and int(profiling.get("epoch", -1)) == int(epoch)
    )


def _validation_skipped_record(
    *,
    pipeline: str,
    seed: int,
    epoch: int,
    optimizer_update: int,
    processed_images: int,
) -> dict[str, Any]:
    return {
        "record_type": "validation_skipped",
        "pipeline": pipeline,
        "seed": seed,
        "epoch": epoch,
        "optimizer_update": optimizer_update,
        "processed_images": processed_images,
        "reason": (
            "profiling-only replay: avoid post-capture inference graph compilation; "
            "the source formal run already contains authoritative validation"
        ),
        "scientific_result": False,
    }


def _run_pipeline(
    *,
    pipeline: str,
    output_dir: Path,
    contract: Mapping[str, Any],
    initial_state: Mapping[str, torch.Tensor],
    initial_model_hash: str,
    train_samples: Sequence[TrainingSample],
    val_samples: Sequence[TrainingSample],
    rgbnomore_root: Path,
    workers: int,
    device: torch.device,
    stop_after_epoch: int | None,
) -> dict[str, Any]:
    run_dir = output_dir / "runs" / pipeline
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "final_result.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        if result.get("contract_hash") != contract["contract_hash"]:
            raise ValueError(f"completed {pipeline} result contract differs")
        return result

    seed = int(contract["seed"])
    seed_everything(seed)
    model = build_model(rgbnomore_root, "rgb", device)
    model.load_state_dict(initial_state, strict=True)
    if tensor_state_sha256(model.state_dict()) != initial_model_hash:
        raise RuntimeError(f"{pipeline} device model differs from canonical state")
    published = recipe_contract(RECIPE_NAME)
    total_updates = int(contract["scheduler"]["total_optimizer_updates"])
    optimizer, weight_decayer, scheduler = build_published_optimizer(
        model,
        learning_rate=float(published["optimizer"]["learning_rate"]),
        weight_decay=float(
            published["optimizer"]["weight_decay"]["coefficient"]
        ),
        warmup_updates=int(published["scheduler"]["warmup_optimizer_updates"]),
        total_updates=total_updates,
    )
    execution_model = compile_published_model(model, published)
    metrics = MetricsWriter(run_dir / "metrics.jsonl")
    completed_epoch = 0
    global_update = 0
    processed_images = 0
    pending_validation_epoch: int | None = None
    latest = run_dir / "latest.pt"
    if latest.exists():
        (
            completed_epoch,
            global_update,
            processed_images,
            pending_validation_epoch,
        ) = _restore_checkpoint(
            latest,
            pipeline=pipeline,
            contract_hash=str(contract["contract_hash"]),
            initial_model_hash=initial_model_hash,
            model=model,
            optimizer=optimizer,
            weight_decayer=weight_decayer,
            scheduler=scheduler,
            metrics=metrics,
        )
    elif metrics.lines:
        metrics.truncate({"lines": 0, "bytes": 0})

    run_status = {
        "schema_version": RESULT_SCHEMA,
        "state": "running",
        "pipeline": pipeline,
        "seed": seed,
        "completed_epoch": completed_epoch,
        "optimizer_update": global_update,
        "processed_images": processed_images,
        "started_at_unix": time.time(),
    }
    _atomic_json(run_dir / "run_status.json", run_status)

    def save(permanent_epoch: int | None, pending: int | None) -> None:
        _save_checkpoint(
            run_dir,
            permanent_epoch=permanent_epoch,
            pipeline=pipeline,
            contract_hash=str(contract["contract_hash"]),
            initial_model_hash=initial_model_hash,
            model=model,
            optimizer=optimizer,
            weight_decayer=weight_decayer,
            scheduler=scheduler,
            completed_epoch=completed_epoch,
            global_update=global_update,
            processed_images=processed_images,
            metrics=metrics,
            pending_validation_epoch=pending,
        )

    if not latest.exists() and metrics.lines == 0:
        metrics.append(
            _evaluate(
                pipeline=pipeline,
                execution_model=execution_model,
                samples=val_samples,
                workers=workers,
                device=device,
                contract=contract,
                seed=seed,
                epoch=0,
                optimizer_update=0,
                processed_images=0,
            )
        )
        save(0, None)

    if pending_validation_epoch is not None:
        if int(pending_validation_epoch) != completed_epoch:
            raise ValueError("pending validation epoch differs from checkpoint epoch")
        if _profiling_skips_validation(contract, completed_epoch):
            metrics.append(
                _validation_skipped_record(
                    pipeline=pipeline,
                    seed=seed,
                    epoch=completed_epoch,
                    optimizer_update=global_update,
                    processed_images=processed_images,
                )
            )
        else:
            metrics.append(
                _evaluate(
                    pipeline=pipeline,
                    execution_model=execution_model,
                    samples=val_samples,
                    workers=workers,
                    device=device,
                    contract=contract,
                    seed=seed,
                    epoch=completed_epoch,
                    optimizer_update=global_update,
                    processed_images=processed_images,
                )
            )
        save(completed_epoch, None)

    for epoch in range(completed_epoch, int(contract["prefix_schedule"]["epochs"])):
        epoch_record, global_update, processed_images = _train_epoch(
            pipeline=pipeline,
            epoch=epoch,
            seed=seed,
            execution_model=execution_model,
            model=model,
            optimizer=optimizer,
            weight_decayer=weight_decayer,
            scheduler=scheduler,
            samples=train_samples,
            workers=workers,
            device=device,
            contract=contract,
            global_update=global_update,
            processed_images=processed_images,
        )
        metrics.append(epoch_record)
        completed_epoch = epoch + 1
        save(None, completed_epoch)
        if _profiling_skips_validation(contract, completed_epoch):
            metrics.append(
                _validation_skipped_record(
                    pipeline=pipeline,
                    seed=seed,
                    epoch=completed_epoch,
                    optimizer_update=global_update,
                    processed_images=processed_images,
                )
            )
        else:
            metrics.append(
                _evaluate(
                    pipeline=pipeline,
                    execution_model=execution_model,
                    samples=val_samples,
                    workers=workers,
                    device=device,
                    contract=contract,
                    seed=seed,
                    epoch=completed_epoch,
                    optimizer_update=global_update,
                    processed_images=processed_images,
                )
            )
        save(completed_epoch, None)
        run_status.update(
            completed_epoch=completed_epoch,
            optimizer_update=global_update,
            processed_images=processed_images,
        )
        _atomic_json(run_dir / "run_status.json", run_status)
        if (
            stop_after_epoch is not None
            and completed_epoch >= stop_after_epoch
            and completed_epoch < int(contract["prefix_schedule"]["epochs"])
        ):
            paused = {
                **run_status,
                "state": "paused-at-epoch-boundary",
                "ended_at_unix": time.time(),
                "latest_checkpoint": str(latest.resolve()),
                "scientific_result": False,
            }
            _atomic_json(run_dir / "pause_result.json", paused)
            _atomic_json(run_dir / "run_status.json", paused)
            return paused

    records = metrics.records()
    epochs = [
        row
        for row in records
        if row.get("record_type") == "train" and row.get("scope") == "epoch"
    ]
    validations = [row for row in records if row.get("record_type") == "validation"]
    skipped_validations = [
        row for row in records if row.get("record_type") == "validation_skipped"
    ]
    total_seconds = sum(float(row["epoch_seconds"]) for row in epochs)
    total_samples = sum(int(row["epoch_samples"]) for row in epochs)
    result = {
        "schema_version": RESULT_SCHEMA,
        "state": "completed",
        "pipeline": pipeline,
        "seed": seed,
        "contract_hash": contract["contract_hash"],
        "initial_model_hash": initial_model_hash,
        "completed_epoch": completed_epoch,
        "optimizer_update": global_update,
        "processed_images": processed_images,
        "training_seconds": total_seconds,
        "images_per_second": total_samples / total_seconds,
        "epoch_records": epochs,
        "validation_records": validations,
        "validation_skipped_records": skipped_validations,
        "primary_warm_epoch": next(
            row for row in epochs if int(row["epoch"]) == DEFAULT_PREFIX_EPOCHS
        ),
        "ended_at_unix": time.time(),
    }
    _atomic_json(final_path, result)
    run_status.update(
        state="completed",
        ended_at_unix=time.time(),
        final_result=str(final_path.resolve()),
    )
    _atomic_json(run_dir / "run_status.json", run_status)
    del execution_model, model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _write_summary(output_dir: Path, results: Sequence[Mapping[str, Any]]) -> None:
    completed = [row for row in results if row.get("state") == "completed"]
    summary = {
        "schema_version": "galp-equal-image-rgb-summary-v2",
        "results": list(results),
        "all_completed": len(completed) == len(results),
        "primary_metric": "epoch 2 images_per_second",
    }
    _atomic_json(output_dir / "results.json", summary)
    fields = [
        "pipeline",
        "epoch",
        "epoch_samples",
        "epoch_microbatches",
        "epoch_optimizer_updates",
        "epoch_seconds",
        "images_per_second",
        "data_preparation_seconds",
        "loader_wait_seconds",
    ]
    temporary = output_dir / ".epoch_performance.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for result in completed:
            for record in result["epoch_records"]:
                writer.writerow({field: record.get(field) for field in fields})
    os.replace(temporary, output_dir / "epoch_performance.csv")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pipelines", default="d2,d3,pytorch")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dali-num-threads", type=int, default=4)
    parser.add_argument("--dali-prefetch-depth", type=int, default=2)
    parser.add_argument(
        "--dali-hybrid-huffman-threshold", type=int, default=1_000_000
    )
    parser.add_argument("--dali-hw-decoder-load", type=float, default=0.65)
    parser.add_argument("--dali-reader-initial-fill", type=int, default=1024)
    parser.add_argument(
        "--dali-reader-dont-use-mmap",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dali-reader-read-ahead",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=11997733)
    parser.add_argument("--epochs", type=int, default=DEFAULT_PREFIX_EPOCHS)
    parser.add_argument(
        "--expected-train-images", type=int, default=EXPECTED_TRAIN_IMAGES
    )
    parser.add_argument(
        "--expected-validation-images",
        type=int,
        default=EXPECTED_VALIDATION_IMAGES,
    )
    parser.add_argument(
        "--required-gpu-name-substring",
        default="H100",
        help="fail closed when the selected logical CUDA device is not the intended GPU",
    )
    parser.add_argument(
        "--rgbnomore-root", type=Path, default=Path("/home/tangyuxin/RGB-no-more")
    )
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--profile-epoch", type=int, choices=(1, 2))
    parser.add_argument("--profile-warmup-microbatches", type=int, default=512)
    parser.add_argument("--profile-microbatches", type=int, default=1024)
    parser.add_argument(
        "--profile-skip-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "skip validation after the profiled epoch; profiling-only replays can "
            "use the authoritative validation records copied from the source run"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.resolved_pipelines = _parse_pipelines(args.pipelines)
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    for name in (
        "dali_num_threads",
        "dali_prefetch_depth",
        "dali_reader_initial_fill",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.dali_hybrid_huffman_threshold < 0:
        raise ValueError("--dali-hybrid-huffman-threshold must be non-negative")
    if not 0.0 <= args.dali_hw_decoder_load <= 1.0:
        raise ValueError("--dali-hw-decoder-load must be in [0, 1]")
    if args.epochs != DEFAULT_PREFIX_EPOCHS:
        raise ValueError(
            f"the registered equal-image benchmark fixes --epochs={DEFAULT_PREFIX_EPOCHS}"
        )
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        raise ValueError("--stop-after-epoch must be in [1, epochs]")
    if args.profile_warmup_microbatches < 0 or args.profile_microbatches <= 0:
        raise ValueError("profiling warmup must be non-negative and capture must be positive")
    if (
        args.profile_warmup_microbatches % GRADIENT_ACCUMULATION
        or args.profile_microbatches % GRADIENT_ACCUMULATION
    ):
        raise ValueError("profiling bounds must align to gradient accumulation")
    if (
        args.profile_warmup_microbatches + args.profile_microbatches
        > schedule_summary(args.expected_train_images, epochs=1)[
            "microbatches_per_epoch"
        ]
    ):
        raise ValueError("profiling range exceeds one epoch")
    if args.profile_skip_validation and args.profile_epoch is None:
        raise ValueError("--profile-skip-validation requires --profile-epoch")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = build_contract(args)
    _write_or_validate_contract(output_dir / "contract.json", contract)
    _atomic_json(
        output_dir / "execution_plan.json",
        {
            "schema_version": "galp-equal-image-rgb-execution-plan-v2",
            "execute_requested": bool(args.execute),
            "pipeline_order": list(args.resolved_pipelines),
            "device": args.device,
            "contract_hash": contract["contract_hash"],
            "resume_mode": "epoch-boundary",
            "expected_outputs": [
                str(output_dir / "runs" / pipeline / "final_result.json")
                for pipeline in args.resolved_pipelines
            ],
        },
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "state": "planned",
                    "output_dir": str(output_dir),
                    "contract_hash": contract["contract_hash"],
                    "schedule": contract["prefix_schedule"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    device = torch.device(args.device)
    environment = _environment(device)
    required = str(args.required_gpu_name_substring)
    if required and required.lower() not in str(environment["gpu_name"]).lower():
        raise RuntimeError(
            f"selected device is {environment['gpu_name']!r}; required substring "
            f"is {required!r}"
        )
    _atomic_json(output_dir / "environment.json", environment)
    train_samples, train_meta = load_training_manifest(
        args.train_manifest, root=None, expected_split="train"
    )
    val_samples, val_meta = load_training_manifest(
        args.val_manifest, root=None, expected_split="val"
    )
    separation = validate_dataset_separation(train_samples, val_samples)
    if not separation["ok"]:
        raise ValueError("training and validation datasets overlap")
    _atomic_json(
        output_dir / "dataset_validation.json",
        {
            "train": train_meta,
            "validation": val_meta,
            "separation": separation,
        },
    )
    initial_state, initial_model_hash = _canonical_initial_state(
        output_dir, rgbnomore_root=args.rgbnomore_root, seed=args.seed
    )
    failures_path = output_dir / "failures.jsonl"
    failures_path.touch(exist_ok=True)
    results: list[dict[str, Any]] = []
    exit_code = 0
    for pipeline in args.resolved_pipelines:
        try:
            result = _run_pipeline(
                pipeline=pipeline,
                output_dir=output_dir,
                contract=contract,
                initial_state=initial_state,
                initial_model_hash=initial_model_hash,
                train_samples=train_samples,
                val_samples=val_samples,
                rgbnomore_root=args.rgbnomore_root,
                workers=args.workers,
                device=device,
                stop_after_epoch=args.stop_after_epoch,
            )
            results.append(result)
        except Exception as error:  # retain the other standard-path result
            failure = {
                "timestamp_unix": time.time(),
                "pipeline": pipeline,
                "error_type": type(error).__name__,
                "error": str(error),
                "scientific_failure": False,
            }
            with failures_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(failure, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
            results.append({"state": "failed", **failure})
            exit_code = 1
    _write_summary(output_dir, results)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
