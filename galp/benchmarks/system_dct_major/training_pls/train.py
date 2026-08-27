#!/usr/bin/env python3
"""Run one full core PLS condition/seed with the shared Direct-DCT backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .contracts import (
    blocking_code_identity,
    code_provenance_differences,
    code_version,
    condition_identity_hash,
)
from .core_schedule import epoch_position_pools, stable_digest
from .layout import (
    LayoutMapping,
    load_layout_mapping,
    sha256_file,
    validate_manifest_against_layout,
)
from .matrix import resolve_condition
from .published_augmentation import (
    PublishedAugmentation,
    apply_published_mixup,
    apply_published_randaugment,
    published_training_augmentation,
    published_validation_augmentation,
)
from .published_optimizer import build_published_optimizer
from .recipe import RECIPE_NAME, assert_recipe_overrides, sha256_json, validation_epochs


REPO_ROOT = Path(__file__).resolve().parents[4]
RGB_BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "system_rgbnomore"
if str(RGB_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(RGB_BENCHMARK_ROOT))

from training.artifacts import nested_state_sha256, tensor_state_sha256  # noqa: E402
from training.direct_dct_reader import DirectDctTrainingReader  # noqa: E402
from training.model_factory import (  # noqa: E402
    build_model,
    capture_rng_state,
    model_configuration,
    restore_rng_state,
    seed_everything,
)
from training.pipeline import (  # noqa: E402
    TrainingSample,
    build_training_adapter,
)
from training.sample_order import SampleIdentity  # noqa: E402
from galp.torch.experimental import DirectDctPlsPipeline  # noqa: E402


RUN_SCHEMA = "galp-pls-core-training-run-v2"
CHECKPOINT_SCHEMA = "galp-pls-core-epoch-checkpoint-v4"
DESCRIPTOR_CHUNK_MICROBATCHES = 256
NATIVE_PHYSICAL_BACKEND = "native-physical-pls"
SEMANTIC_BACKEND = "semantic-emulation"


def build_paired_model(
    rgbnomore_root: Path, *, seed: int, device: torch.device
) -> tuple[torch.nn.Module, str]:
    """Build a device-native model with a device-independent paired state."""

    seed_everything(seed)
    canonical = build_model(rgbnomore_root, "dct", torch.device("cpu"))
    initial_hash = tensor_state_sha256(canonical.state_dict())
    if device.type == "cpu":
        return canonical, initial_hash
    execution_model = build_model(rgbnomore_root, "dct", device)
    incompatible = execution_model.load_state_dict(canonical.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict paired model initialization unexpectedly differed")
    del canonical
    if tensor_state_sha256(execution_model.state_dict()) != initial_hash:
        raise RuntimeError("device-native model differs after loading canonical paired state")
    return execution_model, initial_hash


def compile_published_model(
    model: torch.nn.Module, recipe: Mapping[str, Any]
) -> torch.nn.Module:
    """Compile the fixed-shape model without changing its checkpoint state object."""

    configuration = recipe["execution"]["model_compile"]
    if not bool(configuration["enabled"]):
        raise ValueError("the published recipe requires model compilation")
    precision = str(recipe["execution"]["float32_matmul_precision"])
    torch.set_float32_matmul_precision(precision)
    return torch.compile(
        model,
        backend=str(configuration["backend"]),
        mode=str(configuration["mode"]),
        fullgraph=bool(configuration["fullgraph"]),
        dynamic=bool(configuration["dynamic"]),
    )


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


class MetricsWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lines = 0
        if path.is_file():
            with path.open("rb") as source:
                self.lines = sum(1 for _line in source)

    def append(self, payload: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
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
        expected_bytes = int(cursor["bytes"])
        with self.path.open("r+b") as output:
            output.truncate(expected_bytes)
        self.lines = int(cursor["lines"])


def _load_manifest_fast(
    path: Path, *, expected_split: str
) -> tuple[list[TrainingSample], dict[str, Any]]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = str(payload.get("split", expected_split))
    aliases = {"train": {"train", "training"}, "val": {"val", "validation"}}
    if split not in aliases[expected_split]:
        raise ValueError(f"{path} split is {split!r}, expected {expected_split!r}")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"manifest contains no samples: {path}")
    samples: list[TrainingSample] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_samples):
        logical_id = str(raw.get("logical_sample_id", ""))
        if not logical_id or logical_id in seen:
            raise ValueError(f"manifest sample {index} has a missing or duplicate logical ID")
        seen.add(logical_id)
        sample_path = Path(str(raw.get("path", logical_id)))
        if not sample_path.is_absolute():
            root = Path(str(payload.get("jpeg_root", path.parent)))
            sample_path = root / sample_path
        samples.append(
            TrainingSample(
                logical_sample_id=logical_id,
                path=sample_path.resolve(),
                label=int(raw["label"]),
                width=int(raw["width"]),
                height=int(raw["height"]),
                galp_image_id=int(raw["galp_image_id"]),
                payload_sha256=(
                    None
                    if raw.get("payload_sha256") is None
                    else str(raw["payload_sha256"])
                ),
            )
        )
    population = payload.get("population_count")
    if population is not None and int(population) != len(samples):
        raise ValueError("manifest population_count differs from the sample array")
    raw_galp = payload.get("galp_manifest")
    if raw_galp is None:
        raise ValueError(f"manifest does not declare galp_manifest: {path}")
    galp_manifest = Path(str(raw_galp))
    if not galp_manifest.is_absolute():
        galp_manifest = path.parent / galp_manifest
    metadata = {
        "path": str(path),
        "manifest_hash": sha256_file(path),
        "sample_count": len(samples),
        "split": expected_split,
        "galp_manifest": str(galp_manifest.resolve()),
        "format": payload.get("format"),
        "validation_semantics": payload.get("validation_semantics"),
    }
    return samples, metadata


def _validate_contract(
    path: Path,
    *,
    condition_id: str,
    seed: int,
    recipe_hash: str,
    layout_hash: str,
    execution_backend: str,
    physical_galp_manifest: Path | None,
    premixed_mapping_csv: Path | None,
    expected_mapping_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "run_manifest_hash" in payload:
        expected_manifest_hash = sha256_json(
            {
                key: value
                for key, value in payload.items()
                if key != "run_manifest_hash"
            }
        )
        if payload.get("run_manifest_hash") != expected_manifest_hash:
            raise ValueError("run manifest integrity hash mismatch")
        if payload.get("condition_hash") != condition_identity_hash(payload):
            raise ValueError("run manifest training identity hash mismatch")
    else:
        # Compatibility with v2 condition contracts produced before the
        # resolved run-manifest format introduced a separate integrity hash.
        expected_hash = sha256_json(
            {key: value for key, value in payload.items() if key != "condition_hash"}
        )
        if payload.get("condition_hash") != expected_hash:
            raise ValueError("condition contract hash mismatch")
    native_physical = execution_backend == NATIVE_PHYSICAL_BACKEND
    expected = {
        "condition_id": condition_id,
        "training_seed": seed,
        "recipe_hash": recipe_hash,
        "layout_hash": layout_hash,
        "execution_mode": (
            "native_physical_pls" if native_physical else "semantic_emulation"
        ),
        "semantic_emulation": not native_physical,
        "physical_fls_observed": native_physical,
        "physical_gpu_pool": native_physical,
        "backend_implementation": (
            "galp-native-direct-dct-pls-block-major-v1"
            if native_physical
            else "shared-galp-direct-dct-semantic-backend-v2"
        ),
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"condition contract does not match run arguments: expected {expected}, got {observed}"
        )
    planned_code = payload.get("code_version")
    if not isinstance(planned_code, dict):
        raise ValueError("run manifest lacks code_version")
    current_code = code_version(REPO_ROOT)
    planned_identity = blocking_code_identity(planned_code)
    current_identity = blocking_code_identity(current_code)
    if planned_identity != current_identity:
        raise ValueError(
            "training runtime sources changed after the run manifest was generated: "
            f"planned={planned_identity}, current={current_identity}"
        )
    if native_physical:
        if (
            physical_galp_manifest is None
            or premixed_mapping_csv is None
            or expected_mapping_sha256 is None
        ):
            raise ValueError("native physical execution arguments are incomplete")
        physical = payload.get("physical_execution")
        if not isinstance(physical, dict):
            raise ValueError("native condition contract lacks physical_execution")
        expected_physical = {
            "physical_galp_manifest": str(physical_galp_manifest.resolve()),
            "physical_galp_manifest_sha256": sha256_file(
                physical_galp_manifest.resolve()
            ),
            "premixed_mapping_csv": str(premixed_mapping_csv.resolve()),
            "premixed_mapping_sha256": expected_mapping_sha256,
        }
        observed_physical = {
            key: physical.get(key) for key in expected_physical
        }
        if observed_physical != expected_physical:
            raise ValueError(
                "physical execution contract does not match run arguments: "
                f"expected {expected_physical}, got {observed_physical}"
            )
    provenance_differences = code_provenance_differences(planned_code, current_code)
    validation = {
        "schema_version": "galp-pls-run-manifest-validation-v1",
        "blocking_code_identity": current_identity,
        "blocking_code_identity_matches": True,
        "non_blocking_code_provenance_differences": provenance_differences,
        "non_blocking_code_provenance_changed": bool(provenance_differences),
    }
    return payload, validation


def _device_environment(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        uuid = getattr(properties, "uuid", None)
        payload.update(
            gpu_index=index,
            gpu_name=properties.name,
            gpu_uuid=None if uuid is None else str(uuid),
            gpu_total_memory=int(properties.total_memory),
            cudnn_version=torch.backends.cudnn.version(),
        )
    return payload


def _adapter_config(
    *,
    reader: DirectDctTrainingReader,
    galp_manifest: Path,
    module_path: Path,
    prefetch_depth: int,
) -> dict[str, Any]:
    return {
        "galp_manifest": str(galp_manifest.resolve()),
        "galp_torch_module_path": str(module_path.resolve()),
        "prefetch_depth": int(prefetch_depth),
        "execution_mode": "runtime",
        "_direct_dct_training_reader": reader,
    }


def _move_batch(
    batch: Any, device: torch.device
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    if batch.on_device:
        return tuple(batch.inputs), batch.labels
    return (
        tuple(value.to(device, non_blocking=True) for value in batch.inputs),
        batch.labels.to(device, non_blocking=True),
    )


def _all_finite(values: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in values)


def _merge_numeric(target: dict[str, float], source: Mapping[str, Any], prefix: str = "") -> None:
    for key, value in source.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _merge_numeric(target, value, name)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                target[name] = target.get(name, 0.0) + float(value)


def _schedule_chunk(
    *,
    positions: Sequence[int],
    chunk_begin_microbatch: int,
    chunk_end_microbatch: int,
    epoch: int,
    pool_index: int,
    epoch_position_begin: int,
    microbatch_index_begin: int,
    mapping: LayoutMapping,
    samples: Sequence[TrainingSample],
    condition: Mapping[str, Any],
    seed: int,
) -> tuple[
    list[list[int]],
    list[list[SampleIdentity]],
    list[list[PublishedAugmentation]],
    int,
]:
    microbatch_positions: list[list[int]] = []
    identities: list[list[SampleIdentity]] = []
    augmentations: list[list[PublishedAugmentation]] = []
    local_epoch_position = epoch_position_begin
    for pool_microbatch_index in range(chunk_begin_microbatch, chunk_end_microbatch):
        begin = pool_microbatch_index * 64
        selected = [int(value) for value in positions[begin : begin + 64]]
        batch_identities: list[SampleIdentity] = []
        batch_augmentations: list[PublishedAugmentation] = []
        for planned_position in selected:
            manifest_index = int(mapping.manifest_indices[planned_position])
            sample = samples[manifest_index]
            batch_identities.append(
                SampleIdentity(epoch, local_epoch_position, sample.logical_sample_id)
            )
            batch_augmentations.append(
                published_training_augmentation(
                    training_seed=seed,
                    epoch=epoch,
                    logical_sample_id=sample.logical_sample_id,
                    virtual_pls_id=int(mapping.virtual_pls_ids[planned_position]),
                    crop_policy=str(condition["crop_policy"]),
                    source_width=sample.width,
                    source_height=sample.height,
                )
            )
            local_epoch_position += 1
        microbatch_positions.append(selected)
        identities.append(batch_identities)
        augmentations.append(batch_augmentations)
    return microbatch_positions, identities, augmentations, local_epoch_position


def _make_adapter(
    *,
    samples: Sequence[TrainingSample],
    identities: Sequence[Sequence[SampleIdentity]],
    augmentations: Sequence[Sequence[PublishedAugmentation]],
    device: torch.device,
    workers: int,
    config: dict[str, Any],
) -> Any:
    flat_identities = [value for batch in identities for value in batch]
    flat_augmentations = [value.decision for batch in augmentations for value in batch]
    by_id = {sample.logical_sample_id: sample for sample in samples}
    selected_samples = [by_id[identity.logical_sample_id] for identity in flat_identities]
    adapter = build_training_adapter(
        "galp",
        selected_samples,
        batch_size=64,
        workers=workers,
        device=device,
        config=config,
    )
    adapter.begin(
        flat_identities,
        flat_augmentations,
        [len(batch) for batch in identities],
    )
    return adapter


def _run_validation(
    *,
    model: torch.nn.Module,
    samples: Sequence[TrainingSample],
    reader: DirectDctTrainingReader,
    galp_manifest: Path,
    module_path: Path,
    prefetch_depth: int,
    workers: int,
    device: torch.device,
    condition_id: str,
    seed: int,
    epoch: int,
    optimizer_update: int,
    processed_images: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    model.eval()
    total = 0
    loss_sum = 0.0
    top1 = 0
    top5 = 0
    loader_totals: dict[str, float] = {}
    started = time.perf_counter()
    config = _adapter_config(
        reader=reader,
        galp_manifest=galp_manifest,
        module_path=module_path,
        prefetch_depth=prefetch_depth,
    )
    batch_count = (len(samples) + 63) // 64
    with torch.no_grad():
        for chunk_begin in range(0, batch_count, DESCRIPTOR_CHUNK_MICROBATCHES):
            chunk_end = min(batch_count, chunk_begin + DESCRIPTOR_CHUNK_MICROBATCHES)
            identities: list[list[SampleIdentity]] = []
            decisions: list[list[PublishedAugmentation]] = []
            for batch_index in range(chunk_begin, chunk_end):
                begin = batch_index * 64
                batch_samples = samples[begin : begin + 64]
                identities.append(
                    [
                        SampleIdentity(epoch, begin + offset, sample.logical_sample_id)
                        for offset, sample in enumerate(batch_samples)
                    ]
                )
                decisions.append(
                    [
                        PublishedAugmentation(
                            decision=published_validation_augmentation(
                                logical_sample_id=sample.logical_sample_id,
                                source_width=sample.width,
                                source_height=sample.height,
                                epoch=epoch,
                            ),
                            crop_key="validation-fixed",
                            flip_key="validation-no-flip",
                            crop_seed=0,
                            flip_seed=0,
                        )
                        for sample in batch_samples
                    ]
                )
            chunk_samples = samples[chunk_begin * 64 : min(len(samples), chunk_end * 64)]
            adapter = _make_adapter(
                samples=chunk_samples,
                identities=identities,
                augmentations=decisions,
                device=device,
                workers=workers,
                config=config,
            )
            try:
                for expected in identities:
                    batch = adapter.next_batch()
                    inputs, labels = _move_batch(batch, device)
                    if [value.logical_sample_id for value in batch.identities] != [
                        value.logical_sample_id for value in expected
                    ]:
                        raise RuntimeError("validation adapter changed logical sample order")
                    logits = model(*inputs)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    if not bool(torch.isfinite(loss).item()) or not bool(
                        torch.isfinite(logits).all().item()
                    ):
                        raise FloatingPointError("validation produced NaN/Inf")
                    length = len(expected)
                    predictions = logits.topk(5, dim=1).indices
                    top1 += int((predictions[:, 0] == labels).sum().item())
                    top5 += int((predictions == labels[:, None]).any(dim=1).sum().item())
                    loss_sum += float(loss.item()) * length
                    total += length
                    for name, value in batch.stage_seconds.items():
                        loader_totals[name] = loader_totals.get(name, 0.0) + float(value)
                    adapter.snapshot_batch_metrics(batch)
                    del batch, inputs, labels, logits, loss
                _merge_numeric(loader_totals, adapter.loader_metrics(), "adapter")
            finally:
                adapter.close()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    model.train()
    if total != len(samples):
        raise RuntimeError(f"validation consumed {total} samples; expected {len(samples)}")
    return (
        {
            "record_type": "validation",
            "condition": condition_id,
            "seed": seed,
            "epoch": epoch,
            "optimizer_update": optimizer_update,
            "processed_images": processed_images,
            "validation_top1": 100.0 * top1 / total,
            "validation_top5": 100.0 * top5 / total,
            "validation_loss": loss_sum / total,
            "validation_latency_seconds": seconds,
            "validation_samples": total,
        },
        loader_totals,
    )


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


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    weight_decayer: Any,
    scheduler: Any,
    completed_epoch: int,
    global_optimizer_update: int,
    processed_images: int,
    recipe_hash: str,
    layout_hash: str,
    condition_hash: str,
    initial_model_hash: str,
    metrics_cursor: Mapping[str, int],
    logging_state: Mapping[str, Any],
    integration_checks: Mapping[str, Any],
    loader_totals: Mapping[str, float],
    elapsed_runtime_seconds: float,
    pending_validation_epoch: int | None,
) -> dict[str, Any]:
    rng = capture_rng_state()
    return {
        "format": CHECKPOINT_SCHEMA,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "weight_decay_state": weight_decayer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "gradient_accumulation_state": {
            "microbatches": 0,
            "samples": 0,
            "boundary": "completed-epoch",
        },
        "completed_epoch": int(completed_epoch),
        "global_optimizer_update": int(global_optimizer_update),
        "processed_image_count": int(processed_images),
        "python_rng_state": rng["python"],
        "numpy_rng_state": rng["numpy"],
        "torch_cpu_rng_state": rng["torch_cpu"],
        "cuda_rng_state": rng.get("torch_cuda", []),
        "rng_state": rng,
        "recipe_hash": recipe_hash,
        "layout_hash": layout_hash,
        "condition_contract_hash": condition_hash,
        "initial_model_hash": initial_model_hash,
        "metrics_cursor": dict(metrics_cursor),
        "logging_state": dict(logging_state),
        "integration_checks": dict(integration_checks),
        "loader_totals": dict(loader_totals),
        "elapsed_runtime_seconds": float(elapsed_runtime_seconds),
        "pending_validation_epoch": pending_validation_epoch,
    }


def _restore_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    weight_decayer: Any,
    scheduler: Any,
    recipe_hash: str,
    layout_hash: str,
    condition_hash: str,
    initial_model_hash: str,
    metrics: MetricsWriter,
) -> tuple[int, int, int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "format": CHECKPOINT_SCHEMA,
        "recipe_hash": recipe_hash,
        "layout_hash": layout_hash,
        "condition_contract_hash": condition_hash,
        "initial_model_hash": initial_model_hash,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"resume checkpoint contract mismatch: {observed}")
    accumulation = payload.get("gradient_accumulation_state", {})
    if accumulation.get("microbatches") != 0 or accumulation.get("samples") != 0:
        raise ValueError("resume checkpoint is not at an epoch/accumulation boundary")
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
        {
            "logging_state": dict(payload.get("logging_state", {})),
            "integration_checks": dict(payload.get("integration_checks", {})),
            "loader_totals": dict(payload.get("loader_totals", {})),
            "elapsed_runtime_seconds": float(
                payload.get("elapsed_runtime_seconds", 0.0)
            ),
            "pending_validation_epoch": payload.get("pending_validation_epoch"),
        },
    )


def _save_boundary_checkpoint(
    *,
    output_dir: Path,
    permanent_epoch: int | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    weight_decayer: Any,
    scheduler: Any,
    completed_epoch: int,
    global_optimizer_update: int,
    processed_images: int,
    recipe_hash: str,
    layout_hash: str,
    condition_hash: str,
    initial_model_hash: str,
    metrics: MetricsWriter,
    logging_state: Mapping[str, Any],
    integration_checks: Mapping[str, Any],
    loader_totals: Mapping[str, float],
    elapsed_runtime_seconds: float,
    pending_validation_epoch: int | None,
) -> None:
    latest = output_dir / "latest.pt"
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        weight_decayer=weight_decayer,
        scheduler=scheduler,
        completed_epoch=completed_epoch,
        global_optimizer_update=global_optimizer_update,
        processed_images=processed_images,
        recipe_hash=recipe_hash,
        layout_hash=layout_hash,
        condition_hash=condition_hash,
        initial_model_hash=initial_model_hash,
        metrics_cursor=metrics.cursor(),
        logging_state=logging_state,
        integration_checks=integration_checks,
        loader_totals=loader_totals,
        elapsed_runtime_seconds=elapsed_runtime_seconds,
        pending_validation_epoch=pending_validation_epoch,
    )
    _atomic_checkpoint(latest, payload)
    if permanent_epoch is not None:
        _retain_checkpoint(latest, output_dir / f"checkpoint_epoch_{permanent_epoch:03d}.pt")


def _final_metrics_from_file(path: Path) -> dict[str, Any]:
    final_validation: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("record_type") == "validation":
                final_validation = record
    if final_validation is None:
        raise RuntimeError("completed run has no validation metrics")
    return final_validation


def _train_native_physical_epoch(
    *,
    pipeline: DirectDctPlsPipeline,
    execution_model: torch.nn.Module,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    weight_decayer: Any,
    scheduler: Any,
    device: torch.device,
    recipe: Mapping[str, Any],
    metrics: MetricsWriter,
    loader_totals: dict[str, float],
    integration_checks: dict[str, Any],
    condition_id: str,
    seed: int,
    epoch: int,
    expected_sample_count: int,
    global_update: int,
    processed_images: int,
    loss_since_log: float,
    samples_since_log: int,
    last_logged_update: int,
    integration_check_first_100: bool,
) -> dict[str, Any]:
    """Consume one native-owned physical PLS epoch at the model boundary."""

    epoch_started = time.perf_counter()
    order_policy = str(resolve_condition(condition_id)["order_policy"])
    sample_shuffle_enabled = order_policy != "physical-order"
    epoch_loss_sum = 0.0
    epoch_samples = 0
    epoch_microbatches = 0
    epoch_updates = 0
    pool_count = 0
    pool_load_seconds = 0.0
    pool_boundary_wait_seconds = 0.0
    native_execution_stats: dict[str, Any] = {}
    coverage = np.zeros(expected_sample_count, dtype=np.bool_)
    order_hash = hashlib.sha256()
    pool_membership_hash = hashlib.sha256()
    microbatch_images = int(recipe["training"]["physical_microbatch"])
    accumulation = int(recipe["training"]["gradient_accumulation"])
    pipeline.start_epoch(epoch)

    while pipeline.has_next_pool:
        load_started = time.perf_counter()
        pool = pipeline.next_pool()
        pool_load_seconds += time.perf_counter() - load_started
        if pool.epoch != epoch or pool.pool_index != pool_count:
            raise RuntimeError(
                "native PLS pool epoch/index differs from the streaming cursor"
            )
        for pls_id in pool.virtual_pls_ids:
            pool_membership_hash.update(int(pls_id).to_bytes(8, "little"))
        pool_membership_hash.update((2**63 - 1).to_bytes(8, "little"))

        pool_microbatches = int(pool.microbatch_count)
        pool_images = int(pool.image_count)
        pool_seen_images = 0
        for window_begin in range(0, pool_microbatches, accumulation):
            window_microbatches = min(
                accumulation, pool_microbatches - window_begin
            )
            window_sample_count = min(
                window_microbatches * microbatch_images,
                pool_images - window_begin * microbatch_images,
            )
            if window_sample_count <= 0:
                raise RuntimeError("native PLS accumulation window is empty")
            optimizer.zero_grad(set_to_none=True)
            learning_rate = scheduler.prepare_next_update()
            update_loss_sum = 0.0
            for _local_index in range(window_microbatches):
                native_batch = next(pool)
                y, cbcr, targets = native_batch.tensors
                image_ids = native_batch.global_image_ids
                labels = native_batch.labels
                batch_size = len(image_ids)
                if (
                    batch_size <= 0
                    or batch_size != int(y.shape[0])
                    or batch_size != int(cbcr.shape[0])
                    or batch_size != int(targets.shape[0])
                    or batch_size != len(labels)
                ):
                    raise RuntimeError("native PLS tensor/identity cardinalities differ")
                for image_id in image_ids:
                    if image_id < 0 or image_id >= expected_sample_count:
                        raise RuntimeError(
                            f"native PLS emitted out-of-range physical image ID {image_id}"
                        )
                    if coverage[image_id]:
                        raise RuntimeError(
                            f"native PLS emitted duplicate physical image ID {image_id}"
                        )
                    coverage[image_id] = True
                    order_hash.update(int(image_id).to_bytes(8, "little"))
                logits = execution_model(y, cbcr)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                if not bool(torch.isfinite(loss).item()) or not bool(
                    torch.isfinite(logits).all().item()
                ):
                    raise FloatingPointError(
                        f"non-finite native PLS loss/logits at epoch {epoch} "
                        f"pool {pool_count} microbatch {epoch_microbatches}"
                    )
                (loss * (batch_size / window_sample_count)).backward()
                update_loss_sum += float(loss.detach().item()) * batch_size
                pool_seen_images += batch_size
                epoch_microbatches += 1
                del native_batch, y, cbcr, targets, logits, loss

            if not _all_finite(
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ):
                raise FloatingPointError(
                    f"non-finite gradients at optimizer update {global_update + 1}"
                )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(recipe["optimizer"]["gradient_clipping_norm"]),
            )
            optimizer.step()
            weight_decayer.step(learning_rate)
            scheduler.complete_update()
            global_update += 1
            epoch_updates += 1
            processed_images += window_sample_count
            epoch_samples += window_sample_count
            epoch_loss_sum += update_loss_sum
            loss_since_log += update_loss_sum
            samples_since_log += window_sample_count
            if integration_check_first_100 and global_update <= 100:
                integration_checks["checked_updates"] += 1
                if not _all_finite(model.parameters()):
                    integration_checks["parameters_finite"] = False
                    raise FloatingPointError(
                        f"non-finite model parameters at integration update {global_update}"
                    )
            if global_update % int(
                recipe["logging"]["train_loss_every_optimizer_updates"]
            ) == 0:
                metrics.append(
                    {
                        "record_type": "train",
                        "scope": "optimizer-window",
                        "condition": condition_id,
                        "seed": seed,
                        "epoch": epoch + 1,
                        "optimizer_update": global_update,
                        "processed_images": processed_images,
                        "train_loss": loss_since_log / samples_since_log,
                        "learning_rate": learning_rate,
                        "window_optimizer_updates": global_update
                        - last_logged_update,
                        "window_samples": samples_since_log,
                        "execution_backend": NATIVE_PHYSICAL_BACKEND,
                    }
                )
                loss_since_log = 0.0
                samples_since_log = 0
                last_logged_update = global_update

        if pool_seen_images != pool_images:
            raise RuntimeError(
                f"native PLS pool consumed {pool_seen_images}/{pool_images} images"
            )
        wait_started = time.perf_counter()
        torch.cuda.synchronize(device)
        pool_stats = pool.execution_stats
        for key, value in pool_stats.items():
            if isinstance(value, bool):
                native_execution_stats[key] = bool(
                    native_execution_stats.get(key, True) and value
                )
            elif isinstance(value, (int, float)):
                native_execution_stats[key] = (
                    native_execution_stats.get(key, 0) + value
                )
            elif value:
                previous = native_execution_stats.get(key)
                if previous is None:
                    native_execution_stats[key] = value
                elif previous != value:
                    native_execution_stats[key] = "mixed"
        del pool
        pipeline.reclaim_finished_pools()
        pool_boundary_wait_seconds += time.perf_counter() - wait_started
        pool_count += 1

    if epoch_samples != expected_sample_count or not bool(coverage.all()):
        integration_checks["coverage_counters_valid"] = False
        raise RuntimeError(
            f"native PLS epoch {epoch} consumed {epoch_samples}/"
            f"{expected_sample_count} unique images"
        )
    epoch_seconds = time.perf_counter() - epoch_started
    loader_totals["native_pool_load_seconds"] = (
        loader_totals.get("native_pool_load_seconds", 0.0) + pool_load_seconds
    )
    loader_totals["native_pool_boundary_wait_seconds"] = (
        loader_totals.get("native_pool_boundary_wait_seconds", 0.0)
        + pool_boundary_wait_seconds
    )
    loader_totals["native_pool_count"] = (
        loader_totals.get("native_pool_count", 0.0) + pool_count
    )
    for key, value in native_execution_stats.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            loader_key = f"native_execution.{key}"
            loader_totals[loader_key] = loader_totals.get(loader_key, 0.0) + float(
                value
            )
    return {
        "global_update": global_update,
        "processed_images": processed_images,
        "loss_since_log": loss_since_log,
        "samples_since_log": samples_since_log,
        "last_logged_update": last_logged_update,
        "epoch_record": {
            "record_type": "train",
            "scope": "epoch",
            "condition": condition_id,
            "seed": seed,
            "epoch": epoch + 1,
            "optimizer_update": global_update,
            "processed_images": processed_images,
            "train_loss": epoch_loss_sum / epoch_samples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_samples": epoch_samples,
            "epoch_microbatches": epoch_microbatches,
            "epoch_optimizer_updates": epoch_updates,
            "epoch_seconds": epoch_seconds,
            "images_per_second": epoch_samples / epoch_seconds,
            "data_preparation_seconds": 0.0,
            "native_pool_load_seconds": pool_load_seconds,
            "native_pool_boundary_wait_seconds": pool_boundary_wait_seconds,
            "native_pool_count": pool_count,
            "native_execution_stats": native_execution_stats,
            "sample_order_digest": order_hash.hexdigest(),
            "pool_membership_digest": pool_membership_hash.hexdigest(),
            "crop_key_digest": "native-owned-by-rgbnomore-training-pls-v1",
            "flip_key_digest": "native-owned-by-rgbnomore-training-pls-v1",
            "randaugment_digest": "native-owned-by-rgbnomore-training-pls-v1",
            "mixup_digest": "native-owned-by-rgbnomore-training-pls-v1",
            "execution_backend": NATIVE_PHYSICAL_BACKEND,
            "native_crop_pushdown": True,
            "native_physical_order": True,
            "gpu_resident_closed_pool": True,
            "sample_order_policy": order_policy,
            "sample_shuffle_enabled": sample_shuffle_enabled,
            "cuda_transform": True,
            "cuda_ordered_output_placement": True,
            "cuda_mixup": True,
            "cuda_transform_shuffle_mixup": sample_shuffle_enabled,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    recipe = assert_recipe_overrides(recipe=RECIPE_NAME, epochs=args.epochs)
    condition = resolve_condition(args.condition)
    native_physical = args.execution_backend == NATIVE_PHYSICAL_BACKEND
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = load_layout_mapping(args.layout_plan)
    manifest_path = getattr(args, "run_manifest", None) or getattr(
        args, "condition_contract", None
    )
    if manifest_path is None:
        raise ValueError("a run manifest is required")
    contract, manifest_validation = _validate_contract(
        manifest_path,
        condition_id=args.condition,
        seed=args.seed,
        recipe_hash=recipe["recipe_hash"],
        layout_hash=layout.layout_hash,
        execution_backend=args.execution_backend,
        physical_galp_manifest=args.physical_galp_manifest,
        premixed_mapping_csv=args.premixed_mapping_csv,
        expected_mapping_sha256=args.expected_mapping_sha256,
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the formal PLS core run requires an available CUDA device")
    torch.cuda.set_device(device)
    train_samples, train_meta = _load_manifest_fast(
        args.train_manifest, expected_split="train"
    )
    val_samples, val_meta = _load_manifest_fast(args.val_manifest, expected_split="val")
    validate_manifest_against_layout(
        layout, manifest_path=args.train_manifest, samples=train_samples
    )
    if {sample.logical_sample_id for sample in train_samples} & {
        sample.logical_sample_id for sample in val_samples
    }:
        raise ValueError("training and validation logical sample IDs overlap")
    train_galp_manifest = (
        args.physical_galp_manifest.resolve()
        if native_physical
        else Path(train_meta["galp_manifest"])
    )
    val_galp_manifest = Path(val_meta["galp_manifest"])
    for path in (train_galp_manifest, val_galp_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_reader = (
        None
        if native_physical
        else DirectDctTrainingReader(
            train_galp_manifest, module_path=args.galp_torch_module_path
        )
    )
    val_reader = DirectDctTrainingReader(
        val_galp_manifest, module_path=args.galp_torch_module_path
    )
    if train_reader is not None and train_reader.image_count != len(train_samples):
        raise ValueError("training GALP image count differs from training manifest")
    if val_reader.image_count != len(val_samples):
        raise ValueError("validation GALP image count differs from validation manifest")

    model, initial_model_hash = build_paired_model(
        args.rgbnomore_root, seed=args.seed, device=device
    )
    if contract["initial_model_hash"] != initial_model_hash:
        raise ValueError(
            "runtime initial model hash differs from the paired condition contract"
        )
    total_updates = int(contract["total_optimizer_updates"])
    optimizer, weight_decayer, scheduler = build_published_optimizer(
        model,
        learning_rate=float(recipe["optimizer"]["learning_rate"]),
        weight_decay=float(recipe["optimizer"]["weight_decay"]["coefficient"]),
        warmup_updates=int(recipe["scheduler"]["warmup_optimizer_updates"]),
        total_updates=total_updates,
    )
    metrics = MetricsWriter(output_dir / "metrics.jsonl")
    latest = output_dir / "latest.pt"
    completed_epoch = 0
    global_update = 0
    processed_images = 0
    resume_bookkeeping: dict[str, Any] = {}
    if latest.is_file():
        if not args.resume:
            raise ValueError(
                f"checkpoint already exists at {latest}; pass --resume to continue"
            )
        (
            completed_epoch,
            global_update,
            processed_images,
            resume_bookkeeping,
        ) = _restore_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            weight_decayer=weight_decayer,
            scheduler=scheduler,
            recipe_hash=recipe["recipe_hash"],
            layout_hash=layout.layout_hash,
            condition_hash=contract["condition_hash"],
            initial_model_hash=initial_model_hash,
            metrics=metrics,
        )
    elif any(output_dir.iterdir()):
        allowed = {
            "condition_contract.json",
            "run_manifest.json",
            "run_status.json",
            "environment.json",
            "failure.json",
            "failures.jsonl",
            "attempts",
        }
        unexpected = [
            path.name
            for path in output_dir.iterdir()
            if path.name not in allowed and not path.name.startswith("failure_")
        ]
        if unexpected:
            raise ValueError(f"new run output directory contains unexpected files: {unexpected}")

    execution_model = compile_published_model(model, recipe)

    native_pipeline: DirectDctPlsPipeline | None = None
    if native_physical:
        native_pipeline = DirectDctPlsPipeline(
            train_galp_manifest,
            args.premixed_mapping_csv,
            training_seed=args.seed,
            expected_mapping_sha256=args.expected_mapping_sha256,
            crop_policy=str(condition["crop_policy"]),
            order_policy=str(condition["order_policy"]),
            segments_per_pool=int(condition.get("segments_per_pool") or 4),
            microbatch_images=int(recipe["training"]["physical_microbatch"]),
            segment_images=int(layout.plan["target_pls_size"]),
            model_classes=int(recipe["model"]["classes"]),
            module_path=args.galp_torch_module_path,
        )
        if native_pipeline.sample_count != layout.sample_count:
            raise ValueError(
                "native physical PLS mapping and frozen layout have different populations"
            )

    environment = _device_environment(device)
    environment.update(
        execution_mode=(
            "native_physical_pls" if native_physical else "semantic_emulation"
        ),
        semantic_emulation=not native_physical,
        physical_fls_observed=native_physical,
        physical_gpu_pool=native_physical,
        layout_hash=layout.layout_hash,
        recipe_hash=recipe["recipe_hash"],
        condition_hash=contract["condition_hash"],
        initial_model_hash=initial_model_hash,
        model_execution=recipe["execution"],
        backend_implementation=(
            "galp-native-direct-dct-pls-block-major-v1"
            if native_physical
            else "shared-galp-direct-dct-semantic-backend-v2"
        ),
        train_manifest=train_meta,
        validation_manifest=val_meta,
        galp_torch_module_path=str(args.galp_torch_module_path.resolve()),
        run_manifest=str(manifest_path.resolve()),
        run_manifest_validation=manifest_validation,
    )
    if native_physical:
        environment["physical_execution"] = dict(contract["physical_execution"])
    _atomic_json(output_dir / "environment.json", environment)
    run_status = {
        "schema_version": RUN_SCHEMA,
        "state": "running",
        "condition": args.condition,
        "seed": args.seed,
        "started_at_unix": time.time(),
        "completed_epoch": completed_epoch,
        "optimizer_update": global_update,
        "processed_images": processed_images,
    }
    _atomic_json(output_dir / "run_status.json", run_status)

    validation_grid = set(validation_epochs())
    loader_totals: dict[str, float] = {
        str(key): float(value)
        for key, value in resume_bookkeeping.get("loader_totals", {}).items()
    }
    elapsed_runtime_seconds = float(
        resume_bookkeeping.get("elapsed_runtime_seconds", 0.0)
    )
    runtime_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    logging_state = resume_bookkeeping.get("logging_state", {})
    loss_since_log = float(logging_state.get("loss_since_log", 0.0))
    samples_since_log = int(logging_state.get("samples_since_log", 0))
    last_logged_update = int(logging_state.get("last_logged_update", global_update))
    default_integration_checks = {
        "required_updates": 100 if args.integration_check_first_100 else 0,
        "checked_updates": min(global_update, 100)
        if args.integration_check_first_100
        else 0,
        "loss_finite": True,
        "gradients_finite": True,
        "parameters_finite": True,
        "coverage_counters_valid": True,
        "contract_hash_valid": True,
        "metrics_output_valid": True,
    }
    integration_checks = dict(default_integration_checks)
    integration_checks.update(resume_bookkeeping.get("integration_checks", {}))

    if completed_epoch == 0 and metrics.lines == 0:
        event, validation_loader = _run_validation(
            model=execution_model,
            samples=val_samples,
            reader=val_reader,
            galp_manifest=val_galp_manifest,
            module_path=args.galp_torch_module_path,
            prefetch_depth=args.prefetch_depth,
            workers=args.workers,
            device=device,
            condition_id=args.condition,
            seed=args.seed,
            epoch=0,
            optimizer_update=0,
            processed_images=0,
        )
        metrics.append(event)
        _merge_numeric(loader_totals, validation_loader, "validation")
        _save_boundary_checkpoint(
            output_dir=output_dir,
            permanent_epoch=0,
            model=model,
            optimizer=optimizer,
            weight_decayer=weight_decayer,
            scheduler=scheduler,
            completed_epoch=0,
            global_optimizer_update=0,
            processed_images=0,
            recipe_hash=recipe["recipe_hash"],
            layout_hash=layout.layout_hash,
            condition_hash=contract["condition_hash"],
            initial_model_hash=initial_model_hash,
            metrics=metrics,
            logging_state={
                "loss_since_log": loss_since_log,
                "samples_since_log": samples_since_log,
                "last_logged_update": last_logged_update,
            },
            integration_checks=integration_checks,
            loader_totals=loader_totals,
            elapsed_runtime_seconds=(
                elapsed_runtime_seconds + time.perf_counter() - runtime_started
            ),
            pending_validation_epoch=None,
        )

    adapter_config = (
        None
        if native_physical
        else _adapter_config(
            reader=train_reader,
            galp_manifest=train_galp_manifest,
            module_path=args.galp_torch_module_path,
            prefetch_depth=args.prefetch_depth,
        )
    )
    pending_validation_epoch = resume_bookkeeping.get("pending_validation_epoch")
    if pending_validation_epoch is not None:
        pending_validation_epoch = int(pending_validation_epoch)
        if pending_validation_epoch != completed_epoch:
            raise ValueError("checkpoint pending validation epoch is inconsistent")
        validation, validation_loader = _run_validation(
            model=execution_model,
            samples=val_samples,
            reader=val_reader,
            galp_manifest=val_galp_manifest,
            module_path=args.galp_torch_module_path,
            prefetch_depth=args.prefetch_depth,
            workers=args.workers,
            device=device,
            condition_id=args.condition,
            seed=args.seed,
            epoch=completed_epoch,
            optimizer_update=global_update,
            processed_images=processed_images,
        )
        metrics.append(validation)
        _merge_numeric(loader_totals, validation_loader, "validation")
        _save_boundary_checkpoint(
            output_dir=output_dir,
            permanent_epoch=completed_epoch,
            model=model,
            optimizer=optimizer,
            weight_decayer=weight_decayer,
            scheduler=scheduler,
            completed_epoch=completed_epoch,
            global_optimizer_update=global_update,
            processed_images=processed_images,
            recipe_hash=recipe["recipe_hash"],
            layout_hash=layout.layout_hash,
            condition_hash=contract["condition_hash"],
            initial_model_hash=initial_model_hash,
            metrics=metrics,
            logging_state={
                "loss_since_log": loss_since_log,
                "samples_since_log": samples_since_log,
                "last_logged_update": last_logged_update,
            },
            integration_checks=integration_checks,
            loader_totals=loader_totals,
            elapsed_runtime_seconds=(
                elapsed_runtime_seconds + time.perf_counter() - runtime_started
            ),
            pending_validation_epoch=None,
        )
    for epoch in range(completed_epoch, int(recipe["training"]["epochs"])):
        if native_physical:
            if native_pipeline is None:
                raise RuntimeError("native physical backend was not initialized")
            native_epoch = _train_native_physical_epoch(
                pipeline=native_pipeline,
                execution_model=execution_model,
                model=model,
                optimizer=optimizer,
                weight_decayer=weight_decayer,
                scheduler=scheduler,
                device=device,
                recipe=recipe,
                metrics=metrics,
                loader_totals=loader_totals,
                integration_checks=integration_checks,
                condition_id=args.condition,
                seed=args.seed,
                epoch=epoch,
                expected_sample_count=layout.sample_count,
                global_update=global_update,
                processed_images=processed_images,
                loss_since_log=loss_since_log,
                samples_since_log=samples_since_log,
                last_logged_update=last_logged_update,
                integration_check_first_100=args.integration_check_first_100,
            )
            global_update = int(native_epoch["global_update"])
            processed_images = int(native_epoch["processed_images"])
            loss_since_log = float(native_epoch["loss_since_log"])
            samples_since_log = int(native_epoch["samples_since_log"])
            last_logged_update = int(native_epoch["last_logged_update"])
            metrics.append(native_epoch["epoch_record"])
            completed_epoch = epoch + 1
            if completed_epoch in validation_grid:
                _save_boundary_checkpoint(
                    output_dir=output_dir,
                    permanent_epoch=None,
                    model=model,
                    optimizer=optimizer,
                    weight_decayer=weight_decayer,
                    scheduler=scheduler,
                    completed_epoch=completed_epoch,
                    global_optimizer_update=global_update,
                    processed_images=processed_images,
                    recipe_hash=recipe["recipe_hash"],
                    layout_hash=layout.layout_hash,
                    condition_hash=contract["condition_hash"],
                    initial_model_hash=initial_model_hash,
                    metrics=metrics,
                    logging_state={
                        "loss_since_log": loss_since_log,
                        "samples_since_log": samples_since_log,
                        "last_logged_update": last_logged_update,
                    },
                    integration_checks=integration_checks,
                    loader_totals=loader_totals,
                    elapsed_runtime_seconds=(
                        elapsed_runtime_seconds
                        + time.perf_counter()
                        - runtime_started
                    ),
                    pending_validation_epoch=completed_epoch,
                )
                validation, validation_loader = _run_validation(
                    model=execution_model,
                    samples=val_samples,
                    reader=val_reader,
                    galp_manifest=val_galp_manifest,
                    module_path=args.galp_torch_module_path,
                    prefetch_depth=args.prefetch_depth,
                    workers=args.workers,
                    device=device,
                    condition_id=args.condition,
                    seed=args.seed,
                    epoch=completed_epoch,
                    optimizer_update=global_update,
                    processed_images=processed_images,
                )
                metrics.append(validation)
                _merge_numeric(loader_totals, validation_loader, "validation")
            _save_boundary_checkpoint(
                output_dir=output_dir,
                permanent_epoch=(
                    completed_epoch if completed_epoch in validation_grid else None
                ),
                model=model,
                optimizer=optimizer,
                weight_decayer=weight_decayer,
                scheduler=scheduler,
                completed_epoch=completed_epoch,
                global_optimizer_update=global_update,
                processed_images=processed_images,
                recipe_hash=recipe["recipe_hash"],
                layout_hash=layout.layout_hash,
                condition_hash=contract["condition_hash"],
                initial_model_hash=initial_model_hash,
                metrics=metrics,
                logging_state={
                    "loss_since_log": loss_since_log,
                    "samples_since_log": samples_since_log,
                    "last_logged_update": last_logged_update,
                },
                integration_checks=integration_checks,
                loader_totals=loader_totals,
                elapsed_runtime_seconds=(
                    elapsed_runtime_seconds + time.perf_counter() - runtime_started
                ),
                pending_validation_epoch=None,
            )
            run_status.update(
                completed_epoch=completed_epoch,
                optimizer_update=global_update,
                processed_images=processed_images,
            )
            _atomic_json(output_dir / "run_status.json", run_status)
            if (
                args.stop_after_epoch is not None
                and completed_epoch >= args.stop_after_epoch
                and completed_epoch < int(recipe["training"]["epochs"])
            ):
                native_pipeline.close()
                paused = {
                    "schema_version": RUN_SCHEMA,
                    "state": "paused-at-epoch-boundary",
                    "condition": args.condition,
                    "seed": args.seed,
                    "completed_epoch": completed_epoch,
                    "optimizer_update": global_update,
                    "processed_images": processed_images,
                    "integration_checks": integration_checks,
                    "latest_checkpoint": str((output_dir / "latest.pt").resolve()),
                    "resume_semantics": (
                        "rerun the same command with --resume and without "
                        "--stop-after-epoch"
                    ),
                    "scientific_result": False,
                    "execution_backend": NATIVE_PHYSICAL_BACKEND,
                }
                _atomic_json(output_dir / "pause_result.json", paused)
                run_status.update(
                    state="paused-at-epoch-boundary",
                    ended_at_unix=time.time(),
                    pause_result="pause_result.json",
                )
                _atomic_json(output_dir / "run_status.json", run_status)
                return paused
            continue
        epoch_started = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        epoch_microbatches = 0
        epoch_updates = 0
        epoch_position = 0
        global_microbatch_index = 0
        coverage = np.zeros(layout.sample_count, dtype=np.bool_)
        order_hash = hashlib.sha256()
        crop_hash = hashlib.sha256()
        flip_hash = hashlib.sha256()
        randaugment_hash = hashlib.sha256()
        mixup_hash = hashlib.sha256()
        pool_membership_hash = hashlib.sha256()
        data_preparation_seconds = 0.0
        for pool_index, pool_pls_ids, positions in epoch_position_pools(
            layout,
            condition_id=args.condition,
            seed=args.seed,
            epoch=epoch,
        ):
            for pls_id in pool_pls_ids:
                pool_membership_hash.update(int(pls_id).to_bytes(8, "little"))
            pool_membership_hash.update((2**63 - 1).to_bytes(8, "little"))
            pool_microbatch_count = (len(positions) + 63) // 64
            for chunk_begin in range(
                0, pool_microbatch_count, DESCRIPTOR_CHUNK_MICROBATCHES
            ):
                chunk_end = min(
                    pool_microbatch_count,
                    chunk_begin + DESCRIPTOR_CHUNK_MICROBATCHES,
                )
                preparation_started = time.perf_counter()
                (
                    microbatch_positions,
                    identities,
                    augmentations,
                    next_epoch_position,
                ) = _schedule_chunk(
                    positions=positions,
                    chunk_begin_microbatch=chunk_begin,
                    chunk_end_microbatch=chunk_end,
                    epoch=epoch,
                    pool_index=pool_index,
                    epoch_position_begin=epoch_position,
                    microbatch_index_begin=global_microbatch_index,
                    mapping=layout,
                    samples=train_samples,
                    condition=condition,
                    seed=args.seed,
                )
                chunk_samples = [
                    train_samples[int(layout.manifest_indices[position])]
                    for batch in microbatch_positions
                    for position in batch
                ]
                adapter = _make_adapter(
                    samples=chunk_samples,
                    identities=identities,
                    augmentations=augmentations,
                    device=device,
                    workers=args.workers,
                    config=adapter_config,
                )
                data_preparation_seconds += time.perf_counter() - preparation_started
                try:
                    for window_begin in range(0, len(identities), 16):
                        window_end = min(len(identities), window_begin + 16)
                        window_sample_count = sum(
                            len(batch) for batch in identities[window_begin:window_end]
                        )
                        optimizer.zero_grad(set_to_none=True)
                        learning_rate = scheduler.prepare_next_update()
                        update_loss_sum = 0.0
                        for local_index in range(window_begin, window_end):
                            batch = adapter.next_batch()
                            expected = identities[local_index]
                            if [value.logical_sample_id for value in batch.identities] != [
                                value.logical_sample_id for value in expected
                            ]:
                                raise RuntimeError("training adapter changed scheduled sample order")
                            inputs, labels = _move_batch(batch, device)
                            logical_ids = [value.logical_sample_id for value in expected]
                            inputs, randaugment_records = apply_published_randaugment(
                                (inputs[0], inputs[1]),
                                training_seed=args.seed,
                                epoch=epoch,
                                logical_sample_ids=logical_ids,
                                rgbnomore_root=args.rgbnomore_root,
                            )
                            microbatch_index = global_microbatch_index + local_index
                            inputs, mixed_labels, mixup_record = apply_published_mixup(
                                inputs,
                                labels,
                                training_seed=args.seed,
                                epoch=epoch,
                                microbatch_index=microbatch_index,
                                alpha=float(recipe["augmentation"]["mixup"]["alpha"]),
                                classes=int(recipe["model"]["classes"]),
                            )
                            logits = execution_model(*inputs)
                            loss = torch.nn.functional.cross_entropy(logits, mixed_labels)
                            if not bool(torch.isfinite(loss).item()) or not bool(
                                torch.isfinite(logits).all().item()
                            ):
                                raise FloatingPointError(
                                    f"non-finite loss/logits at epoch {epoch} microbatch {microbatch_index}"
                                )
                            batch_size = len(expected)
                            (loss * (batch_size / window_sample_count)).backward()
                            update_loss_sum += float(loss.detach().item()) * batch_size
                            for planned_position, augmentation in zip(
                                microbatch_positions[local_index], augmentations[local_index]
                            ):
                                if coverage[planned_position]:
                                    raise RuntimeError(
                                        f"duplicate planned position {planned_position} in epoch {epoch}"
                                    )
                                coverage[planned_position] = True
                                order_hash.update(int(planned_position).to_bytes(8, "little"))
                                crop_hash.update(augmentation.crop_key.encode("ascii"))
                                flip_hash.update(augmentation.flip_key.encode("ascii"))
                            randaugment_hash.update(
                                json.dumps(
                                    randaugment_records,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            )
                            mixup_hash.update(
                                json.dumps(
                                    mixup_record,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            )
                            for name, value in batch.stage_seconds.items():
                                loader_totals[name] = loader_totals.get(name, 0.0) + float(value)
                            adapter.snapshot_batch_metrics(batch)
                            epoch_microbatches += 1
                            del batch, inputs, labels, mixed_labels, logits, loss
                        if not _all_finite(
                            parameter.grad
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        ):
                            raise FloatingPointError(
                                f"non-finite gradients at optimizer update {global_update + 1}"
                            )
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=float(recipe["optimizer"]["gradient_clipping_norm"]),
                        )
                        optimizer.step()
                        weight_decayer.step(learning_rate)
                        scheduler.complete_update()
                        global_update += 1
                        epoch_updates += 1
                        processed_images += window_sample_count
                        epoch_samples += window_sample_count
                        epoch_loss_sum += update_loss_sum
                        loss_since_log += update_loss_sum
                        samples_since_log += window_sample_count
                        if args.integration_check_first_100 and global_update <= 100:
                            integration_checks["checked_updates"] += 1
                            if not _all_finite(model.parameters()):
                                integration_checks["parameters_finite"] = False
                                raise FloatingPointError(
                                    f"non-finite model parameters at integration update {global_update}"
                                )
                        if global_update % int(
                            recipe["logging"]["train_loss_every_optimizer_updates"]
                        ) == 0:
                            metrics.append(
                                {
                                    "record_type": "train",
                                    "scope": "optimizer-window",
                                    "condition": args.condition,
                                    "seed": args.seed,
                                    "epoch": epoch + 1,
                                    "optimizer_update": global_update,
                                    "processed_images": processed_images,
                                    "train_loss": loss_since_log / samples_since_log,
                                    "learning_rate": learning_rate,
                                    "window_optimizer_updates": global_update
                                    - last_logged_update,
                                    "window_samples": samples_since_log,
                                }
                            )
                            loss_since_log = 0.0
                            samples_since_log = 0
                            last_logged_update = global_update
                    _merge_numeric(loader_totals, adapter.loader_metrics(), "train_adapter")
                finally:
                    adapter.close()
                epoch_position = next_epoch_position
                global_microbatch_index += len(identities)
        if epoch_position != layout.sample_count or epoch_samples != layout.sample_count:
            integration_checks["coverage_counters_valid"] = False
            raise RuntimeError(
                f"epoch {epoch} consumed {epoch_samples}/{epoch_position} samples; "
                f"expected {layout.sample_count}"
            )
        if not bool(coverage.all()):
            integration_checks["coverage_counters_valid"] = False
            raise RuntimeError(f"epoch {epoch} has missing planned positions")
        epoch_seconds = time.perf_counter() - epoch_started
        epoch_record = {
            "record_type": "train",
            "scope": "epoch",
            "condition": args.condition,
            "seed": args.seed,
            "epoch": epoch + 1,
            "optimizer_update": global_update,
            "processed_images": processed_images,
            "train_loss": epoch_loss_sum / epoch_samples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_samples": epoch_samples,
            "epoch_microbatches": epoch_microbatches,
            "epoch_optimizer_updates": epoch_updates,
            "epoch_seconds": epoch_seconds,
            "images_per_second": epoch_samples / epoch_seconds,
            "data_preparation_seconds": data_preparation_seconds,
            "sample_order_digest": order_hash.hexdigest(),
            "pool_membership_digest": pool_membership_hash.hexdigest(),
            "crop_key_digest": crop_hash.hexdigest(),
            "flip_key_digest": flip_hash.hexdigest(),
            "randaugment_digest": randaugment_hash.hexdigest(),
            "mixup_digest": mixup_hash.hexdigest(),
        }
        metrics.append(epoch_record)
        completed_epoch = epoch + 1
        if completed_epoch in validation_grid:
            _save_boundary_checkpoint(
                output_dir=output_dir,
                permanent_epoch=None,
                model=model,
                optimizer=optimizer,
                weight_decayer=weight_decayer,
                scheduler=scheduler,
                completed_epoch=completed_epoch,
                global_optimizer_update=global_update,
                processed_images=processed_images,
                recipe_hash=recipe["recipe_hash"],
                layout_hash=layout.layout_hash,
                condition_hash=contract["condition_hash"],
                initial_model_hash=initial_model_hash,
                metrics=metrics,
                logging_state={
                    "loss_since_log": loss_since_log,
                    "samples_since_log": samples_since_log,
                    "last_logged_update": last_logged_update,
                },
                integration_checks=integration_checks,
                loader_totals=loader_totals,
                elapsed_runtime_seconds=(
                    elapsed_runtime_seconds + time.perf_counter() - runtime_started
                ),
                pending_validation_epoch=completed_epoch,
            )
        if completed_epoch in validation_grid:
            validation, validation_loader = _run_validation(
                model=execution_model,
                samples=val_samples,
                reader=val_reader,
                galp_manifest=val_galp_manifest,
                module_path=args.galp_torch_module_path,
                prefetch_depth=args.prefetch_depth,
                workers=args.workers,
                device=device,
                condition_id=args.condition,
                seed=args.seed,
                epoch=completed_epoch,
                optimizer_update=global_update,
                processed_images=processed_images,
            )
            metrics.append(validation)
            _merge_numeric(loader_totals, validation_loader, "validation")
        _save_boundary_checkpoint(
            output_dir=output_dir,
            permanent_epoch=(completed_epoch if completed_epoch in validation_grid else None),
            model=model,
            optimizer=optimizer,
            weight_decayer=weight_decayer,
            scheduler=scheduler,
            completed_epoch=completed_epoch,
            global_optimizer_update=global_update,
            processed_images=processed_images,
            recipe_hash=recipe["recipe_hash"],
            layout_hash=layout.layout_hash,
            condition_hash=contract["condition_hash"],
            initial_model_hash=initial_model_hash,
            metrics=metrics,
            logging_state={
                "loss_since_log": loss_since_log,
                "samples_since_log": samples_since_log,
                "last_logged_update": last_logged_update,
            },
            integration_checks=integration_checks,
            loader_totals=loader_totals,
            elapsed_runtime_seconds=(
                elapsed_runtime_seconds + time.perf_counter() - runtime_started
            ),
            pending_validation_epoch=None,
        )
        run_status.update(
            completed_epoch=completed_epoch,
            optimizer_update=global_update,
            processed_images=processed_images,
        )
        _atomic_json(output_dir / "run_status.json", run_status)
        if (
            args.stop_after_epoch is not None
            and completed_epoch >= args.stop_after_epoch
            and completed_epoch < int(recipe["training"]["epochs"])
        ):
            paused = {
                "schema_version": RUN_SCHEMA,
                "state": "paused-at-epoch-boundary",
                "condition": args.condition,
                "seed": args.seed,
                "completed_epoch": completed_epoch,
                "optimizer_update": global_update,
                "processed_images": processed_images,
                "integration_checks": integration_checks,
                "latest_checkpoint": str((output_dir / "latest.pt").resolve()),
                "resume_semantics": "rerun the same command with --resume and without --stop-after-epoch",
                "scientific_result": False,
            }
            _atomic_json(output_dir / "pause_result.json", paused)
            run_status.update(
                state="paused-at-epoch-boundary",
                ended_at_unix=time.time(),
                pause_result="pause_result.json",
            )
            _atomic_json(output_dir / "run_status.json", run_status)
            return paused

    if global_update != total_updates:
        raise RuntimeError(
            f"completed {global_update} optimizer updates; contract requires {total_updates}"
        )
    if processed_images != layout.sample_count * int(recipe["training"]["epochs"]):
        raise RuntimeError("final processed image count is inconsistent with full epochs")
    if args.integration_check_first_100 and integration_checks["checked_updates"] < 100:
        raise RuntimeError("first-seed integration check did not observe 100 optimizer updates")
    if native_pipeline is not None:
        torch.cuda.synchronize(device)
        native_pipeline.reclaim_finished_pools()
        native_pipeline.close()
    final_validation = _final_metrics_from_file(metrics.path)
    runtime = elapsed_runtime_seconds + time.perf_counter() - runtime_started
    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }
    result = {
        "schema_version": RUN_SCHEMA,
        "state": "completed",
        "condition": args.condition,
        "seed": args.seed,
        "execution_mode": (
            "native_physical_pls" if native_physical else "semantic_emulation"
        ),
        "semantic_emulation": not native_physical,
        "physical_fls_observed": native_physical,
        "physical_gpu_pool": native_physical,
        "layout_hash": layout.layout_hash,
        "recipe_hash": recipe["recipe_hash"],
        "condition_hash": contract["condition_hash"],
        "initial_model_hash": initial_model_hash,
        "model_execution": recipe["execution"],
        "backend_implementation": (
            "galp-native-direct-dct-pls-block-major-v1"
            if native_physical
            else "shared-galp-direct-dct-semantic-backend-v2"
        ),
        "final_top1": final_validation["validation_top1"],
        "final_top5": final_validation["validation_top5"],
        "final_validation_loss": final_validation["validation_loss"],
        "total_processed_images": processed_images,
        "total_optimizer_updates": global_update,
        "training_runtime_seconds": runtime,
        "images_per_second": processed_images / runtime,
        "cuda_memory": memory,
        "loader_metrics_numeric_sums": loader_totals,
        "integration_checks": integration_checks,
        "checkpoint": {
            "latest": str((output_dir / "latest.pt").resolve()),
            "latest_sha256": sha256_file(output_dir / "latest.pt"),
            "final_epoch": completed_epoch,
        },
        "claim_boundary": (
            (
                "model training used the registered premixed block-major manifest, "
                "native crop pushdown/physical ordering, one GPU-resident M=4 pool, "
                "and CUDA transform/ordered placement/augmentation; sample shuffle "
                f"policy was {condition['order_policy']}; physical byte reduction "
                "requires the recorded native counters"
            )
            if native_physical
            else (
                "model effect observed with a frozen virtual PLS mapping and the shared "
                "Direct-DCT semantic backend; no full physical FLS or byte-reduction claim"
            )
        ),
    }
    if native_physical:
        result["physical_execution"] = dict(contract["physical_execution"])
    _atomic_json(output_dir / "final_result.json", result)
    run_status.update(
        state="completed",
        ended_at_unix=time.time(),
        final_result="final_result.json",
    )
    _atomic_json(output_dir / "run_status.json", run_status)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--layout-plan", type=Path, required=True)
    manifest_group = parser.add_mutually_exclusive_group(required=True)
    manifest_group.add_argument(
        "--run-manifest",
        type=Path,
        help="resolved run manifest generated by training_pls.run_matrix",
    )
    manifest_group.add_argument(
        "--condition-contract",
        type=Path,
        help="deprecated alias for --run-manifest",
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=("A0", "A1", "B2", "B6", "N6", "N2"),
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--recipe", choices=(RECIPE_NAME,), default=RECIPE_NAME)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--execution-backend",
        choices=(SEMANTIC_BACKEND, NATIVE_PHYSICAL_BACKEND),
        default=SEMANTIC_BACKEND,
    )
    parser.add_argument("--physical-galp-manifest", type=Path)
    parser.add_argument("--premixed-mapping-csv", type=Path)
    parser.add_argument("--expected-mapping-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument(
        "--galp-torch-module-path", type=Path, default=REPO_ROOT / "build/galp/torch"
    )
    parser.add_argument(
        "--rgbnomore-root", type=Path, default=Path("/home/tangyuxin/RGB-no-more")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--integration-check-first-100", action="store_true")
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help=(
            "operational preemption point; pause only after this completed epoch "
            "has been checkpointed (not part of the scientific condition)"
        ),
    )
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.prefetch_depth < 0:
        raise ValueError("prefetch depth must be non-negative")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch < 300:
        raise ValueError("--stop-after-epoch must be in [1, 299]")
    physical_values = (
        args.physical_galp_manifest,
        args.premixed_mapping_csv,
        args.expected_mapping_sha256,
    )
    if args.execution_backend == NATIVE_PHYSICAL_BACKEND:
        if any(value is None for value in physical_values):
            raise ValueError(
                "native-physical-pls requires --physical-galp-manifest, "
                "--premixed-mapping-csv, and --expected-mapping-sha256"
            )
        for path in (args.physical_galp_manifest, args.premixed_mapping_csv):
            if not path.is_file():
                raise FileNotFoundError(path)
        if len(args.expected_mapping_sha256) != 64:
            raise ValueError("--expected-mapping-sha256 must contain 64 hex characters")
        try:
            int(args.expected_mapping_sha256, 16)
        except ValueError as error:
            raise ValueError(
                "--expected-mapping-sha256 must contain 64 hex characters"
            ) from error
        if sha256_file(args.premixed_mapping_csv) != args.expected_mapping_sha256:
            raise ValueError(
                "--expected-mapping-sha256 differs from --premixed-mapping-csv"
            )
    elif any(value is not None for value in physical_values):
        raise ValueError(
            "physical PLS paths require --execution-backend native-physical-pls"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        failed_at = time.time()
        failure = {
            "state": "failed",
            "condition": args.condition,
            "seed": args.seed,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at_unix": failed_at,
        }
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(args.output_dir / "failure.json", failure)
            _atomic_json(
                args.output_dir / f"failure_{time.time_ns()}.json", failure
            )
            with (args.output_dir / "failures.jsonl").open(
                "a", encoding="utf-8"
            ) as output:
                output.write(json.dumps(failure, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
            _atomic_json(args.output_dir / "run_status.json", failure)
        except BaseException:
            pass
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
