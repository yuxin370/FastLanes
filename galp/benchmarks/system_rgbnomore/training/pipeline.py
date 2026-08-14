#!/usr/bin/env python3
"""Four auditable training data-pipeline adapters.

Adapters own data acquisition and preprocessing only.  The training lifecycle,
optimizer, reset, timing window, and artifact generation remain controlled by
``training/run.py``.
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import multiprocessing
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch

from training.artifacts import sha256_file, sha256_json
from training.augmentation import (
    AugmentationDecision,
    apply_rgb_augmentation_staged,
    horizontal_flip_dct,
)
from training.direct_dct_reader import (
    DirectDctTrainingReader,
    optional_native_execution_stats,
    optional_native_execution_stats_snapshot,
)
from training.sample_order import SampleIdentity
from training.schema import DOMAINS, PIPELINES


@dataclass(frozen=True)
class TrainingSample:
    logical_sample_id: str
    path: Path
    label: int
    width: int
    height: int
    galp_image_id: int | None = None
    payload_sha256: str | None = None

    def as_contract_record(self) -> dict[str, Any]:
        return {
            "logical_sample_id": self.logical_sample_id,
            "path": str(self.path),
            "label": self.label,
            "width": self.width,
            "height": self.height,
            "galp_image_id": self.galp_image_id,
            "payload_sha256": self.payload_sha256,
        }


@dataclass
class TrainingBatch:
    inputs: tuple[torch.Tensor, ...]
    labels: torch.Tensor
    identities: list[SampleIdentity]
    augmentations: list[dict[str, Any]]
    on_device: bool
    stage_seconds: dict[str, float] = field(default_factory=dict)
    native_counters: dict[str, int | float] = field(default_factory=dict)
    native_execution_stats: dict[str, Any] = field(default_factory=dict)
    native_stats_finalized: bool = False
    keepalive: list[Any] = field(default_factory=list)
    native_stats_source: Any | None = None


@dataclass
class _AsyncQueueEntry:
    sequence: int
    metadata: Any
    handle: Any


class OrderedAsyncPrefetchQueue:
    """Bounded FIFO ownership for asynchronous producer handles."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("prefetch queue capacity must be positive")
        self.capacity = int(capacity)
        self._entries: deque[_AsyncQueueEntry] = deque()
        self._closed = False
        self._next_sequence = 0
        self._metrics: dict[str, int | float | bool] = {
            "capacity_batches": self.capacity,
            "submitted_batches": 0,
            "consumed_batches": 0,
            "max_queue_depth_batches": 0,
            "queue_hit_batches": 0,
            "queue_miss_batches": 0,
            "backpressure_events": 0,
            "producer_submit_seconds": 0.0,
            "producer_active_seconds": 0.0,
            "producer_planning_seconds": 0.0,
            "producer_io_staging_seconds": 0.0,
            "producer_ordered_submission_seconds": 0.0,
            "consumer_wait_seconds": 0.0,
            "async_worker_started_batches": 0,
            "cancelled_batches": 0,
            "drained_batches": 0,
            "close_errors": 0,
            "closed": False,
        }

    @staticmethod
    def _boolean_property(handle: Any, name: str, default: bool = False) -> bool:
        value = getattr(handle, name, default)
        return bool(value() if callable(value) else value)

    @staticmethod
    def _numeric_property(handle: Any, name: str, default: float = 0.0) -> float:
        value = getattr(handle, name, default)
        return float(value() if callable(value) else value)

    def _accumulate_completed_handle_metrics(self, handle: Any) -> None:
        self._metrics["producer_active_seconds"] += (
            self._numeric_property(handle, "producer_active_ms") / 1000.0
        )
        self._metrics["producer_planning_seconds"] += (
            self._numeric_property(handle, "planning_ms") / 1000.0
        )
        self._metrics["producer_io_staging_seconds"] += (
            self._numeric_property(handle, "io_staging_ms") / 1000.0
        )
        self._metrics["producer_ordered_submission_seconds"] += (
            self._numeric_property(handle, "ordered_submission_ms") / 1000.0
        )

    def submit(self, metadata: Any, factory: Callable[[], Any]) -> None:
        if self._closed:
            raise RuntimeError("prefetch queue is closed")
        if len(self._entries) >= self.capacity:
            self._metrics["backpressure_events"] += 1
            raise RuntimeError(
                f"prefetch queue capacity {self.capacity} reached; consumer must make progress"
            )
        begin = time.perf_counter()
        handle = factory()
        self._metrics["producer_submit_seconds"] += time.perf_counter() - begin
        self._entries.append(_AsyncQueueEntry(self._next_sequence, metadata, handle))
        self._next_sequence += 1
        self._metrics["submitted_batches"] += 1
        self._metrics["max_queue_depth_batches"] = max(
            int(self._metrics["max_queue_depth_batches"]), len(self._entries)
        )

    def pop(self) -> tuple[Any, Any]:
        if not self._entries:
            raise StopIteration
        entry = self._entries.popleft()
        ready = self._boolean_property(entry.handle, "ready")
        self._metrics["queue_hit_batches" if ready else "queue_miss_batches"] += 1
        begin = time.perf_counter()
        try:
            value = entry.handle.read()
        finally:
            self._metrics["consumer_wait_seconds"] += time.perf_counter() - begin
            if self._boolean_property(entry.handle, "started"):
                self._metrics["async_worker_started_batches"] += 1
            self._accumulate_completed_handle_metrics(entry.handle)
        self._metrics["consumed_batches"] += 1
        return entry.metadata, value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        entries = list(self._entries)
        self._entries.clear()
        cancellation: list[bool] = []
        # Request cancellation for every queued successor before joining the
        # active head; otherwise the ordered native chain could start each
        # successor while close was blocked draining its predecessor.
        for entry in entries:
            cancelled = False
            cancel = getattr(entry.handle, "cancel", None)
            if callable(cancel):
                try:
                    cancelled = bool(cancel())
                    self._metrics["cancelled_batches"] += int(cancelled)
                except Exception:
                    self._metrics["close_errors"] += 1
            cancellation.append(cancelled)
        for entry, cancelled in zip(entries, cancellation):
            try:
                entry.handle.read()
                self._metrics["drained_batches"] += 1
            except Exception:
                if not cancelled:
                    self._metrics["close_errors"] += 1
            if self._boolean_property(entry.handle, "started"):
                self._metrics["async_worker_started_batches"] += 1
            self._accumulate_completed_handle_metrics(entry.handle)
        self._metrics["closed"] = True

    def metrics(self) -> dict[str, Any]:
        result = dict(self._metrics)
        result["current_queue_depth_batches"] = len(self._entries)
        consumed = int(result["consumed_batches"])
        result["queue_hit_rate"] = (
            float(result["queue_hit_batches"]) / consumed if consumed else None
        )
        return result


def _manifest_samples(payload: dict[str, Any], manifest_path: Path, root: Path | None) -> list[TrainingSample]:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"training manifest has no samples: {manifest_path}")
    samples: list[TrainingSample] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise ValueError(f"training manifest sample {index} is not an object")
        sample_id = str(raw.get("logical_sample_id", raw.get("sample_id", "")))
        if not sample_id:
            raise ValueError(f"training manifest sample {index} lacks logical_sample_id")
        if sample_id in seen:
            raise ValueError(f"training manifest duplicates logical_sample_id {sample_id!r}")
        seen.add(sample_id)
        raw_path = Path(str(raw["path"]))
        path = raw_path if raw_path.is_absolute() else (root or manifest_path.parent) / raw_path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        label = int(raw["label"])
        if not 0 <= label < 1000:
            raise ValueError(f"sample {sample_id!r} has invalid ImageNet-1K label {label}")
        width = int(raw.get("width", 0))
        height = int(raw.get("height", 0))
        if width <= 0 or height <= 0:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
        expected_sha = raw.get("payload_sha256", raw.get("sha256"))
        samples.append(
            TrainingSample(
                logical_sample_id=sample_id,
                path=path,
                label=label,
                width=width,
                height=height,
                galp_image_id=None if raw.get("galp_image_id") is None else int(raw["galp_image_id"]),
                payload_sha256=None if expected_sha is None else str(expected_sha),
            )
        )
    return samples


def load_training_manifest(path: Path, *, root: Path | None, expected_split: str) -> tuple[list[TrainingSample], dict[str, Any]]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = str(payload.get("split", ""))
    aliases = {"train": {"train", "training"}, "val": {"val", "validation"}}
    if split not in aliases[expected_split]:
        raise ValueError(f"{path} split is {split!r}, expected {expected_split!r}")
    samples = _manifest_samples(payload, path, root)
    label_mapping = payload.get("label_mapping", "imagenet-1k-zero-based")
    if label_mapping != "imagenet-1k-zero-based":
        raise ValueError(
            f"unsupported label mapping {label_mapping!r}; expected 'imagenet-1k-zero-based'"
        )
    index_rows = [
        {
            "logical_sample_id": sample.logical_sample_id,
            "label": sample.label,
            "path": str(sample.path),
            "width": sample.width,
            "height": sample.height,
            "galp_image_id": sample.galp_image_id,
            "declared_payload_sha256": sample.payload_sha256,
        }
        for sample in samples
    ]
    payload_records = []
    for sample in samples:
        actual = sha256_file(sample.path)
        if sample.payload_sha256 is not None and sample.payload_sha256 != actual:
            raise ValueError(f"payload hash mismatch for {sample.logical_sample_id}: {sample.path}")
        payload_records.append({"logical_sample_id": sample.logical_sample_id, "sha256": actual})
    metadata = {
        "path": str(path),
        "split": expected_split,
        "sample_count": len(samples),
        "manifest_sha256": sha256_file(path),
        "index_sha256": sha256_json(index_rows),
        "payload_fingerprint_sha256": sha256_json(payload_records),
        "label_mapping": label_mapping,
        "source_split": payload.get("source_split"),
        "validation_semantics": payload.get("validation_semantics"),
        "declared_galp_manifest": (
            None
            if payload.get("galp_manifest") is None
            else str(
                (
                    Path(str(payload["galp_manifest"]))
                    if Path(str(payload["galp_manifest"])).is_absolute()
                    else path.parent / Path(str(payload["galp_manifest"]))
                ).resolve()
            )
        ),
    }
    return samples, metadata


def validate_dataset_separation(train: Sequence[TrainingSample], val: Sequence[TrainingSample]) -> dict[str, Any]:
    train_ids = {sample.logical_sample_id for sample in train}
    val_ids = {sample.logical_sample_id for sample in val}
    train_paths = {sample.path for sample in train}
    val_paths = {sample.path for sample in val}
    id_overlap = sorted(train_ids & val_ids)
    path_overlap = sorted(map(str, train_paths & val_paths))
    return {
        "ok": not id_overlap and not path_overlap,
        "logical_sample_id_overlap": id_overlap,
        "path_overlap": path_overlap,
    }


def load_index_csv(path: Path, root: Path | None = None) -> list[TrainingSample]:
    """Small utility for fixtures and converted RGB-no-more indexes."""

    result: list[TrainingSample] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            sample_path = Path(row["Filepath"])
            if not sample_path.is_absolute():
                sample_path = (root or path.parent) / sample_path
            from PIL import Image

            with Image.open(sample_path) as image:
                width, height = image.size
            result.append(
                TrainingSample(row["Filepath"], sample_path.resolve(), int(row["Label"]), width, height)
            )
    return result


def _resolve_plan(
    samples: dict[str, TrainingSample],
    identities: Sequence[SampleIdentity],
    decisions: Sequence[AugmentationDecision],
) -> list[tuple[TrainingSample, SampleIdentity, AugmentationDecision]]:
    if len(identities) != len(decisions):
        raise ValueError("identity and augmentation plan lengths differ")
    result = []
    for identity, decision in zip(identities, decisions):
        sample = samples.get(identity.logical_sample_id)
        if sample is None:
            raise KeyError(f"sample order references unknown ID {identity.logical_sample_id!r}")
        if decision.logical_sample_id != identity.logical_sample_id or decision.epoch != identity.epoch:
            raise ValueError("augmentation decision does not match sample identity")
        result.append((sample, identity, decision))
    return result


def _uniform_dali_tensor(output: Any) -> tuple[Any, Any]:
    """Collapse a uniform DALI TensorList using DALI's native shape object."""

    tensor = output.as_tensor()
    return tensor, tensor.shape()


class TrainingPipelineAdapter:
    pipeline: str
    domain: str
    gpu_only = False

    def __init__(
        self,
        samples: Sequence[TrainingSample],
        *,
        batch_size: int,
        workers: int,
        device: torch.device,
        config: dict[str, Any],
    ) -> None:
        self.samples = {sample.logical_sample_id: sample for sample in samples}
        self.batch_size = batch_size
        self.workers = workers
        self.device = device
        self.config = config
        self._iterator: Iterator[Any] | None = None
        self._planned: list[tuple[TrainingSample, SampleIdentity, AugmentationDecision]] = []
        self._read_indices: Any = []
        self._manager: Any = None

    def begin(
        self,
        identities: Sequence[SampleIdentity],
        decisions: Sequence[AugmentationDecision],
        batch_lengths: Sequence[int] | None = None,
    ) -> None:
        self._planned = _resolve_plan(self.samples, identities, decisions)
        self._batch_lengths = list(batch_lengths or [])
        if not self._batch_lengths:
            self._batch_lengths = [
                min(self.batch_size, len(self._planned) - begin)
                for begin in range(0, len(self._planned), self.batch_size)
            ]
        if any(length <= 0 or length > self.batch_size for length in self._batch_lengths):
            raise ValueError("invalid planned batch length")
        if sum(self._batch_lengths) != len(self._planned):
            raise ValueError("planned batch lengths do not cover the identity plan")
        self._batch_ranges: list[list[int]] = []
        cursor = 0
        for length in self._batch_lengths:
            self._batch_ranges.append(list(range(cursor, cursor + length)))
            cursor += length

    def next_batch(self) -> TrainingBatch:
        raise NotImplementedError

    def finalize_batch_metrics(self, batch: TrainingBatch) -> None:
        """Materialize metrics that may synchronize a device, after timing."""

    def snapshot_batch_metrics(self, batch: TrainingBatch) -> None:
        """Capture metrics without retaining a device batch when supported."""

        self.finalize_batch_metrics(batch)

    def loader_metrics(self) -> dict[str, Any]:
        return {
            "worker_semantics": "framework-native workers",
            "configured_workers": self.workers,
            "prefetch": "framework-managed",
        }

    def prefetched_read_identities(self) -> list[SampleIdentity]:
        return [self._planned[int(index)][1] for index in list(self._read_indices)]

    def end(self) -> None:
        self._iterator = None
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None

    def close(self) -> None:
        self.end()


class _PlannedRgbDataset(torch.utils.data.Dataset):
    def __init__(self, plan: Sequence[tuple[TrainingSample, SampleIdentity, AugmentationDecision]], read_indices: Any) -> None:
        self.plan = list(plan)
        self.read_indices = read_indices

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int):
        from PIL import Image

        sample, _identity, decision = self.plan[index]
        start = time.perf_counter()
        encoded = sample.path.read_bytes()
        read_seconds = time.perf_counter() - start
        start = time.perf_counter()
        with Image.open(io.BytesIO(encoded)) as image:
            rgb = image.convert("RGB")
            rgb.load()
        decode_seconds = time.perf_counter() - start
        tensor, stages = apply_rgb_augmentation_staged(rgb, decision)
        self.read_indices.append(index)
        return (
            tensor,
            sample.label,
            index,
            read_seconds,
            decode_seconds,
            stages["augmentation"],
            stages["preprocess"],
        )


class PyTorchTrainingAdapter(TrainingPipelineAdapter):
    pipeline = "pytorch"
    domain = "rgb"

    def begin(
        self,
        identities: Sequence[SampleIdentity],
        decisions: Sequence[AugmentationDecision],
        batch_lengths: Sequence[int] | None = None,
    ) -> None:
        super().begin(identities, decisions, batch_lengths)
        if self.workers > 0:
            self._manager = multiprocessing.Manager()
            self._read_indices = self._manager.list()
        else:
            self._read_indices = []
        loader = torch.utils.data.DataLoader(
            _PlannedRgbDataset(self._planned, self._read_indices),
            batch_sampler=self._batch_ranges,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.workers > 0,
            prefetch_factor=int(self.config.get("prefetch_depth", 2)) if self.workers > 0 else None,
        )
        self._iterator = iter(loader)

    def next_batch(self) -> TrainingBatch:
        if self._iterator is None:
            raise RuntimeError("pipeline repeat has not begun")
        begin = time.perf_counter()
        images, labels, indices, read, decode, augmentation, preprocess = next(self._iterator)
        wait = time.perf_counter() - begin
        indices_list = [int(index) for index in indices.tolist()]
        plan = [self._planned[index] for index in indices_list]
        return TrainingBatch(
            inputs=(images,),
            labels=labels.long(),
            identities=[item[1] for item in plan],
            augmentations=[item[2].as_dict() for item in plan],
            on_device=False,
            stage_seconds={
                "loader_data_wait": wait,
                "read": float(read.sum().item()),
                "decode": float(decode.sum().item()),
                "augmentation": float(augmentation.sum().item()),
                "preprocess": float(preprocess.sum().item()),
            },
        )


class _RgbNoMoreDctDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        plan: Sequence[tuple[TrainingSample, SampleIdentity, AugmentationDecision]],
        read_indices: Any,
        rgbnomore_root: Path,
    ) -> None:
        self.plan = list(plan)
        self.read_indices = read_indices
        root_text = str(rgbnomore_root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

    def __len__(self) -> int:
        return len(self.plan)

    @staticmethod
    def _component_crop(
        decision: AugmentationDecision,
        *,
        block_width: int,
        block_height: int,
    ) -> tuple[int, int, int, int]:
        """Map a source-pixel crop to one JPEG component's padded block grid."""

        if block_width <= 0 or block_height <= 0:
            raise ValueError("DCT component block dimensions must be positive")

        def axis(origin: int, extent: int, blocks: int, pixels: int) -> tuple[int, int]:
            begin = origin * blocks // pixels
            end = min(
                blocks,
                ((origin + extent) * blocks + pixels - 1) // pixels,
            )
            if begin >= end:
                raise ValueError("source-pixel crop maps to an empty DCT component")
            return begin, end - begin

        left, width = axis(
            decision.crop_x,
            decision.crop_width,
            block_width,
            decision.source_width,
        )
        top, height = axis(
            decision.crop_y,
            decision.crop_height,
            block_height,
            decision.source_height,
        )
        return top, left, height, width

    def __getitem__(self, index: int):
        sample, _identity, decision = self.plan[index]
        start = time.perf_counter()
        dct_manip = importlib.import_module("dct_manip")
        dops = importlib.import_module("utils.dct_ops")
        _dimensions, quantization, y, cbcr = dct_manip.read_coefficients(str(sample.path))
        read_decode_seconds = time.perf_counter() - start

        start = time.perf_counter()
        y = torch.clamp(y * quantization[0], min=-1024, max=1016)
        if cbcr is None:
            cbcr = torch.zeros((2, y.shape[1] // 2, y.shape[2] // 2, 8, 8), dtype=y.dtype)
        else:
            cbcr = torch.clamp(cbcr * quantization[1:3, None, None], min=-1024, max=1016)
        y_crop = self._component_crop(
            decision,
            block_width=int(y.shape[2]),
            block_height=int(y.shape[1]),
        )
        cbcr_crop = self._component_crop(
            decision,
            block_width=int(cbcr.shape[2]),
            block_height=int(cbcr.shape[1]),
        )
        y = dops.crop_dct(y, *y_crop)
        cbcr = dops.crop_dct(cbcr, *cbcr_crop)
        y = dops.resize_dct(y, 28, dtype=torch.float32)
        cbcr = dops.resize_dct(cbcr, 14, dtype=torch.float32)
        if decision.horizontal_flip:
            y, cbcr = horizontal_flip_dct(y, cbcr)
        augmentation_seconds = time.perf_counter() - start

        start = time.perf_counter()
        y = (y.float() + 1024.0) / 2040.0 * 2.0 - 1.0
        cbcr = (cbcr.float() + 1024.0) / 2040.0 * 2.0 - 1.0
        preprocess_seconds = time.perf_counter() - start
        self.read_indices.append(index)
        return (
            y,
            cbcr,
            sample.label,
            index,
            read_decode_seconds,
            augmentation_seconds,
            preprocess_seconds,
        )


class RgbNoMoreTrainingAdapter(TrainingPipelineAdapter):
    pipeline = "rgbnomore"
    domain = "dct"

    def begin(
        self,
        identities: Sequence[SampleIdentity],
        decisions: Sequence[AugmentationDecision],
        batch_lengths: Sequence[int] | None = None,
    ) -> None:
        super().begin(identities, decisions, batch_lengths)
        if self.workers > 0:
            self._manager = multiprocessing.Manager()
            self._read_indices = self._manager.list()
        else:
            self._read_indices = []
        dataset = _RgbNoMoreDctDataset(
            self._planned, self._read_indices, Path(self.config["rgbnomore_root"])
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=self._batch_ranges,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.workers > 0,
            prefetch_factor=int(self.config.get("prefetch_depth", 2)) if self.workers > 0 else None,
        )
        self._iterator = iter(loader)

    def next_batch(self) -> TrainingBatch:
        if self._iterator is None:
            raise RuntimeError("pipeline repeat has not begun")
        begin = time.perf_counter()
        y, cbcr, labels, indices, read_decode, augmentation, preprocess = next(self._iterator)
        wait = time.perf_counter() - begin
        indices_list = [int(index) for index in indices.tolist()]
        plan = [self._planned[index] for index in indices_list]
        return TrainingBatch(
            inputs=(y, cbcr),
            labels=labels.long(),
            identities=[item[1] for item in plan],
            augmentations=[item[2].as_dict() for item in plan],
            on_device=False,
            stage_seconds={
                "loader_data_wait": wait,
                "read_decode": float(read_decode.sum().item()),
                "augmentation": float(augmentation.sum().item()),
                "preprocess": float(preprocess.sum().item()),
            },
        )


class GalpTrainingAdapter(TrainingPipelineAdapter):
    pipeline = "galp"
    domain = "dct"
    gpu_only = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        injected_reader = self.config.get("_direct_dct_training_reader")
        if self.device.type != "cuda" and injected_reader is None:
            raise RuntimeError("GALP training adapter requires a CUDA device")
        module_path = self.config.get("galp_torch_module_path")
        self.reader = injected_reader or DirectDctTrainingReader(
            Path(self.config["galp_manifest"]),
            module_path=None if module_path is None else Path(module_path),
        )
        self._read_indices = []
        self.execution_mode = str(self.config.get("execution_mode", "audit"))
        if self.execution_mode not in ("audit", "runtime"):
            raise ValueError(f"invalid GALP execution mode {self.execution_mode!r}")
        if self.workers <= 0:
            raise ValueError(
                "GALP --workers controls native rowgroup-prefetch workers and must be at least 1"
            )
        self._queue: OrderedAsyncPrefetchQueue | None = None
        self._reported_consumer_wait = 0.0
        pls_pool_config = self.config.get("pls_gpu_pool") or {}
        self._pls_gpu_pool = bool(pls_pool_config.get("enabled", False))
        self._pls_closed_pool_specs = list(
            pls_pool_config.get("closed_pool_batches", [])
        )
        self._pls_pool_ranges: list[list[int]] = []
        self._active_pls_pool: dict[str, Any] | None = None
        self._next_emit_batch = 0
        self._pls_pool_metrics: dict[str, int | bool | str] = {
            "enabled": self._pls_gpu_pool,
            "pool_lifetime": (
                "load-complete-pool; consume-completely; release; load-next"
                if self._pls_gpu_pool
                else "not-enabled"
            ),
            "configured_pool_count": len(self._pls_closed_pool_specs),
            "materialized_pool_count": 0,
            "materialized_sample_count": 0,
            "fully_emitted_release_eligible_pool_count": 0,
            "max_materialized_pool_samples": 0,
            "simultaneously_active_pool_limit": 1 if self._pls_gpu_pool else 0,
        }

    def _native_arguments(self) -> dict[str, Any]:
        return {
            "dct_coeffs": "all",
            "layout": "transformed-dct-grid",
            "grid_transform": {
                "y_output_width_blocks": 28,
                "y_output_height_blocks": 28,
                "cbcr_output_width_blocks": 14,
                "cbcr_output_height_blocks": 14,
                "crop_reference_width_blocks": 32,
                "crop_reference_height_blocks": 32,
                "crop_origin_alignment_blocks": 2,
                "chroma_crop_scale_x": 2,
                "chroma_crop_scale_y": 2,
                "clamp_min": -1024,
                "clamp_max": 1016,
                "output_dtype": "float32",
                "output_add": 4.0,
                "output_scale": 1.0 / 1020.0,
                "dequantize": True,
                "require_all_coefficients": True,
                "allow_grayscale": True,
                "preferred_small_crop_width_blocks": [2, 4, 14, 28],
                "preferred_small_crop_height_blocks": [2, 4, 14, 28],
                "allowed_chroma_sampling_ratios": [[1, 1, 1, 1], [1, 2, 1, 2]],
            },
            "cache_capacity_mib": int(self.config.get("galp_cache_capacity_mib", 0)),
            "rowgroup_prefetch_workers": self.workers,
        }

    def _enqueue_next(self) -> None:
        ranges = self._pls_pool_ranges if self._pls_gpu_pool else self._batch_ranges
        if self._next_enqueue >= len(ranges):
            return
        indices = ranges[self._next_enqueue]
        plan = [self._planned[index] for index in indices]
        image_ids: list[int] = []
        transforms: list[dict[str, Any]] = []
        for sample, _identity, decision in plan:
            if sample.galp_image_id is None:
                raise ValueError(f"GALP sample {sample.logical_sample_id!r} lacks galp_image_id")
            image_ids.append(sample.galp_image_id)
            transforms.append(decision.native_dct_descriptor())
        if self._queue is None:
            raise RuntimeError("GALP prefetch queue is not initialized")
        self._queue.submit(
            (indices, image_ids, transforms),
            lambda: self.reader.prefetch_batch(
                image_ids, transforms=transforms, **self._native_arguments()
            ),
        )
        self._read_indices.extend(indices)
        self._next_enqueue += 1

    def begin(
        self,
        identities: Sequence[SampleIdentity],
        decisions: Sequence[AugmentationDecision],
        batch_lengths: Sequence[int] | None = None,
    ) -> None:
        super().begin(identities, decisions, batch_lengths)
        self._next_enqueue = 0
        self._read_indices = []
        self._reported_consumer_wait = 0.0
        self._prefetch_depth = int(self.config.get("prefetch_depth", 2))
        self._active_pls_pool = None
        self._next_emit_batch = 0
        self._pls_pool_ranges = []
        if self._pls_gpu_pool:
            batch_cursor = 0
            for spec in self._pls_closed_pool_specs:
                optimizer_batches = int(spec["optimizer_batches"])
                if optimizer_batches <= 0:
                    raise ValueError("PLS pool must contain at least one optimizer batch")
                selected_ranges = self._batch_ranges[
                    batch_cursor : batch_cursor + optimizer_batches
                ]
                if len(selected_ranges) != optimizer_batches:
                    raise ValueError("PLS pool specifications exceed the planned batches")
                indices = [index for values in selected_ranges for index in values]
                if len(indices) != int(spec["sample_count"]):
                    raise ValueError("PLS pool sample count does not match optimizer batches")
                self._pls_pool_ranges.append(indices)
                batch_cursor += optimizer_batches
            if batch_cursor != len(self._batch_ranges):
                raise ValueError("PLS pool specifications do not cover every planned batch")
        capacity = 1 if self._pls_gpu_pool else self._prefetch_depth + 1
        self._queue = OrderedAsyncPrefetchQueue(capacity)
        initial_depth = min(
            len(self._pls_pool_ranges if self._pls_gpu_pool else self._batch_ranges),
            capacity,
        )
        for _ in range(initial_depth):
            self._enqueue_next()

    @staticmethod
    def _validate_native_provenance(
        batch: Any,
        image_ids: Sequence[int],
        transforms: Sequence[dict[str, Any]],
    ) -> None:
        actual_descriptors = list(batch.transform_descriptors)
        actual_image_ids = list(batch.global_image_ids)
        if len(actual_descriptors) != len(transforms) or len(actual_image_ids) != len(image_ids):
            raise RuntimeError("GALP native batch changed the requested batch cardinality")
        for actual, actual_image_id, expected, image_id in zip(
            actual_descriptors, actual_image_ids, transforms, image_ids
        ):
            if int(actual_image_id) != int(image_id):
                raise RuntimeError("GALP native batch changed the requested image ID order")
            if int(actual["global_image_id"]) != int(image_id):
                raise RuntimeError("GALP native transform provenance changed the requested image ID")
            for field in ("crop", "horizontal_flip", "logical_sample_id", "augmentation_key"):
                if actual[field] != expected[field]:
                    raise RuntimeError(f"GALP native transform provenance mismatch for {field}")

    def _load_active_pls_pool(self) -> None:
        if self._queue is None:
            raise RuntimeError("GALP prefetch queue is not initialized")
        if int(self._queue.metrics()["current_queue_depth_batches"]) == 0:
            self._enqueue_next()
        try:
            (indices, image_ids, transforms), native_batch = self._queue.pop()
            self._validate_native_provenance(native_batch, image_ids, transforms)
        except Exception:
            self._queue.close()
            raise
        queue_wait = float(self._queue.metrics()["consumer_wait_seconds"])
        wait = queue_wait - self._reported_consumer_wait
        self._reported_consumer_wait = queue_wait
        self._active_pls_pool = {
            "indices": indices,
            "batch": native_batch,
            "offset": 0,
            "consumer_wait_seconds": wait,
            "native_stats_pending": True,
        }
        sample_count = len(indices)
        self._pls_pool_metrics["materialized_pool_count"] = int(
            self._pls_pool_metrics["materialized_pool_count"]
        ) + 1
        self._pls_pool_metrics["materialized_sample_count"] = int(
            self._pls_pool_metrics["materialized_sample_count"]
        ) + sample_count
        self._pls_pool_metrics["max_materialized_pool_samples"] = max(
            int(self._pls_pool_metrics["max_materialized_pool_samples"]),
            sample_count,
        )

    def _next_pls_pool_batch(self) -> TrainingBatch:
        if self._active_pls_pool is None:
            self._load_active_pls_pool()
        assert self._active_pls_pool is not None
        if self._next_emit_batch >= len(self._batch_ranges):
            raise StopIteration
        expected_indices = self._batch_ranges[self._next_emit_batch]
        offset = int(self._active_pls_pool["offset"])
        indices = self._active_pls_pool["indices"]
        observed_indices = indices[offset : offset + len(expected_indices)]
        if observed_indices != expected_indices:
            raise RuntimeError("PLS GPU pool does not preserve the scheduled optimizer order")
        native_batch = self._active_pls_pool["batch"]
        plan = [self._planned[index] for index in expected_indices]
        tensors = tuple(
            tensor[offset : offset + len(expected_indices)] for tensor in native_batch.tensors
        )
        stats_pending = bool(self._active_pls_pool["native_stats_pending"])
        training_batch = TrainingBatch(
            inputs=tensors,
            labels=torch.tensor(
                [item[0].label for item in plan], dtype=torch.long, device=self.device
            ),
            identities=[item[1] for item in plan],
            augmentations=[item[2].as_dict() for item in plan],
            on_device=all(getattr(value, "device", None) == self.device for value in tensors),
            stage_seconds={
                "loader_data_wait": (
                    float(self._active_pls_pool["consumer_wait_seconds"])
                    if stats_pending
                    else 0.0
                ),
            },
            keepalive=[native_batch],
            native_stats_source=native_batch if stats_pending else None,
        )
        self._active_pls_pool["native_stats_pending"] = False
        self._active_pls_pool["offset"] = offset + len(expected_indices)
        self._next_emit_batch += 1
        if int(self._active_pls_pool["offset"]) == len(indices):
            self._active_pls_pool = None
            self._pls_pool_metrics["fully_emitted_release_eligible_pool_count"] = int(
                self._pls_pool_metrics["fully_emitted_release_eligible_pool_count"]
            ) + 1
        if self.execution_mode == "audit":
            self.finalize_batch_metrics(training_batch)
        return training_batch

    def next_batch(self) -> TrainingBatch:
        if self._queue is None:
            raise RuntimeError("pipeline repeat has not begun")
        if self._pls_gpu_pool:
            return self._next_pls_pool_batch()
        if int(self._queue.metrics()["current_queue_depth_batches"]) == 0:
            if self._next_enqueue >= len(self._batch_ranges):
                raise StopIteration
            self._enqueue_next()
        try:
            (indices, image_ids, transforms), batch = self._queue.pop()
        except Exception:
            # Join/cancel successors before propagating the producer failure.
            self._queue.close()
            raise
        plan = [self._planned[index] for index in indices]
        queue_wait = float(self._queue.metrics()["consumer_wait_seconds"])
        wait = queue_wait - self._reported_consumer_wait
        self._reported_consumer_wait = queue_wait
        while (
            int(self._queue.metrics()["current_queue_depth_batches"])
            < self._prefetch_depth
            and self._next_enqueue < len(self._batch_ranges)
        ):
            self._enqueue_next()
        try:
            self._validate_native_provenance(batch, image_ids, transforms)
        except Exception:
            self._queue.close()
            raise
        inputs = tuple(batch.tensors)
        training_batch = TrainingBatch(
            inputs=inputs,
            labels=torch.tensor([item[0].label for item in plan], dtype=torch.long, device=self.device),
            identities=[item[1] for item in plan],
            augmentations=[item[2].as_dict() for item in plan],
            on_device=all(getattr(value, "device", None) == self.device for value in inputs),
            stage_seconds={
                "loader_data_wait": wait,
            },
            keepalive=[batch],
            native_stats_source=batch,
        )
        if self.execution_mode == "audit":
            self.finalize_batch_metrics(training_batch)
        return training_batch

    @staticmethod
    def _apply_batch_metrics(
        batch: TrainingBatch, execution_stats: dict[str, Any]
    ) -> None:
        batch.native_execution_stats = execution_stats
        batch.native_stats_finalized = True
        projection = float(execution_stats.get("projection_ms", 0.0)) / 1000.0
        batch.stage_seconds.update(
            {
                "read": float(execution_stats.get("sync_rowgroup_read_ms", 0.0)) / 1000.0,
                "decode": float(execution_stats.get("decode_ms", 0.0)) / 1000.0,
                "augmentation": projection,
                "preprocess": projection,
            }
        )
        for name in (
            "rowgroup_count",
            "decode_kernel_launch_count",
            "fixed_transform_image_count",
            "internal_sync_count",
            "cached_gather_sync_count",
            "decoded_batch_sync_count",
            "rowgroup_storage_bytes_read",
            "pinned_rowgroup_read_count",
            "pinned_rowgroup_read_bytes",
            "compact_batch_buffer_acquire_count",
            "compact_batch_buffer_growth_count",
            "compact_batch_buffer_reuse_count",
            "compact_batch_buffer_requested_bytes",
            "compact_batch_buffer_capacity_bytes",
            "compact_batch_buffer_high_water_bytes",
            "compact_batch_buffer_pageable_fallback_count",
            "compact_batch_read_group_count",
            "compact_batch_read_worker_count",
            "plan_device_batch_ms",
            "compile_io_plan_ms",
            "reader_lookup_ms",
            "descriptor_open_ms",
            "schema_plan_build_ms",
            "static_metadata_wait_ms",
            "parallel_reader_resolve_ms",
            "parallel_reader_resolve_workers",
            "dynamic_image_planning_ms",
            "crop_geometry_planning_ms",
            "crop_interval_planning_ms",
            "axis_program_planning_ms",
            "rowgroup_binding_planning_ms",
            "plan_finalize_ms",
            "reader_cache_hit_count",
            "reader_cache_miss_count",
            "reader_cache_eviction_count",
            "static_metadata_cache_hit_count",
            "static_metadata_cache_miss_count",
            "descriptor_map_count",
            "static_metadata_wait_count",
            "active_reader_count",
            "active_reader_peak_count",
            "static_metadata_count",
            "static_metadata_peak_count",
            "static_metadata_bytes",
            "static_metadata_peak_bytes",
            "planning_unique_shard_count",
            "planning_rowgroup_binding_count",
            "static_metadata_prewarm_ms",
            "static_metadata_prewarm_shards",
            "static_metadata_prewarm_workers",
            "payload_fd_current_count",
            "payload_fd_peak_count",
            "payload_fd_open_count",
            "payload_fd_close_count",
            "descriptor_mapping_current_count",
            "descriptor_mapping_peak_count",
            "descriptor_map_process_count",
            "descriptor_unmap_count",
            "descriptor_mapped_current_bytes",
            "descriptor_mapped_peak_bytes",
            "compact_batch_pool_prewarmed_slots",
            "compact_batch_pool_prewarmed_bytes",
            "compact_batch_pool_largest_size_class_bytes",
            "compact_batch_pool_capacity_contract_images",
            "compact_batch_pool_capacity_contract_groups",
            "compact_batch_pool_capacity_contract_batches",
            "compact_batch_pool_capacity_contract_bytes",
            "compact_read_group_planning_ms",
            "decode_workset_capacity_plan_image_count",
            "decode_workset_output_arena_capacity_plan_bytes",
            "decode_workset_output_arena_requested_bytes",
            "decode_workset_output_arena_capacity_bytes",
            "decode_workset_output_arena_growth_count",
            "decode_workset_output_arena_growth_bytes",
            "decode_workset_chunk_arena_capacity_plan_bytes",
            "decode_workset_chunk_arena_requested_bytes",
            "decode_workset_chunk_arena_capacity_bytes",
            "decode_workset_chunk_arena_growth_count",
            "decode_workset_chunk_arena_growth_bytes",
            "galp_native_device_in_use_bytes",
            "galp_native_device_peak_in_use_bytes",
            "galp_native_device_cached_bytes",
            "galp_native_device_allocation_requests",
            "galp_native_device_cuda_allocation_count",
            "galp_native_device_cuda_allocation_bytes",
            "galp_native_pinned_in_use_bytes",
            "galp_native_pinned_peak_in_use_bytes",
            "galp_native_pinned_cached_bytes",
            "galp_native_pinned_allocation_requests",
            "galp_native_pinned_cuda_allocation_count",
            "galp_native_pinned_cuda_allocation_bytes",
        ):
            value = execution_stats.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                batch.native_counters[name] = value

    def finalize_batch_metrics(self, batch: TrainingBatch) -> None:
        source = batch.native_stats_source
        if source is None or batch.native_stats_finalized:
            return
        stats_method = getattr(source, "native_execution_stats", None)
        execution_stats = (
            dict(stats_method())
            if callable(stats_method)
            else optional_native_execution_stats(source)
        )
        self._apply_batch_metrics(batch, execution_stats)

    def snapshot_batch_metrics(self, batch: TrainingBatch) -> None:
        source = batch.native_stats_source
        if source is None or batch.native_stats_finalized:
            return
        stats_method = getattr(source, "native_execution_stats_snapshot", None)
        execution_stats = (
            dict(stats_method())
            if callable(stats_method)
            else optional_native_execution_stats_snapshot(source)
        )
        self._apply_batch_metrics(batch, execution_stats)

    def loader_metrics(self) -> dict[str, Any]:
        metrics = self._queue.metrics() if self._queue is not None else {}
        metrics.update(
            {
                "worker_semantics": (
                    "native rowgroup-prefetch workers within one ordered GALP producer"
                ),
                "configured_workers": self.workers,
                "actual_batch_producer_workers": 1,
                "native_rowgroup_prefetch_workers": self.workers,
                "execution_mode": self.execution_mode,
                "physical_load_segment_gpu_pool": dict(self._pls_pool_metrics),
            }
        )
        return metrics

    def end(self) -> None:
        if self._queue is not None:
            self._queue.close()
        super().end()


class DaliTrainingAdapter(TrainingPipelineAdapter):
    pipeline = "dali"
    domain = "rgb"
    gpu_only = True

    def begin(
        self,
        identities: Sequence[SampleIdentity],
        decisions: Sequence[AugmentationDecision],
        batch_lengths: Sequence[int] | None = None,
    ) -> None:
        super().begin(identities, decisions, batch_lengths)
        if self.device.type != "cuda":
            raise RuntimeError("DALI training adapter requires a CUDA device")
        from nvidia.dali import fn, types
        from nvidia.dali.pipeline import Pipeline

        self._cursor = 0
        self._read_indices = []
        batches: list[tuple[list[Path], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for indices in self._batch_ranges:
            plan = [self._planned[index] for index in indices]
            paths = [item[0].path for item in plan]
            anchors = np.asarray(
                [[item[2].crop_y / item[2].source_height, item[2].crop_x / item[2].source_width] for item in plan],
                dtype=np.float32,
            )
            shapes = np.asarray(
                [[item[2].crop_height / item[2].source_height, item[2].crop_width / item[2].source_width] for item in plan],
                dtype=np.float32,
            )
            mirrors = np.asarray([int(item[2].horizontal_flip) for item in plan], dtype=np.int32)
            indices = np.asarray(indices, dtype=np.int64)
            batches.append((paths, anchors, shapes, mirrors, indices))
        source_iterator = iter(batches)
        self._dali_read_seconds: dict[tuple[int, ...], float] = {}

        def source():
            value = next(source_iterator)
            read_begin = time.perf_counter()
            encoded = [np.fromfile(path, dtype=np.uint8) for path in value[0]]
            read_seconds = time.perf_counter() - read_begin
            self._read_indices.extend(int(index) for index in value[4].tolist())
            self._dali_read_seconds[tuple(int(index) for index in value[4].tolist())] = read_seconds
            return encoded, value[1], value[2], value[3], value[4]

        pipeline = Pipeline(
            batch_size=self.batch_size,
            num_threads=max(1, self.workers),
            device_id=self.device.index or 0,
            prefetch_queue_depth=int(self.config.get("prefetch_depth", 2)),
            exec_pipelined=True,
            exec_async=True,
        )
        with pipeline:
            encoded, anchors, shapes, mirrors, indices = fn.external_source(
                source=source,
                num_outputs=5,
                batch=True,
                dtype=[types.UINT8, types.FLOAT, types.FLOAT, types.INT32, types.INT64],
                ndim=[1, 1, 1, 0, 0],
            )
            images = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
            images = fn.slice(images, anchors, shapes, axes=[0, 1], normalized_anchor=True, normalized_shape=True)
            images = fn.resize(images, device="gpu", resize_x=224, resize_y=224, interp_type=types.INTERP_LINEAR)
            images = fn.crop_mirror_normalize(
                images,
                device="gpu",
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                mirror=mirrors,
            )
            pipeline.set_outputs(images, indices)
        pipeline.build()
        self._dali_pipeline = pipeline

    def next_batch(self) -> TrainingBatch:
        from nvidia.dali.plugin.pytorch import feed_ndarray, to_torch_type

        begin = time.perf_counter()
        outputs = self._dali_pipeline.run()
        wait = time.perf_counter() - begin
        image_output, index_output = outputs
        image_tensor, image_shape = _uniform_dali_tensor(image_output)
        images = torch.empty(
            image_shape,
            device=self.device,
            dtype=to_torch_type[image_tensor.dtype],
        )
        feed_ndarray(image_tensor, images)
        indices_cpu = index_output.as_cpu().as_array().reshape(-1).tolist()
        indices = [int(value) for value in indices_cpu]
        read_seconds = self._dali_read_seconds.pop(tuple(indices), 0.0)
        plan = [self._planned[index] for index in indices]
        return TrainingBatch(
            inputs=(images,),
            labels=torch.tensor([item[0].label for item in plan], dtype=torch.long, device=self.device),
            identities=[item[1] for item in plan],
            augmentations=[item[2].as_dict() for item in plan],
            on_device=True,
            stage_seconds={
                "loader_data_wait": wait,
                "read": read_seconds,
                "read_decode_augmentation_preprocess": wait,
            },
            keepalive=list(outputs),
        )

    def end(self) -> None:
        if hasattr(self, "_dali_pipeline"):
            del self._dali_pipeline
        super().end()


def build_training_adapter(
    pipeline: str,
    samples: Sequence[TrainingSample],
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    config: dict[str, Any],
) -> TrainingPipelineAdapter:
    if pipeline not in PIPELINES:
        raise ValueError(f"unknown training pipeline: {pipeline}")
    classes = {
        "galp": GalpTrainingAdapter,
        "rgbnomore": RgbNoMoreTrainingAdapter,
        "dali": DaliTrainingAdapter,
        "pytorch": PyTorchTrainingAdapter,
    }
    adapter = classes[pipeline](
        samples, batch_size=batch_size, workers=workers, device=device, config=config
    )
    if adapter.domain != DOMAINS[pipeline]:
        raise AssertionError("training pipeline domain regression")
    return adapter
