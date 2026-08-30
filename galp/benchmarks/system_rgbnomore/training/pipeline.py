#!/usr/bin/env python3
"""Four auditable training data-pipeline adapters.

Adapters own data acquisition and preprocessing only.  The training lifecycle,
optimizer, reset, timing window, and artifact generation remain controlled by
``training/run.py``.
"""

from __future__ import annotations

import contextlib
import csv
import importlib
import io
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from galp.benchmarks.system_rgbnomore.training.artifacts import sha256_file, sha256_json
from galp.benchmarks.system_rgbnomore.training.augmentation import (
    AugmentationDecision,
    apply_rgb_augmentation_staged,
    horizontal_flip_dct,
)
from galp.benchmarks.system_rgbnomore.training.direct_dct_reader import (
    DirectDctTrainingReader,
    optional_native_execution_stats,
    optional_native_execution_stats_observation,
    optional_native_execution_stats_snapshot,
)
from galp.benchmarks.system_rgbnomore.training.sample_order import SampleIdentity
from galp.benchmarks.system_rgbnomore.training.schema import DOMAINS, PIPELINES


CANONICAL_PIPELINE_LOOKAHEAD_BATCHES = 2
DALI_VARIANTS = ("d2", "d3")
DALI_VARIANT_CONFIGS = {
    "d2": {
        "source_mode": "reader",
        "decoder_mode": "roi",
        "augmentation_mode": "planned",
        "preserves_canonical_order": True,
        "role": "fair-native-dali-baseline",
    },
    "d3": {
        "source_mode": "reader",
        "decoder_mode": "roi",
        "augmentation_mode": "native",
        "preserves_canonical_order": False,
        "role": "native-dali-performance-ceiling",
    },
}


def resolve_dali_variant(variant: str) -> dict[str, Any]:
    name = str(variant).strip().lower()
    if name not in DALI_VARIANT_CONFIGS:
        raise ValueError(f"unknown DALI variant {variant!r}; expected {DALI_VARIANTS}")
    return {"variant": name, **DALI_VARIANT_CONFIGS[name]}


_FINE_NSYS_RANGES = os.environ.get("GALP_NSYS_FINE", "0") == "1"


@contextlib.contextmanager
def _fine_nsys_range(name: str):
    """Emit profiling-only NVTX without creating CUDA in loader workers."""

    if not _FINE_NSYS_RANGES:
        yield
        return
    import nvtx

    with nvtx.annotate(name, domain="galp-training"):
        yield


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
    native_host_snapshot_taken: bool = False
    native_gpu_timings_finalized: bool = False
    # Compatibility alias for historical report readers. It now means GPU
    # timing finalization, not merely that a host snapshot was taken.
    native_stats_finalized: bool = False
    keepalive: list[Any] = field(default_factory=list)
    native_stats_source: Any | None = None


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
        self._set_batch_ranges(batch_lengths)

    def _set_batch_ranges(
        self, batch_lengths: Sequence[int] | None = None
    ) -> None:
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

    def preserves_canonical_order(self) -> bool:
        return True

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
        with _fine_nsys_range("pytorch.worker.read"):
            start = time.perf_counter()
            encoded = sample.path.read_bytes()
            read_seconds = time.perf_counter() - start
        with _fine_nsys_range("pytorch.worker.jpeg_decode"):
            start = time.perf_counter()
            with Image.open(io.BytesIO(encoded)) as image:
                rgb = image.convert("RGB")
                rgb.load()
            decode_seconds = time.perf_counter() - start
        with _fine_nsys_range("pytorch.worker.augment_preprocess"):
            tensor, stages = apply_rgb_augmentation_staged(rgb, decision)
        with _fine_nsys_range("pytorch.worker.result_enqueue"):
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
            prefetch_factor=CANONICAL_PIPELINE_LOOKAHEAD_BATCHES if self.workers > 0 else None,
        )
        self._iterator = iter(loader)

    def next_batch(self) -> TrainingBatch:
        if self._iterator is None:
            raise RuntimeError("pipeline repeat has not begun")
        with _fine_nsys_range("pytorch.main.dataloader_wait"):
            begin = time.perf_counter()
            images, labels, indices, read, decode, augmentation, preprocess = next(
                self._iterator
            )
            wait = time.perf_counter() - begin
        with _fine_nsys_range("pytorch.main.batch_metadata"):
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
            prefetch_factor=CANONICAL_PIPELINE_LOOKAHEAD_BATCHES if self.workers > 0 else None,
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

    _NATIVE_SUM_METRICS = {
        "producer_active_seconds": "producer_ms",
        "producer_planning_seconds": "planning_ms",
        "producer_io_staging_seconds": "io_ms",
        "producer_decode_seconds": "decode_ms",
        "producer_transform_seconds": "transform_ms",
        "consumer_wait_seconds": "consumer_wait_ms",
        "logical_bytes": "logical_bytes",
        "physical_bytes": "physical_bytes",
    }

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
        self._repeat_active = False
        self._pipeline_active = False
        self._native_schedule: list[
            tuple[list[int], list[int], list[dict[str, Any]]]
        ] = []
        self._next_native_batch = 0
        self._prefetched_batch_count = 0
        self._prefetched_indices: list[int] = []
        self._committed_native_metric_snapshots: list[dict[str, Any]] = []
        self._committed_native_metrics: dict[str, Any] = {}
        self._loader_metrics: dict[str, Any] = {}
        pls_pool_config = self.config.get("pls_gpu_pool") or {}
        self._pls_gpu_pool = bool(pls_pool_config.get("enabled", False))
        self._pls_closed_pool_specs = list(
            pls_pool_config.get("closed_pool_batches", [])
        )
        self._pls_pool_ranges: list[list[int]] = []
        self._active_pls_pool: dict[str, Any] | None = None
        self._next_emit_batch = 0
        self._next_pls_pool = 0
        self._pls_pool_release_pending = False
        self._pls_pool_metrics = self._new_pls_pool_metrics()

    def _new_pls_pool_metrics(self) -> dict[str, int | bool | str]:
        return {
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
            "released_pool_count": 0,
            "active_pool_count": 0,
            "max_materialized_pool_samples": 0,
            "max_simultaneously_active_pool_count": 0,
            "simultaneously_active_pool_limit": 1 if self._pls_gpu_pool else 0,
        }

    def _reset_native_repeat(self, scheduled_batches: int) -> None:
        self._native_schedule = []
        self._next_native_batch = 0
        self._prefetched_batch_count = 0
        self._prefetched_indices = []
        self._committed_native_metric_snapshots = []
        self._committed_native_metrics = {}
        self._loader_metrics = {
            "native_pipeline_owned": True,
            "scheduled_batches": scheduled_batches,
            "consumed_batches": 0,
            "max_queue_depth_batches": 0,
            "closed": False,
        }

    def _start_native_pipeline(self, ranges: Sequence[Sequence[int]]) -> None:
        self._native_schedule = []
        image_id_batches: list[list[int]] = []
        transforms_by_batch: list[list[dict[str, Any]]] = []
        for raw_indices in ranges:
            indices = [int(index) for index in raw_indices]
            plan = [self._planned[index] for index in indices]
            image_ids: list[int] = []
            transforms: list[dict[str, Any]] = []
            for sample, _identity, decision in plan:
                if sample.galp_image_id is None:
                    raise ValueError(
                        f"GALP sample {sample.logical_sample_id!r} lacks galp_image_id"
                    )
                image_ids.append(int(sample.galp_image_id))
                transforms.append(decision.native_dct_descriptor())
            self._native_schedule.append((indices, image_ids, transforms))
            image_id_batches.append(image_ids)
            transforms_by_batch.append(transforms)
        self._next_native_batch = 0
        self._prefetched_batch_count = 0
        self.reader.start(
            image_id_batches,
            transforms_by_batch=transforms_by_batch,
        )
        self._pipeline_active = True

    def _consume_native_batch(
        self,
    ) -> tuple[list[int], list[int], list[dict[str, Any]], Any, float]:
        if not self._pipeline_active:
            raise RuntimeError("native GALP pipeline has not been started")
        if self._next_native_batch >= len(self._native_schedule):
            raise StopIteration
        indices, image_ids, transforms = self._native_schedule[self._next_native_batch]
        begin = time.perf_counter()
        try:
            batch = self.reader.next_batch()
        except Exception:
            self.reader.close()
            self._pipeline_active = False
            raise
        observed_wait = time.perf_counter() - begin
        self._next_native_batch += 1
        try:
            self._validate_native_provenance(batch, image_ids, transforms)
        except Exception:
            self.reader.close()
            self._pipeline_active = False
            raise
        metrics = getattr(batch, "metrics", None)
        wait = (
            float(metrics.consumer_wait_ms) / 1000.0
            if metrics is not None
            else observed_wait
        )
        self._loader_metrics["consumed_batches"] += 1
        return indices, image_ids, transforms, batch, wait

    def begin(
        self,
        identities: Sequence[SampleIdentity],
        decisions: Sequence[AugmentationDecision],
        batch_lengths: Sequence[int] | None = None,
    ) -> None:
        if self._repeat_active:
            self.end()
        super().begin(identities, decisions, batch_lengths)
        self._read_indices = []
        self._active_pls_pool = None
        self._next_emit_batch = 0
        self._next_pls_pool = 0
        self._pls_pool_release_pending = False
        self._pls_pool_ranges = []
        self._pls_pool_metrics = self._new_pls_pool_metrics()
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
        scheduled_batches = (
            len(self._pls_pool_ranges) if self._pls_gpu_pool else len(self._batch_ranges)
        )
        self._reset_native_repeat(scheduled_batches)
        self._repeat_active = True
        if self._pls_gpu_pool:
            self._start_next_pls_pipeline()
        else:
            self._start_native_pipeline(self._batch_ranges)

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

    @classmethod
    def _native_loader_metrics(cls, native: Any) -> dict[str, Any]:
        def value(name: str) -> Any:
            if isinstance(native, Mapping):
                return native[name]
            return getattr(native, name)

        values: dict[str, Any] = {
            "native_metrics_complete": bool(value("complete")),
            "peak_transient_bytes": int(value("peak_transient_bytes")),
        }
        for loader_name, native_name in cls._NATIVE_SUM_METRICS.items():
            metric_value = value(native_name)
            values[loader_name] = (
                float(metric_value) / 1000.0
                if native_name.endswith("_ms")
                else int(metric_value)
            )
        return values

    @classmethod
    def _merge_native_loader_metrics(
        cls, committed: dict[str, Any], current: dict[str, Any]
    ) -> dict[str, Any]:
        if not committed:
            return dict(current)
        merged = dict(committed)
        merged["native_metrics_complete"] = bool(
            committed.get("native_metrics_complete", True)
        ) and bool(current["native_metrics_complete"])
        merged["peak_transient_bytes"] = max(
            int(committed.get("peak_transient_bytes", 0)),
            int(current["peak_transient_bytes"]),
        )
        for loader_name in cls._NATIVE_SUM_METRICS:
            merged[loader_name] = committed.get(loader_name, 0) + current[loader_name]
        return merged

    def _current_native_metrics(self) -> dict[str, Any]:
        snapshot = self._current_native_metric_snapshot()
        return self._native_loader_metrics(
            snapshot if snapshot is not None else self.reader.metrics()
        )

    def _current_native_metric_snapshot(self) -> dict[str, Any] | None:
        snapshot = getattr(self.reader, "metrics_snapshot", None)
        if not callable(snapshot):
            return None
        return dict(snapshot())

    def _aggregate_native_metric_snapshots(
        self, snapshots: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any] | None:
        aggregate = getattr(self.reader, "aggregate_metrics_snapshots", None)
        if not callable(aggregate):
            return None
        return dict(aggregate(snapshots))

    def _combined_native_metrics(self) -> dict[str, Any]:
        if not self._pipeline_active:
            return dict(self._committed_native_metrics)
        current_snapshot = self._current_native_metric_snapshot()
        if current_snapshot is not None:
            aggregated = self._aggregate_native_metric_snapshots(
                [*self._committed_native_metric_snapshots, current_snapshot]
            )
            if aggregated is not None:
                return self._native_loader_metrics(aggregated)
        return self._merge_native_loader_metrics(
            self._committed_native_metrics,
            self._native_loader_metrics(
                current_snapshot
                if current_snapshot is not None
                else self.reader.metrics()
            ),
        )

    def _freeze_active_prefetch_evidence(self) -> None:
        if not self._pipeline_active:
            return
        self._prefetched_batch_count = min(
            self.reader.prefetched_batch_count(), len(self._native_schedule)
        )
        for indices, _image_ids, _transforms in self._native_schedule[
            : self._prefetched_batch_count
        ]:
            self._prefetched_indices.extend(indices)

    def _close_native_pipeline(
        self,
        *,
        synchronize_consumer: bool,
        require_complete_metrics: bool,
        released_pls_pool: bool,
    ) -> None:
        if not self._pipeline_active:
            return
        try:
            if synchronize_consumer and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            current_snapshot = self._current_native_metric_snapshot()
            current = self._native_loader_metrics(
                current_snapshot
                if current_snapshot is not None
                else self.reader.metrics()
            )
            if require_complete_metrics and not current["native_metrics_complete"]:
                raise RuntimeError(
                    "GALP native metrics remained incomplete at a PLS pool boundary"
                )
            if current_snapshot is not None:
                self._committed_native_metric_snapshots.append(current_snapshot)
                aggregated = self._aggregate_native_metric_snapshots(
                    self._committed_native_metric_snapshots
                )
                if aggregated is None:
                    raise RuntimeError(
                        "native metrics snapshots require the native canonical aggregator"
                    )
                self._committed_native_metrics = self._native_loader_metrics(aggregated)
            else:
                # Test-injected/legacy readers may not expose the private native
                # snapshot API. This compatibility path is not production
                # metrics authority.
                self._committed_native_metrics = self._merge_native_loader_metrics(
                    self._committed_native_metrics, current
                )
            self._loader_metrics.update(self._committed_native_metrics)
            self._freeze_active_prefetch_evidence()
        finally:
            self.reader.close()
            self._pipeline_active = False
            self._native_schedule = []
            self._next_native_batch = 0
            self._prefetched_batch_count = 0
            if released_pls_pool:
                self._pls_pool_metrics["released_pool_count"] = int(
                    self._pls_pool_metrics["released_pool_count"]
                ) + 1
                self._pls_pool_metrics["active_pool_count"] = 0

    def _start_next_pls_pipeline(self) -> None:
        if self._next_pls_pool >= len(self._pls_pool_ranges):
            raise StopIteration
        self._start_native_pipeline([self._pls_pool_ranges[self._next_pls_pool]])
        self._next_pls_pool += 1
        self._pls_pool_metrics["active_pool_count"] = 1
        self._pls_pool_metrics["max_simultaneously_active_pool_count"] = max(
            int(self._pls_pool_metrics["max_simultaneously_active_pool_count"]),
            int(self._pls_pool_metrics["active_pool_count"]),
        )

    def _load_active_pls_pool(self) -> None:
        if self._pls_pool_release_pending:
            self._close_native_pipeline(
                synchronize_consumer=True,
                require_complete_metrics=True,
                released_pls_pool=True,
            )
            self._pls_pool_release_pending = False
            self._start_next_pls_pipeline()
        elif not self._pipeline_active:
            self._start_next_pls_pipeline()
        indices, _image_ids, _transforms, native_batch, wait = (
            self._consume_native_batch()
        )
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
            native_stats_source=native_batch if stats_pending else None,
        )
        self._active_pls_pool["native_stats_pending"] = False
        self._active_pls_pool["offset"] = offset + len(expected_indices)
        self._next_emit_batch += 1
        if int(self._active_pls_pool["offset"]) == len(indices):
            self._active_pls_pool = None
            self._pls_pool_release_pending = True
            self._pls_pool_metrics["fully_emitted_release_eligible_pool_count"] = int(
                self._pls_pool_metrics["fully_emitted_release_eligible_pool_count"]
            ) + 1
        if self.execution_mode == "audit":
            self.finalize_batch_metrics(training_batch)
        return training_batch

    def next_batch(self) -> TrainingBatch:
        if not self._repeat_active:
            raise RuntimeError("pipeline repeat has not begun")
        if self._pls_gpu_pool:
            if self._next_emit_batch >= len(self._batch_ranges):
                raise StopIteration
            return self._next_pls_pool_batch()
        if not self._pipeline_active:
            raise RuntimeError("pipeline repeat has not begun")
        indices, image_ids, transforms, batch, wait = self._consume_native_batch()
        plan = [self._planned[index] for index in indices]
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
            native_stats_source=batch,
        )
        if self.execution_mode == "audit":
            self.finalize_batch_metrics(training_batch)
        return training_batch

    @staticmethod
    def _apply_batch_metrics(
        batch: TrainingBatch,
        execution_stats: dict[str, Any],
        *,
        host_snapshot_taken: bool,
        gpu_timings_finalized: bool,
    ) -> None:
        batch.native_execution_stats = execution_stats
        batch.native_host_snapshot_taken = host_snapshot_taken
        batch.native_gpu_timings_finalized = gpu_timings_finalized
        batch.native_stats_finalized = gpu_timings_finalized
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
        if source is None or batch.native_gpu_timings_finalized:
            return
        stats_method = getattr(source, "native_execution_stats", None)
        execution_stats = (
            dict(stats_method())
            if callable(stats_method)
            else optional_native_execution_stats(source)
        )
        self._apply_batch_metrics(
            batch,
            execution_stats,
            host_snapshot_taken=True,
            gpu_timings_finalized=True,
        )

    def snapshot_batch_metrics(self, batch: TrainingBatch) -> None:
        source = batch.native_stats_source
        if source is None or batch.native_gpu_timings_finalized:
            return
        observation_method = getattr(
            source, "native_execution_stats_observation", None
        )
        observation = (
            dict(observation_method())
            if callable(observation_method)
            else optional_native_execution_stats_observation(source)
        )
        if observation and isinstance(observation.get("stats"), Mapping):
            execution_stats = dict(observation["stats"])
            host_snapshot_taken = bool(
                observation.get("host_snapshot_taken", True)
            )
            gpu_timings_finalized = bool(
                observation.get("gpu_timings_finalized", False)
            )
        else:
            # Compatibility path for injected/legacy readers. A host snapshot
            # alone never claims that CUDA event timings are finalized.
            stats_method = getattr(source, "native_execution_stats_snapshot", None)
            execution_stats = (
                dict(stats_method())
                if callable(stats_method)
                else optional_native_execution_stats_snapshot(source)
            )
            host_snapshot_taken = True
            gpu_timings_finalized = False
        self._apply_batch_metrics(
            batch,
            execution_stats,
            host_snapshot_taken=host_snapshot_taken,
            gpu_timings_finalized=gpu_timings_finalized,
        )

    def _capture_native_metrics(self) -> None:
        self._loader_metrics.update(self._combined_native_metrics())

    def loader_metrics(self) -> dict[str, Any]:
        if self._pipeline_active:
            self._capture_native_metrics()
        metrics = dict(self._loader_metrics)
        metrics.update(
            {
                "worker_semantics": (
                    "native runtime profile; --workers is not forwarded to GALP"
                ),
                "configured_workers": self.workers,
                "execution_mode": self.execution_mode,
                "physical_load_segment_gpu_pool": dict(self._pls_pool_metrics),
            }
        )
        return metrics

    def prefetched_read_identities(self) -> list[SampleIdentity]:
        active_indices: list[int] = []
        if self._pipeline_active:
            self._prefetched_batch_count = min(
                self.reader.prefetched_batch_count(), len(self._native_schedule)
            )
            active_indices = [
                index
                for batch_indices, _image_ids, _transforms in self._native_schedule[
                    : self._prefetched_batch_count
                ]
                for index in batch_indices
            ]
        indices = [*self._prefetched_indices, *active_indices]
        return [self._planned[index][1] for index in indices]

    def end(self) -> None:
        if self._pipeline_active:
            self._close_native_pipeline(
                synchronize_consumer=self._pls_gpu_pool,
                require_complete_metrics=self._pls_gpu_pool,
                released_pls_pool=self._pls_gpu_pool,
            )
            self._pls_pool_release_pending = False
        self._repeat_active = False
        self._loader_metrics["closed"] = True
        super().end()


class _DaliMetadataBatchSource:
    """Metadata stream aligned with a deterministic ``readers.file`` stream."""

    def __init__(
        self,
        anchors: np.ndarray,
        shapes: np.ndarray,
        mirrors: np.ndarray,
        batch_ranges: Sequence[Sequence[int]],
        batch_size: int,
    ) -> None:
        self.anchors = anchors
        self.shapes = shapes
        self.mirrors = mirrors
        self.batch_ranges = [list(values) for values in batch_ranges]
        self.batch_size = int(batch_size)

    def __call__(self, iteration: int):
        if iteration >= len(self.batch_ranges):
            raise StopIteration
        indices = list(self.batch_ranges[iteration])
        if len(indices) < self.batch_size:
            indices.extend([indices[-1]] * (self.batch_size - len(indices)))
        values = np.asarray(indices, dtype=np.int64)
        return self.anchors[values], self.shapes[values], self.mirrors[values]


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
        dali_config = dict(self.config.get("dali", {}))
        variant = resolve_dali_variant(str(dali_config.get("variant", "d2")))
        requested_augmentation = str(variant["augmentation_mode"])
        dali_phase = str(self.config.get("phase", "train"))
        effective_augmentation = (
            requested_augmentation if dali_phase == "train" else "planned"
        )
        if effective_augmentation == "native" and not decisions:
            self._planned = []
            for identity in identities:
                sample = self.samples.get(identity.logical_sample_id)
                if sample is None:
                    raise KeyError(
                        f"sample order references unknown ID {identity.logical_sample_id!r}"
                    )
                self._planned.append((sample, identity, None))
            self._set_batch_ranges(batch_lengths)
        else:
            super().begin(identities, decisions, batch_lengths)
        if self.device.type != "cuda":
            raise RuntimeError("DALI training adapter requires a CUDA device")
        from nvidia.dali import fn, types
        from nvidia.dali.pipeline import Pipeline

        self._cursor = 0
        self._read_indices = []
        self._dali_variant = str(variant["variant"])
        self._dali_source_mode = "reader"
        self._dali_decoder_mode = "roi"
        self._dali_phase = dali_phase
        self._dali_augmentation_mode = effective_augmentation

        paths = [str(item[0].path) for item in self._planned]
        if self._dali_augmentation_mode == "planned":
            anchors = np.asarray(
                [
                    [
                        item[2].crop_y / item[2].source_height,
                        item[2].crop_x / item[2].source_width,
                    ]
                    for item in self._planned
                ],
                dtype=np.float32,
            )
            shapes = np.asarray(
                [
                    [
                        item[2].crop_height / item[2].source_height,
                        item[2].crop_width / item[2].source_width,
                    ]
                    for item in self._planned
                ],
                dtype=np.float32,
            )
            mirrors_host = np.asarray(
                [int(item[2].horizontal_flip) for item in self._planned],
                dtype=np.int32,
            )
        else:
            anchors = np.empty((0, 2), dtype=np.float32)
            shapes = np.empty((0, 2), dtype=np.float32)
            mirrors_host = np.empty((0,), dtype=np.int32)
        self._dali_encoded_bytes = 0
        self._dali_source_samples = 0
        self._dali_consumed_samples = 0
        self._dali_handoff_seconds = 0.0
        self._dali_run_wait_seconds = 0.0

        num_threads = int(dali_config.get("num_threads", max(1, self.workers)))
        prefetch_depth = int(
            dali_config.get(
                "prefetch_queue_depth", CANONICAL_PIPELINE_LOOKAHEAD_BATCHES
            )
        )
        if num_threads <= 0 or prefetch_depth <= 0:
            raise ValueError("DALI thread/prefetch values must be positive")
        epoch = int(identities[0].epoch) if identities else 0
        dali_seed = int(dali_config.get("seed", 11997733)) + epoch

        pipeline = Pipeline(
            batch_size=self.batch_size,
            num_threads=num_threads,
            device_id=self.device.index or 0,
            seed=dali_seed,
            prefetch_queue_depth=prefetch_depth,
            exec_pipelined=True,
            exec_async=True,
            # DALI's PyTorch plugin can safely expose dynamic-executor outputs
            # through DLPack.  Keeping the executor mode explicit makes the
            # zero-copy handoff below part of this adapter's contract.
            exec_dynamic=True,
        )
        with pipeline:
            encoded, indices = fn.readers.file(
                files=paths,
                labels=list(range(len(paths))),
                random_shuffle=self._dali_augmentation_mode == "native",
                initial_fill=int(dali_config.get("reader_initial_fill", 1024)),
                pad_last_batch=True,
                dont_use_mmap=bool(dali_config.get("reader_dont_use_mmap", False)),
                read_ahead=bool(dali_config.get("reader_read_ahead", False)),
                seed=dali_seed,
                name="dali_training_reader",
            )
            encoded_bytes = encoded.shape(dtype=types.INT64)
            if self._dali_augmentation_mode == "planned":
                metadata_source = _DaliMetadataBatchSource(
                    anchors,
                    shapes,
                    mirrors_host,
                    self._batch_ranges,
                    self.batch_size,
                )
                anchors_node, shapes_node, mirrors = fn.external_source(
                    source=metadata_source,
                    num_outputs=3,
                    batch=True,
                    dtype=[types.FLOAT, types.FLOAT, types.INT32],
                    ndim=[1, 1, 0],
                )
            else:
                anchors_node = None
                shapes_node = None
                mirrors = fn.random.coin_flip(probability=0.5, seed=dali_seed + 2)

            decoder_kwargs = {
                "device": "mixed",
                "output_type": types.RGB,
                "hybrid_huffman_threshold": int(
                    dali_config.get("hybrid_huffman_threshold", 1_000_000)
                ),
                "hw_decoder_load": float(dali_config.get("hw_decoder_load", 0.65)),
            }
            if self._dali_augmentation_mode == "native":
                images = fn.decoders.image_random_crop(
                    encoded,
                    random_area=[0.05, 1.0],
                    random_aspect_ratio=[0.75, 4.0 / 3.0],
                    num_attempts=10,
                    seed=dali_seed + 1,
                    **decoder_kwargs,
                )
            else:
                images = fn.decoders.image_slice(
                    encoded,
                    anchors_node,
                    shapes_node,
                    axes=[0, 1],
                    normalized_anchor=True,
                    normalized_shape=True,
                    **decoder_kwargs,
                )
            images = fn.resize(
                images,
                device="gpu",
                resize_x=224,
                resize_y=224,
                interp_type=types.INTERP_LINEAR,
            )
            images = fn.crop_mirror_normalize(
                images,
                device="gpu",
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                mirror=mirrors,
            )
            pipeline.set_outputs(images, indices, encoded_bytes)
        pipeline.build()
        self._dali_pipeline = pipeline

    def preserves_canonical_order(self) -> bool:
        return self._dali_augmentation_mode != "native"

    def next_batch(self) -> TrainingBatch:
        from nvidia.dali.plugin.pytorch.torch_utils import to_torch_tensor

        with _fine_nsys_range("dali.adapter.pipeline_run"):
            begin = time.perf_counter()
            outputs = self._dali_pipeline.run()
            wait = time.perf_counter() - begin
        handoff_started = time.perf_counter()
        image_output, index_output = outputs[:2]
        with _fine_nsys_range("dali.adapter.dlpack_handoff"):
            image_tensor, _image_shape = _uniform_dali_tensor(image_output)
            # Match DALI's official PyTorch iterator: dynamic-executor outputs have
            # independent storage and can be handed to PyTorch through DLPack;
            # static-executor outputs require a defensive device-to-device copy.
            images = to_torch_tensor(
                image_tensor,
                copy=not bool(self._dali_pipeline.exec_dynamic),
            )
        with _fine_nsys_range("dali.adapter.index_and_byte_metadata"):
            all_indices = [
                int(value)
                for value in index_output.as_cpu().as_array().reshape(-1).tolist()
            ]
            expected_length = self._batch_lengths[len(self._read_indices)]
            indices = all_indices[:expected_length]
            if int(images.shape[0]) != expected_length:
                images = images[:expected_length]
            if len(outputs) != 3:
                raise RuntimeError(f"unexpected DALI output count: {len(outputs)}")
            encoded_bytes = outputs[2].as_cpu().as_array().reshape(-1)
            read_seconds = 0.0
            source_bytes = int(encoded_bytes.sum())
            source_samples = int(encoded_bytes.size)
            plan = [self._planned[index] for index in indices]
        self._read_indices.append(indices)
        self._dali_run_wait_seconds += wait
        self._dali_encoded_bytes += source_bytes
        self._dali_source_samples += source_samples
        self._dali_consumed_samples += len(indices)
        handoff = time.perf_counter() - handoff_started
        self._dali_handoff_seconds += handoff
        with _fine_nsys_range("dali.adapter.training_batch_construct"):
            return TrainingBatch(
                inputs=(images,),
                labels=torch.tensor(
                    [item[0].label for item in plan],
                    dtype=torch.long,
                    device=self.device,
                ),
                identities=[item[1] for item in plan],
                augmentations=(
                    [item[2].as_dict() for item in plan]
                    if self._dali_augmentation_mode == "planned"
                    else [
                        {
                            "source": "dali-native-random-resized-crop",
                            "epoch": item[1].epoch,
                            "logical_sample_id": item[1].logical_sample_id,
                        }
                        for item in plan
                    ]
                ),
                on_device=True,
                stage_seconds={
                    "loader_data_wait": wait,
                    "read": read_seconds,
                    "dali_handoff": handoff,
                    "read_decode_augmentation_preprocess": wait,
                },
                keepalive=list(outputs),
            )

    def loader_metrics(self) -> dict[str, Any]:
        return {
            "worker_semantics": "DALI native file reader and operator threads",
            "configured_workers": self.workers,
            "variant": self._dali_variant,
            "source_mode": self._dali_source_mode,
            "decoder_mode": self._dali_decoder_mode,
            "augmentation_mode": self._dali_augmentation_mode,
            "preserves_canonical_order": self.preserves_canonical_order(),
            "encoded_source_bytes": self._dali_encoded_bytes,
            "source_samples_read": self._dali_source_samples,
            "consumed_samples": self._dali_consumed_samples,
            "reader_overread_samples": max(
                0, self._dali_source_samples - self._dali_consumed_samples
            ),
            "read_work_seconds": 0.0,
            "read_work_timing_semantics": "unavailable for native reader",
            "pipeline_run_wait_seconds": self._dali_run_wait_seconds,
            "torch_handoff_host_seconds": self._dali_handoff_seconds,
            "config": dict(self.config.get("dali", {})),
        }

    def prefetched_read_identities(self) -> list[SampleIdentity]:
        # The native reader is owned by DALI, so a complete main-process prefetch
        # ledger is not available.  Full-epoch coverage is validated by the runner.
        indices = [index for batch in self._read_indices for index in batch]
        return [self._planned[int(index)][1] for index in indices]

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
