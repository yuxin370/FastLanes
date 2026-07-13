#!/usr/bin/env python3
"""Run one pipeline under a shared end-to-end benchmark contract."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from common import (
    RESULT_SCHEMA,
    distribution,
    load_contract,
    load_sample_manifest,
    measured_samples,
    normalize_device,
    require,
    sample_trace,
    sha256_file,
    sha256_json,
    verify_file_fingerprint,
    write_json,
)
from model_factory import build_dct_model, build_rgb_model


HERE = Path(__file__).resolve().parent
DIAGNOSTICS_DIR = HERE / "diagnostics"


@dataclass
class LoadedBatch:
    inputs: tuple[torch.Tensor, ...]
    labels: torch.Tensor
    ordinals: list[int]
    on_device: bool
    native_stage_seconds: dict[str, float] = field(default_factory=dict)
    native_counters: dict[str, int] = field(default_factory=dict)
    keepalive: list[Any] = field(default_factory=list)


def _add_diagnostics_to_path() -> None:
    text = str(DIAGNOSTICS_DIR)
    if text not in sys.path:
        sys.path.insert(0, text)


def _build_rgb_model(contract: dict[str, Any], device: torch.device) -> torch.nn.Module:
    config = contract["pipelines"]["rgbnomore"]
    model = build_rgb_model(
        Path(config["root"]),
        Path(contract["models"]["rgb"]["checkpoint"]),
        device,
    )
    model.eval()
    return model


def _build_dct_model(contract: dict[str, Any], device: torch.device) -> torch.nn.Module:
    config = contract["pipelines"]["rgbnomore"]
    model = build_dct_model(
        Path(config["root"]),
        Path(contract["models"]["dct"]["checkpoint"]),
        device,
    )
    model.eval()
    return model


def _autocast(contract: dict[str, Any], device: torch.device):
    precision = contract["execution"]["precision"]
    if precision == "fp32":
        return contextlib.nullcontext()
    if device.type != "cuda":
        raise ValueError(f"{precision} is supported only on CUDA by this benchmark")
    dtype = torch.float16 if precision == "amp_fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _forward(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], contract: dict[str, Any], device: torch.device) -> torch.Tensor:
    with torch.inference_mode(), _autocast(contract, device):
        logits = model(*inputs)
    if logits.ndim != 2 or logits.shape[1] != 1000:
        raise RuntimeError(f"expected [B,1000] logits, got {tuple(logits.shape)}")
    return logits


class CanonicalRgbDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[dict[str, Any]], transform: Any) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        from PIL import Image

        sample = self.samples[index]
        with Image.open(sample["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(sample["label"]), int(sample["ordinal"])


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        inputs, label = self.dataset[index]
        return inputs, int(label), index


def _loader_kwargs(contract: dict[str, Any]) -> dict[str, Any]:
    execution = contract["execution"]
    workers = int(execution["workers"])
    result: dict[str, Any] = {
        "batch_size": int(execution["batch_size"]),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if workers > 0:
        result["persistent_workers"] = True
        result["prefetch_factor"] = int(contract["pipelines"]["pytorch"]["prefetch_factor"])
    return result


class PipelineAdapter:
    domain: str
    worker_semantics: str

    def __init__(self, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device) -> None:
        self.contract = contract
        self.samples = list(samples)
        self.device = device
        self._iterator: Iterator[Any] | None = None

    def begin_repeat(self) -> None:
        raise NotImplementedError

    def load(
        self,
        expected: Sequence[dict[str, Any]],
        next_expected: Sequence[dict[str, Any]] | None = None,
    ) -> LoadedBatch:
        raise NotImplementedError

    def end_repeat(self) -> None:
        self._iterator = None

    def close(self) -> None:
        pass


class PyTorchAdapter(PipelineAdapter):
    domain = "rgb"
    worker_semantics = "torch_dataloader_processes"

    def __init__(self, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device) -> None:
        super().__init__(contract, samples, device)
        root = Path(contract["pipelines"]["rgbnomore"]["root"])
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        datasets = importlib.import_module("datasets")
        transform = datasets.get_transform(dataset="imagenet", type="test", dtype=torch.float32)
        self.loader = torch.utils.data.DataLoader(CanonicalRgbDataset(samples, transform), **_loader_kwargs(contract))

    def begin_repeat(self) -> None:
        self._iterator = iter(self.loader)

    def load(
        self,
        expected: Sequence[dict[str, Any]],
        next_expected: Sequence[dict[str, Any]] | None = None,
    ) -> LoadedBatch:
        assert self._iterator is not None
        images, labels, ordinals = next(self._iterator)
        return LoadedBatch((images,), labels.long(), [int(item) for item in ordinals.tolist()], False)


class RgbNoMoreAdapter(PipelineAdapter):
    domain = "dct"
    worker_semantics = "torch_dataloader_processes"

    def __init__(self, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device) -> None:
        super().__init__(contract, samples, device)
        root = Path(contract["pipelines"]["rgbnomore"]["root"])
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        datasets = importlib.import_module("datasets")
        base = datasets.imagenet_dataset_indexing(
            indexfile=contract["dataset"]["canonical_index_csv"],
            type="test",
            basepath="",
            load_mode="DCT",
            dtype=torch.float32,
        )
        transform = datasets.get_transform(dataset="imagenet_dct", type="test", dtype=torch.float32)
        transformed = datasets.SubsetWithTransform(base, dataset="imagenet_dct", transform=transform)
        self.loader = torch.utils.data.DataLoader(IndexedDataset(transformed), **_loader_kwargs(contract))

    def begin_repeat(self) -> None:
        self._iterator = iter(self.loader)

    def load(
        self,
        expected: Sequence[dict[str, Any]],
        next_expected: Sequence[dict[str, Any]] | None = None,
    ) -> LoadedBatch:
        assert self._iterator is not None
        (input_y, input_cbcr), labels, ordinals = next(self._iterator)
        return LoadedBatch(
            (input_y, input_cbcr),
            labels.long(),
            [int(self.samples[int(item)]["ordinal"]) for item in ordinals.tolist()],
            False,
        )


class DaliAdapter(PipelineAdapter):
    domain = "rgb"
    worker_semantics = "dali_cpu_threads"

    def __init__(self, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device) -> None:
        super().__init__(contract, samples, device)
        if device.type != "cuda":
            raise ValueError("DALI pipeline requires a CUDA device")
        self.iterator: Any | None = None

    def begin_repeat(self) -> None:
        from nvidia.dali import fn, pipeline_def, types
        from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

        execution = self.contract["execution"]
        config = self.contract["pipelines"]["dali"]
        paths = [str(sample["path"]) for sample in self.samples]
        ordinals = [int(sample["ordinal"]) for sample in self.samples]

        @pipeline_def
        def create_pipeline():
            encoded, ordinal = fn.readers.file(
                files=paths,
                labels=ordinals,
                random_shuffle=False,
                pad_last_batch=False,
                name="Reader",
            )
            images = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
            images = fn.resize(
                images,
                device="gpu",
                resize_shorter=int(self.contract["preprocess"]["rgb"]["resize_shorter"]),
                interp_type=types.INTERP_LINEAR,
                antialias=True,
            )
            images = fn.crop_mirror_normalize(
                images,
                device="gpu",
                dtype=types.FLOAT,
                output_layout="CHW",
                crop=tuple(self.contract["preprocess"]["rgb"]["crop_size"]),
                # torchvision/PIL CenterCrop floors the half-pixel when the
                # resized extent minus crop extent is odd. DALI rounds a
                # literal 0.5 upward, so use the left-limit of the midpoint.
                crop_pos_x=0.499999,
                crop_pos_y=0.499999,
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            )
            return images, ordinal

        pipeline = create_pipeline(
            batch_size=int(execution["batch_size"]),
            num_threads=int(execution["workers"]),
            device_id=int(config["device_id"]),
            seed=int(execution["seed"]),
            prefetch_queue_depth=int(config["prefetch_queue_depth"]),
        )
        pipeline.build()
        self.iterator = DALIGenericIterator(
            [pipeline],
            output_map=["image", "ordinal"],
            reader_name="Reader",
            auto_reset=False,
            last_batch_policy=LastBatchPolicy.DROP,
        )

    def load(
        self,
        expected: Sequence[dict[str, Any]],
        next_expected: Sequence[dict[str, Any]] | None = None,
    ) -> LoadedBatch:
        assert self.iterator is not None
        batch = next(self.iterator)[0]
        images = batch["image"]
        ordinals_tensor = batch["ordinal"].reshape(-1).long()
        ordinals = [int(item) for item in ordinals_tensor.cpu().tolist()]
        labels = torch.tensor([self.samples[item]["label"] for item in ordinals], dtype=torch.long, device=self.device)
        return LoadedBatch((images,), labels, ordinals, True)

    def end_repeat(self) -> None:
        if self.iterator is not None:
            self.iterator.reset()
        self.iterator = None


class GalpAdapter(PipelineAdapter):
    domain = "dct"
    worker_semantics = "galp_internal_runtime_not_configured_by_worker_count"

    def __init__(self, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device) -> None:
        super().__init__(contract, samples, device)
        if device.type != "cuda":
            raise ValueError("GALP pipeline requires a CUDA device")
        _add_diagnostics_to_path()
        self.module = importlib.import_module("direct_dct")
        galp_dct = importlib.import_module("_galp_direct_dct")
        config = contract["pipelines"]["galp"]
        self.reader = galp_dct.DirectDctReader(str(config["manifest"]))
        self.args = SimpleNamespace(
            preprocess=config["preprocess"],
            cache_capacity_mib=int(config["cache_capacity_mib"]),
            decode_batch_rowgroups=int(config.get("decode_batch_rowgroups", 2)),
            rowgroup_prefetch_depth=int(config.get("rowgroup_prefetch_depth", 16)),
            rowgroup_prefetch_workers=int(config.get("rowgroup_prefetch_workers", 4)),
            no_dequantize=False,
            no_scale=False,
        )
        self.transform = (
            self.module.build_rgbnomore_dct_val_transform(Path(contract["pipelines"]["rgbnomore"]["root"]))
            if config["preprocess"] == "rgbnomore-val"
            else None
        )
        self.pending: Any | None = None
        self.pending_image_ids: list[int] | None = None

    def begin_repeat(self) -> None:
        self.pending = None
        self.pending_image_ids = None

    def load(
        self,
        expected: Sequence[dict[str, Any]],
        next_expected: Sequence[dict[str, Any]] | None = None,
    ) -> LoadedBatch:
        image_ids = [int(sample["galp_image_id"]) for sample in expected]
        if self.args.preprocess == "rgbnomore-val-pushdown":
            if self.pending is None:
                self.pending = self.module._prefetch_pushdown_batch(self.reader, self.args, image_ids)
                self.pending_image_ids = image_ids
            if self.pending_image_ids != image_ids:
                raise RuntimeError(
                    f"GALP pending batch mismatch: expected {image_ids}, queued {self.pending_image_ids}"
                )
            input_y, input_cbcr, source_batches = self.module._adapt_prefetched_pushdown_batch(
                self.reader,
                self.args,
                image_ids,
                self.pending,
            )
            self.pending = None
            self.pending_image_ids = None
            if next_expected is not None:
                next_image_ids = [int(sample["galp_image_id"]) for sample in next_expected]
                self.pending = self.module._prefetch_pushdown_batch(self.reader, self.args, next_image_ids)
                self.pending_image_ids = next_image_ids
        else:
            input_y, input_cbcr, source_batches = self.module.read_and_adapt_batch(
                self.reader,
                self.args,
                image_ids,
                None,
                self.transform,
            )
        totals = self.module._empty_totals()
        self.module._accumulate_many_stats(totals, source_batches)
        if self.args.preprocess == "rgbnomore-val-pushdown":
            for source_batch in source_batches:
                stats = dict(source_batch.execution_stats)
                if int(stats.get("fixed_transform_item_count", 0)) <= 0:
                    raise RuntimeError("GALP benchmark did not exercise the fused transformed-grid path")
                if (
                    int(stats.get("projection_item_count", 0)) != 0
                    or int(stats.get("decoded_projection_item_count", 0)) != 0
                    or int(stats.get("project_decoded_ycbcr_grid_launch_count", 0)) != 0
                ):
                    raise RuntimeError("GALP benchmark unexpectedly used generic projection")
        labels = torch.tensor([sample["label"] for sample in expected], dtype=torch.long, device=self.device)
        native_stage_seconds = {
            key: float(value)
            for key, value in totals.items()
            if key.endswith("_seconds") and isinstance(value, (int, float))
        }
        native_counters = {
            key: int(value)
            for key, value in totals.items()
            if not key.endswith("_seconds") and isinstance(value, int)
        }
        return LoadedBatch(
            inputs=(input_y, input_cbcr),
            labels=labels,
            ordinals=[int(sample["ordinal"]) for sample in expected],
            on_device=True,
            native_stage_seconds=native_stage_seconds,
            native_counters=native_counters,
            keepalive=source_batches,
        )

    def end_repeat(self) -> None:
        self.pending = None
        self.pending_image_ids = None


def _make_adapter(name: str, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device) -> PipelineAdapter:
    adapters = {
        "galp": GalpAdapter,
        "rgbnomore": RgbNoMoreAdapter,
        "dali": DaliAdapter,
        "pytorch": PyTorchAdapter,
    }
    return adapters[name](contract, samples, device)


def _to_device(batch: LoadedBatch, device: torch.device) -> LoadedBatch:
    if batch.on_device:
        return batch
    inputs = tuple(item.to(device, non_blocking=True) for item in batch.inputs)
    labels = batch.labels.to(device, non_blocking=True)
    return LoadedBatch(
        inputs=inputs,
        labels=labels,
        ordinals=batch.ordinals,
        on_device=True,
        native_stage_seconds=batch.native_stage_seconds,
        native_counters=batch.native_counters,
        keepalive=batch.keepalive,
    )


def _validate_batch_identity(batch: LoadedBatch, expected: Sequence[dict[str, Any]]) -> None:
    expected_ordinals = [int(sample["ordinal"]) for sample in expected]
    if batch.ordinals != expected_ordinals:
        raise RuntimeError(f"sample order mismatch: expected {expected_ordinals}, got {batch.ordinals}")
    expected_labels = [int(sample["label"]) for sample in expected]
    actual_labels = [int(item) for item in batch.labels.detach().cpu().tolist()]
    if actual_labels != expected_labels:
        raise RuntimeError(f"label mismatch for ordinals {expected_ordinals}: expected {expected_labels}, got {actual_labels}")


def _accuracy_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    top5 = logits.topk(5, dim=1).indices
    matches = top5.eq(labels.reshape(-1, 1))
    return int(matches[:, :1].sum().item()), int(matches.sum().item())


def _new_event(device: torch.device) -> torch.cuda.Event | None:
    return torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None


def _event_ms(start: torch.cuda.Event | None, end: torch.cuda.Event | None) -> float:
    return float(start.elapsed_time(end)) if start is not None and end is not None else 0.0


def _capture_semantic(
    store: dict[str, list[np.ndarray]],
    batch: LoadedBatch,
    logits: torch.Tensor,
    expected: Sequence[dict[str, Any]],
    remaining: int,
) -> int:
    count = min(remaining, len(expected))
    if count <= 0:
        return 0
    store.setdefault("ordinals", []).append(np.asarray(batch.ordinals[:count], dtype=np.int64))
    store.setdefault("labels", []).append(np.asarray([item["label"] for item in expected[:count]], dtype=np.int64))
    store.setdefault("logits", []).append(logits[:count].detach().float().cpu().numpy())
    for index, tensor in enumerate(batch.inputs):
        store.setdefault(f"input_{index}", []).append(tensor[:count].detach().float().cpu().numpy())
    return count


def _write_semantic(path: Path, store: dict[str, list[np.ndarray]], metadata: dict[str, Any]) -> None:
    arrays = {key: np.concatenate(parts, axis=0) for key, parts in store.items()}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _model_metadata(contract: dict[str, Any], domain: str) -> dict[str, Any]:
    model = contract["models"][domain]
    return {
        "architecture": model["architecture"],
        "input_domain": model["input_domain"],
        "recipe_id": model["recipe_id"],
        "checkpoint": model["checkpoint"],
        "checkpoint_sha256": model["checkpoint_sha256"],
    }


def run_pipeline(name: str, contract_path: Path, output: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    if name not in contract["pipelines"]["enabled"]:
        raise ValueError(f"pipeline {name} is not enabled by the contract")
    manifest_path = Path(contract["dataset"]["sample_manifest"])
    manifest, samples = load_sample_manifest(manifest_path, contract["dataset"]["manifest_sha256"])
    for domain in ("rgb", "dct"):
        checkpoint = Path(contract["models"][domain]["checkpoint"])
        require(
            sha256_file(checkpoint) == contract["models"][domain]["checkpoint_sha256"],
            f"{domain} checkpoint SHA-256 changed after contract creation",
        )
    canonical_index = Path(contract["dataset"]["canonical_index_csv"])
    require(
        sha256_file(canonical_index) == contract["dataset"]["canonical_index_sha256"],
        "canonical RGB-no-more index changed after contract creation",
    )
    if name == "galp":
        galp_config = contract["pipelines"]["galp"]
        verify_file_fingerprint(
            Path(galp_config["manifest"]), galp_config["manifest_fingerprint"], "GALP manifest"
        )
        for payload in galp_config["payload_fingerprints"]:
            verify_file_fingerprint(Path(payload["path"]), payload, f"GALP {payload['kind']} payload")
    execution = contract["execution"]
    batch_size = int(execution["batch_size"])
    warmup_batches = int(execution["warmup_batches"])
    measurement_batches = int(execution["measurement_batches"])
    repeats = int(execution["repeats"])
    required_samples = batch_size * (warmup_batches + measurement_batches)
    require(len(samples) == required_samples, f"v1 requires exactly {required_samples} manifest samples")

    device = torch.device(normalize_device(execution["device"]))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)
    adapter = _make_adapter(name, contract, samples, device)
    model = _build_rgb_model(contract, device) if adapter.domain == "rgb" else _build_dct_model(contract, device)
    semantic_count = int(contract["semantic_validation"]["sample_count"])
    semantic_store: dict[str, list[np.ndarray]] = {}
    repeat_records: list[dict[str, Any]] = []

    measured_expected = measured_samples(samples, batch_size, warmup_batches, measurement_batches)
    expected_trace = sample_trace(measured_expected)
    total_batches = warmup_batches + measurement_batches

    def expected_batch(batch_index: int) -> list[dict[str, Any]]:
        begin = batch_index * batch_size
        return list(samples[begin : begin + batch_size])

    def following_batch(batch_index: int) -> list[dict[str, Any]] | None:
        return expected_batch(batch_index + 1) if batch_index + 1 < total_batches else None

    for repeat in range(repeats):
        adapter.begin_repeat()
        for warmup_index in range(warmup_batches):
            expected = expected_batch(warmup_index)
            batch = _to_device(adapter.load(expected, following_batch(warmup_index)), device)
            _validate_batch_identity(batch, expected)
            _forward(model, batch.inputs, contract, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        cpu_process_seconds = 0.0
        latency_ms: list[float] = []
        loader_submit_ms: list[float] = []
        h2d_gpu_ms: list[float] = []
        forward_gpu_ms: list[float] = []
        actual_samples: list[dict[str, Any]] = []
        native_stage_seconds: dict[str, float] = {}
        native_counters: dict[str, int] = {}
        correct1 = 0
        correct5 = 0
        semantic_captured = 0

        for measured_index in range(measurement_batches):
            batch_index = warmup_batches + measured_index
            expected = expected_batch(batch_index)
            process_started = time.process_time()
            wall_started_ns = time.perf_counter_ns()
            load_started_ns = wall_started_ns
            batch = adapter.load(expected, following_batch(batch_index))
            load_ended_ns = time.perf_counter_ns()
            _validate_batch_identity(batch, expected)

            h2d_start = _new_event(device)
            h2d_end = _new_event(device)
            if h2d_start is not None:
                h2d_start.record()
            batch = _to_device(batch, device)
            if h2d_end is not None:
                h2d_end.record()

            forward_start = _new_event(device)
            forward_end = _new_event(device)
            if forward_start is not None:
                forward_start.record()
            logits = _forward(model, batch.inputs, contract, device)
            if forward_end is not None:
                forward_end.record()
            batch_correct1, batch_correct5 = _accuracy_counts(logits, batch.labels)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall_ended_ns = time.perf_counter_ns()
            cpu_process_seconds += time.process_time() - process_started

            if repeat == 0 and semantic_captured < semantic_count:
                semantic_captured += _capture_semantic(
                    semantic_store,
                    batch,
                    logits,
                    expected,
                    semantic_count - semantic_captured,
                )
            latency_ms.append((wall_ended_ns - wall_started_ns) / 1e6)
            loader_submit_ms.append((load_ended_ns - load_started_ns) / 1e6)
            h2d_gpu_ms.append(_event_ms(h2d_start, h2d_end))
            forward_gpu_ms.append(_event_ms(forward_start, forward_end))
            correct1 += batch_correct1
            correct5 += batch_correct5
            actual_samples.extend(expected)
            for key, value in batch.native_stage_seconds.items():
                native_stage_seconds[key] = native_stage_seconds.get(key, 0.0) + float(value)
            for key, value in batch.native_counters.items():
                native_counters[key] = native_counters.get(key, 0) + int(value)

        measured_seconds = sum(latency_ms) / 1000.0
        images = measurement_batches * batch_size
        actual_trace = sample_trace(actual_samples)
        if actual_trace["sha256"] != expected_trace["sha256"]:
            raise RuntimeError("measured sample trace does not match the contract")
        record = {
            "repeat": repeat,
            "images": images,
            "seconds": measured_seconds,
            "throughput_images_per_s": images / measured_seconds,
            "end_to_end_latency_ms": distribution(latency_ms),
            "accuracy_top1": correct1 / images,
            "accuracy_top5": correct5 / images,
            "correct_top1": correct1,
            "correct_top5": correct5,
            "cpu_process_seconds": cpu_process_seconds,
            "cpu_time_scope": "main_process_only_excludes_loader_workers",
            "stage_breakdown_ms": {
                "loader_and_preprocess_submit": distribution(loader_submit_ms),
                "host_to_device_gpu": distribution(h2d_gpu_ms),
                "model_forward_gpu": distribution(forward_gpu_ms),
                "native_totals_seconds": native_stage_seconds,
            },
            "native_counters": native_counters,
            "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
            "peak_gpu_memory_scope": (
                "torch_allocator_only_excludes_galp_native_allocations"
                if name == "galp"
                else "torch_allocator_only_excludes_dali_native_allocations"
                if name == "dali"
                else "torch_allocator"
            ),
            "sample_trace": actual_trace,
        }
        repeat_records.append(record)
        adapter.end_repeat()

    semantic_path = output.parent / f"semantic_{name}.npz"
    _write_semantic(
        semantic_path,
        semantic_store,
        {
            "pipeline": name,
            "domain": adapter.domain,
            "sample_count": semantic_count,
            "contract_sha256": sha256_json(contract),
        },
    )
    dependency_versions: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
    }
    if name == "dali":
        import nvidia.dali

        dependency_versions["dali"] = nvidia.dali.__version__
    result = {
        "schema_version": RESULT_SCHEMA,
        "pipeline": name,
        "domain": adapter.domain,
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256_json(contract),
        "sample_manifest": str(manifest_path.resolve()),
        "sample_manifest_sha256": contract["dataset"]["manifest_sha256"],
        "model": _model_metadata(contract, adapter.domain),
        "execution": dict(execution),
        "worker_semantics": adapter.worker_semantics,
        "timing": dict(contract["timing"]),
        "preprocess": dict(contract["preprocess"][adapter.domain]),
        "pipeline_config": dict(contract["pipelines"][name]),
        "dataset_full_size": manifest["full_dataset_size"],
        "semantic_artifact": str(semantic_path.resolve()),
        "semantic_artifact_sha256": sha256_file(semantic_path),
        "dependency_versions": dependency_versions,
        "device_metadata": (
            {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
            if device.type == "cuda"
            else {"name": "cpu"}
        ),
        "repeats": repeat_records,
    }
    write_json(output, result)
    adapter.close()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("galp", "rgbnomore", "dali", "pytorch"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_pipeline(args.pipeline, args.contract, args.output)
    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "pipeline": result["pipeline"],
                "output": str(args.output.resolve()),
                "repeats": len(result["repeats"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
