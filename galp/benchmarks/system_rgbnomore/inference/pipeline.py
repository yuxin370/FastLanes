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
from typing import Any, Iterator, Sequence

import numpy as np
import torch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from galp.profiles.rgbnomore import VALIDATION
from galp.torch import DirectDctReader
from galp.diagnostics.direct_dct import execution_stats, execution_stats_snapshot

from shared.common import (
    RESULT_SCHEMA,
    GALP_PIPELINES,
    GALP_RUNTIME_PROFILE,
    INFERENCE_PIPELINES,
    contract_pipeline_name,
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
from inference.model_factory import build_dct_model, build_rgb_model


HERE = Path(__file__).resolve().parent
DIAGNOSTICS_DIR = BENCHMARK_ROOT / "diagnostics"


_NATIVE_MAX_COUNTER_KEYS = frozenset(
    {
        "compact_batch_buffer_capacity_bytes",
        "compact_batch_buffer_high_water_bytes",
        "planless_transform_max_blocks_per_launch",
        "planless_transform_max_output_blocks_per_launch",
        "planless_transform_registers_per_thread",
        "planless_transform_static_shared_bytes_per_cta",
        "planless_transform_local_bytes_per_thread",
        "planless_transform_threads_per_cta",
        "planless_transform_max_active_ctas_per_sm",
        "cuda_max_threads_per_sm",
        "cuda_warp_size",
    }
)

_NATIVE_INVARIANT_COUNTER_KEYS = frozenset(
    {
        "direct_dct_stream_priority",
        "direct_dct_h2d_stream_priority",
        "direct_dct_decode_stream_priority",
        "direct_dct_transform_stream_priority",
        "direct_dct_round_stream_priority",
        "cuda_least_stream_priority",
        "cuda_greatest_stream_priority",
    }
)


def _accumulate_native_counter(totals: dict[str, int], key: str, value: int) -> None:
    """Merge one per-batch native statistic according to its counter semantics."""
    value = int(value)
    if (
        key.startswith("galp_native_device_")
        or key.startswith("galp_native_pinned_")
        or key in _NATIVE_MAX_COUNTER_KEYS
    ):
        # Allocator fields are process-global snapshots, including their
        # monotonic request/allocation counters. Capacity/high-water fields are
        # gauges. Summing either class once per batch produces fictitious peaks.
        totals[key] = max(totals.get(key, 0), value)
    elif key in _NATIVE_INVARIANT_COUNTER_KEYS:
        current = totals.get(key)
        if current is not None and current != value:
            raise RuntimeError(f"invariant native counter changed within repeat: {key}")
        totals[key] = value
    else:
        totals[key] = totals.get(key, 0) + value


def _process_memory_snapshot(status_path: Path = Path("/proc/self/status")) -> dict[str, int]:
    """Return Linux main-process resident and lifetime-peak resident bytes."""
    values: dict[str, int] = {}
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"cannot read process memory status from {status_path}: {error}") from error
    for line in lines:
        name, separator, raw_value = line.partition(":")
        if separator and name in ("VmRSS", "VmHWM"):
            fields = raw_value.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise RuntimeError(f"unexpected {name} format in {status_path}: {line}")
            values[name] = int(fields[0]) * 1024
    if values.get("VmRSS", 0) <= 0 or values.get("VmHWM", 0) <= 0:
        raise RuntimeError(f"process memory status lacks positive VmRSS/VmHWM values: {status_path}")
    if values["VmHWM"] < values["VmRSS"]:
        raise RuntimeError(f"process VmHWM is smaller than VmRSS in {status_path}")
    return {
        "rss_bytes": values["VmRSS"],
        "peak_rss_bytes": values["VmHWM"],
    }


@dataclass
class LoadedBatch:
    inputs: tuple[torch.Tensor, ...]
    labels: torch.Tensor | None
    ordinals: list[int]
    on_device: bool
    native_stage_seconds: dict[str, float] = field(default_factory=dict)
    native_counters: dict[str, int] = field(default_factory=dict)
    native_properties: dict[str, Any] = field(default_factory=dict)
    audit_sources: list[Any] = field(default_factory=list)


def _resolve_model_stream_priority(
    requested: str | int, priority_range: tuple[int, int]
) -> int:
    least_priority, greatest_priority = (int(value) for value in priority_range)
    if requested == "greatest":
        return greatest_priority
    if requested == "least":
        return least_priority
    if isinstance(requested, str):
        raise ValueError(f"invalid model stream priority: {requested}")
    return int(requested)


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


def _forward(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    contract: dict[str, Any],
    device: torch.device,
    *,
    validate_output: bool = True,
) -> torch.Tensor:
    with torch.inference_mode(), _autocast(contract, device):
        logits = model(*inputs)
    if validate_output and (logits.ndim != 2 or logits.shape[1] != 1000):
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
    audit_enabled = True

    def __init__(self, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device) -> None:
        self.contract = contract
        self.samples = list(samples)
        self.device = device
        self.audit_enabled = True
        self._iterator: Iterator[Any] | None = None
        self.setup_metrics: dict[str, Any] = {}

    def begin_repeat(self) -> None:
        raise NotImplementedError

    def load(
        self,
        expected: Sequence[dict[str, Any]],
        next_expected: Sequence[dict[str, Any]] | None = None,
    ) -> LoadedBatch:
        raise NotImplementedError

    def after_model_complete(self) -> None:
        """Release GPU work intentionally serialized after the current model."""

    def finalize_batch_metrics(self, batch: LoadedBatch) -> None:
        """Collect audit-only metrics outside the timed loader/model interval."""

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
        if not self.audit_enabled:
            return LoadedBatch((images,), None, [], False)
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
        if not self.audit_enabled:
            return LoadedBatch((input_y, input_cbcr), None, [], False)
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
        if not self.audit_enabled:
            return LoadedBatch((images,), None, [], True)
        ordinals_tensor = batch["ordinal"].reshape(-1).long()
        ordinals = [int(item) for item in ordinals_tensor.cpu().tolist()]
        labels = torch.tensor([self.samples[item]["label"] for item in ordinals], dtype=torch.long, device=self.device)
        return LoadedBatch((images,), labels, ordinals, True)

    def end_repeat(self) -> None:
        if self.iterator is not None:
            self.iterator.reset()
        self.iterator = None


def _validate_profile_contract(
    profile_info: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    expected = {
        "layout": "transformed_dct_grid",
        "output_dtype": "float32",
        "y_output_blocks": (28, 28),
        "cbcr_output_blocks": (14, 14),
    }
    mismatches: dict[str, Any] = {}
    for key, expected_value in expected.items():
        observed = profile_info.get(key)
        normalized = tuple(observed) if key.endswith("_blocks") and observed is not None else observed
        if normalized != expected_value:
            mismatches[key] = observed
    if mismatches:
        raise RuntimeError(
            f"GALP semantic profile contract failed for {context}: {mismatches}"
        )
    return {
        "status": "passed",
        "runtime_profile": GALP_RUNTIME_PROFILE,
        "profile_id": profile_info.get("id"),
        **expected,
    }


def _validate_transform_execution_stats(
    stats: dict[str, Any],
    image_count: int,
) -> None:
    planless = (
        int(stats.get("planless_image_descriptor_count", 0)) == image_count
        and int(stats.get("fixed_transform_item_count", 0)) == 0
        and int(stats.get("host_expanded_transform_items_created", 0)) == 0
        and int(stats.get("host_output_block_source_lists_created", 0)) == 0
        and int(stats.get("host_global_transform_sort_items", 0)) == 0
        and bool(stats.get("device_mapping_fused", False))
    )
    if not planless:
        raise RuntimeError(
            "GALP runtime violated the planless production profile: "
            f"descriptors={stats.get('planless_image_descriptor_count', 0)} "
            f"items={stats.get('fixed_transform_item_count', 0)}"
        )
    if (
        int(stats.get("projection_item_count", 0)) != 0
        or int(stats.get("decoded_projection_item_count", 0)) != 0
        or int(stats.get("project_decoded_ycbcr_grid_launch_count", 0)) != 0
    ):
        raise RuntimeError("GALP planless benchmark unexpectedly used generic projection")


class GalpAdapter(PipelineAdapter):
    domain = "dct"
    worker_semantics = "galp_internal_runtime_not_configured_by_worker_count"
    config_name = "galp"

    def __init__(self, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device) -> None:
        super().__init__(contract, samples, device)
        if device.type != "cuda":
            raise ValueError("GALP pipeline requires a CUDA device")
        _add_diagnostics_to_path()
        self.stats_module = importlib.import_module("direct_dct")
        config = contract["pipelines"][self.config_name]
        if config.get("runtime_profile") != GALP_RUNTIME_PROFILE:
            raise ValueError(
                f"GALP production pipeline requires runtime_profile={GALP_RUNTIME_PROFILE!r}"
            )
        self.reader = DirectDctReader(config["manifest"])
        profile_info = self.reader.profile_info(VALIDATION)
        if profile_info["runtime_policy_id"] != GALP_RUNTIME_PROFILE:
            raise RuntimeError("GALP native profile does not match the benchmark contract")
        if config.get("preprocess") != "rgbnomore-val-pushdown":
            raise ValueError("GALP production pipeline supports only native RGB-no-more preprocessing")
        self.batch_size = int(contract["execution"]["batch_size"])
        self.total_batches = len(self.samples) // self.batch_size
        self.pipeline = self.reader.pipeline(VALIDATION)
        self.last_batch_prefetch_metrics: dict[str, float | int] = {}
        context = (
            f"pipeline={self.config_name}, manifest_version={config.get('manifest_version', 'unknown')}, "
            f"runtime_profile={GALP_RUNTIME_PROFILE}"
        )
        self.setup_metrics["semantic_profile_contract"] = _validate_profile_contract(
            profile_info,
            context=context,
        )

    def begin_repeat(self) -> None:
        self.last_batch_prefetch_metrics = {}
        image_id_batches = [
            [
                int(sample["galp_image_id"])
                for sample in self.samples[
                    begin : begin + self.batch_size
                ]
            ]
            for begin in range(0, self.total_batches * self.batch_size, self.batch_size)
        ]
        self.pipeline.start(image_id_batches)

    def load(
        self,
        expected: Sequence[dict[str, Any]],
        next_expected: Sequence[dict[str, Any]] | None = None,
    ) -> LoadedBatch:
        image_ids = [int(sample["galp_image_id"]) for sample in expected]
        native_batch = next(self.pipeline)
        observed_image_ids = [int(value) for value in native_batch.global_image_ids]
        if observed_image_ids != image_ids:
            raise RuntimeError(
                f"GALP native batch mismatch: expected {image_ids}, got {observed_image_ids}"
            )
        input_y = native_batch.y
        input_cbcr = native_batch.cbcr
        if (
            native_batch.layout != "transformed_dct_grid"
            or input_y.dtype != torch.float32
            or input_cbcr.dtype != torch.float32
            or tuple(input_y.shape[1:]) != (1, 28, 28, 8, 8)
            or tuple(input_cbcr.shape[1:]) != (2, 14, 14, 8, 8)
        ):
            raise RuntimeError("GALP production profile returned an invalid model-ready DCT batch")
        source_batches = [native_batch]
        metrics = native_batch.metrics
        self.last_batch_prefetch_metrics = {
            "batch_prefetch_read_wait_seconds": metrics.consumer_wait_ms / 1000.0,
            "batch_prefetch_producer_active_seconds": metrics.producer_ms / 1000.0,
            "batch_prefetch_planning_seconds": metrics.planning_ms / 1000.0,
            "batch_prefetch_io_staging_seconds": metrics.io_ms / 1000.0,
        }
        if not self.audit_enabled:
            return LoadedBatch(
                inputs=(input_y, input_cbcr),
                labels=None,
                ordinals=[],
                on_device=True,
            )
        labels = torch.tensor([sample["label"] for sample in expected], dtype=torch.long, device=self.device)
        native_stage_seconds = {
            key: float(value)
            for key, value in self.last_batch_prefetch_metrics.items()
            if key.endswith("_seconds") and isinstance(value, (int, float))
        }
        return LoadedBatch(
            inputs=(input_y, input_cbcr),
            labels=labels,
            ordinals=[int(sample["ordinal"]) for sample in expected],
            on_device=True,
            native_stage_seconds=native_stage_seconds,
            native_counters={},
            audit_sources=source_batches,
        )

    def close(self) -> None:
        self.pipeline.close()

    def finalize_batch_metrics(self, batch: LoadedBatch) -> None:
        if not self.audit_enabled or not batch.audit_sources:
            return
        source_execution_stats: list[dict[str, Any]] = []
        for source_batch in batch.audit_sources:
            try:
                stats = execution_stats(source_batch)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                stats = execution_stats_snapshot(source_batch)
            source_execution_stats.append(stats)
            _validate_transform_execution_stats(stats, len(batch.ordinals))
            if (
                not bool(stats.get("fixed_grid_output_float32", False))
                or not bool(stats.get("fixed_grid_output_affine_applied", False))
                or int(stats.get("fixed_grid_finalize_kernel_launch_count", 0)) != 1
            ):
                raise RuntimeError(
                    "GALP transformed-grid output did not use the required native FP32 finalize"
                )
        totals = self.stats_module._empty_totals()
        self.stats_module._accumulate_many_stats(totals, source_execution_stats)
        batch.native_stage_seconds.update(
            {
                key: float(value)
                for key, value in totals.items()
                if key.endswith("_seconds") and isinstance(value, (int, float))
            }
        )
        batch.native_counters.update(
            {
                key: int(value)
                for key, value in totals.items()
                if not key.endswith("_seconds") and isinstance(value, int)
            }
        )
        batch.native_properties.update(
            {
                key: value
                for key, value in totals.items()
                if key in {"storage_read_granularity", "decode_granularity", "sparse_read_fallback_reason"}
            }
        )
        full_bytes = int(totals.get("full_compressed_payload_bytes", 0))
        batch.native_properties["read_amplification"] = (
            int(totals.get("compressed_payload_bytes_read", 0)) / full_bytes if full_bytes > 0 else 0.0
        )
        batch.native_properties["sparse_read_supported"] = bool(
            int(totals.get("sparse_read_supported_batches", 0))
        )

    def end_repeat(self) -> None:
        self.pipeline.close()


def _make_adapter(
    name: str,
    contract: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    audit_enabled: bool = True,
) -> PipelineAdapter:
    adapters = {
        "galp": GalpAdapter,
        "rgbnomore": RgbNoMoreAdapter,
        "dali": DaliAdapter,
        "pytorch": PyTorchAdapter,
    }
    adapter = adapters[name](contract, samples, device)
    adapter.audit_enabled = audit_enabled
    return adapter


def _to_device(batch: LoadedBatch, device: torch.device) -> LoadedBatch:
    if batch.on_device:
        return batch
    inputs = tuple(item.to(device, non_blocking=True) for item in batch.inputs)
    labels = batch.labels.to(device, non_blocking=True) if batch.labels is not None else None
    return LoadedBatch(
        inputs=inputs,
        labels=labels,
        ordinals=batch.ordinals,
        on_device=True,
        native_stage_seconds=batch.native_stage_seconds,
        native_counters=batch.native_counters,
        native_properties=batch.native_properties,
        audit_sources=batch.audit_sources,
    )


def _validate_batch_identity(batch: LoadedBatch, expected: Sequence[dict[str, Any]]) -> None:
    if batch.labels is None:
        raise RuntimeError("batch identity audit requires labels")
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


def _synchronize_model_stream(device: torch.device) -> None:
    """Finish the consumed batch without draining independent next-batch preprocessing streams."""
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


@contextlib.contextmanager
def _nvtx_range(message: str, *, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


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


def run_pipeline(
    name: str,
    contract_path: Path,
    output: Path,
    *,
    runtime_profile: bool = False,
    emit_nvtx: bool = False,
) -> dict[str, Any]:
    if emit_nvtx and not runtime_profile:
        raise ValueError("--emit-nvtx requires --runtime-profile")
    contract = load_contract(contract_path)
    name = contract_pipeline_name(contract, name)
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
    if name in GALP_PIPELINES:
        galp_config = contract["pipelines"][name]
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
    runtime_tail_batches = 0
    adapter_samples = list(samples)
    if runtime_profile:
        galp_prefetch_depth = 2 if "galp" in contract["pipelines"]["enabled"] else 0
        runtime_tail_batches = max(
            int(contract["pipelines"]["dali"]["prefetch_queue_depth"]),
            galp_prefetch_depth,
        )
        tail_begin = warmup_batches * batch_size
        tail_end = tail_begin + runtime_tail_batches * batch_size
        tail_samples = list(samples[tail_begin:tail_end])
        require(
            len(tail_samples) == runtime_tail_batches * batch_size,
            "runtime profile lacks samples for the steady-state prefetch tail",
        )
        adapter_samples.extend(tail_samples)

    device = torch.device(normalize_device(execution["device"]))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)
    adapter_setup_memory_before = _process_memory_snapshot()
    adapter_setup_started = time.perf_counter()
    adapter = _make_adapter(
        name,
        contract,
        adapter_samples,
        device,
        audit_enabled=not runtime_profile,
    )
    adapter_setup_elapsed_ms = 1000.0 * (time.perf_counter() - adapter_setup_started)
    adapter_setup_memory_after = _process_memory_snapshot()
    adapter.setup_metrics["adapter_construction"] = {
        "elapsed_ms": adapter_setup_elapsed_ms,
        "host_process_rss_before_bytes": adapter_setup_memory_before["rss_bytes"],
        "host_process_rss_after_bytes": adapter_setup_memory_after["rss_bytes"],
        "host_process_rss_delta_bytes": (
            adapter_setup_memory_after["rss_bytes"] - adapter_setup_memory_before["rss_bytes"]
        ),
        "host_process_peak_rss_bytes": adapter_setup_memory_after["peak_rss_bytes"],
    }
    model = _build_rgb_model(contract, device) if adapter.domain == "rgb" else _build_dct_model(contract, device)
    model_stream: torch.cuda.Stream | None = None
    previous_stream: torch.cuda.Stream | None = None
    model_stream_priority_requested = execution.get("model_stream_priority", "greatest")
    model_stream_priority_resolved: int | None = None
    torch_stream_priority_range: tuple[int, int] | None = None
    if device.type == "cuda":
        previous_stream = torch.cuda.current_stream(device)
        previous_stream.synchronize()
        torch_stream_priority_range = tuple(
            int(value) for value in torch.cuda.Stream.priority_range()
        )
        model_stream_priority_resolved = _resolve_model_stream_priority(
            model_stream_priority_requested, torch_stream_priority_range
        )
        model_stream = torch.cuda.Stream(device=device, priority=model_stream_priority_resolved)
        torch.cuda.set_stream(model_stream)
    semantic_count = 0 if runtime_profile else int(contract["semantic_validation"]["sample_count"])
    semantic_store: dict[str, list[np.ndarray]] = {}
    prediction_store: dict[str, list[np.ndarray]] = {}
    repeat_records: list[dict[str, Any]] = []

    measured_expected = measured_samples(samples, batch_size, warmup_batches, measurement_batches)
    expected_trace = sample_trace(measured_expected)
    total_batches = len(adapter_samples) // batch_size

    def expected_batch(batch_index: int) -> list[dict[str, Any]]:
        begin = batch_index * batch_size
        return list(adapter_samples[begin : begin + batch_size])

    def following_batch(batch_index: int) -> list[dict[str, Any]] | None:
        return expected_batch(batch_index + 1) if batch_index + 1 < total_batches else None

    for repeat in range(repeats):
        adapter.begin_repeat()
        for warmup_index in range(warmup_batches):
            expected = expected_batch(warmup_index)
            batch = _to_device(adapter.load(expected, following_batch(warmup_index)), device)
            if not runtime_profile:
                _validate_batch_identity(batch, expected)
            _forward(
                model,
                batch.inputs,
                contract,
                device,
                validate_output=not runtime_profile,
            )
            _synchronize_model_stream(device)
            adapter.after_model_complete()

        if device.type == "cuda":
            _synchronize_model_stream(device)
            torch.cuda.reset_peak_memory_stats(device)
        host_memory_before = _process_memory_snapshot()
        cpu_process_seconds = 0.0
        latency_ms: list[float] = []
        loader_submit_ms: list[float] = []
        h2d_gpu_ms: list[float] = []
        forward_gpu_ms: list[float] = []
        actual_samples: list[dict[str, Any]] = []
        native_stage_seconds: dict[str, float] = {}
        native_stage_ms_by_batch: dict[str, list[float]] = {}
        native_counters: dict[str, int] = {}
        native_properties: dict[str, Any] = {}
        correct1 = 0
        correct5 = 0
        semantic_captured = 0
        h2d_event_pairs: list[tuple[torch.cuda.Event | None, torch.cuda.Event | None]] = []
        forward_event_pairs: list[tuple[torch.cuda.Event | None, torch.cuda.Event | None]] = []
        nvtx_enabled = emit_nvtx and device.type == "cuda"

        with _nvtx_range("runtime_e2e", enabled=nvtx_enabled):
            for measured_index in range(measurement_batches):
                batch_index = warmup_batches + measured_index
                expected = expected_batch(batch_index)
                with _nvtx_range(f"runtime_batch_{measured_index:04d}", enabled=nvtx_enabled):
                    process_started = time.process_time()
                    wall_started_ns = time.perf_counter_ns()
                    load_started_ns = wall_started_ns
                    with _nvtx_range("load_and_preprocess", enabled=nvtx_enabled):
                        batch = adapter.load(expected, following_batch(batch_index))
                    load_ended_ns = time.perf_counter_ns()
                    if not runtime_profile:
                        _validate_batch_identity(batch, expected)

                    h2d_start = _new_event(device)
                    h2d_end = _new_event(device)
                    with _nvtx_range("host_to_device", enabled=nvtx_enabled):
                        if h2d_start is not None:
                            h2d_start.record()
                        batch = _to_device(batch, device)
                        if h2d_end is not None:
                            h2d_end.record()

                    forward_start = _new_event(device)
                    forward_end = _new_event(device)
                    with _nvtx_range("model_forward", enabled=nvtx_enabled):
                        if forward_start is not None:
                            forward_start.record()
                        logits = _forward(
                            model,
                            batch.inputs,
                            contract,
                            device,
                            validate_output=not runtime_profile,
                        )
                        if forward_end is not None:
                            forward_end.record()
                        if not runtime_profile:
                            if batch.labels is None:
                                raise RuntimeError("accuracy audit requires labels")
                            batch_correct1, batch_correct5 = _accuracy_counts(logits, batch.labels)
                        else:
                            batch_correct1, batch_correct5 = 0, 0
                        _synchronize_model_stream(device)
                        adapter.after_model_complete()
                    wall_ended_ns = time.perf_counter_ns()
                    cpu_process_seconds += time.process_time() - process_started

                adapter.finalize_batch_metrics(batch)
                if not runtime_profile:
                    if repeat == 0 and semantic_captured < semantic_count:
                        semantic_captured += _capture_semantic(
                            semantic_store,
                            batch,
                            logits,
                            expected,
                            semantic_count - semantic_captured,
                        )
                    if repeat == 0:
                        prediction_store.setdefault("prediction_ordinals", []).append(
                            np.asarray(batch.ordinals, dtype=np.int64)
                        )
                        prediction_store.setdefault("prediction_labels", []).append(
                            np.asarray([item["label"] for item in expected], dtype=np.int64)
                        )
                        top5 = torch.topk(
                            logits.detach(), k=min(5, logits.shape[1]), dim=1
                        ).indices.cpu().numpy()
                        prediction_store.setdefault("top1_predictions", []).append(
                            top5[:, 0].astype(np.int64)
                        )
                        prediction_store.setdefault("top5_predictions", []).append(
                            top5.astype(np.int64)
                        )
                latency_ms.append((wall_ended_ns - wall_started_ns) / 1e6)
                loader_submit_ms.append((load_ended_ns - load_started_ns) / 1e6)
                h2d_event_pairs.append((h2d_start, h2d_end))
                forward_event_pairs.append((forward_start, forward_end))
                correct1 += batch_correct1
                correct5 += batch_correct5
                if not runtime_profile:
                    actual_samples.extend(expected)
                    for key, value in batch.native_stage_seconds.items():
                        seconds = float(value)
                        native_stage_seconds[key] = native_stage_seconds.get(key, 0.0) + seconds
                        native_stage_ms_by_batch.setdefault(key, []).append(seconds * 1000.0)
                    if (
                        "device_mapping_seconds" in batch.native_stage_seconds
                        or "fixed_transform_kernel_seconds" in batch.native_stage_seconds
                    ):
                        combined_ms = 1000.0 * (
                            float(batch.native_stage_seconds.get("device_mapping_seconds", 0.0))
                            + float(batch.native_stage_seconds.get("fixed_transform_kernel_seconds", 0.0))
                        )
                        native_stage_ms_by_batch.setdefault(
                            "device_mapping_plus_fixed_transform_seconds", []
                        ).append(combined_ms)
                    for key, value in batch.native_counters.items():
                        _accumulate_native_counter(native_counters, key, int(value))
                    for key, value in batch.native_properties.items():
                        current = native_properties.get(key)
                        if current is None:
                            native_properties[key] = value
                        elif key == "read_amplification":
                            native_properties[key] = max(float(current), float(value))
                        elif current != value:
                            native_properties[key] = "mixed"

        h2d_gpu_ms.extend(_event_ms(start, end) for start, end in h2d_event_pairs)
        forward_gpu_ms.extend(_event_ms(start, end) for start, end in forward_event_pairs)

        measured_seconds = sum(latency_ms) / 1000.0
        images = measurement_batches * batch_size
        actual_trace = expected_trace if runtime_profile else sample_trace(actual_samples)
        if not runtime_profile and actual_trace["sha256"] != expected_trace["sha256"]:
            raise RuntimeError("measured sample trace does not match the contract")
        host_memory_after = _process_memory_snapshot()
        record = {
            "repeat": repeat,
            "images": images,
            "seconds": measured_seconds,
            "throughput_images_per_s": images / measured_seconds,
            "end_to_end_latency_ms": distribution(latency_ms),
            "accuracy_top1": None if runtime_profile else correct1 / images,
            "accuracy_top5": None if runtime_profile else correct5 / images,
            "correct_top1": None if runtime_profile else correct1,
            "correct_top5": None if runtime_profile else correct5,
            "cpu_process_seconds": cpu_process_seconds,
            "cpu_time_scope": "main_process_only_excludes_loader_workers",
            "host_process_rss_before_measurement_bytes": host_memory_before["rss_bytes"],
            "host_process_rss_after_measurement_bytes": host_memory_after["rss_bytes"],
            "host_process_peak_rss_bytes": host_memory_after["peak_rss_bytes"],
            "host_process_memory_scope": (
                "linux_main_pipeline_process_VmRSS_and_lifetime_VmHWM_includes_python_torch_"
                "and_native_pipeline_allocations_excludes_loader_worker_processes"
            ),
            "stage_breakdown_ms": {
                "loader_and_preprocess_submit": distribution(loader_submit_ms),
                "host_to_device_gpu": distribution(h2d_gpu_ms),
                "model_forward_gpu": distribution(forward_gpu_ms),
                "native_totals_seconds": native_stage_seconds,
                "native_per_batch_ms": {
                    key: distribution(values) for key, values in native_stage_ms_by_batch.items()
                },
            },
            "native_counters": native_counters,
            "native_properties": native_properties,
            "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
            "peak_gpu_memory_scope": (
                "torch_allocator_only_excludes_galp_native_allocations"
                if name in GALP_PIPELINES
                else "torch_allocator_only_excludes_dali_native_allocations"
                if name == "dali"
                else "torch_allocator"
            ),
            "sample_trace": actual_trace,
        }
        repeat_records.append(record)
        adapter.end_repeat()

    semantic_path: Path | None = None
    semantic_sha256: str | None = None
    if not runtime_profile:
        semantic_path = output.parent / f"semantic_{name}.npz"
        semantic_store.update(prediction_store)
        _write_semantic(
            semantic_path,
            semantic_store,
            {
                "pipeline": name,
                "domain": adapter.domain,
                "sample_count": semantic_count,
                "prediction_agreement_sample_count": int(
                    contract["semantic_validation"]["prediction_agreement_sample_count"]
                ),
                "contract_sha256": sha256_json(contract),
            },
        )
        semantic_sha256 = sha256_file(semantic_path)
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
        "mode": "runtime_profile" if runtime_profile else "validated_benchmark",
        "measurement_scope": (
            "steady_state_data_read_decode_preprocess_h2d_model_forward_and_batch_completion"
            if runtime_profile
            else "validated_end_to_end_benchmark"
        ),
        "nvtx_capture_range": "runtime_e2e" if emit_nvtx else None,
        "audit_enabled": not runtime_profile,
        "runtime_tail_batches": runtime_tail_batches,
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256_json(contract),
        "sample_manifest": str(manifest_path.resolve()),
        "sample_manifest_sha256": contract["dataset"]["manifest_sha256"],
        "model": _model_metadata(contract, adapter.domain),
        "execution": dict(execution),
        "cuda_scheduling": {
            "model_stream_priority_requested": model_stream_priority_requested,
            "model_stream_priority_resolved": model_stream_priority_resolved,
            "model_stream_priority_actual": (
                int(model_stream.priority) if model_stream is not None else None
            ),
            "torch_least_stream_priority": (
                torch_stream_priority_range[0] if torch_stream_priority_range is not None else None
            ),
            "torch_greatest_stream_priority": (
                torch_stream_priority_range[1] if torch_stream_priority_range is not None else None
            ),
            "model_stream_is_explicit": model_stream is not None,
        },
        "worker_semantics": adapter.worker_semantics,
        "pipeline_setup": dict(adapter.setup_metrics),
        "timing": dict(contract["timing"]),
        "preprocess": dict(contract["preprocess"][adapter.domain]),
        "pipeline_config": dict(contract["pipelines"][name]),
        "dataset_full_size": manifest["full_dataset_size"],
        "semantic_artifact": str(semantic_path.resolve()) if semantic_path is not None else None,
        "semantic_artifact_sha256": semantic_sha256,
        "dependency_versions": dependency_versions,
        "device_metadata": (
            {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
                "multi_processor_count": torch.cuda.get_device_properties(device).multi_processor_count,
            }
            if device.type == "cuda"
            else {"name": "cpu"}
        ),
        "repeats": repeat_records,
    }
    if previous_stream is not None:
        torch.cuda.set_stream(previous_stream)
    write_json(output, result)
    adapter.close()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline",
        choices=INFERENCE_PIPELINES,
        required=True,
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-profile",
        action="store_true",
        help=(
            "After warmup, measure only the production runtime path; "
            "disable identity/accuracy/native-stat audits and semantic artifact export."
        ),
    )
    parser.add_argument(
        "--emit-nvtx",
        action="store_true",
        help="Emit the runtime_e2e capture trigger and nested runtime NVTX ranges.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_pipeline(
        args.pipeline,
        args.contract,
        args.output,
        runtime_profile=args.runtime_profile,
        emit_nvtx=args.emit_nvtx,
    )
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
