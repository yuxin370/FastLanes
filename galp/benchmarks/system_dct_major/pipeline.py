#!/usr/bin/env python3
"""Execute one no-shuffle DCT-major, DALI, or PyTorch pipeline."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from common import (
    PIPELINES,
    PIPELINE_RESULT_SCHEMA,
    REPO_ROOT,
    RGBNOMORE_BENCHMARK_ROOT,
    chunked,
    distribution,
    file_identity,
    load_contract,
    load_sample_manifest,
    parse_manifest,
    sample_trace,
    selected_batches,
    sha256_file,
    sha256_json,
    write_json,
)
from feature_model import build_workload_model, expected_output_width


@dataclass
class LoadedBatch:
    inputs: tuple[torch.Tensor, ...]
    labels: torch.Tensor
    ordinals: list[int]
    label_values: list[int]
    on_device: bool
    native_stats: list[dict[str, Any]] = field(default_factory=list)
    keepalive: list[Any] = field(default_factory=list)


def _process_memory_snapshot(status_path: Path = Path("/proc/self/status")) -> dict[str, int]:
    values = {"rss_bytes": 0, "peak_rss_bytes": 0}
    if not status_path.is_file():
        return values
    mapping = {"VmRSS": "rss_bytes", "VmHWM": "peak_rss_bytes"}
    for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        name, _, tail = line.partition(":")
        if name not in mapping:
            continue
        fields = tail.strip().split()
        if fields:
            values[mapping[name]] = int(fields[0]) * 1024
    return values


def _loader_kwargs(contract: dict[str, Any], pipeline_name: str) -> dict[str, Any]:
    execution = contract["execution"]
    workers = int(execution["workers"])
    config = contract["pipelines"].get(pipeline_name, {})
    result: dict[str, Any] = {
        "batch_size": int(execution["batch_size"]),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": str(execution["device"]).startswith("cuda"),
        "drop_last": False,
    }
    if workers > 0:
        result["persistent_workers"] = True
        result["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return result


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


class Adapter:
    domain: str
    worker_semantics: str

    def __init__(self, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device, name: str) -> None:
        self.contract = contract
        self.samples = list(samples)
        self.device = device
        self.name = name

    def begin_repeat(self) -> None:
        raise NotImplementedError

    def prime_cold_start(self) -> None:
        """Start reversible loader preparation while the model is being constructed."""

        return None

    def load(self, expected: Sequence[dict[str, Any]]) -> LoadedBatch:
        raise NotImplementedError

    def begin_measurement(self) -> None:
        pass

    def end_repeat(self) -> None:
        pass

    def close(self) -> None:
        pass


class PyTorchAdapter(Adapter):
    domain = "rgb"
    worker_semantics = "torch_dataloader_processes"

    def __init__(self, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device, name: str) -> None:
        super().__init__(contract, samples, device, name)
        root = Path(contract["models"]["rgbnomore_root"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        datasets = importlib.import_module("datasets")
        transform = datasets.get_transform(dataset="imagenet", type="test", dtype=torch.float32)
        self.loader = torch.utils.data.DataLoader(
            CanonicalRgbDataset(samples, transform),
            **_loader_kwargs(contract, name),
        )
        self.iterator: Iterator[Any] | None = None
        self._cold_iterator_primed = False

    def begin_repeat(self) -> None:
        if self._cold_iterator_primed:
            self._cold_iterator_primed = False
            return
        self.iterator = iter(self.loader)

    def prime_cold_start(self) -> None:
        self.iterator = iter(self.loader)
        self._cold_iterator_primed = True

    def load(self, expected: Sequence[dict[str, Any]]) -> LoadedBatch:
        if self.iterator is None:
            raise RuntimeError("PyTorch adapter was not started")
        images, labels, ordinals = next(self.iterator)
        return LoadedBatch(
            inputs=(images,),
            labels=labels.long(),
            ordinals=[int(item) for item in ordinals.tolist()],
            label_values=[int(item) for item in labels.tolist()],
            on_device=False,
        )


class RgbNoMoreAdapter(Adapter):
    domain = "dct"
    worker_semantics = "torch_dataloader_processes"

    def __init__(self, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device, name: str) -> None:
        super().__init__(contract, samples, device, name)
        root = Path(contract["models"]["rgbnomore_root"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
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
        self.loader = torch.utils.data.DataLoader(
            IndexedDataset(transformed),
            **_loader_kwargs(contract, name),
        )
        self.iterator: Iterator[Any] | None = None
        self._cold_iterator_primed = False

    def begin_repeat(self) -> None:
        if self._cold_iterator_primed:
            self._cold_iterator_primed = False
            return
        self.iterator = iter(self.loader)

    def prime_cold_start(self) -> None:
        self.iterator = iter(self.loader)
        self._cold_iterator_primed = True

    def load(self, expected: Sequence[dict[str, Any]]) -> LoadedBatch:
        if self.iterator is None:
            raise RuntimeError("RGB-no-more adapter was not started")
        (input_y, input_cbcr), labels, ordinals = next(self.iterator)
        return LoadedBatch(
            inputs=(input_y, input_cbcr),
            labels=labels.long(),
            ordinals=[int(self.samples[int(item)]["ordinal"]) for item in ordinals.tolist()],
            label_values=[int(item) for item in labels.tolist()],
            on_device=False,
        )


class DaliAdapter(Adapter):
    domain = "rgb"
    worker_semantics = "dali_cpu_threads"

    def __init__(self, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device, name: str) -> None:
        super().__init__(contract, samples, device, name)
        if device.type != "cuda":
            raise ValueError("DALI requires CUDA")
        self.iterator: Any | None = None
        self._cold_iterator_primed = False

    def begin_repeat(self) -> None:
        if self._cold_iterator_primed:
            self._cold_iterator_primed = False
            return
        from nvidia.dali import fn, pipeline_def, types
        from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

        execution = self.contract["execution"]
        config = self.contract["pipelines"][self.name]
        paths = [str(sample["path"]) for sample in self.samples]
        ordinals = [int(sample["ordinal"]) for sample in self.samples]
        preprocess = self.contract["preprocess"]["rgb"]

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
                resize_shorter=int(preprocess["resize_shorter"]),
                interp_type=types.INTERP_LINEAR,
                antialias=True,
            )
            images = fn.crop_mirror_normalize(
                images,
                device="gpu",
                dtype=types.FLOAT,
                output_layout="CHW",
                crop=tuple(preprocess["crop_size"]),
                crop_pos_x=0.499999,
                crop_pos_y=0.499999,
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            )
            return images, ordinal

        pipeline = create_pipeline(
            batch_size=int(execution["batch_size"]),
            num_threads=max(1, int(execution["workers"])),
            device_id=int(config.get("device_id", self.device.index or 0)),
            seed=int(execution["seed"]),
            prefetch_queue_depth=int(config.get("prefetch_queue_depth", 2)),
        )
        pipeline.build()
        self.iterator = DALIGenericIterator(
            [pipeline],
            output_map=["image", "ordinal"],
            reader_name="Reader",
            auto_reset=False,
            last_batch_policy=LastBatchPolicy.PARTIAL,
            last_batch_padded=False,
        )

    def prime_cold_start(self) -> None:
        self.begin_repeat()
        self._cold_iterator_primed = True

    def load(self, expected: Sequence[dict[str, Any]]) -> LoadedBatch:
        if self.iterator is None:
            raise RuntimeError("DALI adapter was not started")
        batch = next(self.iterator)[0]
        images = batch["image"][: len(expected)]
        ordinals = [int(item) for item in batch["ordinal"].reshape(-1)[: len(expected)].cpu().tolist()]
        label_values = [int(self.samples[item]["label"]) for item in ordinals]
        labels = torch.tensor(label_values, dtype=torch.long, device=self.device)
        return LoadedBatch(
            inputs=(images,),
            labels=labels,
            ordinals=ordinals,
            label_values=label_values,
            on_device=True,
        )

    def end_repeat(self) -> None:
        if self.iterator is not None:
            self.iterator.reset()
        self.iterator = None


def _load_direct_dct_modules(
    binding_dir: Path,
    *,
    load_postdecode_diagnostics: bool,
) -> tuple[Any, Any | None, dict[str, Any], dict[str, Any]]:
    diagnostics = RGBNOMORE_BENCHMARK_ROOT / "diagnostics"
    torch_source = REPO_ROOT / "galp/torch"
    for path in (binding_dir.resolve(), torch_source.resolve(), diagnostics.resolve()):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    binding_started_ns = time.perf_counter_ns()
    binding = importlib.import_module("_galp_direct_dct")
    binding_ready_ns = time.perf_counter_ns()
    profile = importlib.import_module("rgbnomore_dct_profile")
    profile_ready_ns = time.perf_counter_ns()
    direct_dct = None
    if load_postdecode_diagnostics:
        direct_dct = importlib.import_module("direct_dct")
    diagnostics_ready_ns = time.perf_counter_ns()
    return (
        binding,
        direct_dct,
        dict(profile.RGBNOMORE_VAL_DCT_GRID_TRANSFORM_FP32),
        {
            "binding_extension_import_ms": (binding_ready_ns - binding_started_ns) / 1.0e6,
            "direct_dct_profile_import_ms": (profile_ready_ns - binding_ready_ns) / 1.0e6,
            "postdecode_diagnostics_import_ms": (diagnostics_ready_ns - profile_ready_ns) / 1.0e6,
            "postdecode_diagnostics_imported": load_postdecode_diagnostics,
        },
    )


def _native_args(config: dict[str, Any], preprocess: str) -> SimpleNamespace:
    return SimpleNamespace(
        preprocess=preprocess,
        cache_capacity_mib=int(config.get("cache_capacity_mib", 0)),
        plan_cache_capacity=int(config.get("plan_cache_capacity", 0)),
        decode_batch_rowgroups=int(config.get("decode_batch_rowgroups", 64)),
        decode_workset_capacity_mib=int(config.get("decode_workset_capacity_mib", 512)),
        rowgroup_prefetch_depth=int(config.get("rowgroup_prefetch_depth", 16)),
        rowgroup_prefetch_workers=int(config.get("rowgroup_prefetch_workers", 4)),
        rowgroup_prefetch_min_decode_batches=int(config.get("rowgroup_prefetch_min_decode_batches", 1)),
        no_dequantize=False,
        no_scale=False,
        enable_planless_execution=bool(config.get("enable_planless_execution", True)),
        crop_execution_mode=str(config.get("crop_execution_mode", "auto")),
        scheduling_policy=str(config.get("scheduling_policy", "limited-overlap")),
        transform_blocks_per_launch=int(config.get("transform_blocks_per_launch", 0)),
        transform_ctas_per_launch=int(config.get("transform_ctas_per_launch", 0)),
        use_low_priority_streams=bool(config.get("use_low_priority_streams", True)),
        block_major_double_buffer=str(config.get("block_major_double_buffer", "auto")),
    )


def _batch_native_stats(batch: Any) -> dict[str, Any]:
    stats = dict(batch.execution_stats)
    cache = dict(getattr(batch, "cache_stats", {}))
    if cache:
        stats.update(
            {
                "decoded_rowgroup_cache_capacity_bytes": int(cache.get("capacity_bytes", 0)),
                "decoded_rowgroup_cache_current_bytes": int(cache.get("resident_bytes", 0)),
                "decoded_rowgroup_cache_peak_bytes": int(
                    cache.get("peak_resident_bytes", cache.get("resident_bytes", 0))
                ),
                "decoded_rowgroup_cache_current_rowgroups": int(cache.get("resident_rowgroups", 0)),
                "decoded_rowgroup_cache_peak_rowgroups": int(
                    cache.get("peak_resident_rowgroups", cache.get("resident_rowgroups", 0))
                ),
                "decoded_rowgroup_cache_hits": int(cache.get("hits", 0)),
                "decoded_rowgroup_cache_misses": int(cache.get("misses", 0)),
                "decoded_rowgroup_cache_inserts": int(cache.get("inserts", 0)),
                "decoded_rowgroup_cache_evictions": int(cache.get("evictions", 0)),
            }
        )
    return stats


def _present_component_by_slot(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    components = [item for item in metadata.get("components", []) if bool(item.get("present"))]
    by_slot = {
        int(item.get("semantic_slot_id", -1)): item
        for item in components
        if int(item.get("semantic_slot_id", -1)) in (0, 1, 2)
    }
    if 0 not in by_slot:
        by_slot = {
            int(item.get("local_component_index", -1)): item
            for item in components
            if int(item.get("local_component_index", -1)) in (0, 1, 2)
        }
    return by_slot


def _is_grayscale_metadata(metadata: dict[str, Any]) -> bool:
    by_slot = _present_component_by_slot(metadata)
    return 0 in by_slot and 1 not in by_slot and 2 not in by_slot


def _rebuild_grayscale_full_grid(
    batch: Any,
    metadata: dict[str, Any],
    image_id: int,
) -> tuple[SimpleNamespace, int]:
    """Rebuild a full Y grid from a compact full decode and synthesize zero chroma."""

    if batch.layout != "compact":
        raise RuntimeError(f"grayscale full fallback requires compact layout, got {batch.layout!r}")
    if list(batch.selected_coefficients) != list(range(64)):
        raise RuntimeError("grayscale full fallback requires all 64 DCT coefficients")
    by_slot = _present_component_by_slot(metadata)
    if not _is_grayscale_metadata(metadata):
        raise RuntimeError(f"image_id={image_id} is not a grayscale Y-only JPEG")
    y_component = by_slot[0]
    height = int(y_component.get("height_in_blocks", 0))
    width = int(y_component.get("width_in_blocks", 0))
    if height < 2 or width < 2:
        raise RuntimeError(f"image_id={image_id} has invalid grayscale Y grid {height}x{width}")

    coefficients = batch.coefficients
    blocks = list(batch.block_metadata)
    expected_blocks = height * width
    if coefficients.ndim != 2 or tuple(coefficients.shape) != (expected_blocks, 64):
        raise RuntimeError(
            f"image_id={image_id} compact full decode returned coefficients {tuple(coefficients.shape)}, "
            f"expected ({expected_blocks},64)"
        )
    if len(blocks) != expected_blocks:
        raise RuntimeError(
            f"image_id={image_id} compact full decode returned {len(blocks)} block metadata entries, "
            f"expected {expected_blocks}"
        )

    coordinates: list[tuple[int, int]] = []
    for block in blocks:
        request_index = int(block.get("request_index", -1))
        global_image_index = int(block.get("global_image_index", -1))
        semantic_slot = int(block.get("semantic_slot_id", -1))
        block_y = int(block.get("block_y", -1))
        block_x = int(block.get("block_x", -1))
        if request_index != 0 or global_image_index != image_id or semantic_slot != 0:
            raise RuntimeError(f"image_id={image_id} compact fallback contains a non-Y or foreign block: {block}")
        if not (0 <= block_y < height and 0 <= block_x < width):
            raise RuntimeError(f"image_id={image_id} compact fallback block is outside the Y grid: {block}")
        coordinates.append((block_y, block_x))
    if len(set(coordinates)) != expected_blocks:
        raise RuntimeError(f"image_id={image_id} compact fallback Y block coordinates are not complete and unique")

    block_y = torch.tensor([item[0] for item in coordinates], dtype=torch.long, device=coefficients.device)
    block_x = torch.tensor([item[1] for item in coordinates], dtype=torch.long, device=coefficients.device)
    y = torch.empty((1, 1, height, width, 8, 8), dtype=coefficients.dtype, device=coefficients.device)
    y[0, 0, block_y, block_x] = coefficients.reshape(expected_blocks, 8, 8)
    cbcr = torch.zeros(
        (1, 2, height // 2, width // 2, 8, 8),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    return (
        SimpleNamespace(
            layout="ycbcr_dct_grid",
            selected_coefficients=list(range(64)),
            y=y,
            cbcr=cbcr,
        ),
        expected_blocks,
    )


class GalpAdapter(Adapter):
    domain = "dct"
    worker_semantics = "galp_native_segment_prefetch"

    def __init__(self, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device, name: str) -> None:
        super().__init__(contract, samples, device, name)
        if device.type != "cuda":
            raise ValueError("GALP Direct-DCT requires CUDA")
        self.config = contract["pipelines"][name]
        self.preprocess = str(self.config["preprocess"])
        block_major_access_dir = self.config.get("block_major_access_dir")
        if block_major_access_dir is not None:
            access_dir = Path(block_major_access_dir).resolve()
            companion_index = access_dir / "manifest.block_major_access.bin"
            if not companion_index.is_file():
                raise FileNotFoundError(
                    f"configured block-major access companion index is missing: {companion_index}"
                )
            os.environ["GALP_BLOCK_MAJOR_ACCESS_DIR"] = str(access_dir)
        binding, self.direct_dct, self.grid_transform, self.startup_timings = _load_direct_dct_modules(
            Path(self.config["torch_binding_dir"]),
            load_postdecode_diagnostics=self.preprocess == "rgbnomore-val",
        )
        reader_started_ns = time.perf_counter_ns()
        self.reader = binding.DirectDctReader(str(Path(self.config["manifest"]).resolve()))
        reader_ready_ns = time.perf_counter_ns()
        self.startup_timings["direct_dct_reader_python_constructor_ms"] = (
            reader_ready_ns - reader_started_ns
        ) / 1.0e6
        self.startup_timings["native_reader_initialization"] = dict(self.reader.initialization_stats)
        if int(self.reader.image_count) < len(samples):
            raise ValueError(f"GALP manifest has {self.reader.image_count} images for {len(samples)} samples")
        self.args = _native_args(self.config, self.preprocess)
        self.rgbnomore_transform = (
            self.direct_dct.build_rgbnomore_dct_val_transform(Path(contract["models"]["rgbnomore_root"]))
            if self.preprocess == "rgbnomore-val"
            else None
        )
        segment_size = int(self.config.get("segment_size", contract["execution"]["batch_size"]))
        self.segment_size = max(1, segment_size)
        warmup_images = int(contract["execution"]["batch_size"]) * int(
            contract["execution"]["warmup_batches"]
        )
        self._warmup_segments = list(chunked(self.samples[:warmup_images], self.segment_size))
        self._measurement_segments = list(chunked(self.samples[warmup_images:], self.segment_size))
        self.segments: list[list[dict[str, Any]]] = []
        self._next_segment = 0
        self._pending: Any | None = None
        self._current: dict[str, Any] | None = None
        self._cold_measurement_primed = False
        self._reuse_cold_measurement = False

    def startup_diagnostics(self) -> dict[str, Any]:
        result = dict(self.startup_timings)
        # Descriptor mmap/ValidateSource is lazy and normally occurs in the
        # real first-segment prefetch, so refresh the native snapshot at report
        # time rather than freezing a construction-only zero.
        result["native_reader_initialization"] = dict(self.reader.initialization_stats)
        return result

    def _activate_segments(self, segments: Sequence[Sequence[dict[str, Any]]]) -> None:
        self.segments = [list(segment) for segment in segments]
        self._next_segment = 0
        self._pending = None
        self._current = None

    def _prefetch(self, segment: Sequence[dict[str, Any]]) -> Any:
        image_ids = [int(sample["galp_image_id"]) for sample in segment]
        return self.reader.prefetch_batch(
            image_ids,
            crop=None,
            dct_coeffs="all",
            cache_capacity_mib=self.args.cache_capacity_mib,
            decode_batch_rowgroups=self.args.decode_batch_rowgroups,
            decode_workset_capacity_mib=getattr(self.args, "decode_workset_capacity_mib", 512),
            rowgroup_prefetch_depth=self.args.rowgroup_prefetch_depth,
            rowgroup_prefetch_workers=self.args.rowgroup_prefetch_workers,
            rowgroup_prefetch_min_decode_batches=self.args.rowgroup_prefetch_min_decode_batches,
            plan_cache_capacity=self.args.plan_cache_capacity,
            enable_planless_execution=self.args.enable_planless_execution,
            scheduling_policy=self.args.scheduling_policy,
            transform_blocks_per_launch=self.args.transform_blocks_per_launch,
            transform_ctas_per_launch=self.args.transform_ctas_per_launch,
            use_low_priority_streams=self.args.use_low_priority_streams,
            block_major_double_buffer=getattr(self.args, "block_major_double_buffer", "auto"),
            crop_execution_mode=self.args.crop_execution_mode,
            layout="transformed_dct_grid",
            grid_transform=self.grid_transform,
        )

    def _start_next_prefetch(self) -> None:
        if self._next_segment >= len(self.segments):
            self._pending = None
            return
        self._pending = self._prefetch(self.segments[self._next_segment])
        self._next_segment += 1

    def _load_next_segment(self) -> None:
        if self._pending is None:
            raise StopIteration("GALP segment stream is exhausted")
        pending = self._pending
        batch = pending.read()
        image_ids = [int(item) for item in batch.global_image_ids]
        y = batch.y
        cbcr = batch.cbcr
        if batch.layout != "transformed_dct_grid" or y.dtype != torch.float32 or cbcr.dtype != torch.float32:
            raise RuntimeError(
                f"expected native FP32 transformed grid, got layout={batch.layout} y={y.dtype} cbcr={cbcr.dtype}"
            )
        if tuple(y.shape[1:]) != (1, 28, 28, 8, 8) or tuple(cbcr.shape[1:]) != (2, 14, 14, 8, 8):
            raise RuntimeError(f"unexpected transformed grid shapes: y={tuple(y.shape)} cbcr={tuple(cbcr.shape)}")
        self._current = {
            "image_ids": image_ids,
            "y": y,
            "cbcr": cbcr,
            "offset": 0,
            "batch": batch,
            "stats_pending": True,
            "prefetch_telemetry": {
                "producer_active_ms": float(pending.producer_active_ms),
                "planning_ms": float(pending.planning_ms),
                "io_staging_ms": float(pending.io_staging_ms),
                "ordered_submission_ms": float(pending.ordered_submission_ms),
            },
        }
        self._start_next_prefetch()

    def begin_repeat(self) -> None:
        if getattr(self, "_cold_measurement_primed", False):
            self._cold_measurement_primed = False
            self._reuse_cold_measurement = True
            return
        initial = self._warmup_segments if self._warmup_segments else self._measurement_segments
        self._activate_segments(initial)

    def begin_measurement(self) -> None:
        if getattr(self, "_reuse_cold_measurement", False):
            self._reuse_cold_measurement = False
            return
        self._activate_segments(self._measurement_segments)

    def prime_cold_start(self) -> None:
        if self._warmup_segments or not self._measurement_segments or self._cold_measurement_primed:
            return
        self._activate_segments(self._measurement_segments)
        self._start_next_prefetch()
        self._cold_measurement_primed = True

    def _load_pushdown(self, expected: Sequence[dict[str, Any]]) -> LoadedBatch:
        remaining = [int(sample["galp_image_id"]) for sample in expected]
        y_parts: list[torch.Tensor] = []
        cbcr_parts: list[torch.Tensor] = []
        native_stats: list[dict[str, Any]] = []
        keepalive: list[Any] = []
        while remaining:
            if self._current is None or int(self._current["offset"]) >= len(self._current["image_ids"]):
                if self._pending is None:
                    self._start_next_prefetch()
                self._load_next_segment()
            assert self._current is not None
            offset = int(self._current["offset"])
            available = len(self._current["image_ids"]) - offset
            take = min(len(remaining), available)
            observed = self._current["image_ids"][offset : offset + take]
            if observed != remaining[:take]:
                raise RuntimeError(f"GALP segment order mismatch: expected {remaining[:take]}, got {observed}")
            y_parts.append(self._current["y"][offset : offset + take])
            cbcr_parts.append(self._current["cbcr"][offset : offset + take])
            keepalive.append(self._current["batch"])
            if self._current["stats_pending"]:
                stats = _batch_native_stats(self._current["batch"])
                stats["segment_image_count"] = len(self._current["image_ids"])
                stats["segment_first_image_id"] = self._current["image_ids"][0]
                stats["segment_last_image_id"] = self._current["image_ids"][-1]
                stats.update({f"prefetch_{key}": value for key, value in self._current["prefetch_telemetry"].items()})
                native_stats.append(stats)
                self._current["stats_pending"] = False
            self._current["offset"] = offset + take
            remaining = remaining[take:]
        y = y_parts[0] if len(y_parts) == 1 else torch.cat(y_parts, dim=0)
        cbcr = cbcr_parts[0] if len(cbcr_parts) == 1 else torch.cat(cbcr_parts, dim=0)
        label_values = [int(sample["label"]) for sample in expected]
        labels = torch.tensor(label_values, dtype=torch.long, device=self.device)
        return LoadedBatch(
            inputs=(y, cbcr),
            labels=labels,
            ordinals=[int(sample["ordinal"]) for sample in expected],
            label_values=label_values,
            on_device=True,
            native_stats=native_stats,
            keepalive=keepalive,
        )

    def _load_postdecode(self, expected: Sequence[dict[str, Any]]) -> LoadedBatch:
        image_ids = [int(sample["galp_image_id"]) for sample in expected]
        y_items: list[torch.Tensor] = []
        cbcr_items: list[torch.Tensor] = []
        source_batches: list[Any] = []
        native_stats: list[dict[str, Any]] = []
        for image_id in image_ids:
            grayscale_full_blocks = 0
            try:
                image_y, image_cbcr, image_batches = self.direct_dct.read_and_adapt_batch(
                    self.reader,
                    self.args,
                    [image_id],
                    None,
                    self.rgbnomore_transform,
                )
            except RuntimeError as error:
                if "YCbCr DCT grid layout requires Y, Cb, and Cr components per image" not in str(error):
                    raise
                metadata = self.reader.image_metadata(image_id)
                if not _is_grayscale_metadata(metadata):
                    raise
                full_batch = self.reader.read_batch(
                    [image_id],
                    crop=None,
                    dct_coeffs="all",
                    cache_capacity_mib=self.args.cache_capacity_mib,
                    decode_batch_rowgroups=self.args.decode_batch_rowgroups,
                    decode_workset_capacity_mib=getattr(self.args, "decode_workset_capacity_mib", 512),
                    rowgroup_prefetch_depth=self.args.rowgroup_prefetch_depth,
                    rowgroup_prefetch_workers=self.args.rowgroup_prefetch_workers,
                    rowgroup_prefetch_min_decode_batches=self.args.rowgroup_prefetch_min_decode_batches,
                    plan_cache_capacity=self.args.plan_cache_capacity,
                    enable_planless_execution=self.args.enable_planless_execution,
                    scheduling_policy=self.args.scheduling_policy,
                    transform_blocks_per_launch=self.args.transform_blocks_per_launch,
                    transform_ctas_per_launch=self.args.transform_ctas_per_launch,
                    use_low_priority_streams=self.args.use_low_priority_streams,
                    crop_execution_mode="full-rowgroup-decode",
                    layout="compact",
                )
                full_grid, grayscale_full_blocks = _rebuild_grayscale_full_grid(
                    full_batch,
                    metadata,
                    image_id,
                )
                image_y, image_cbcr = self.direct_dct.adapt_galp_batch_to_rgbnomore(
                    self.reader,
                    full_grid,
                    [image_id],
                    dequantize=not self.args.no_dequantize,
                    scale=not self.args.no_scale,
                    preprocess=self.args.preprocess,
                    rgbnomore_dct_val_transform=self.rgbnomore_transform,
                )
                image_batches = [full_batch]
            y_items.append(image_y[0])
            cbcr_items.append(image_cbcr[0])
            source_batches.extend(image_batches)
            for batch in image_batches:
                stats = _batch_native_stats(batch)
                if grayscale_full_blocks:
                    stats["grayscale_full_fallback_count"] = 1
                    stats["grayscale_zero_chroma_image_count"] = 1
                    stats["grayscale_full_y_block_count"] = grayscale_full_blocks
                native_stats.append(stats)
        y = torch.stack(y_items, dim=0)
        cbcr = torch.stack(cbcr_items, dim=0)
        label_values = [int(sample["label"]) for sample in expected]
        labels = torch.tensor(label_values, dtype=torch.long, device=self.device)
        return LoadedBatch(
            inputs=(y, cbcr),
            labels=labels,
            ordinals=[int(sample["ordinal"]) for sample in expected],
            label_values=label_values,
            on_device=True,
            native_stats=native_stats,
            keepalive=list(source_batches),
        )

    def load(self, expected: Sequence[dict[str, Any]]) -> LoadedBatch:
        if self.preprocess == "rgbnomore-val-pushdown":
            return self._load_pushdown(expected)
        if self.preprocess == "rgbnomore-val":
            return self._load_postdecode(expected)
        raise RuntimeError(f"unsupported GALP preprocess: {self.preprocess}")

    def end_repeat(self) -> None:
        self._pending = None
        self._current = None


def _is_galp_pipeline(name: str) -> bool:
    return name.startswith("dct_major_") or name.startswith("image_major_")


def make_adapter(name: str, contract: dict[str, Any], samples: Sequence[dict[str, Any]], device: torch.device) -> Adapter:
    if _is_galp_pipeline(name):
        return GalpAdapter(contract, samples, device, name)
    if name == "rgbnomore":
        return RgbNoMoreAdapter(contract, samples, device, name)
    if name == "dali":
        return DaliAdapter(contract, samples, device, name)
    if name == "pytorch":
        return PyTorchAdapter(contract, samples, device, name)
    raise ValueError(f"unsupported pipeline: {name}")


def _to_device(batch: LoadedBatch, device: torch.device) -> LoadedBatch:
    if batch.on_device:
        return batch
    return LoadedBatch(
        inputs=tuple(tensor.to(device, non_blocking=True) for tensor in batch.inputs),
        labels=batch.labels.to(device, non_blocking=True),
        ordinals=batch.ordinals,
        label_values=batch.label_values,
        on_device=True,
        native_stats=batch.native_stats,
        keepalive=batch.keepalive,
    )


def _validate_identity(batch: LoadedBatch, expected: Sequence[dict[str, Any]]) -> None:
    expected_ordinals = [int(item["ordinal"]) for item in expected]
    if batch.ordinals != expected_ordinals:
        raise RuntimeError(f"sample order mismatch: {batch.ordinals} != {expected_ordinals}")
    expected_labels = [int(item["label"]) for item in expected]
    if batch.label_values != expected_labels:
        raise RuntimeError(f"label mismatch: {batch.label_values} != {expected_labels}")


def _forward(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], expected_width: int) -> torch.Tensor:
    with torch.inference_mode():
        output = model(*inputs)
    if output.ndim != 2 or output.shape[1] != expected_width:
        raise RuntimeError(f"expected [B,{expected_width}] model output, got {tuple(output.shape)}")
    return output


def _prime_model_for_cold_start(
    model: torch.nn.Module,
    domain: str,
    batch_size: int,
    device: torch.device,
    expected_width: int,
) -> None:
    if batch_size <= 0:
        return
    if domain == "rgb":
        inputs = (torch.zeros((batch_size, 3, 224, 224), dtype=torch.float32, device=device),)
    elif domain == "dct":
        inputs = (
            torch.zeros((batch_size, 1, 28, 28, 8, 8), dtype=torch.float32, device=device),
            torch.zeros((batch_size, 2, 14, 14, 8, 8), dtype=torch.float32, device=device),
        )
    else:
        raise ValueError(f"unsupported model domain: {domain}")
    _forward(model, inputs, expected_width)
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


def _record_keepalive(batch: LoadedBatch) -> None:
    seen: set[int] = set()
    for item in batch.keepalive:
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        record = getattr(item, "record_stream", None)
        if callable(record):
            record()


_NATIVE_ALLOCATOR_SNAPSHOT_PREFIXES = (
    "galp_native_device_",
    "galp_native_pinned_",
)

_NATIVE_ALLOCATOR_LAST_VALUE_FIELDS = frozenset(
    {
        "galp_native_device_in_use_bytes",
        "galp_native_device_cached_bytes",
        "galp_native_pinned_in_use_bytes",
        "galp_native_pinned_cached_bytes",
    }
)


def _merge_allocator_snapshot(totals: dict[str, Any], key: str, value: int | float) -> None:
    """Merge a process-global allocator snapshot without inventing segment work."""

    if key in _NATIVE_ALLOCATOR_LAST_VALUE_FIELDS:
        # These are instantaneous gauges.  Segment stats are consumed in
        # logical order, so the last observation preserves their meaning.
        totals[key] = value
        return
    # Allocation requests/counts/bytes are monotonic snapshots; peak fields
    # are high-water gauges.  Max is robust to asynchronous observation order
    # and is also the safe default for future allocator snapshot fields.
    totals[key] = max(totals.get(key, value), value)


def _accumulate_native(totals: dict[str, Any], stats: dict[str, Any]) -> None:
    totals["segment_count"] = int(totals.get("segment_count", 0)) + 1
    for key, value in stats.items():
        if isinstance(value, bool):
            totals[key] = int(totals.get(key, 0)) + int(value)
        elif isinstance(value, int):
            if key.startswith(_NATIVE_ALLOCATOR_SNAPSHOT_PREFIXES):
                _merge_allocator_snapshot(totals, key, value)
            elif any(token in key for token in ("peak", "max_", "capacity", "registers_per_thread", "threads_per_cta")):
                totals[key] = max(int(totals.get(key, 0)), value)
            elif key in {
                "decoded_rowgroup_cache_capacity_bytes",
                "decoded_rowgroup_cache_current_bytes",
                "decoded_rowgroup_cache_current_rowgroups",
            }:
                totals[key] = value
            elif key in {"segment_first_image_id"}:
                totals[key] = min(int(totals.get(key, value)), value)
            elif key in {"segment_last_image_id"}:
                totals[key] = max(int(totals.get(key, value)), value)
            else:
                totals[key] = int(totals.get(key, 0)) + value
        elif isinstance(value, float) and math.isfinite(value):
            if key.startswith(_NATIVE_ALLOCATOR_SNAPSHOT_PREFIXES):
                _merge_allocator_snapshot(totals, key, value)
            elif key != "coordinate_group_index_density":
                totals[key] = float(totals.get(key, 0.0)) + value
        elif isinstance(value, str) and value:
            previous = str(totals.get(key, ""))
            if not previous:
                totals[key] = value
            elif previous != value:
                totals[key] = "mixed"
    coordinate_entries = int(totals.get("coordinate_group_index_entries", 0))
    if coordinate_entries > 0:
        totals["coordinate_group_index_density"] = (
            int(totals.get("coordinate_group_index_populated", 0)) / coordinate_entries
        )


def _capture_semantic(
    store: dict[str, list[np.ndarray]],
    batch: LoadedBatch,
    output: torch.Tensor,
    expected: Sequence[dict[str, Any]],
    remaining: int,
) -> int:
    count = min(remaining, len(expected))
    if count <= 0:
        return 0
    store.setdefault("ordinals", []).append(np.asarray(batch.ordinals[:count], dtype=np.int64))
    store.setdefault("labels", []).append(np.asarray([item["label"] for item in expected[:count]], dtype=np.int64))
    store.setdefault("output", []).append(output[:count].detach().float().cpu().numpy())
    for index, tensor in enumerate(batch.inputs):
        store.setdefault(f"input_{index}", []).append(tensor[:count].detach().float().cpu().numpy())
    return count


def _write_semantic(path: Path, store: dict[str, list[np.ndarray]], metadata: dict[str, Any]) -> None:
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in store.items() if parts}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _verify_runtime_inputs(name: str, contract: dict[str, Any]) -> None:
    model_key = "rgb" if name in {"dali", "pytorch"} else "dct"
    checkpoint = Path(contract["models"][model_key]["checkpoint"])
    if sha256_file(checkpoint) != contract["models"][model_key]["checkpoint_sha256"]:
        raise RuntimeError(f"{model_key} checkpoint changed after contract creation")
    if _is_galp_pipeline(name):
        config = contract["pipelines"][name]
        observed = parse_manifest(Path(config["manifest"]))
        expected_version = 1 if name.startswith("dct_major") else int(config["manifest_version"])
        if int(observed["version"]) != expected_version:
            raise RuntimeError(f"{name} manifest version changed: {observed['version']} != {expected_version}")
        snapshot_key = config.get(
            "storage_snapshot_key",
            "dct_major_storage" if name.startswith("dct_major") else "image_major_storage",
        )
        snapshot = contract["dataset"][snapshot_key]
        if not isinstance(snapshot, dict):
            raise RuntimeError(f"{name} storage snapshot is missing: {snapshot_key}")
        for payload in snapshot["payloads"]:
            path = Path(payload["path"])
            if not path.is_file() or file_identity(path) != payload["file_identity"]:
                raise RuntimeError(f"{name} payload identity changed after contract creation: {path}")
            if payload.get("sha256") and sha256_file(path) != payload["sha256"]:
                raise RuntimeError(f"{name} payload SHA-256 changed after contract creation: {path}")
        if name.startswith("dct_major") and config.get("block_major_access_dir") is not None:
            access = contract["dataset"].get("block_major_access")
            if not isinstance(access, dict):
                raise RuntimeError(f"{name} has a descriptor directory without a descriptor contract")
            fingerprints = [access.get("companion_index"), *access.get("shards", [])]
            for fingerprint in fingerprints:
                if not isinstance(fingerprint, dict):
                    raise RuntimeError(f"{name} block-major descriptor fingerprint is missing")
                path = Path(fingerprint["path"])
                if not path.is_file() or file_identity(path) != fingerprint["file_identity"]:
                    raise RuntimeError(f"{name} block-major descriptor identity changed: {path}")
                if sha256_file(path) != fingerprint["sha256"]:
                    raise RuntimeError(f"{name} block-major descriptor SHA-256 changed: {path}")


def run_pipeline(name: str, contract_path: Path, output_path: Path) -> dict[str, Any]:
    pipeline_started_ns = time.perf_counter_ns()
    contract = load_contract(contract_path)
    if name not in contract["pipelines"]["enabled"]:
        raise ValueError(f"pipeline is not enabled: {name}")
    _verify_runtime_inputs(name, contract)
    samples = load_sample_manifest(
        Path(contract["dataset"]["sample_manifest"]),
        contract["dataset"]["sample_manifest_sha256"],
    )
    if sha256_file(Path(contract["dataset"]["canonical_index_csv"])) != contract["dataset"]["canonical_index_sha256"]:
        raise RuntimeError("canonical no-shuffle index changed after contract creation")
    for sample in samples:
        path = Path(sample["path"])
        if not path.is_file() or file_identity(path) != sample["file_identity"]:
            raise RuntimeError(f"sample identity changed after contract creation: {path}")
        if sample.get("sha256") and sha256_file(path) != sample["sha256"]:
            raise RuntimeError(f"sample SHA-256 changed after contract creation: {path}")
    warmup_batches, measured_batches = selected_batches(contract, samples)
    adapter_samples = [sample for batch in (*warmup_batches, *measured_batches) for sample in batch]
    device = torch.device(contract["execution"]["device"])
    cuda_probe_started_ns = time.perf_counter_ns()
    cuda_available_ready_ns = cuda_probe_started_ns
    cuda_device_ready_ns = cuda_probe_started_ns
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        cuda_available_ready_ns = time.perf_counter_ns()
        torch.cuda.set_device(device)
        cuda_device_ready_ns = time.perf_counter_ns()
    adapter_started_ns = time.perf_counter_ns()
    adapter = make_adapter(name, contract, adapter_samples, device)
    adapter_ready_ns = time.perf_counter_ns()
    cold_prime_started_ns = time.perf_counter_ns()
    adapter.prime_cold_start()
    cold_prime_submitted_ns = time.perf_counter_ns()
    model_key = "rgb" if adapter.domain == "rgb" else "dct"
    model_config = contract["models"][model_key]
    model_started_ns = time.perf_counter_ns()
    model = build_workload_model(
        domain=adapter.domain,
        workload=contract["workload"]["kind"],
        rgbnomore_root=Path(contract["models"]["rgbnomore_root"]),
        checkpoint=Path(model_config["checkpoint"]),
        device=device,
        feature_stage=contract["workload"].get("feature_stage", "penultimate"),
    )
    model_ready_ns = time.perf_counter_ns()
    expected_width = expected_output_width(contract["workload"]["kind"])
    previous_stream: torch.cuda.Stream | None = None
    model_stream: torch.cuda.Stream | None = None
    model_stream_priority: int | None = None
    if device.type == "cuda":
        previous_stream = torch.cuda.current_stream(device)
        previous_stream.synchronize()
        _least_priority, greatest_priority = torch.cuda.Stream.priority_range()
        model_stream_priority = int(greatest_priority)
        model_stream = torch.cuda.Stream(device=device, priority=model_stream_priority)
        torch.cuda.set_stream(model_stream)
    model_prime_started_ns = time.perf_counter_ns()
    if bool(contract["execution"].get("cold_start_model_prime", True)):
        _prime_model_for_cold_start(
            model,
            adapter.domain,
            int(contract["execution"]["batch_size"]),
            device,
            expected_width,
        )
    model_prime_ready_ns = time.perf_counter_ns()
    semantic_limit = int(contract["semantic_validation"]["sample_count"])
    semantic_store: dict[str, list[np.ndarray]] = {}
    semantic_captured = 0
    repeat_records: list[dict[str, Any]] = []
    measured_trace = sample_trace(measured_batches)
    semantic_artifact = output_path.with_name(f"semantic_{name}.npz")

    for repeat in range(int(contract["execution"]["repeats"])):
        repeat_started_ns = time.perf_counter_ns()
        adapter.begin_repeat()
        for expected in warmup_batches:
            batch = _to_device(adapter.load(expected), device)
            _validate_identity(batch, expected)
            _forward(model, batch.inputs, expected_width)
            if device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()
            _record_keepalive(batch)

        adapter.begin_measurement()
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        latency_ms: list[float] = []
        load_ms: list[float] = []
        h2d_ms: list[float] = []
        model_ms: list[float] = []
        sink_ms: list[float] = []
        native_totals: dict[str, Any] = {}
        native_segments: list[dict[str, Any]] = []
        correct1 = 0
        correct5 = 0
        feature_sum = torch.zeros(expected_width, dtype=torch.float64, device=device)
        feature_square_sum = torch.zeros(expected_width, dtype=torch.float64, device=device)
        predictions_top1: list[np.ndarray] = []
        predictions_top5: list[np.ndarray] = []
        process_started = time.process_time()
        host_before = _process_memory_snapshot()
        feature_sink: np.memmap | None = None
        feature_sink_offset = 0
        repeat_to_first_batch_ms: float | None = None
        process_scope_to_first_batch_ms: float | None = None
        if contract["workload"]["kind"] == "feature-extraction" and contract["workload"].get("materialize_features") and repeat == 0:
            feature_count = sum(len(batch) for batch in measured_batches)
            feature_sink = np.lib.format.open_memmap(
                output_path.with_name(f"features_{name}.npy"),
                mode="w+",
                dtype=np.float32,
                shape=(feature_count, expected_width),
            )

        for expected in measured_batches:
            batch_started = time.perf_counter_ns()
            load_started = batch_started
            batch = adapter.load(expected)
            load_finished = time.perf_counter_ns()
            _validate_identity(batch, expected)
            h2d_started = time.perf_counter_ns()
            batch = _to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()
            h2d_finished = time.perf_counter_ns()
            model_started = time.perf_counter_ns()
            output = _forward(model, batch.inputs, expected_width)
            if contract["workload"]["kind"] == "evaluation":
                top5 = output.topk(5, dim=1).indices
                matches = top5.eq(batch.labels.reshape(-1, 1))
                correct1 += int(matches[:, :1].sum().item())
                correct5 += int(matches.sum().item())
                predictions_top1.append(top5[:, 0].detach().cpu().numpy())
                predictions_top5.append(top5.detach().cpu().numpy())
            else:
                feature_sum.add_(output.detach().double().sum(dim=0))
                feature_square_sum.add_(output.detach().double().square().sum(dim=0))
            if device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()
            model_finished = time.perf_counter_ns()
            batch_finished = model_finished
            if repeat_to_first_batch_ms is None:
                repeat_to_first_batch_ms = (batch_finished - repeat_started_ns) / 1.0e6
                if repeat == 0:
                    process_scope_to_first_batch_ms = (batch_finished - pipeline_started_ns) / 1.0e6
                    print(
                        "GALP_FIRST_OUTPUT_READY "
                        + json.dumps(
                            {
                                "pipeline": name,
                                "process_scope_time_to_first_batch_ms": process_scope_to_first_batch_ms,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            _record_keepalive(batch)
            if repeat == 0 and semantic_captured < semantic_limit:
                semantic_captured += _capture_semantic(
                    semantic_store,
                    batch,
                    output,
                    expected,
                    semantic_limit - semantic_captured,
                )
            if feature_sink is not None:
                sink_started = time.perf_counter_ns()
                host_features = output.detach().float().cpu().numpy()
                feature_sink[feature_sink_offset : feature_sink_offset + len(expected)] = host_features
                feature_sink_offset += len(expected)
                sink_ms.append((time.perf_counter_ns() - sink_started) / 1.0e6)
            for stats in batch.native_stats:
                _accumulate_native(native_totals, stats)
                native_segments.append(
                    {
                        key: stats.get(key)
                        for key in (
                            "segment_image_count",
                            "segment_first_image_id",
                            "segment_last_image_id",
                            "planned_vector_count",
                            "actual_vector_count",
                            "full_vector_count",
                            "compressed_payload_bytes_read",
                            "full_compressed_payload_bytes",
                            "pread_count",
                            "rowgroup_count",
                            "planning_ms",
                            "prefetch_producer_active_ms",
                            "prefetch_ordered_submission_ms",
                            "sync_rowgroup_read_ms",
                            "workset_count",
                            "workset_build_ms",
                            "workset_upload_ms",
                            "decode_ms",
                            "fixed_transform_ms",
                            "planless_transform_active_output_planning_ms",
                            "planless_transform_group_workset_build_ms",
                            "planless_transform_active_output_count_ms",
                            "planless_transform_active_output_prefix_ms",
                            "planless_transform_active_output_fill_ms",
                            "planless_transform_gpu_kernel_ms",
                            "planless_transform_output_block_count",
                            "planless_transform_source_contribution_count",
                            "planless_transform_source_contribution_visit_count",
                            "planless_transform_output_workset_ownership_count",
                            "planless_transform_active_output_schedule_build_count",
                            "coordinate_group_lookup_count",
                            "coordinate_group_index_entries",
                            "coordinate_group_index_populated",
                            "coordinate_group_index_holes",
                            "coordinate_group_index_bytes",
                            "coordinate_group_index_density",
                        )
                        if key in stats
                    }
                )
            latency_ms.append((batch_finished - batch_started) / 1.0e6)
            load_ms.append((load_finished - load_started) / 1.0e6)
            h2d_ms.append((h2d_finished - h2d_started) / 1.0e6)
            model_ms.append((model_finished - model_started) / 1.0e6)

        if feature_sink is not None:
            feature_sink.flush()
        host_after = _process_memory_snapshot()
        seconds = sum(latency_ms) / 1000.0
        images = sum(len(batch) for batch in measured_batches)
        steady_images = sum(len(batch) for batch in measured_batches[1:])
        steady_seconds = sum(latency_ms[1:]) / 1000.0
        record: dict[str, Any] = {
            "repeat": repeat,
            "images": images,
            "batches": len(measured_batches),
            "seconds": seconds,
            "throughput_images_per_s": images / seconds,
            "time_to_first_batch_ms": latency_ms[0],
            "repeat_scope_time_to_first_batch_ms": repeat_to_first_batch_ms,
            "repeat_scope_seconds": (time.perf_counter_ns() - repeat_started_ns) / 1.0e9,
            "process_scope_time_to_first_batch_ms": process_scope_to_first_batch_ms,
            "process_scope_throughput_images_per_s": (
                images / ((time.perf_counter_ns() - pipeline_started_ns) / 1.0e9)
                if repeat == 0
                else None
            ),
            "steady_images": steady_images,
            "steady_seconds": steady_seconds,
            "steady_throughput_images_per_s": (
                steady_images / steady_seconds if steady_images and steady_seconds else None
            ),
            "latency_ms": distribution(latency_ms),
            "loader_submit_ms": distribution(load_ms),
            "top_level_h2d_ms": distribution(h2d_ms),
            "model_ms": distribution(model_ms),
            "feature_sink_ms": distribution(sink_ms) if sink_ms else {"count": 0},
            "throughput_scope": "load+top-level-H2D+model+metrics; optional feature sink reported separately",
            "cpu_process_seconds": time.process_time() - process_started,
            "host_rss_before_bytes": host_before["rss_bytes"],
            "host_rss_after_bytes": host_after["rss_bytes"],
            "host_peak_rss_bytes": host_after["peak_rss_bytes"],
            "peak_torch_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "peak_torch_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
            "native_totals": native_totals,
            "native_segments": native_segments,
            "sample_trace": measured_trace,
        }
        if contract["workload"]["kind"] == "evaluation":
            record.update(
                {
                    "correct_top1": correct1,
                    "correct_top5": correct5,
                    "accuracy_top1": correct1 / images,
                    "accuracy_top5": correct5 / images,
                    "top1_predictions_sha256": hashlib.sha256(np.concatenate(predictions_top1).tobytes()).hexdigest(),
                    "top5_predictions_sha256": hashlib.sha256(np.concatenate(predictions_top5).tobytes()).hexdigest(),
                }
            )
            if repeat == 0:
                semantic_store["top1_predictions"] = [np.concatenate(predictions_top1)]
                semantic_store["top5_predictions"] = [np.concatenate(predictions_top5)]
        else:
            record.update(
                {
                    "feature_sum": feature_sum.detach().cpu().tolist(),
                    "feature_square_sum": feature_square_sum.detach().cpu().tolist(),
                }
            )
        repeat_records.append(record)
        adapter.end_repeat()

    _write_semantic(
        semantic_artifact,
        semantic_store,
        {
            "pipeline": name,
            "workload": contract["workload"],
            "sample_count": semantic_captured,
            "output_width": expected_width,
        },
    )
    result = {
        "schema_version": PIPELINE_RESULT_SCHEMA,
        "pipeline": name,
        "domain": adapter.domain,
        "worker_semantics": adapter.worker_semantics,
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256_json(contract),
        "sample_manifest_sha256": contract["dataset"]["sample_manifest_sha256"],
        "execution": contract["execution"],
        "workload": contract["workload"],
        "model": model_config,
        "pipeline_config": contract["pipelines"][name],
        "model_stream_priority": model_stream_priority,
        "cold_start_scope": {
            "definition": (
                "run_pipeline entry through first measured repeat; includes input verification, reader/adapter "
                "construction, model/checkpoint construction, explicitly submitted loader preparation, data path, "
                "model, and metrics; includes adapter-local binding/profile imports but excludes interpreter and "
                "top-level module import before run_pipeline"
            ),
            "adapter_construction_ms": (adapter_ready_ns - adapter_started_ns) / 1.0e6,
            "adapter_subphases": (
                adapter.startup_diagnostics() if hasattr(adapter, "startup_diagnostics") else {}
            ),
            "cuda_availability_probe_ms": (cuda_available_ready_ns - cuda_probe_started_ns) / 1.0e6,
            "cuda_set_device_ms": (cuda_device_ready_ns - cuda_available_ready_ns) / 1.0e6,
            "loader_prime_submit_ms": (cold_prime_submitted_ns - cold_prime_started_ns) / 1.0e6,
            "model_construction_ms": (model_ready_ns - model_started_ns) / 1.0e6,
            "model_prime_ms": (model_prime_ready_ns - model_prime_started_ns) / 1.0e6,
            "model_prime_enabled": bool(contract["execution"].get("cold_start_model_prime", True)),
            "loader_preparation_precedes_model_construction": True,
        },
        "semantic_artifact": str(semantic_artifact.resolve()),
        "repeats": repeat_records,
    }
    write_json(output_path, result)
    adapter.close()
    if previous_stream is not None:
        previous_stream.synchronize()
        torch.cuda.set_stream(previous_stream)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=PIPELINES, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args.pipeline, args.contract, args.output)
    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "pipeline": args.pipeline,
                "repeats": len(result["repeats"]),
                "throughput_images_per_s": [item["throughput_images_per_s"] for item in result["repeats"]],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
