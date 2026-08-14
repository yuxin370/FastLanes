#!/usr/bin/env python3
"""Stable training-facing wrapper around the public DirectDctReader API."""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_compatible(child) for child in value]
    return {"type": type(value).__name__, "repr": repr(value)}


def _optional_native_execution_stats_attribute(
    source: Any, attribute: str
) -> dict[str, Any]:
    """Return one JSON-safe optional native stats attribute."""

    try:
        value = getattr(source, attribute)
        value = value() if callable(value) else value
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            value = dict(value)
        return _json_compatible(value)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return {}


def optional_native_execution_stats(source: Any) -> dict[str, Any]:
    """Return complete optional stats, allowing the native completion wait."""

    return _optional_native_execution_stats_attribute(source, "execution_stats")


def optional_native_execution_stats_snapshot(source: Any) -> dict[str, Any]:
    """Return a non-blocking native stats snapshot when the runtime exposes it."""

    return _optional_native_execution_stats_attribute(
        source, "execution_stats_snapshot"
    )


@dataclass
class DirectDctTrainingBatch:
    """Layout-independent semantic result returned to the training adapter."""

    native_batch: Any

    @property
    def tensors(self) -> tuple[Any, Any]:
        return self.native_batch.y, self.native_batch.cbcr

    @property
    def global_image_ids(self) -> list[int]:
        return [int(value) for value in self.native_batch.global_image_ids]

    @property
    def transform_descriptors(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self.native_batch.transform_descriptors]

    def native_execution_stats(self) -> dict[str, Any]:
        return optional_native_execution_stats(self.native_batch)

    def native_execution_stats_snapshot(self) -> dict[str, Any]:
        return optional_native_execution_stats_snapshot(self.native_batch)


class DirectDctTrainingHandle:
    """Preserve the native asynchronous-handle surface while wrapping its result."""

    def __init__(self, native_handle: Any) -> None:
        self._native_handle = native_handle

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native_handle, name)

    def read(self) -> DirectDctTrainingBatch:
        return DirectDctTrainingBatch(self._native_handle.read())

    def cancel(self) -> bool:
        cancel = getattr(self._native_handle, "cancel", None)
        return bool(cancel()) if callable(cancel) else False


class DirectDctTrainingReader:
    """Training-only facade that is intentionally blind to manifest layout/version."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        module_path: Path | None = None,
        native_module: Any | None = None,
    ) -> None:
        if module_path is not None:
            resolved = str(module_path.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
        module = native_module or importlib.import_module("_galp_direct_dct")
        self._reader = module.DirectDctReader(str(manifest_path.resolve()))

    @property
    def image_count(self) -> int:
        return int(self._reader.image_count)

    def prefetch_batch(
        self,
        image_ids: Sequence[int],
        *,
        transforms: Sequence[dict[str, Any]],
        **reader_options: Any,
    ) -> DirectDctTrainingHandle:
        native_handle = self._reader.prefetch_batch(
            [int(value) for value in image_ids],
            transforms=[dict(value) for value in transforms],
            **reader_options,
        )
        return DirectDctTrainingHandle(native_handle)


class NativeExecutionStatsAccumulator:
    """Generic aggregation that tolerates added, removed, or missing native fields."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.observed_batches = 0
        self.missing_batches = 0
        self.latest: dict[str, Any] = {}
        self.numeric: dict[str, dict[str, int | float]] = {}

    def observe(self, stats: Mapping[str, Any] | None) -> None:
        if not stats:
            self.missing_batches += 1
            return
        normalized = _json_compatible(stats)
        if not isinstance(normalized, dict):
            self.missing_batches += 1
            return
        self.observed_batches += 1
        self.latest = normalized
        for name, value in normalized.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            aggregate = self.numeric.setdefault(
                name,
                {"count": 0, "sum": 0.0, "min": float(value), "max": float(value), "last": float(value)},
            )
            aggregate["count"] = int(aggregate["count"]) + 1
            aggregate["sum"] = float(aggregate["sum"]) + float(value)
            aggregate["min"] = min(float(aggregate["min"]), float(value))
            aggregate["max"] = max(float(aggregate["max"]), float(value))
            aggregate["last"] = float(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "optional": True,
            "correctness_dependency": False,
            "observed_batches": self.observed_batches,
            "missing_batches": self.missing_batches,
            "latest": self.latest,
            "numeric_aggregates": {name: dict(value) for name, value in sorted(self.numeric.items())},
        }


_GLOBAL_CUMULATIVE_ALLOCATION_FIELDS = (
    "galp_native_device_allocation_requests",
    "galp_native_device_cuda_allocation_count",
    "galp_native_device_cuda_allocation_bytes",
    "galp_native_pinned_allocation_requests",
    "galp_native_pinned_cuda_allocation_count",
    "galp_native_pinned_cuda_allocation_bytes",
)

_ALLOCATION_STABILITY_FIELDS = (
    "galp_native_device_cuda_allocation_count",
    "galp_native_pinned_cuda_allocation_count",
)

_PER_BATCH_STABILITY_FIELDS = (
    "compact_batch_buffer_growth_count",
    "compact_batch_buffer_pageable_fallback_count",
    "decode_workset_output_arena_growth_count",
    "decode_workset_chunk_arena_growth_count",
    "planless_axis_program_device_growth_count",
    "planless_axis_program_pinned_growth_count",
)

_CAPACITY_CONTRACT_FIELDS = (
    "planless_axis_program_capacity_contract_complete",
    "compact_batch_pool_capacity_contract_complete",
)


def merge_native_counter_snapshot(
    aggregate: dict[str, float], counters: Mapping[str, int | float]
) -> None:
    """Fold per-batch counters without summing process-global snapshots."""

    for name, value in counters.items():
        numeric = float(value)
        if name.startswith("galp_native_"):
            aggregate[name] = max(aggregate.get(name, numeric), numeric)
        else:
            aggregate[name] = aggregate.get(name, 0.0) + numeric


def native_allocation_stability(
    warmup_stats: Sequence[Mapping[str, Any]],
    measured_stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove whether native allocation growth stopped after the warmup boundary."""

    warmup = [stats for stats in warmup_stats if stats]
    measured = [stats for stats in measured_stats if stats]
    result: dict[str, Any] = {
        "criterion": (
            "zero new native CUDA allocations, zero output/chunk arena growth, and zero "
            "compact/planless-axis buffer growth or pageable fallback in measured batches, "
            "with complete dataset-derived planless-axis and compact-pool capacity contracts"
        ),
        "warmup_observed_batches": len(warmup),
        "measured_observed_batches": len(measured),
        "global_counter_deltas": {},
        "measured_per_batch_totals": {},
        "verifiable": False,
        "stable_after_warmup": None,
        "capacity_contract_complete": None,
        "missing_fields": [],
    }
    if not warmup or not measured:
        result["reason"] = "both warmup and measured native snapshots are required"
        return result

    missing = [
        name
        for name in _ALLOCATION_STABILITY_FIELDS
        if any(
            not isinstance(stats.get(name), (int, float))
            or isinstance(stats.get(name), bool)
            for stats in warmup + measured
        )
    ]
    missing.extend(
        name
        for name in _PER_BATCH_STABILITY_FIELDS
        if any(
            not isinstance(stats.get(name), (int, float))
            or isinstance(stats.get(name), bool)
            for stats in measured
        )
    )
    missing.extend(
        name
        for name in _CAPACITY_CONTRACT_FIELDS
        if any(not isinstance(stats.get(name), bool) for stats in warmup + measured)
    )
    result["missing_fields"] = sorted(set(missing))
    if missing:
        result["reason"] = "required native allocation diagnostics are missing"
        return result

    deltas = {
        name: max(float(stats[name]) for stats in warmup + measured)
        - max(float(stats[name]) for stats in warmup)
        for name in _GLOBAL_CUMULATIVE_ALLOCATION_FIELDS
        if all(
            isinstance(stats.get(name), (int, float))
            and not isinstance(stats.get(name), bool)
            for stats in warmup + measured
        )
    }
    per_batch_totals = {
        name: sum(float(stats[name]) for stats in measured)
        for name in _PER_BATCH_STABILITY_FIELDS
    }
    result["global_counter_deltas"] = deltas
    result["measured_per_batch_totals"] = per_batch_totals
    capacity_contract_complete = all(
        bool(stats[name]) for name in _CAPACITY_CONTRACT_FIELDS for stats in warmup + measured
    )
    result["capacity_contract_complete"] = capacity_contract_complete
    result["verifiable"] = True
    result["stable_after_warmup"] = (
        all(deltas[name] == 0.0 for name in _ALLOCATION_STABILITY_FIELDS)
        and all(value == 0.0 for value in per_batch_totals.values())
        and capacity_contract_complete
    )
    return result
