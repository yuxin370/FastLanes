#!/usr/bin/env python3
"""Stable training-facing wrapper around the public DirectDctReader API."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galp.profiles.rgbnomore import VALIDATION
from galp.benchmarks.system_rgbnomore.shared.common import GALP_RUNTIME_PROFILE
from galp.torch import DirectDctMetrics, DirectDctReader
from galp.diagnostics.direct_dct import (
    execution_stats_observation as public_execution_stats_observation,
    execution_stats as public_execution_stats,
    execution_stats_snapshot as public_execution_stats_snapshot,
    pipeline_stats,
)


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

    try:
        return _json_compatible(public_execution_stats(source))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        pass
    return _optional_native_execution_stats_attribute(source, "execution_stats")


def optional_native_execution_stats_snapshot(source: Any) -> dict[str, Any]:
    """Return a non-blocking native stats snapshot when the runtime exposes it."""

    try:
        return _json_compatible(public_execution_stats_snapshot(source))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        pass
    return _optional_native_execution_stats_attribute(
        source, "execution_stats_snapshot"
    )


def optional_native_execution_stats_observation(source: Any) -> dict[str, Any]:
    """Return non-blocking stats plus independent host/GPU completion state."""

    try:
        return _json_compatible(public_execution_stats_observation(source))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        pass
    return _optional_native_execution_stats_attribute(
        source, "_execution_stats_observation"
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

    @property
    def metrics(self) -> Any:
        return self.native_batch.metrics

    def native_execution_stats(self) -> dict[str, Any]:
        return optional_native_execution_stats(self.native_batch)

    def native_execution_stats_snapshot(self) -> dict[str, Any]:
        return optional_native_execution_stats_snapshot(self.native_batch)

    def native_execution_stats_observation(self) -> dict[str, Any]:
        return optional_native_execution_stats_observation(self.native_batch)


class DirectDctTrainingReader:
    """Training-only facade that is intentionally blind to manifest layout/version."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        module_path: Path | None = None,
        native_module: Any | None = None,
    ) -> None:
        self._reader = DirectDctReader(
            manifest_path,
            module_path=module_path,
            native_module=native_module,
        )
        profile_info = self._reader.profile_info(VALIDATION)
        runtime_policy_id = str(profile_info.get("runtime_policy_id", ""))
        if runtime_policy_id != GALP_RUNTIME_PROFILE:
            raise RuntimeError(
                "GALP native profile does not match the training contract: "
                f"expected {GALP_RUNTIME_PROFILE!r}, got {runtime_policy_id!r}"
            )
        self._pipeline = self._reader.pipeline(VALIDATION)

    @property
    def image_count(self) -> int:
        return int(self._reader.image_count)

    def start(
        self,
        image_id_batches: Sequence[Sequence[int]],
        *,
        transforms_by_batch: Sequence[Sequence[dict[str, Any]]],
    ) -> None:
        self._pipeline.start(
            [
                [int(value) for value in image_ids]
                for image_ids in image_id_batches
            ],
            transforms_by_batch=[
                [dict(value) for value in transforms]
                for transforms in transforms_by_batch
            ],
        )

    def next_batch(self) -> DirectDctTrainingBatch:
        return DirectDctTrainingBatch(next(self._pipeline))

    def close(self) -> None:
        self._pipeline.close()

    def metrics(self) -> DirectDctMetrics:
        return self._pipeline.metrics

    def metrics_snapshot(self) -> dict[str, Any]:
        """Return the native stable-schema snapshot without Python reduction."""

        return dict(self._pipeline._native.metrics)

    def aggregate_metrics_snapshots(
        self, snapshots: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Combine pipeline scopes using the native descriptor-driven reducer."""

        return dict(
            self._reader._module._aggregate_direct_dct_metrics(
                [dict(snapshot) for snapshot in snapshots]
            )
        )

    def prefetched_batch_count(self) -> int:
        return int(pipeline_stats(self._pipeline)["prefetched_batch_count"])


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
