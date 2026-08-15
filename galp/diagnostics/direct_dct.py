"""Unstable native diagnostics for Direct-DCT benchmark and debugging tools.

These helpers intentionally live outside :mod:`galp.torch`. Counter names may
change with planner, allocator, I/O, storage, or kernel implementations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from galp.profiles import DirectDctProfile
from galp.torch.direct_dct import DirectDctBatch, DirectDctPipeline, DirectDctReader


def _native_batch(batch: DirectDctBatch) -> Any:
    if not isinstance(batch, DirectDctBatch):
        raise TypeError("expected galp.torch.DirectDctBatch")
    return batch._native


def _native_reader(reader: DirectDctReader) -> Any:
    if not isinstance(reader, DirectDctReader):
        raise TypeError("expected galp.torch.DirectDctReader")
    return reader._native


def execution_stats(batch: DirectDctBatch) -> dict[str, Any]:
    """Return complete implementation counters, waiting for completion if needed."""

    return dict(_native_batch(batch).execution_stats)


def execution_stats_snapshot(batch: DirectDctBatch) -> dict[str, Any]:
    """Return the currently available implementation counter snapshot."""

    return dict(_native_batch(batch).execution_stats_snapshot)


def cache_stats(batch: DirectDctBatch) -> dict[str, Any]:
    """Return decoded-rowgroup cache implementation counters."""

    return dict(_native_batch(batch).cache_stats)


def pipeline_stats(pipeline: DirectDctPipeline) -> dict[str, float | int | bool]:
    """Return unstable queue state for benchmark diagnostics."""

    if not isinstance(pipeline, DirectDctPipeline):
        raise TypeError("expected galp.torch.DirectDctPipeline")
    native = pipeline._native
    metrics = dict(native.prefetch_metrics)
    return {
        "ready": bool(native.ready),
        "started": bool(native.started),
        "prefetched_batch_count": int(native.prefetched_batch_count),
        "producer_active_ms": float(metrics.get("producer_ms", 0.0)),
        "planning_ms": float(metrics.get("planning_ms", 0.0)),
        "io_staging_ms": float(metrics.get("io_ms", 0.0)),
        "ordered_submission_ms": float(metrics.get("ordered_submission_ms", 0.0)),
    }


def initialization_stats(reader: DirectDctReader) -> dict[str, Any]:
    """Return private reader startup diagnostics."""

    return dict(_native_reader(reader).initialization_stats)


def binding_import_ms(reader: DirectDctReader) -> float:
    """Return native extension import latency for benchmark startup audits."""

    if not isinstance(reader, DirectDctReader):
        raise TypeError("expected galp.torch.DirectDctReader")
    return float(reader._binding_import_ms)


def plan_preview(
    reader: DirectDctReader,
    image_ids: Sequence[int],
    profile: DirectDctProfile | str,
    *,
    transforms: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return an unstable native planning preview for diagnostics only."""

    profile_id = reader.profile_info(profile)["id"]
    native_transforms = (
        None if transforms is None else [dict(value) for value in transforms]
    )
    return dict(
        _native_reader(reader).plan(
            [int(value) for value in image_ids],
            profile_id,
            transforms=native_transforms,
        )
    )


def image_metadata(reader: DirectDctReader, global_image_index: int) -> dict[str, Any]:
    """Return private JPEG component metadata for dataset/diagnostic tooling."""

    return dict(_native_reader(reader).image_metadata(int(global_image_index)))


def rowgroup_storage_bytes(
    reader: DirectDctReader,
    shard_id: int,
    rowgroup_indices: Sequence[int],
) -> int:
    """Return physical rowgroup bytes for storage diagnostics."""

    return int(
        _native_reader(reader).rowgroup_storage_bytes(
            int(shard_id), [int(value) for value in rowgroup_indices]
        )
    )


__all__ = [
    "binding_import_ms",
    "cache_stats",
    "execution_stats",
    "execution_stats_snapshot",
    "image_metadata",
    "initialization_stats",
    "pipeline_stats",
    "plan_preview",
    "rowgroup_storage_bytes",
]
