"""Unstable native diagnostics for Direct-DCT benchmark and debugging tools.

These helpers intentionally sit outside the stable batch API. Counter names
may change with planner, allocator, I/O, or kernel implementations.
"""

from __future__ import annotations

from typing import Any

from .direct_dct import DirectDctBatch, DirectDctFuture


def _native_batch(batch: DirectDctBatch) -> Any:
    if not isinstance(batch, DirectDctBatch):
        raise TypeError("expected galp.torch.DirectDctBatch")
    return batch._native


def execution_stats(batch: DirectDctBatch) -> dict[str, Any]:
    """Return complete implementation counters, waiting for completion if needed."""

    return dict(_native_batch(batch).execution_stats)


def execution_stats_snapshot(batch: DirectDctBatch) -> dict[str, Any]:
    """Return the currently available implementation counter snapshot."""

    return dict(_native_batch(batch).execution_stats_snapshot)


def cache_stats(batch: DirectDctBatch) -> dict[str, Any]:
    """Return decoded-rowgroup cache implementation counters."""

    return dict(_native_batch(batch).cache_stats)


def prefetch_stats(future: DirectDctFuture) -> dict[str, float]:
    """Return implementation timing for one native prefetch operation."""

    if not isinstance(future, DirectDctFuture):
        raise TypeError("expected galp.torch.DirectDctFuture")
    native = future._native
    return {
        "producer_active_ms": float(native.producer_active_ms),
        "planning_ms": float(native.planning_ms),
        "io_staging_ms": float(native.io_staging_ms),
        "ordered_submission_ms": float(native.ordered_submission_ms),
    }


__all__ = [
    "cache_stats",
    "execution_stats",
    "execution_stats_snapshot",
    "prefetch_stats",
]
