#!/usr/bin/env python3
"""Summarize a bounded equal-image Nsight Systems SQLite capture.

The report deliberately separates wall-time ranges, GPU interval unions, and
summed per-thread/operator work.  Those quantities have different overlap
semantics and must not be added together as a sequential time breakdown.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STAGE_NAMES = (
    "training.loader.next_batch",
    "training.input_handoff",
    "training.model.forward",
    "training.loss",
    "training.model.backward",
    "training.optimizer",
)

DALI_INPUT_KERNEL_FRAGMENTS = (
    "BatchedSeparableResampleKernel",
    "Hwc2HwcChwNormalize",
    "dctQuantInvJpegKernelMultiChannel",
    "implicit_convolve_sgemm",
    "nchwToNhwcKernel",
    "splitKreduce_kernel",
    "ycbcr_to_format_kernel_roi",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--window-name",
        default="profile-rgb-d2-microbatches_512_1535",
    )
    parser.add_argument("--captured-images", type=int, default=65_536)
    parser.add_argument("--top", type=int, default=25)
    return parser.parse_args(argv)


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _duration_summary(values_ns: Sequence[int]) -> dict[str, Any]:
    if not values_ns:
        return {
            "count": 0,
            "sum_seconds": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(values_ns),
        "sum_seconds": sum(values_ns) / 1.0e9,
        "mean_ms": (sum(values_ns) / len(values_ns)) / 1.0e6,
        "p50_ms": _percentile(values_ns, 0.50) / 1.0e6,
        "p95_ms": _percentile(values_ns, 0.95) / 1.0e6,
        "max_ms": max(values_ns) / 1.0e6,
    }


def _clip(
    start: int, end: int, window_start: int, window_end: int
) -> tuple[int, int] | None:
    clipped_start = max(int(start), window_start)
    clipped_end = min(int(end), window_end)
    if clipped_end <= clipped_start:
        return None
    return clipped_start, clipped_end


def _union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _intervals(
    connection: sqlite3.Connection,
    table: str,
    window_start: int,
    window_end: int,
    *,
    predicate: str = "1",
) -> list[tuple[int, int]]:
    rows = connection.execute(
        f"SELECT start, end FROM {table} "
        f"WHERE start < ? AND end > ? AND ({predicate})",
        (window_end, window_start),
    )
    result: list[tuple[int, int]] = []
    for start, end in rows:
        clipped = _clip(start, end, window_start, window_end)
        if clipped is not None:
            result.append(clipped)
    return result


def _named_duration_rows(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[Any, ...],
    window_start: int,
    window_end: int,
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for name, start, end in connection.execute(query, parameters):
        clipped = _clip(start, end, window_start, window_end)
        if clipped is not None:
            grouped[str(name)].append(clipped[1] - clipped[0])
    return grouped


def _top_summed(grouped: dict[str, list[int]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(grouped.items(), key=lambda item: sum(item[1]), reverse=True)
    return [
        {"name": name, **_duration_summary(durations)}
        for name, durations in ordered[:limit]
    ]


def _capture_cardinality(
    stage_grouped: Mapping[str, Sequence[int]],
) -> tuple[int, int]:
    microbatches = len(stage_grouped.get("training.loader.next_batch", ()))
    optimizer_updates = len(stage_grouped.get("training.optimizer", ()))
    if microbatches <= 0 or optimizer_updates <= 0:
        raise ValueError("capture lacks loader or optimizer NVTX ranges")
    return microbatches, optimizer_updates


def summarize(
    sqlite_path: Path,
    *,
    window_name: str,
    captured_images: int,
    top: int,
) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        windows = connection.execute(
            "SELECT start, end, globalTid FROM NVTX_EVENTS "
            "WHERE text = ? AND end IS NOT NULL",
            (window_name,),
        ).fetchall()
        if len(windows) != 1:
            raise ValueError(f"expected one {window_name!r} range, found {len(windows)}")
        window_start, window_end, profile_tid = map(int, windows[0])
        window_ns = window_end - window_start

        stage_grouped = _named_duration_rows(
            connection,
            "SELECT text, start, end FROM NVTX_EVENTS "
            f"WHERE globalTid = ? AND text IN ({','.join('?' for _ in STAGE_NAMES)}) "
            "AND start < ? AND end > ?",
            (profile_tid, *STAGE_NAMES, window_end, window_start),
            window_start,
            window_end,
        )
        captured_microbatches, captured_optimizer_updates = _capture_cardinality(
            stage_grouped
        )
        stage_intervals = [
            interval
            for name in STAGE_NAMES
            for interval in connection.execute(
                "SELECT start, end FROM NVTX_EVENTS "
                "WHERE globalTid = ? AND text = ? AND start < ? AND end > ?",
                (profile_tid, name, window_end, window_start),
            )
        ]
        clipped_stage_intervals = [
            clipped
            for start, end in stage_intervals
            if (clipped := _clip(start, end, window_start, window_end)) is not None
        ]
        named_stage_union_ns = _union_ns(clipped_stage_intervals)

        kernel_intervals = _intervals(
            connection,
            "CUPTI_ACTIVITY_KIND_KERNEL",
            window_start,
            window_end,
        )
        memcpy_intervals = _intervals(
            connection,
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            window_start,
            window_end,
        )
        memset_intervals = _intervals(
            connection,
            "CUPTI_ACTIVITY_KIND_MEMSET",
            window_start,
            window_end,
        )
        gpu_active_ns = _union_ns(
            [*kernel_intervals, *memcpy_intervals, *memset_intervals]
        )

        transfer_rows: dict[str, dict[str, Any]] = {}
        for label, count, bytes_count, duration_ns in connection.execute(
            "SELECT e.label, COUNT(*), SUM(m.bytes), SUM(m.end-m.start) "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
            "JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind=e.id "
            "WHERE m.start < ? AND m.end > ? GROUP BY e.label",
            (window_end, window_start),
        ):
            transfer_rows[str(label)] = {
                "count": int(count),
                "bytes": int(bytes_count),
                "gib": int(bytes_count) / (1024.0**3),
                "summed_gpu_seconds": int(duration_ns) / 1.0e9,
            }

        kernel_grouped = _named_duration_rows(
            connection,
            "SELECT s.value, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.shortName=s.id "
            "WHERE k.start < ? AND k.end > ?",
            (window_end, window_start),
            window_start,
            window_end,
        )
        input_kernel_grouped = {
            name: durations
            for name, durations in kernel_grouped.items()
            if any(fragment in name for fragment in DALI_INPUT_KERNEL_FRAGMENTS)
        }

        runtime_grouped = _named_duration_rows(
            connection,
            "SELECT s.value, r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON r.nameId=s.id "
            "WHERE r.start < ? AND r.end > ?",
            (window_end, window_start),
            window_start,
            window_end,
        )
        sync_runtime_intervals: list[tuple[int, int]] = []
        for name in runtime_grouped:
            if "Synchronize" not in name:
                continue
            for start, end in connection.execute(
                "SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON r.nameId=s.id "
                "WHERE s.value=? AND r.start < ? AND r.end > ?",
                (name, window_end, window_start),
            ):
                clipped = _clip(start, end, window_start, window_end)
                if clipped is not None:
                    sync_runtime_intervals.append(clipped)

        dali_grouped = _named_duration_rows(
            connection,
            "SELECT COALESCE(n.text, s.value), n.start, n.end "
            "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
            "WHERE COALESCE(n.text, s.value) LIKE '[DALI]%' "
            "AND n.start < ? AND n.end > ? AND n.end IS NOT NULL",
            (window_end, window_start),
            window_start,
            window_end,
        )
        nvjpeg_grouped = _named_duration_rows(
            connection,
            "SELECT COALESCE(n.text, s.value), n.start, n.end "
            "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
            "WHERE (COALESCE(n.text, s.value) LIKE 'nvjpeg%' "
            "OR COALESCE(n.text, s.value) LIKE 'nvimgcodec%') "
            "AND n.start < ? AND n.end > ? AND n.end IS NOT NULL",
            (window_end, window_start),
            window_start,
            window_end,
        )

        osrt_grouped = _named_duration_rows(
            connection,
            "SELECT s.value, o.start, o.end FROM OSRT_API o "
            "JOIN StringIds s ON o.nameId=s.id "
            "WHERE o.start < ? AND o.end > ?",
            (window_end, window_start),
            window_start,
            window_end,
        )
        file_access: dict[str, dict[str, Any]] = {}
        for label, count, bytes_count, duration_ns in connection.execute(
            "SELECT e.label, COUNT(*), SUM(f.bytesProcessed), "
            "SUM(f.endedAt-f.startedAt) FROM OSRT_FILE_ACCESS_EVENTS f "
            "JOIN ENUM_OSRT_FILE_ACCESS_EVENT_TYPE e ON f.eventType=e.id "
            "WHERE f.startedAt < ? AND f.endedAt > ? GROUP BY e.label",
            (window_end, window_start),
        ):
            file_access[str(label)] = {
                "count": int(count),
                "bytes_processed": int(bytes_count),
                "summed_seconds": int(duration_ns) / 1.0e9,
            }

        return {
            "schema_version": "galp-equal-image-nsys-summary-v1",
            "source_sqlite": str(sqlite_path.resolve()),
            "window": {
                "name": window_name,
                "start_ns": window_start,
                "end_ns": window_end,
                "seconds": window_ns / 1.0e9,
                "captured_images": captured_images,
                "images_per_second": captured_images / (window_ns / 1.0e9),
                "microbatches": captured_microbatches,
                "optimizer_updates": captured_optimizer_updates,
            },
            "host_wall_stages": {
                "semantics": (
                    "NVTX wall time on the training thread; stages are sequential, "
                    "while DALI worker/GPU work may overlap them"
                ),
                "stages": {
                    name: _duration_summary(stage_grouped.get(name, []))
                    for name in STAGE_NAMES
                },
                "named_stage_union_seconds": named_stage_union_ns / 1.0e9,
                "unnamed_gap_seconds": (window_ns - named_stage_union_ns) / 1.0e9,
            },
            "gpu_timeline": {
                "semantics": "interval unions clipped to the outer NVTX window",
                "active_seconds": gpu_active_ns / 1.0e9,
                "active_percent": 100.0 * gpu_active_ns / window_ns,
                "idle_seconds": (window_ns - gpu_active_ns) / 1.0e9,
                "idle_percent": 100.0 * (window_ns - gpu_active_ns) / window_ns,
                "kernel_union_seconds": _union_ns(kernel_intervals) / 1.0e9,
                "memcpy_union_seconds": _union_ns(memcpy_intervals) / 1.0e9,
                "memset_union_seconds": _union_ns(memset_intervals) / 1.0e9,
            },
            "transfers": transfer_rows,
            "cuda_runtime": {
                "top_summed_thread_work": _top_summed(runtime_grouped, top),
                "synchronization_summed_seconds": sum(
                    sum(values)
                    for name, values in runtime_grouped.items()
                    if "Synchronize" in name
                )
                / 1.0e9,
                "synchronization_union_seconds": _union_ns(sync_runtime_intervals)
                / 1.0e9,
            },
            "kernels": {
                "top_summed_gpu_work": _top_summed(kernel_grouped, top),
                "dali_input_summed_gpu_seconds": sum(
                    sum(values) for values in input_kernel_grouped.values()
                )
                / 1.0e9,
                "dali_input_kernels": _top_summed(input_kernel_grouped, top),
            },
            "dali_nvtx": {
                "semantics": (
                    "summed operator/thread work; parallel decoder ranges overlap and "
                    "must not be added to the capture wall time"
                ),
                "top": _top_summed(dali_grouped, top),
                "nvjpeg_top": _top_summed(nvjpeg_grouped, top),
            },
            "os_runtime": {
                "top_summed_thread_work": _top_summed(osrt_grouped, top),
                "file_access": file_access,
                "physical_read_limitation": (
                    "DALI readers.file used mmap (dont_use_mmap=false). OSRT recorded "
                    "file opens but no read byte events, so physical disk bytes are not "
                    "observable in this trace. Use the full-epoch logical byte counter "
                    "and a separate cold-cache block-device experiment."
                ),
            },
        }
    finally:
        connection.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.captured_images <= 0 or args.top <= 0:
        raise ValueError("--captured-images and --top must be positive")
    report = summarize(
        args.sqlite.resolve(),
        window_name=args.window_name,
        captured_images=args.captured_images,
        top=args.top,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
