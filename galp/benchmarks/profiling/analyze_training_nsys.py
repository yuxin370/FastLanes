#!/usr/bin/env python3
"""Build a detailed key-path report and SVGs from one training Nsys export."""

from __future__ import annotations

import argparse
import html
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


MAIN_STAGES = (
    "training.loader.next_batch",
    "training.input_handoff",
    "training.batch.audit",
    "training.model.forward",
    "training.loss",
    "training.model.backward",
    "training.optimizer",
)

STAGE_LABELS = {
    "training.loader.next_batch": "Loader wait/next",
    "training.input_handoff": "Input handoff",
    "training.batch.audit": "Batch audit",
    "training.model.forward": "Forward",
    "training.loss": "Loss",
    "training.model.backward": "Backward",
    "training.optimizer": "Optimizer",
}

COLORS = {
    "training.loader.next_batch": "#e69f00",
    "training.input_handoff": "#56b4e9",
    "training.batch.audit": "#cc79a7",
    "training.model.forward": "#009e73",
    "training.loss": "#f0e442",
    "training.model.backward": "#0072b2",
    "training.optimizer": "#d55e00",
    "backend": "#8c6bb1",
    "gpu_input": "#fdae6b",
    "gpu_model": "#31a354",
    "h2d": "#3182bd",
    "d2h": "#9ecae1",
    "d2d": "#756bb1",
    "sync": "#de2d26",
    "osrt": "#969696",
    "idle": "#eeeeee",
}

DALI_INPUT_KERNELS = (
    "BatchedSeparableResampleKernel",
    "Hwc2HwcChwNormalize",
    "dctQuantInvJpegKernelMultiChannel",
    "implicit_convolve_sgemm",
    "nchwToNhwcKernel",
    "splitKreduce_kernel",
    "ycbcr_to_format_kernel_roi",
)

GALP_INPUT_KERNELS = (
    "transformed_dct",
    "direct_dct",
    "decompress",
    "decode",
    "randaugment",
    "mixup",
    "ordered_output",
    "placement",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--kind", choices=("dali", "pytorch", "galp"), required=True)
    parser.add_argument("--window-name", required=True)
    parser.add_argument("--captured-images", type=int, required=True)
    parser.add_argument("--captured-microbatches", type=int, required=True)
    parser.add_argument(
        "--capture-metrics",
        type=Path,
        help="optional profiling-only loader-stage counters written by the runner",
    )
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args(argv)


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _clip(
    start: int, end: int, window_start: int, window_end: int
) -> tuple[int, int] | None:
    clipped = max(start, window_start), min(end, window_end)
    return clipped if clipped[1] > clipped[0] else None


def _merge(
    intervals: Iterable[tuple[int, int]], *, gap_ns: int = 0
) -> list[tuple[int, int]]:
    ordered = sorted(intervals)
    if not ordered:
        return []
    merged: list[tuple[int, int]] = []
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end + gap_ns:
            end = max(end, next_end)
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))
    return merged


def _union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge(intervals))


def _percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return float(ordered[index])


def _duration(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "sum_seconds": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(values),
        "sum_seconds": sum(values) / 1e9,
        "mean_ms": sum(values) / len(values) / 1e6,
        "p50_ms": _percentile(values, 0.50) / 1e6,
        "p95_ms": _percentile(values, 0.95) / 1e6,
        "max_ms": max(values) / 1e6,
    }


def _named_nvtx(
    connection: sqlite3.Connection,
    window_start: int,
    window_end: int,
) -> list[tuple[str, int, int, int]]:
    rows = connection.execute(
        "SELECT COALESCE(n.text,s.value),n.start,n.end,n.globalTid "
        "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
        "WHERE n.end IS NOT NULL AND n.start < ? AND n.end > ?",
        (window_end, window_start),
    )
    result: list[tuple[str, int, int, int]] = []
    for name, start, end, tid in rows:
        clipped = _clip(int(start), int(end), window_start, window_end)
        if name is not None and tid is not None and clipped is not None:
            result.append((str(name), clipped[0], clipped[1], int(tid)))
    return result


def _gpu_events(
    connection: sqlite3.Connection,
    window_start: int,
    window_end: int,
    *,
    kind: str,
) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int, int]]]:
    kernels: list[tuple[str, int, int]] = []
    if _table_exists(connection, "CUPTI_ACTIVITY_KIND_KERNEL"):
        for name, start, end in connection.execute(
            "SELECT s.value,k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.shortName=s.id "
            "WHERE k.start < ? AND k.end > ?",
            (window_end, window_start),
        ):
            clipped = _clip(int(start), int(end), window_start, window_end)
            if clipped is not None:
                kernels.append((str(name), clipped[0], clipped[1]))
    copies: list[tuple[str, int, int, int]] = []
    if _table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for label, start, end, bytes_count in connection.execute(
            "SELECT e.label,m.start,m.end,m.bytes "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
            "JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind=e.id "
            "WHERE m.start < ? AND m.end > ?",
            (window_end, window_start),
        ):
            clipped = _clip(int(start), int(end), window_start, window_end)
            if clipped is not None:
                copies.append((str(label), clipped[0], clipped[1], int(bytes_count)))
    return kernels, copies


def _is_input_kernel(name: str, kind: str) -> bool:
    fragments = DALI_INPUT_KERNELS if kind == "dali" else GALP_INPUT_KERNELS
    return kind != "pytorch" and any(fragment.lower() in name.lower() for fragment in fragments)


def _runtime_sync_intervals(
    connection: sqlite3.Connection, window_start: int, window_end: int
) -> tuple[list[tuple[int, int]], dict[str, list[int]]]:
    intervals: list[tuple[int, int]] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    if not _table_exists(connection, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return intervals, grouped
    for name, start, end in connection.execute(
        "SELECT s.value,r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId=s.id "
        "WHERE r.start < ? AND r.end > ?",
        (window_end, window_start),
    ):
        clipped = _clip(int(start), int(end), window_start, window_end)
        if clipped is None:
            continue
        grouped[str(name)].append(clipped[1] - clipped[0])
        if "Synchronize" in str(name):
            intervals.append(clipped)
    return intervals, grouped


def _osrt(
    connection: sqlite3.Connection, window_start: int, window_end: int
) -> tuple[list[tuple[int, int]], dict[str, list[int]], dict[str, Any]]:
    intervals: list[tuple[int, int]] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    if _table_exists(connection, "OSRT_API"):
        for name, start, end in connection.execute(
            "SELECT s.value,o.start,o.end FROM OSRT_API o "
            "JOIN StringIds s ON o.nameId=s.id "
            "WHERE o.start < ? AND o.end > ?",
            (window_end, window_start),
        ):
            clipped = _clip(int(start), int(end), window_start, window_end)
            if clipped is not None:
                grouped[str(name)].append(clipped[1] - clipped[0])
                intervals.append(clipped)
    file_access: dict[str, Any] = {}
    if _table_exists(connection, "OSRT_FILE_ACCESS_EVENTS"):
        for label, count, bytes_count, duration in connection.execute(
            "SELECT e.label,COUNT(*),COALESCE(SUM(f.bytesProcessed),0),"
            "COALESCE(SUM(f.endedAt-f.startedAt),0) "
            "FROM OSRT_FILE_ACCESS_EVENTS f "
            "JOIN ENUM_OSRT_FILE_ACCESS_EVENT_TYPE e ON f.eventType=e.id "
            "WHERE f.startedAt < ? AND f.endedAt > ? GROUP BY e.label",
            (window_end, window_start),
        ):
            file_access[str(label)] = {
                "count": int(count),
                "bytes": int(bytes_count),
                "summed_seconds": int(duration) / 1e9,
            }
    return intervals, grouped, file_access


def _top(grouped: dict[str, list[int]], limit: int) -> list[dict[str, Any]]:
    return [
        {"name": name, **_duration(values)}
        for name, values in sorted(
            grouped.items(), key=lambda item: sum(item[1]), reverse=True
        )[:limit]
    ]


def _idle_gaps(
    active: Sequence[tuple[int, int]], start: int, end: int
) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    cursor = start
    for active_start, active_end in _merge(active):
        if active_start > cursor:
            gaps.append((cursor, active_start))
        cursor = max(cursor, active_end)
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def _representative_window(
    nvtx: Sequence[tuple[str, int, int, int]],
    profile_tid: int,
    outer_start: int,
    outer_end: int,
) -> tuple[int, int]:
    optimizers = sorted(
        (start, end)
        for name, start, end, tid in nvtx
        if tid == profile_tid and name == "training.optimizer"
    )
    if not optimizers:
        duration = outer_end - outer_start
        return outer_start + duration // 3, outer_start + duration * 2 // 3
    index = len(optimizers) // 2
    end = optimizers[index][1]
    previous_end = outer_start if index == 0 else optimizers[index - 1][1]
    loaders = sorted(
        (start, stop)
        for name, start, stop, tid in nvtx
        if tid == profile_tid
        and name == "training.loader.next_batch"
        and previous_end <= start < end
    )
    start = loaders[0][0] if loaders else previous_end
    return start, end


def _svg_rects(
    intervals: Sequence[tuple[int, int]],
    *,
    start: int,
    end: int,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    opacity: float = 1.0,
) -> str:
    scale = width / max(1, end - start)
    parts: list[str] = []
    for left, right in _merge(intervals, gap_ns=20_000):
        clipped = _clip(left, right, start, end)
        if clipped is None:
            continue
        px = x + (clipped[0] - start) * scale
        pw = max(0.7, (clipped[1] - clipped[0]) * scale)
        parts.append(
            f'<rect x="{px:.2f}" y="{y:.2f}" width="{pw:.2f}" '
            f'height="{height:.2f}" fill="{color}" opacity="{opacity:.3f}"/>'
        )
    return "".join(parts)


def _write_timeline(
    path: Path,
    *,
    label: str,
    rep_start: int,
    rep_end: int,
    profile_tid: int,
    nvtx: Sequence[tuple[str, int, int, int]],
    kernels: Sequence[tuple[str, int, int]],
    copies: Sequence[tuple[str, int, int, int]],
    sync: Sequence[tuple[int, int]],
    osrt: Sequence[tuple[int, int]],
    kind: str,
    view_name: str,
) -> None:
    width = 1600
    left = 250
    plot_width = 1300
    top = 80
    lane_height = 30
    lane_gap = 13
    lanes = [
        "Main training thread",
        "Backend workers / pool",
        "OS runtime",
        "GPU input kernels",
        "GPU model kernels",
        "H2D copies",
        "Other copies",
        "CUDA sync API",
    ]
    height = top + len(lanes) * (lane_height + lane_gap) + 90
    body: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="30" font-family="sans-serif" font-size="22" '
        f'font-weight="bold">{html.escape(label)} — {html.escape(view_name)}</text>',
        f'<text x="20" y="55" font-family="sans-serif" font-size="14" fill="#555">'
        f'{(rep_end-rep_start)/1e6:.3f} ms; rectangles are real intervals from the Nsys SQLite export</text>',
    ]
    for index, lane in enumerate(lanes):
        y = top + index * (lane_height + lane_gap)
        body.append(
            f'<text x="{left-12}" y="{y+20}" text-anchor="end" '
            f'font-family="sans-serif" font-size="14">{html.escape(lane)}</text>'
        )
        body.append(
            f'<rect x="{left}" y="{y}" width="{plot_width}" height="{lane_height}" '
            'fill="#f7f7f7" stroke="#dddddd"/>'
        )

    main_y = top
    main_rows = [
        (name, start, end)
        for name, start, end, tid in nvtx
        if tid == profile_tid and name in MAIN_STAGES
    ]
    for name, start, end in main_rows:
        body.append(
            _svg_rects(
                [(start, end)],
                start=rep_start,
                end=rep_end,
                x=left,
                y=main_y,
                width=plot_width,
                height=lane_height,
                color=COLORS[name],
            )
        )

    if kind == "dali":
        backend_predicate = lambda name: name.startswith("[DALI]") or name.startswith(
            "nvjpeg"
        ) or name.startswith("dali.adapter")
    elif kind == "pytorch":
        backend_predicate = lambda name: name.startswith("pytorch.")
    else:
        backend_predicate = lambda name: name in (
            "galp.pool.load",
            "galp.pool.sync_stats_reclaim",
        )
    backend = [
        (start, end)
        for name, start, end, _tid in nvtx
        if backend_predicate(name)
    ]
    body.append(
        _svg_rects(
            backend,
            start=rep_start,
            end=rep_end,
            x=left,
            y=top + lane_height + lane_gap,
            width=plot_width,
            height=lane_height,
            color=COLORS["backend"],
            opacity=0.85,
        )
    )
    body.append(
        _svg_rects(
            osrt,
            start=rep_start,
            end=rep_end,
            x=left,
            y=top + 2 * (lane_height + lane_gap),
            width=plot_width,
            height=lane_height,
            color=COLORS["osrt"],
            opacity=0.8,
        )
    )
    input_kernels = [(start, end) for name, start, end in kernels if _is_input_kernel(name, kind)]
    model_kernels = [(start, end) for name, start, end in kernels if not _is_input_kernel(name, kind)]
    body.append(
        _svg_rects(
            input_kernels,
            start=rep_start,
            end=rep_end,
            x=left,
            y=top + 3 * (lane_height + lane_gap),
            width=plot_width,
            height=lane_height,
            color=COLORS["gpu_input"],
        )
    )
    body.append(
        _svg_rects(
            model_kernels,
            start=rep_start,
            end=rep_end,
            x=left,
            y=top + 4 * (lane_height + lane_gap),
            width=plot_width,
            height=lane_height,
            color=COLORS["gpu_model"],
        )
    )
    h2d = [(start, end) for label, start, end, _bytes in copies if label == "Host-to-Device"]
    other = [(start, end) for label, start, end, _bytes in copies if label != "Host-to-Device"]
    body.append(
        _svg_rects(
            h2d,
            start=rep_start,
            end=rep_end,
            x=left,
            y=top + 5 * (lane_height + lane_gap),
            width=plot_width,
            height=lane_height,
            color=COLORS["h2d"],
        )
    )
    body.append(
        _svg_rects(
            other,
            start=rep_start,
            end=rep_end,
            x=left,
            y=top + 6 * (lane_height + lane_gap),
            width=plot_width,
            height=lane_height,
            color=COLORS["d2d"],
        )
    )
    body.append(
        _svg_rects(
            sync,
            start=rep_start,
            end=rep_end,
            x=left,
            y=top + 7 * (lane_height + lane_gap),
            width=plot_width,
            height=lane_height,
            color=COLORS["sync"],
        )
    )
    duration_ms = (rep_end - rep_start) / 1e6
    for tick in range(11):
        x = left + plot_width * tick / 10
        body.append(
            f'<line x1="{x:.2f}" y1="{top-8}" x2="{x:.2f}" '
            f'y2="{top + len(lanes)*(lane_height+lane_gap)-lane_gap}" '
            'stroke="#dddddd" stroke-dasharray="3,4"/>'
        )
        body.append(
            f'<text x="{x:.2f}" y="{height-48}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{duration_ms*tick/10:.1f} ms</text>'
        )
    legend_x = left
    legend_y = height - 18
    for name in MAIN_STAGES:
        if not any(row[0] == name for row in main_rows):
            continue
        body.append(
            f'<rect x="{legend_x}" y="{legend_y-12}" width="14" height="14" '
            f'fill="{COLORS[name]}"/>'
        )
        body.append(
            f'<text x="{legend_x+19}" y="{legend_y}" font-family="sans-serif" '
            f'font-size="12">{html.escape(STAGE_LABELS[name])}</text>'
        )
        legend_x += 150
    body.append("</svg>")
    path.write_text("".join(body), encoding="utf-8")


def _write_breakdown(path: Path, *, label: str, summary: Mapping[str, Any]) -> None:
    width, height = 1500, 360
    left, bar_width = 250, 1180
    stage_y, gpu_y = 100, 220
    bar_height = 55
    window = float(summary["window"]["seconds"])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="25" y="35" font-family="sans-serif" font-size="22" '
        f'font-weight="bold">{html.escape(label)} — wall-time and GPU occupancy</text>',
        '<text x="235" y="134" text-anchor="end" font-family="sans-serif" font-size="15">'
        'Training thread</text>',
        '<text x="235" y="254" text-anchor="end" font-family="sans-serif" font-size="15">'
        'GPU interval union</text>',
    ]
    cursor = left
    for name in MAIN_STAGES:
        value = float(summary["host_wall_stages"]["stages"][name]["sum_seconds"])
        segment = bar_width * value / window
        if segment <= 0:
            continue
        parts.append(
            f'<rect x="{cursor:.2f}" y="{stage_y}" width="{segment:.2f}" '
            f'height="{bar_height}" fill="{COLORS[name]}"/>'
        )
        if segment > 60:
            parts.append(
                f'<text x="{cursor+segment/2:.2f}" y="{stage_y+33}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="13" fill="white">{100*value/window:.1f}%</text>'
            )
        cursor += segment
    unnamed = max(0.0, window - sum(
        float(summary["host_wall_stages"]["stages"][name]["sum_seconds"])
        for name in MAIN_STAGES
    ))
    segment = bar_width * unnamed / window
    parts.append(
        f'<rect x="{cursor:.2f}" y="{stage_y}" width="{segment:.2f}" '
        f'height="{bar_height}" fill="#bdbdbd"/>'
    )
    active = float(summary["gpu"]["active_seconds"])
    active_width = bar_width * active / window
    parts.extend(
        [
            f'<rect x="{left}" y="{gpu_y}" width="{active_width:.2f}" '
            f'height="{bar_height}" fill="#31a354"/>',
            f'<rect x="{left+active_width:.2f}" y="{gpu_y}" '
            f'width="{bar_width-active_width:.2f}" height="{bar_height}" fill="#eeeeee"/>',
            f'<text x="{left+active_width/2:.2f}" y="{gpu_y+33}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" fill="white">GPU active '
            f'{100*active/window:.1f}%</text>',
            f'<text x="{left+active_width+(bar_width-active_width)/2:.2f}" y="{gpu_y+33}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="14">idle '
            f'{100*(window-active)/window:.1f}%</text>',
        ]
    )
    legend_x, legend_y = left, 325
    for name in MAIN_STAGES:
        if summary["host_wall_stages"]["stages"][name]["count"] == 0:
            continue
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y-13}" width="14" height="14" '
            f'fill="{COLORS[name]}"/><text x="{legend_x+19}" y="{legend_y}" '
            f'font-family="sans-serif" font-size="12">{html.escape(STAGE_LABELS[name])}</text>'
        )
        legend_x += 155
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def analyze(
    sqlite_path: Path,
    *,
    label: str,
    kind: str,
    window_name: str,
    captured_images: int,
    captured_microbatches: int,
    top: int,
    output_dir: Path,
    capture_metrics_path: Path | None = None,
) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{sqlite_path.resolve()}?mode=ro", uri=True)
    try:
        window_rows = connection.execute(
            "SELECT start,end,globalTid FROM NVTX_EVENTS WHERE text=? AND end IS NOT NULL",
            (window_name,),
        ).fetchall()
        if len(window_rows) != 1:
            raise ValueError(
                f"expected one outer NVTX range {window_name!r}, found {len(window_rows)}"
            )
        window_start, window_end, profile_tid = map(int, window_rows[0])
        window_ns = window_end - window_start
        nvtx = _named_nvtx(connection, window_start, window_end)
        kernels, copies = _gpu_events(
            connection, window_start, window_end, kind=kind
        )
        sync_intervals, runtime_grouped = _runtime_sync_intervals(
            connection, window_start, window_end
        )
        osrt_intervals, osrt_grouped, file_access = _osrt(
            connection, window_start, window_end
        )
        stage_values: dict[str, list[int]] = defaultdict(list)
        stage_intervals: list[tuple[int, int]] = []
        for name, start, end, tid in nvtx:
            if tid == profile_tid and name in MAIN_STAGES:
                stage_values[name].append(end - start)
                stage_intervals.append((start, end))

        backend_grouped: dict[str, list[int]] = defaultdict(list)
        for name, start, end, _tid in nvtx:
            if (
                (kind == "dali" and (
                    name.startswith("[DALI]")
                    or name.startswith("nvjpeg")
                    or name.startswith("dali.adapter")
                ))
                or (kind == "pytorch" and name.startswith("pytorch."))
                or (kind == "galp" and name.startswith("galp.pool"))
            ):
                backend_grouped[name].append(end - start)

        input_kernel_intervals = [
            (start, end)
            for name, start, end in kernels
            if _is_input_kernel(name, kind)
        ]
        model_kernel_intervals = [
            (start, end)
            for name, start, end in kernels
            if not _is_input_kernel(name, kind)
        ]
        memcpy_intervals = [(start, end) for _name, start, end, _bytes in copies]
        memset_intervals: list[tuple[int, int]] = []
        if _table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMSET"):
            for start, end in connection.execute(
                "SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET "
                "WHERE start < ? AND end > ?",
                (window_end, window_start),
            ):
                clipped = _clip(int(start), int(end), window_start, window_end)
                if clipped is not None:
                    memset_intervals.append(clipped)
        gpu_active = _merge(
            [
                *((start, end) for _name, start, end in kernels),
                *memcpy_intervals,
                *memset_intervals,
            ]
        )
        idle = _idle_gaps(gpu_active, window_start, window_end)
        transfer: dict[str, dict[str, Any]] = {}
        transfer_grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        for copy_kind, start, end, bytes_count in copies:
            transfer_grouped[copy_kind].append((start, end, bytes_count))
        for copy_kind, rows in transfer_grouped.items():
            transfer[copy_kind] = {
                "count": len(rows),
                "bytes": sum(row[2] for row in rows),
                "gib": sum(row[2] for row in rows) / 1024**3,
                "summed_gpu_seconds": sum(row[1] - row[0] for row in rows) / 1e9,
            }
        kernel_grouped: dict[str, list[int]] = defaultdict(list)
        for name, start, end in kernels:
            kernel_grouped[name].append(end - start)
        idle_durations = [end - start for start, end in idle]
        summary = {
            "schema_version": "galp-training-nsys-keypath-v1",
            "label": label,
            "kind": kind,
            "source_sqlite": str(sqlite_path.resolve()),
            "window": {
                "name": window_name,
                "start_ns": window_start,
                "end_ns": window_end,
                "seconds": window_ns / 1e9,
                "captured_images": captured_images,
                "captured_microbatches": captured_microbatches,
                "images_per_second_with_instrumentation": captured_images
                / (window_ns / 1e9),
            },
            "host_wall_stages": {
                "semantics": "sequential wall ranges on the main training thread",
                "stages": {
                    name: _duration(stage_values.get(name, []))
                    for name in MAIN_STAGES
                },
                "union_seconds": _union_ns(stage_intervals) / 1e9,
                "unnamed_seconds": (window_ns - _union_ns(stage_intervals)) / 1e9,
            },
            "backend_nvtx": {
                "semantics": "summed overlapping worker/operator work",
                "top": _top(backend_grouped, top),
            },
            "gpu": {
                "active_seconds": _union_ns(gpu_active) / 1e9,
                "active_percent": 100 * _union_ns(gpu_active) / window_ns,
                "idle_seconds": (window_ns - _union_ns(gpu_active)) / 1e9,
                "idle_percent": 100 * (window_ns - _union_ns(gpu_active)) / window_ns,
                "input_kernel_union_seconds": _union_ns(input_kernel_intervals) / 1e9,
                "model_kernel_union_seconds": _union_ns(model_kernel_intervals) / 1e9,
                "memcpy_union_seconds": _union_ns(memcpy_intervals) / 1e9,
                "memset_union_seconds": _union_ns(memset_intervals) / 1e9,
                "idle_gaps": _duration(idle_durations),
            },
            "transfers": transfer,
            "cuda_runtime": {
                "synchronization_union_seconds": _union_ns(sync_intervals) / 1e9,
                "synchronization_summed_seconds": sum(
                    end - start for start, end in sync_intervals
                )
                / 1e9,
                "top": _top(runtime_grouped, top),
            },
            "os_runtime": {
                "union_seconds": _union_ns(osrt_intervals) / 1e9,
                "top": _top(osrt_grouped, top),
                "file_access": file_access,
            },
            "kernels": {
                "top": _top(kernel_grouped, top),
            },
        }
        if capture_metrics_path is not None:
            capture_metrics = json.loads(
                capture_metrics_path.read_text(encoding="utf-8")
            )
            if int(capture_metrics.get("captured_images", -1)) != captured_images:
                raise ValueError("capture metrics image count differs from Nsys window")
            if (
                int(capture_metrics.get("captured_microbatches", -1))
                != captured_microbatches
            ):
                raise ValueError(
                    "capture metrics microbatch count differs from Nsys window"
                )
            summary["loader_worker_work"] = {
                "semantics": capture_metrics["loader_stage_semantics"],
                "stage_seconds": capture_metrics["loader_stage_seconds"],
                "source": str(capture_metrics_path.resolve()),
            }
        rep_start, rep_end = _representative_window(
            nvtx, profile_tid, window_start, window_end
        )
        summary["representative_optimizer_window"] = {
            "start_ns": rep_start,
            "end_ns": rep_end,
            "seconds": (rep_end - rep_start) / 1e9,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_timeline(
            output_dir / "timeline.svg",
            label=label,
            rep_start=rep_start,
            rep_end=rep_end,
            profile_tid=profile_tid,
            nvtx=nvtx,
            kernels=kernels,
            copies=copies,
            sync=sync_intervals,
            osrt=osrt_intervals,
            kind=kind,
            view_name="representative optimizer window",
        )
        _write_timeline(
            output_dir / "overview.svg",
            label=label,
            rep_start=window_start,
            rep_end=window_end,
            profile_tid=profile_tid,
            nvtx=nvtx,
            kernels=kernels,
            copies=copies,
            sync=sync_intervals,
            osrt=osrt_intervals,
            kind=kind,
            view_name="full capture overview",
        )
        _write_breakdown(output_dir / "breakdown.svg", label=label, summary=summary)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage_lines = []
        for name in MAIN_STAGES:
            row = summary["host_wall_stages"]["stages"][name]
            stage_lines.append(
                f"| {STAGE_LABELS[name]} | {row['count']} | "
                f"{row['sum_seconds']:.3f} | {row['mean_ms']:.3f} | "
                f"{row['p95_ms']:.3f} |"
            )
        h2d = transfer.get("Host-to-Device", {})
        worker_lines = []
        if "loader_worker_work" in summary:
            for name, seconds in summary["loader_worker_work"]["stage_seconds"].items():
                worker_lines.append(
                    f"| {name} | {seconds:.3f} | "
                    f"{1000 * seconds / captured_images:.3f} |"
                )
        worker_markdown = (
            "\n## Loader worker work (not wall time)\n\n"
            "These values sum per-sample work across parallel workers and must not "
            "be added to the capture wall time.\n\n"
            "| Worker stage | Summed work (s) | Per image (ms) |\n"
            "|---|---:|---:|\n"
            + "\n".join(worker_lines)
            + "\n"
            if worker_lines
            else ""
        )
        markdown = f"""# {label} Nsys key-path summary

Capture: {captured_images:,} images / {captured_microbatches:,} microbatches; window {window_ns/1e9:.3f} s. Trace throughput is attribution-only.

![Full capture overview](overview.svg)

![Representative key path](timeline.svg)

![Wall/GPU breakdown](breakdown.svg)

| Main-thread stage | Count | Sum (s) | Mean (ms) | p95 (ms) |
|---|---:|---:|---:|---:|
{chr(10).join(stage_lines)}

- GPU active: {summary['gpu']['active_seconds']:.3f} s / {summary['gpu']['active_percent']:.2f}%.
- GPU idle: {summary['gpu']['idle_seconds']:.3f} s / {summary['gpu']['idle_percent']:.2f}%.
- CUDA synchronization union: {summary['cuda_runtime']['synchronization_union_seconds']:.3f} s.
- H2D: {h2d.get('gib', 0.0):.3f} GiB / {h2d.get('summed_gpu_seconds', 0.0):.3f} summed GPU seconds / {h2d.get('count', 0)} calls.
- Representative optimizer window: {(rep_end-rep_start)/1e6:.3f} ms.

Backend/operator and kernel entries in `summary.json` are summed work and may overlap; they are not additive wall-time components.
{worker_markdown}"""
        (output_dir / "README.md").write_text(markdown, encoding="utf-8")
        return summary
    finally:
        connection.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = analyze(
        args.sqlite,
        label=args.label,
        kind=args.kind,
        window_name=args.window_name,
        captured_images=args.captured_images,
        captured_microbatches=args.captured_microbatches,
        top=args.top,
        output_dir=args.output_dir,
        capture_metrics_path=args.capture_metrics,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
