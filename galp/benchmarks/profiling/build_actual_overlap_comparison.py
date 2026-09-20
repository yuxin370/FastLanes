#!/usr/bin/env python3
"""Build a compact, time-aligned overlap view from existing Nsys SQLite traces.

The figure intentionally distinguishes interval overlap from causal exposure.
Every rectangle is derived from a real interval in the representative optimizer
window selected by ``analyze_training_nsys.py``.  The output is analysis-only;
it does not modify or execute a training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from galp.benchmarks.profiling.analyze_training_nsys import (
    MAIN_STAGES,
    _gpu_events,
    _is_input_kernel,
    _merge,
    _named_nvtx,
)


PIPELINES = (
    ("d2", "DALI D2", "dali"),
    ("d3", "DALI D3", "dali"),
    ("galp", "GALP", "galp"),
    ("pytorch", "PyTorch-4", "pytorch"),
)

LANES = (
    ("main_input", "Main: loader / handoff", "#e69f00"),
    ("main_model", "Main: model host scopes", "#56b4e9"),
    ("backend", "Backend host / pool NVTX scopes", "#8c6bb1"),
    ("h2d", "H2D copies", "#cc79a7"),
    ("gpu_input", "GPU input kernels", "#fdae6b"),
    ("gpu_model", "GPU model kernels", "#009e73"),
    ("gpu_idle", "GPU idle", "#d9d9d9"),
)


def _clip(intervals: Iterable[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    return [
        (max(left, start), min(right, end))
        for left, right in intervals
        if left < end and right > start and min(right, end) > max(left, start)
    ]


def _union_ns(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(right - left for left, right in _merge(intervals))


def _intersection_ns(
    left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]
) -> int:
    a = _merge(left)
    b = _merge(right)
    i = 0
    j = 0
    total = 0
    while i < len(a) and j < len(b):
        begin = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > begin:
            total += end - begin
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _complement(
    intervals: Sequence[tuple[int, int]], start: int, end: int
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = start
    for left, right in _merge(_clip(intervals, start, end)):
        if left > cursor:
            result.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        result.append((cursor, end))
    return result


def _backend_predicate(name: str, kind: str) -> bool:
    if kind == "dali":
        return (
            name.startswith("[DALI]")
            or name.startswith("nvjpeg")
            or name.startswith("dali.adapter")
        )
    if kind == "pytorch":
        return name.startswith("pytorch.")
    return name.startswith("galp.pool")


def _load_panel(result_root: Path, key: str, label: str, kind: str) -> dict[str, Any]:
    analysis = result_root / "nsys" / key / "analysis" / "summary.json"
    summary = json.loads(analysis.read_text(encoding="utf-8"))
    sqlite_path = result_root / "nsys" / key / "trace.sqlite"
    rep_start = int(summary["representative_optimizer_window"]["start_ns"])
    rep_end = int(summary["representative_optimizer_window"]["end_ns"])
    window_name = str(summary["window"]["name"])
    connection = sqlite3.connect(f"file:{sqlite_path.resolve()}?mode=ro", uri=True)
    try:
        outer = connection.execute(
            "SELECT globalTid FROM NVTX_EVENTS WHERE text=? AND end IS NOT NULL",
            (window_name,),
        ).fetchall()
        if len(outer) != 1:
            raise ValueError(f"{key}: expected one outer NVTX range, found {len(outer)}")
        profile_tid = int(outer[0][0])
        nvtx = _named_nvtx(connection, rep_start, rep_end)
        kernels, copies = _gpu_events(connection, rep_start, rep_end, kind=kind)
        memsets: list[tuple[int, int]] = []
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
            memsets = [
                (max(int(start), rep_start), min(int(end), rep_end))
                for start, end in connection.execute(
                    "SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET "
                    "WHERE start < ? AND end > ?",
                    (rep_end, rep_start),
                )
            ]
    finally:
        connection.close()

    input_stage_names = {
        "training.loader.next_batch",
        "training.input_handoff",
        "training.batch.audit",
    }
    model_stage_names = set(MAIN_STAGES) - input_stage_names
    main_input = [
        (start, end)
        for name, start, end, tid in nvtx
        if tid == profile_tid and name in input_stage_names
    ]
    main_model = [
        (start, end)
        for name, start, end, tid in nvtx
        if tid == profile_tid and name in model_stage_names
    ]
    backend = [
        (start, end)
        for name, start, end, _tid in nvtx
        if _backend_predicate(name, kind)
    ]
    gpu_input = [
        (start, end)
        for name, start, end in kernels
        if _is_input_kernel(name, kind)
    ]
    gpu_model = [
        (start, end)
        for name, start, end in kernels
        if not _is_input_kernel(name, kind)
    ]
    h2d = [
        (start, end)
        for copy_kind, start, end, _bytes in copies
        if copy_kind == "Host-to-Device"
    ]
    all_copies = [(start, end) for _kind, start, end, _bytes in copies]
    gpu_active = [
        *((start, end) for _name, start, end in kernels),
        *all_copies,
        *memsets,
    ]
    lanes = {
        "main_input": _clip(main_input, rep_start, rep_end),
        "main_model": _clip(main_model, rep_start, rep_end),
        "backend": _clip(backend, rep_start, rep_end),
        "h2d": _clip(h2d, rep_start, rep_end),
        "gpu_input": _clip(gpu_input, rep_start, rep_end),
        "gpu_model": _clip(gpu_model, rep_start, rep_end),
        "gpu_idle": _complement(gpu_active, rep_start, rep_end),
    }
    h2d_union = _union_ns(lanes["h2d"])
    input_union = _union_ns(lanes["gpu_input"])
    model_union = _union_ns(lanes["gpu_model"])
    return {
        "key": key,
        "label": label,
        "kind": kind,
        "start": rep_start,
        "end": rep_end,
        "lanes": lanes,
        "metrics": {
            "window_ms": (rep_end - rep_start) / 1e6,
            "main_input_ms": _union_ns(lanes["main_input"]) / 1e6,
            "backend_scope_union_ms": _union_ns(lanes["backend"]) / 1e6,
            "h2d_ms": h2d_union / 1e6,
            "h2d_model_overlap_ms": _intersection_ns(
                lanes["h2d"], lanes["gpu_model"]
            )
            / 1e6,
            "h2d_non_model_overlap_ms": (
                h2d_union - _intersection_ns(lanes["h2d"], lanes["gpu_model"])
            )
            / 1e6,
            "gpu_input_ms": input_union / 1e6,
            "gpu_input_model_overlap_ms": _intersection_ns(
                lanes["gpu_input"], lanes["gpu_model"]
            )
            / 1e6,
            "gpu_input_non_model_overlap_ms": (
                input_union
                - _intersection_ns(lanes["gpu_input"], lanes["gpu_model"])
            )
            / 1e6,
            "gpu_model_ms": model_union / 1e6,
            "gpu_idle_ms": _union_ns(lanes["gpu_idle"]) / 1e6,
        },
    }


def _rectangles(
    intervals: Sequence[tuple[int, int]],
    *,
    start: int,
    x: float,
    y: float,
    scale: float,
    height: float,
    color: str,
) -> str:
    parts: list[str] = []
    for left, right in _merge(intervals, gap_ns=20_000):
        px = x + (left - start) * scale
        width = max(0.8, (right - left) * scale)
        parts.append(
            f'<rect x="{px:.2f}" y="{y:.2f}" width="{width:.2f}" '
            f'height="{height:.2f}" fill="{color}" opacity="0.90"/>'
        )
    return "".join(parts)


def _write_svg(path: Path, panels: Sequence[dict[str, Any]]) -> None:
    width = 1800
    left = 265
    right = 40
    plot_width = width - left - right
    top = 112
    lane_height = 17
    lane_gap = 5
    panel_gap = 48
    panel_height = len(LANES) * (lane_height + lane_gap)
    max_ns = max(panel["end"] - panel["start"] for panel in panels)
    scale = plot_width / max_ns
    height = top + len(panels) * (panel_height + panel_gap) + 80
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="sans-serif" font-size="24" '
        'font-weight="bold">Actual Nsys overlap — one optimizer window per pipeline</text>',
        '<text x="24" y="60" font-family="sans-serif" font-size="14" fill="#444">'
        'Each panel contains 16 real microbatches; all panels share the same millisecond scale. '
        'Overlap is temporal evidence, not a causal critical-path proof.</text>',
    ]
    ticks = 7
    for index in range(ticks + 1):
        offset = max_ns * index / ticks
        x = left + offset * scale
        parts.append(
            f'<line x1="{x:.2f}" y1="82" x2="{x:.2f}" y2="{height-55}" '
            'stroke="#e5e5e5" stroke-dasharray="3,4"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="78" text-anchor="middle" font-family="sans-serif" '
            f'font-size="12" fill="#555">{offset/1e6:.0f} ms</text>'
        )
    for panel_index, panel in enumerate(panels):
        panel_y = top + panel_index * (panel_height + panel_gap)
        metrics = panel["metrics"]
        parts.append(
            f'<text x="24" y="{panel_y-16}" font-family="sans-serif" font-size="18" '
            f'font-weight="bold">{html.escape(panel["label"])}</text>'
        )
        parts.append(
            f'<text x="180" y="{panel_y-16}" font-family="sans-serif" font-size="12" '
            f'fill="#555">window {metrics["window_ms"]:.1f} ms; H2D {metrics["h2d_ms"]:.1f} ms; '
            f'GPU input {metrics["gpu_input_ms"]:.1f} ms; model kernels {metrics["gpu_model_ms"]:.1f} ms; '
            f'GPU idle {metrics["gpu_idle_ms"]:.1f} ms</text>'
        )
        panel_end_x = left + (panel["end"] - panel["start"]) * scale
        parts.append(
            f'<line x1="{panel_end_x:.2f}" y1="{panel_y-8}" x2="{panel_end_x:.2f}" '
            f'y2="{panel_y+panel_height-5}" stroke="#444" stroke-width="1.2"/>'
        )
        for lane_index, (key, lane_label, color) in enumerate(LANES):
            y = panel_y + lane_index * (lane_height + lane_gap)
            parts.append(
                f'<text x="{left-12}" y="{y+13}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{html.escape(lane_label)}</text>'
            )
            parts.append(
                f'<rect x="{left}" y="{y}" width="{plot_width}" height="{lane_height}" '
                'fill="#fafafa" stroke="#eeeeee"/>'
            )
            parts.append(
                _rectangles(
                    panel["lanes"][key],
                    start=panel["start"],
                    x=left,
                    y=y,
                    scale=scale,
                    height=lane_height,
                    color=color,
                )
            )
    parts.append(
        f'<text x="24" y="{height-25}" font-family="sans-serif" font-size="12" '
        'fill="#555">GPU input means preprocessing CUDA kernels, excluding H2D and model kernels. '
        'Gray GPU-idle intervals are the complement of kernels, copies, and memsets inside each panel.</text>'
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def _write_csv(path: Path, panels: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pipeline", *panels[0]["metrics"].keys()]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for panel in panels:
            writer.writerow({"pipeline": panel["key"], **panel["metrics"]})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root.resolve()
    panels = [_load_panel(root, *spec) for spec in PIPELINES]
    _write_svg(args.output_svg.resolve(), panels)
    _write_csv(args.output_csv.resolve(), panels)
    print(json.dumps({panel["key"]: panel["metrics"] for panel in panels}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
