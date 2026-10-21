#!/usr/bin/env python3
"""Combine four per-pipeline Nsys summaries into one self-contained dashboard."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGES = (
    ("training.loader.next_batch", "Loader", "#e69f00"),
    ("training.input_handoff", "Handoff", "#56b4e9"),
    ("training.batch.audit", "Audit", "#cc79a7"),
    ("training.model.forward", "Forward", "#009e73"),
    ("training.loss", "Loss", "#f0e442"),
    ("training.model.backward", "Backward", "#0072b2"),
    ("training.optimizer", "Optimizer", "#d55e00"),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("summaries", nargs="+", type=Path)
    return parser.parse_args(argv)


def _h2d(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return summary.get("transfers", {}).get("Host-to-Device", {})


def _write_comparison(path: Path, reports: Sequence[Mapping[str, Any]]) -> None:
    width = 1650
    left = 220
    plot_width = 1320
    row_height = 58
    row_gap = 30
    top = 100
    lower_top = top + len(reports) * (row_height + row_gap) + 90
    height = lower_top + len(reports) * (row_height + row_gap) + 120
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="25" y="35" font-family="sans-serif" font-size="24" '
        'font-weight="bold">Unified training Nsys comparison</text>',
        '<text x="25" y="65" font-family="sans-serif" font-size="14" fill="#555">'
        'Host bars use each capture wall time as 100%; GPU bars are interval unions.</text>',
        f'<text x="{left}" y="{top-20}" font-family="sans-serif" font-size="18" '
        'font-weight="bold">Main training thread wall path</text>',
    ]
    for index, report in enumerate(reports):
        y = top + index * (row_height + row_gap)
        window = float(report["window"]["seconds"])
        parts.append(
            f'<text x="{left-15}" y="{y+34}" text-anchor="end" '
            f'font-family="sans-serif" font-size="15">{html.escape(str(report["label"]))}</text>'
        )
        cursor = left
        named = 0.0
        for name, _label, color in STAGES:
            value = float(report["host_wall_stages"]["stages"][name]["sum_seconds"])
            named += value
            segment = plot_width * value / window
            if segment > 0:
                parts.append(
                    f'<rect x="{cursor:.2f}" y="{y}" width="{segment:.2f}" '
                    f'height="{row_height}" fill="{color}"/>'
                )
                if segment > 65:
                    parts.append(
                        f'<text x="{cursor+segment/2:.2f}" y="{y+34}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="13" fill="white">'
                        f'{100*value/window:.1f}%</text>'
                    )
                cursor += segment
        unnamed = max(0.0, window - named)
        segment = plot_width * unnamed / window
        parts.append(
            f'<rect x="{cursor:.2f}" y="{y}" width="{segment:.2f}" '
            f'height="{row_height}" fill="#bdbdbd"/>'
        )
    parts.append(
        f'<text x="{left}" y="{lower_top-20}" font-family="sans-serif" font-size="18" '
        'font-weight="bold">GPU active versus idle</text>'
    )
    for index, report in enumerate(reports):
        y = lower_top + index * (row_height + row_gap)
        window = float(report["window"]["seconds"])
        active = float(report["gpu"]["active_seconds"])
        active_width = plot_width * active / window
        parts.extend(
            [
                f'<text x="{left-15}" y="{y+34}" text-anchor="end" '
                f'font-family="sans-serif" font-size="15">{html.escape(str(report["label"]))}</text>',
                f'<rect x="{left}" y="{y}" width="{active_width:.2f}" '
                f'height="{row_height}" fill="#31a354"/>',
                f'<rect x="{left+active_width:.2f}" y="{y}" '
                f'width="{plot_width-active_width:.2f}" height="{row_height}" fill="#eeeeee"/>',
                f'<text x="{left+active_width/2:.2f}" y="{y+34}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="14" fill="white">active '
                f'{100*active/window:.1f}%</text>',
                f'<text x="{left+active_width+(plot_width-active_width)/2:.2f}" '
                f'y="{y+34}" text-anchor="middle" font-family="sans-serif" font-size="14">'
                f'idle {100*(window-active)/window:.1f}%</text>',
            ]
        )
    legend_x = left
    legend_y = height - 35
    for _name, label, color in STAGES:
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y-13}" width="14" height="14" '
            f'fill="{color}"/><text x="{legend_x+19}" y="{legend_y}" '
            f'font-family="sans-serif" font-size="13">{html.escape(label)}</text>'
        )
        legend_x += 165
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y-13}" width="14" height="14" '
        'fill="#bdbdbd"/><text x="{legend_x+19}" y="'
        f'{legend_y}" font-family="sans-serif" font-size="13">Control/gaps</text>'
    )
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def _bottleneck_signals(report: Mapping[str, Any]) -> list[str]:
    window = float(report["window"]["seconds"])
    stages = report["host_wall_stages"]["stages"]
    loader = float(stages["training.loader.next_batch"]["sum_seconds"]) / window
    compute = (
        float(stages["training.model.forward"]["sum_seconds"])
        + float(stages["training.model.backward"]["sum_seconds"])
    ) / window
    optimizer = float(stages["training.optimizer"]["sum_seconds"]) / window
    gpu_idle = float(report["gpu"]["idle_percent"]) / 100
    sync = float(report["cuda_runtime"]["synchronization_union_seconds"]) / window
    signals = [
        f"loader {loader*100:.1f}%",
        f"forward+backward {compute*100:.1f}%",
        f"optimizer {optimizer*100:.1f}%",
        f"GPU idle {gpu_idle*100:.1f}%",
        f"sync {sync*100:.1f}%",
    ]
    if loader > 0.20 and gpu_idle > 0.35:
        signals.append("input supply is exposed on the critical path")
    if compute > 0.65:
        signals.append("model forward/backward dominates the host path")
    if optimizer > 0.10:
        signals.append("optimizer cadence is material")
    if sync > 0.08:
        signals.append("explicit synchronization is material")
    return signals


def build(output_dir: Path, summaries: Sequence[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
    _write_comparison(output_dir / "comparison.svg", reports)
    rows = []
    cards = []
    for report, source in zip(reports, summaries):
        window = report["window"]
        h2d = _h2d(report)
        rel = Path(os.path.relpath(source.parent.resolve(), output_dir.resolve()))
        signals = "; ".join(_bottleneck_signals(report))
        stage_rows = []
        for name, display, _color in STAGES:
            stage = report["host_wall_stages"]["stages"][name]
            if int(stage["count"]) == 0:
                continue
            stage_rows.append(
                "<tr>"
                f"<td>{html.escape(display)}</td>"
                f"<td>{stage['count']}</td>"
                f"<td>{stage['sum_seconds']:.3f}</td>"
                f"<td>{stage['p50_ms']:.3f}</td>"
                f"<td>{stage['p95_ms']:.3f}</td>"
                "</tr>"
            )
        backend_rows = []
        for item in report["backend_nvtx"]["top"][:8]:
            backend_rows.append(
                "<tr>"
                f"<td>{html.escape(str(item['name']))}</td>"
                f"<td>{item['count']}</td>"
                f"<td>{item['sum_seconds']:.3f}</td>"
                f"<td>{item['p50_ms']:.3f}</td>"
                f"<td>{item['p95_ms']:.3f}</td>"
                "</tr>"
            )
        worker_html = ""
        if "loader_worker_work" in report:
            worker_rows = []
            captured_images = float(window["captured_images"])
            for name, seconds in report["loader_worker_work"]["stage_seconds"].items():
                worker_rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(name))}</td>"
                    f"<td>{seconds:.3f}</td>"
                    f"<td>{1000*seconds/captured_images:.3f}</td>"
                    "</tr>"
                )
            worker_html = (
                "<h3>DataLoader worker work (summed, not wall time)</h3>"
                "<p class=\"note\">These per-sample counters overlap across workers; "
                "do not add them to capture wall time.</p>"
                "<table><thead><tr><th>Stage</th><th>Summed s</th>"
                "<th>ms/image</th></tr></thead><tbody>"
                + "".join(worker_rows)
                + "</tbody></table>"
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(report['label']))}</td>"
            f"<td>{window['seconds']:.3f}</td>"
            f"<td>{window['images_per_second_with_instrumentation']:.1f}</td>"
            f"<td>{report['gpu']['active_percent']:.1f}%</td>"
            f"<td>{h2d.get('gib', 0.0):.3f}</td>"
            f"<td>{report['cuda_runtime']['synchronization_union_seconds']:.3f}</td>"
            f"<td>{html.escape(signals)}</td>"
            "</tr>"
        )
        cards.append(
            f"<section><h2>{html.escape(str(report['label']))}</h2>"
            f'<object data="{html.escape(str(rel / "overview.svg"))}" '
            'type="image/svg+xml"></object>'
            f'<object data="{html.escape(str(rel / "timeline.svg"))}" '
            'type="image/svg+xml"></object>'
            f'<object data="{html.escape(str(rel / "breakdown.svg"))}" '
            'type="image/svg+xml"></object>'
            '<div class="grid">'
            '<div><h3>Main-thread wall stages</h3><table><thead><tr>'
            '<th>Stage</th><th>Count</th><th>Sum s</th><th>p50 ms</th>'
            '<th>p95 ms</th></tr></thead><tbody>'
            f'{"".join(stage_rows)}</tbody></table></div>'
            '<div><h3>Backend/operator work</h3><table><thead><tr>'
            '<th>Range</th><th>Count</th><th>Summed s</th><th>p50 ms</th>'
            '<th>p95 ms</th></tr></thead><tbody>'
            f'{"".join(backend_rows)}</tbody></table></div></div>'
            f'{worker_html}</section>'
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Training Nsys dashboard</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#222}} object{{width:100%;min-height:360px;border:1px solid #ddd;margin:8px 0 20px}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccc;padding:7px;text-align:right}} th:first-child,td:first-child,th:last-child,td:last-child{{text-align:left}} section{{margin-top:40px}} .note{{color:#555}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start}} @media(max-width:1100px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>RTX 4090 unified fine-grained training profile</h1>
<p class="note">Trace throughput includes instrumentation overhead and is not the formal performance metric. All time bars use interval-union or main-thread wall semantics.</p>
<object data="comparison.svg" type="image/svg+xml"></object>
<table><thead><tr><th>Pipeline</th><th>Window s</th><th>Trace img/s</th><th>GPU active</th><th>H2D GiB</th><th>Sync union s</th><th>Bottleneck signals</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{''.join(cards)}
</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    build(args.output_dir, args.summaries)
    print(args.output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
