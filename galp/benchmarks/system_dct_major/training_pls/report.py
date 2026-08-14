#!/usr/bin/env python3
"""Aggregate core PLS curves, paired effects, confidence intervals, and plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .matrix import CORE_CONDITION_IDS, PAIRED_SEEDS, core_matrix


REPORT_SCHEMA = "galp-pls-core-model-effect-report-v2"
COLORS = {
    "A0": (31, 119, 180),
    "A1": (255, 127, 14),
    "B2": (44, 160, 44),
    "B6": (214, 39, 40),
}
T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def mean_ci(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95_low": None, "ci95_high": None}
    std = statistics.stdev(values)
    critical = T_975.get(len(values) - 1, 1.96)
    margin = critical * std / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def normalized_auc(validation: Sequence[Mapping[str, Any]], total_images: int) -> float:
    if total_images <= 0 or len(validation) < 2:
        raise ValueError("normalized AUC requires at least two points and positive total images")
    points = sorted(
        (
            float(event["processed_images"]) / total_images,
            float(event["validation_top1"]),
        )
        for event in validation
    )
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) * 0.5
    return area


def _read_run(path: Path, condition: str, seed: int) -> dict[str, Any]:
    status_path = path / "run_status.json"
    result_path = path / "final_result.json"
    metrics_path = path / "metrics.jsonl"
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {"state": "missing"}
    )
    record: dict[str, Any] = {
        "condition": condition,
        "seed": seed,
        "path": str(path),
        "status": status,
        "completed": result_path.is_file(),
        "result": None,
        "metrics": [],
        "validation": [],
        "train": [],
    }
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"malformed {metrics_path}:{line_number}: {error}") from error
                record["metrics"].append(event)
                if event.get("record_type") == "validation":
                    record["validation"].append(event)
                elif event.get("record_type") == "train":
                    record["train"].append(event)
    if result_path.is_file():
        record["result"] = json.loads(result_path.read_text(encoding="utf-8"))
        result = record["result"]
        if result.get("state") != "completed":
            raise ValueError(f"{result_path} is not a completed result")
        if result.get("condition") != condition or int(result.get("seed")) != seed:
            raise ValueError(f"{result_path} identity differs from its run directory")
        if not record["validation"]:
            raise ValueError(f"{result_path} has no validation curve")
    return record


def _curve_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = (
        "record_type",
        "scope",
        "epoch",
        "optimizer_update",
        "processed_images",
        "train_loss",
        "learning_rate",
        "validation_top1",
        "validation_top5",
        "validation_loss",
        "validation_latency_seconds",
        "images_per_second",
        "data_preparation_seconds",
    )
    for run in runs:
        for event in run["metrics"]:
            row = {"condition": run["condition"], "seed": run["seed"]}
            row.update({field: event.get(field) for field in fields})
            rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["condition"]),
            int(row["seed"]),
            int(row.get("optimizer_update") or 0),
            str(row.get("record_type")),
        )
    )
    return rows


def _final_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not run["completed"]:
            continue
        result = run["result"]
        auc = normalized_auc(run["validation"], int(result["total_processed_images"]))
        rows.append(
            {
                "condition": run["condition"],
                "seed": run["seed"],
                "final_top1": float(result["final_top1"]),
                "final_top5": float(result["final_top5"]),
                "final_validation_loss": float(result["final_validation_loss"]),
                "normalized_top1_auc": auc,
                "total_processed_images": int(result["total_processed_images"]),
                "total_optimizer_updates": int(result["total_optimizer_updates"]),
                "training_runtime_seconds": float(result["training_runtime_seconds"]),
                "images_per_second": float(result["images_per_second"]),
                "cuda_peak_allocated_bytes": result["cuda_memory"]["peak_allocated_bytes"],
                "cuda_peak_reserved_bytes": result["cuda_memory"]["peak_reserved_bytes"],
                "execution_mode": result["execution_mode"],
                "semantic_emulation": result["semantic_emulation"],
                "physical_fls_observed": result["physical_fls_observed"],
                "physical_gpu_pool": result["physical_gpu_pool"],
                "layout_hash": result["layout_hash"],
                "recipe_hash": result["recipe_hash"],
                "condition_hash": result["condition_hash"],
                "initial_model_hash": result["initial_model_hash"],
            }
        )
    rows.sort(key=lambda row: (str(row["condition"]), int(row["seed"])))
    return rows


EFFECTS: dict[str, dict[str, float]] = {
    "crop_global": {"A1": 1.0, "A0": -1.0},
    "crop_closed_pool": {"B6": 1.0, "B2": -1.0},
    "shuffle_per_sample_crop": {"B2": 1.0, "A0": -1.0},
    "shuffle_per_pls_crop": {"B6": 1.0, "A1": -1.0},
    "crop_main": {"A1": 0.5, "A0": -0.5, "B6": 0.5, "B2": -0.5},
    "shuffle_main": {"B2": 0.5, "A0": -0.5, "B6": 0.5, "A1": -0.5},
    "crop_x_shuffle": {"B6": 1.0, "B2": -1.0, "A1": -1.0, "A0": 1.0},
    "target_vs_standard": {"B6": 1.0, "A0": -1.0},
}
SIMPLE_EFFECTS = {
    "crop_global",
    "crop_closed_pool",
    "shuffle_per_sample_crop",
    "shuffle_per_pls_crop",
}


def _effect_rows(final_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(row["condition"]), int(row["seed"])): row for row in final_rows
    }
    endpoints = (
        "final_top1",
        "final_top5",
        "final_validation_loss",
        "normalized_top1_auc",
    )
    rows: list[dict[str, Any]] = []
    for effect_id, coefficients in EFFECTS.items():
        paired_seeds = [
            seed
            for seed in PAIRED_SEEDS
            if all((condition, seed) in by_key for condition in coefficients)
        ]
        for endpoint in endpoints:
            points = [
                {
                    "seed": seed,
                    "difference": sum(
                        coefficient * float(by_key[(condition, seed)][endpoint])
                        for condition, coefficient in coefficients.items()
                    ),
                }
                for seed in paired_seeds
            ]
            summary = mean_ci([point["difference"] for point in points])
            interpretation = None
            if endpoint == "final_top1" and summary["ci95_low"] is not None:
                low = float(summary["ci95_low"])
                high = float(summary["ci95_high"])
                if low > 0.3 or high < -0.3:
                    interpretation = "CI entirely beyond the +/-0.3 pp practical boundary"
                elif low >= -0.3 and high <= 0.3:
                    interpretation = "CI entirely within the +/-0.3 pp practical boundary"
                else:
                    interpretation = "CI overlaps a +/-0.3 pp practical boundary; evidence is insufficient"
            rows.append(
                {
                    "effect_id": effect_id,
                    "effect_class": "simple" if effect_id in SIMPLE_EFFECTS else "factorial",
                    "endpoint": endpoint,
                    "formula_coefficients": json.dumps(coefficients, sort_keys=True),
                    "paired_mean_difference": summary["mean"],
                    "paired_standard_deviation": summary["std"],
                    "paired_seed_count": summary["n"],
                    "ci95_low": summary["ci95_low"],
                    "ci95_high": summary["ci95_high"],
                    "individual_seed_points": json.dumps(points, sort_keys=True),
                    "practical_interpretation": interpretation,
                }
            )
    return rows


def _pointwise_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        if not run["completed"]:
            continue
        for event in run["validation"]:
            grouped[(str(run["condition"]), int(event["epoch"]))].append(event)
    rows: list[dict[str, Any]] = []
    for (condition, epoch), events in sorted(grouped.items()):
        for endpoint in ("validation_top1", "validation_top5", "validation_loss"):
            summary = mean_ci([float(event[endpoint]) for event in events])
            rows.append(
                {
                    "condition": condition,
                    "epoch": epoch,
                    "processed_images_mean": statistics.fmean(
                        float(event["processed_images"]) for event in events
                    ),
                    "endpoint": endpoint,
                    "mean": summary["mean"],
                    "std": summary["std"],
                    "n": summary["n"],
                    "pointwise_ci95_low": summary["ci95_low"],
                    "pointwise_ci95_high": summary["ci95_high"],
                    "band_type": "pointwise-not-simultaneous",
                }
            )
    return rows


def _font(size: int = 18) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _placeholder(path: Path, title: str, reason: str) -> None:
    image = Image.new("RGB", (1400, 850), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 65), title, fill="black", font=_font(30))
    draw.text((70, 145), reason, fill=(120, 120, 120), font=_font(20))
    image.save(path)


def _line_chart(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Sequence[Mapping[str, Any]],
    bands: Sequence[Mapping[str, Any]] = (),
) -> None:
    usable = [item for item in series if len(item["points"]) >= 1]
    if not usable:
        _placeholder(path, title, "No completed model runs are available; no curve is fabricated.")
        return
    width, height = 1600, 950
    left, right, top, bottom = 125, 55, 90, 115
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    all_points = [point for item in usable for point in item["points"]]
    xs = [float(point[0]) for point in all_points]
    ys = [float(point[1]) for point in all_points]
    for band in bands:
        for x, low, high in band["points"]:
            xs.append(float(x))
            if low is not None:
                ys.append(float(low))
            if high is not None:
                ys.append(float(high))
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    ypad = (ymax - ymin) * 0.06
    ymin -= ypad
    ymax += ypad

    def xy(x: float, y: float) -> tuple[int, int]:
        px = left + int((x - xmin) / (xmax - xmin) * (width - left - right))
        py = top + int((ymax - y) / (ymax - ymin) * (height - top - bottom))
        return px, py

    for tick in range(6):
        fraction = tick / 5
        x = xmin + fraction * (xmax - xmin)
        px, _ = xy(x, ymin)
        draw.line((px, top, px, height - bottom), fill=(225, 225, 225, 255), width=1)
        draw.text((px - 35, height - bottom + 18), f"{x:.2g}", fill="black", font=_font(15))
        y = ymin + fraction * (ymax - ymin)
        _, py = xy(xmin, y)
        draw.line((left, py, width - right, py), fill=(225, 225, 225, 255), width=1)
        draw.text((15, py - 10), f"{y:.3g}", fill="black", font=_font(15))
    for band in bands:
        valid = [(x, low, high) for x, low, high in band["points"] if low is not None and high is not None]
        if len(valid) >= 2:
            polygon = [xy(float(x), float(high)) for x, _low, high in valid]
            polygon.extend(xy(float(x), float(low)) for x, low, _high in reversed(valid))
            color = tuple(band["color"])
            draw.polygon(polygon, fill=(*color, 38))
    for item in usable:
        points = [xy(float(x), float(y)) for x, y in item["points"]]
        color = tuple(item["color"])
        draw.line(points, fill=(*color, int(item.get("alpha", 255))), width=int(item.get("width", 3)))
        if item.get("markers"):
            for px, py in points:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(*color, 255))
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    draw.text((left, 28), title, fill="black", font=_font(30))
    draw.text((width // 2 - 90, height - 50), xlabel, fill="black", font=_font(19))
    draw.text((15, 55), ylabel, fill="black", font=_font(19))
    legend_x, legend_y = width - 315, 105
    shown: set[str] = set()
    for item in usable:
        label = str(item["label"])
        if label in shown:
            continue
        shown.add(label)
        color = tuple(item["color"])
        draw.line((legend_x, legend_y + 9, legend_x + 42, legend_y + 9), fill=(*color, 255), width=4)
        draw.text((legend_x + 52, legend_y), label, fill="black", font=_font(16))
        legend_y += 28
    image.convert("RGB").save(path)


def _effect_chart(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    selected = [row for row in rows if row["endpoint"] == "final_top1" and row["paired_mean_difference"] is not None]
    if not selected:
        _placeholder(path, title, "No complete paired seeds are available; no effect is fabricated.")
        return
    width, height = 1500, max(650, 150 + 90 * len(selected))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    values = []
    for row in selected:
        values.append(float(row["paired_mean_difference"]))
        for key in ("ci95_low", "ci95_high"):
            if row[key] is not None:
                values.append(float(row[key]))
        values.extend(
            point["difference"] for point in json.loads(row["individual_seed_points"])
        )
    extent = max(0.35, max(abs(value) for value in values) * 1.15)
    left, right, top, bottom = 430, 70, 100, 90

    def x(value: float) -> int:
        return left + int((value + extent) / (2 * extent) * (width - left - right))

    draw.text((left, 30), title, fill="black", font=_font(28))
    draw.line((x(0), top, x(0), height - bottom), fill=(80, 80, 80), width=2)
    draw.line((x(-0.3), top, x(-0.3), height - bottom), fill=(180, 180, 180), width=2)
    draw.line((x(0.3), top, x(0.3), height - bottom), fill=(180, 180, 180), width=2)
    for index, row in enumerate(selected):
        y = top + 55 + index * 90
        draw.text((30, y - 12), str(row["effect_id"]), fill="black", font=_font(18))
        low, high = row["ci95_low"], row["ci95_high"]
        if low is not None and high is not None:
            draw.line((x(float(low)), y, x(float(high)), y), fill=(20, 20, 20), width=4)
        mean = float(row["paired_mean_difference"])
        draw.ellipse((x(mean) - 7, y - 7, x(mean) + 7, y + 7), fill=(0, 0, 0))
        for point_index, point in enumerate(json.loads(row["individual_seed_points"])):
            px = x(float(point["difference"]))
            offset = -22 + point_index * 12
            draw.ellipse((px - 4, y + offset - 4, px + 4, y + offset + 4), fill=(31, 119, 180))
    draw.text((left, height - 50), "Final top-1 paired difference (percentage points)", fill="black", font=_font(18))
    image.save(path)


def _make_plots(
    output_dir: Path,
    runs: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    pointwise: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
) -> None:
    for endpoint, filename, ylabel in (
        ("validation_top1", "top1_convergence.png", "Validation top-1 (%)"),
        ("validation_top5", "top5_convergence.png", "Validation top-5 (%)"),
    ):
        series: list[dict[str, Any]] = []
        for run in runs:
            if not run["completed"]:
                continue
            condition = str(run["condition"])
            series.append(
                {
                    "label": condition,
                    "color": COLORS[condition],
                    "alpha": 75,
                    "width": 2,
                    "points": [
                        (float(event["processed_images"]), float(event[endpoint]))
                        for event in run["validation"]
                    ],
                }
            )
        bands: list[dict[str, Any]] = []
        for condition in CORE_CONDITION_IDS:
            rows = [
                row
                for row in pointwise
                if row["condition"] == condition and row["endpoint"] == endpoint
            ]
            if rows:
                series.append(
                    {
                        "label": condition,
                        "color": COLORS[condition],
                        "width": 5,
                        "points": [
                            (row["processed_images_mean"], row["mean"]) for row in rows
                        ],
                    }
                )
                bands.append(
                    {
                        "color": COLORS[condition],
                        "points": [
                            (
                                row["processed_images_mean"],
                                row["pointwise_ci95_low"],
                                row["pointwise_ci95_high"],
                            )
                            for row in rows
                        ],
                    }
                )
        _line_chart(
            output_dir / filename,
            title=f"PLS {ylabel} convergence (thin=seed, thick=mean; bands=pointwise 95% CI)",
            xlabel="Processed images",
            ylabel=ylabel,
            series=series,
            bands=bands,
        )

    loss_series = []
    for run in runs:
        epoch_points = [event for event in run["train"] if event.get("scope") == "epoch"]
        if epoch_points:
            condition = str(run["condition"])
            loss_series.append(
                {
                    "label": condition,
                    "color": COLORS[condition],
                    "alpha": 120,
                    "width": 2,
                    "points": [
                        (float(event["processed_images"]), float(event["train_loss"]))
                        for event in epoch_points
                    ],
                }
            )
    _line_chart(
        output_dir / "train_loss_curves.png",
        title="PLS epoch training-loss curves",
        xlabel="Processed images",
        ylabel="Train loss",
        series=loss_series,
    )

    final_series = []
    for condition in CORE_CONDITION_IDS:
        rows = [row for row in final_rows if row["condition"] == condition]
        if rows:
            final_series.append(
                {
                    "label": condition,
                    "color": COLORS[condition],
                    "markers": True,
                    "width": 1,
                    "points": [(float(row["seed"]), float(row["final_top1"])) for row in rows],
                }
            )
    _line_chart(
        output_dir / "final_top1_by_seed.png",
        title="Final-checkpoint top-1 by paired training seed",
        xlabel="Training seed",
        ylabel="Final top-1 (%)",
        series=final_series,
    )
    _effect_chart(
        output_dir / "paired_differences.png",
        "Crop and shuffle simple effects (mean, seed points, 95% t-CI)",
        [row for row in effects if row["effect_class"] == "simple"],
    )
    _effect_chart(
        output_dir / "factorial_effects.png",
        "Factorial effects and B6-A0 (mean, seed points, 95% t-CI)",
        [row for row in effects if row["effect_class"] == "factorial"],
    )
    auc_series = []
    for condition in CORE_CONDITION_IDS:
        rows = [row for row in final_rows if row["condition"] == condition]
        if rows:
            auc_series.append(
                {
                    "label": condition,
                    "color": COLORS[condition],
                    "markers": True,
                    "width": 1,
                    "points": [(float(row["seed"]), float(row["normalized_top1_auc"])) for row in rows],
                }
            )
    _line_chart(
        output_dir / "normalized_auc.png",
        title="Normalized validation top-1 AUC by paired seed",
        xlabel="Training seed",
        ylabel="Normalized top-1 AUC",
        series=auc_series,
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return result


def aggregate(
    runs_root: Path,
    output_dir: Path,
    *,
    conditions: Sequence[str],
    require_complete: bool,
) -> dict[str, Any]:
    runs_root = runs_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in PAIRED_SEEDS:
            runs.append(
                _read_run(
                    runs_root / "runs" / condition / f"seed_{seed}", condition, seed
                )
            )
    completed = [run for run in runs if run["completed"]]
    missing = [
        {
            "condition": run["condition"],
            "seed": run["seed"],
            "status": run["status"],
            "path": run["path"],
        }
        for run in runs
        if not run["completed"]
    ]
    if require_complete and missing:
        raise ValueError(f"core matrix is incomplete: {missing}")
    curve_rows = _curve_rows(runs)
    final_rows = _final_rows(runs)
    effects = _effect_rows(final_rows)
    pointwise = _pointwise_rows(runs)
    curve_fields = (
        "condition",
        "seed",
        "record_type",
        "scope",
        "epoch",
        "optimizer_update",
        "processed_images",
        "train_loss",
        "learning_rate",
        "validation_top1",
        "validation_top5",
        "validation_loss",
        "validation_latency_seconds",
        "images_per_second",
        "data_preparation_seconds",
    )
    final_fields = (
        "condition",
        "seed",
        "final_top1",
        "final_top5",
        "final_validation_loss",
        "normalized_top1_auc",
        "total_processed_images",
        "total_optimizer_updates",
        "training_runtime_seconds",
        "images_per_second",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "execution_mode",
        "semantic_emulation",
        "physical_fls_observed",
        "physical_gpu_pool",
        "layout_hash",
        "recipe_hash",
        "condition_hash",
        "initial_model_hash",
    )
    effect_fields = (
        "effect_id",
        "effect_class",
        "endpoint",
        "formula_coefficients",
        "paired_mean_difference",
        "paired_standard_deviation",
        "paired_seed_count",
        "ci95_low",
        "ci95_high",
        "individual_seed_points",
        "practical_interpretation",
    )
    _write_csv(output_dir / "convergence_curves.csv", curve_rows, curve_fields)
    _write_csv(output_dir / "final_metrics.csv", final_rows, final_fields)
    _write_csv(
        output_dir / "paired_effects.csv",
        [row for row in effects if row["effect_class"] == "simple"],
        effect_fields,
    )
    _write_csv(
        output_dir / "factorial_effects.csv",
        [row for row in effects if row["effect_class"] == "factorial"],
        effect_fields,
    )
    _write_csv(
        output_dir / "pointwise_condition_statistics.csv",
        pointwise,
        (
            "condition",
            "epoch",
            "processed_images_mean",
            "endpoint",
            "mean",
            "std",
            "n",
            "pointwise_ci95_low",
            "pointwise_ci95_high",
            "band_type",
        ),
    )
    _make_plots(output_dir, runs, final_rows, pointwise, effects)

    condition_summaries = {}
    for condition in conditions:
        condition_rows = [row for row in final_rows if row["condition"] == condition]
        condition_summaries[condition] = {
            endpoint: mean_ci([float(row[endpoint]) for row in condition_rows])
            for endpoint in (
                "final_top1",
                "final_top5",
                "final_validation_loss",
                "normalized_top1_auc",
            )
        }
    result = {
        "schema_version": REPORT_SCHEMA,
        "conditions": list(conditions),
        "expected_seeds": list(PAIRED_SEEDS),
        "expected_runs": len(conditions) * len(PAIRED_SEEDS),
        "completed_runs": len(completed),
        "missing_or_failed_runs": missing,
        "complete": not missing,
        "condition_summaries": condition_summaries,
        "final_metrics": final_rows,
        "effects": effects,
        "pointwise_confidence_band_semantics": "pointwise, not simultaneous",
        "strategy_selection_performed": False,
        "claim_boundary": (
            "Results, when present, are semantic-emulation model effects under the "
            "frozen virtual PLS mapping. They are not full physical-FLS byte-reduction evidence."
        ),
    }
    _write_json(output_dir / "model_results.json", result)

    lines = [
        "# Physical Load Segment core model-effect report",
        "",
        "## Core objective status",
        "",
        f"Completed formal runs: **{len(completed)}/{len(runs)}**.",
        "",
        "No condition was ranked, filtered, or selected using model or system metrics.",
        "",
        "## Final checkpoint metrics",
        "",
    ]
    if final_rows:
        lines.extend(
            _markdown_table(
                ("Condition", "Seed", "Top-1 (%)", "Top-5 (%)", "Val loss", "Normalized AUC"),
                [
                    (
                        row["condition"],
                        row["seed"],
                        f"{row['final_top1']:.4f}",
                        f"{row['final_top5']:.4f}",
                        f"{row['final_validation_loss']:.6f}",
                        f"{row['normalized_top1_auc']:.6f}",
                    )
                    for row in final_rows
                ],
            )
        )
    else:
        lines.append("No completed formal training run is available; no model metric is fabricated.")
    lines.extend(["", "## Paired final top-1 effects", ""])
    top1_effects = [row for row in effects if row["endpoint"] == "final_top1"]
    if top1_effects:
        lines.extend(
            _markdown_table(
                ("Effect", "n", "Mean pp", "SD", "95% t-CI", "Interpretation"),
                [
                    (
                        row["effect_id"],
                        row["paired_seed_count"],
                        "NA" if row["paired_mean_difference"] is None else f"{row['paired_mean_difference']:.4f}",
                        "NA" if row["paired_standard_deviation"] is None else f"{row['paired_standard_deviation']:.4f}",
                        (
                            "NA"
                            if row["ci95_low"] is None
                            else f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
                        ),
                        row["practical_interpretation"] or "insufficient paired seeds",
                    )
                    for row in top1_effects
                ],
            )
        )
    lines.extend(["", "## Convergence curves", ""])
    lines.extend(
        [
            "- `top1_convergence.png`: every seed (thin), condition mean (thick), and pointwise 95% t-CI.",
            "- `top5_convergence.png`: corresponding top-5 curves.",
            "- `train_loss_curves.png`: epoch training-loss curves against processed images.",
            "- `normalized_auc.png`: per-seed normalized top-1 AUC.",
            "",
            "Confidence bands are pointwise, not simultaneous.",
            "",
            "## System metrics",
            "",
            "Runtime, images/s, CUDA peak memory, loader wait, and data-preparation time are explanatory only and were not used as scientific gates.",
            "",
            "## Physical evidence and claim boundary",
            "",
            "The core runner records `semantic_emulation=true`, `physical_fls_observed=false`, and `physical_gpu_pool=false`. Therefore this report does not claim full physical block-major FLS execution or reduced compressed payload bytes.",
            "",
            "## Missing or failed runs",
            "",
        ]
    )
    if missing:
        lines.extend(
            f"- {item['condition']} seed {item['seed']}: {item['status'].get('state', 'missing')} ({item['path']})"
            for item in missing
        )
    else:
        lines.append("None.")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", default=",".join(CORE_CONDITION_IDS))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    conditions = tuple(
        value.strip().upper() for value in args.conditions.split(",") if value.strip()
    )
    unexpected = sorted(set(conditions) - set(CORE_CONDITION_IDS))
    if unexpected:
        raise ValueError(f"unknown core conditions: {unexpected}")
    result = aggregate(
        args.runs_root,
        args.output_dir,
        conditions=conditions,
        require_complete=args.require_complete,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
