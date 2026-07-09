#!/usr/bin/env python3
"""Summarize GALP/RGB-no-more benchmark JSON outputs.

Inputs can be JSON files written by the benchmark wrappers or log files that
contain lines starting with "RESULT_JSON ". The script emits a normalized CSV
or Markdown table for comparing loader and forward throughput across backends.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_COLUMNS = [
    "source",
    "backend",
    "phase",
    "model",
    "device",
    "manifest",
    "data_dir",
    "data_root",
    "index_file",
    "split",
    "preprocess",
    "rgb_preprocess",
    "dct_preprocess",
    "eval_transform",
    "dct_coeffs",
    "output_layout",
    "dequantize",
    "scale_to_rgbnomore_range",
    "dataset_size",
    "batch_size",
    "steps",
    "warmup",
    "workers",
    "prefetch_queue_depth",
    "images",
    "seconds",
    "images_per_s",
    "loss",
    "optimizer",
    "train_lr",
    "label_source",
    "label_map_json",
    "input_shape",
    "input_y_shape",
    "input_cbcr_shape",
    "logits_shape",
    "checkpoint",
    "selected_vectors",
    "full_vectors",
    "decode_kernels",
    "rowgroups",
    "worksets",
    "projection_items",
    "internal_syncs",
]


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        records.extend(item for item in payload if isinstance(item, dict))
    elif isinstance(payload, dict):
        records.append(payload)
    if records:
        return records

    for line in text.splitlines():
        marker = "RESULT_JSON "
        if marker not in line:
            continue
        _, json_text = line.split(marker, 1)
        payload = json.loads(json_text)
        if isinstance(payload, list):
            records.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            records.append(payload)
    if not records:
        raise RuntimeError(f"{path} does not contain JSON records or RESULT_JSON lines")
    return records


def _shape_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    return str(value)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return _shape_string(value)
    return str(value)


def _normalize_record(source: Path, record: dict[str, Any]) -> dict[str, Any]:
    normalized = {column: "" for column in DEFAULT_COLUMNS}
    normalized["source"] = str(source)
    for key in DEFAULT_COLUMNS:
        if key in record:
            normalized[key] = record[key]
    normalized["input_shape"] = record.get("input_shape", normalized["input_shape"])
    normalized["input_y_shape"] = record.get("input_y_shape", normalized["input_y_shape"])
    normalized["input_cbcr_shape"] = record.get("input_cbcr_shape", normalized["input_cbcr_shape"])
    normalized["logits_shape"] = record.get("logits_shape", normalized["logits_shape"])
    return normalized


def _read_all(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for record in _load_json_records(path):
            if "backend" not in record or "phase" not in record:
                continue
            rows.append(_normalize_record(path, record))
    if not rows:
        raise RuntimeError("no benchmark records with backend/phase were found")
    return rows


def _write_csv(rows: list[dict[str, Any]], columns: list[str], output: Path | None) -> None:
    stream = output.open("w", encoding="utf-8", newline="") if output is not None else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_value(row.get(column)) for column in columns})
    finally:
        if output is not None:
            stream.close()


def _write_markdown(rows: list[dict[str, Any]], columns: list[str], output: Path | None) -> None:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [_format_value(row.get(column)).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    text = "\n".join(lines) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GALP/RGB-no-more benchmark JSON outputs")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--format", choices=("csv", "markdown"), default="csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--columns",
        default=",".join(DEFAULT_COLUMNS),
        help="Comma-separated output columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    columns = [column.strip() for column in args.columns.split(",") if column.strip()]
    unknown = [column for column in columns if column not in DEFAULT_COLUMNS]
    if unknown:
        raise ValueError(f"unknown columns: {', '.join(unknown)}")
    rows = _read_all(args.inputs)
    rows.sort(key=lambda item: (str(item.get("backend", "")), str(item.get("phase", "")), str(item.get("source", ""))))
    if args.format == "csv":
        _write_csv(rows, columns, args.output)
    else:
        _write_markdown(rows, columns, args.output)


if __name__ == "__main__":
    main()
