#!/usr/bin/env python3
"""Summarize galp_cli pipeline_benchmark output as Markdown or CSV."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable, TextIO


DEFAULT_COLUMNS = [
    "dataset",
    "image_size",
    "crop_size",
    "mode",
    "outputs_match",
    "pushdown_selected_vector_ratio",
    "full_then_crop_selected_vector_ratio",
    "pushdown_total_ms",
    "full_then_crop_total_ms",
    "pushdown_speedup_vs_full_then_crop",
    "pushdown_saved_ms_vs_full_then_crop",
    "pushdown_plan_ms",
    "pushdown_read_decode_ms",
    "pushdown_decode_ms",
    "pushdown_gather_ms",
    "pushdown_workset_count",
    "pushdown_decode_kernel_launch_count",
    "pushdown_gather_kernel_launch_count",
    "pushdown_scratch_allocation_count",
    "pushdown_internal_sync_count",
    "pushdown_runtime_policy_decision",
]


KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z0-9_]+):\s*(.*?)\s*$")
HEADER = "Pipeline benchmark results:"


def read_text(path: str | None) -> tuple[str, str]:
    if path is None or path == "-":
        return "stdin", sys.stdin.read()
    return Path(path).stem, Path(path).read_text(encoding="utf-8")


def parse_blocks(text: str, dataset: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for line in text.splitlines():
        if HEADER in line:
            if current:
                blocks.append(current)
            current = {"dataset": dataset}
            continue
        if current is None:
            continue
        match = KEY_VALUE_RE.match(line)
        if not match:
            continue
        key, value = match.groups()
        current[key] = value

    if current:
        blocks.append(current)
    return blocks


def parse_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def derive_crop_size(crop: str) -> str | None:
    parts = [part.strip() for part in crop.split(",")]
    if len(parts) != 4:
        return None
    width, height = parts[2], parts[3]
    if width == "" or height == "":
        return None
    return f"{width}x{height}"


def add_derived_fields(row: dict[str, str]) -> None:
    if "crop_size" not in row and "crop" in row:
        crop_size = derive_crop_size(row["crop"])
        if crop_size is not None:
            row["crop_size"] = crop_size

    push_total = parse_float(row, "pushdown_total_ms")
    full_total = parse_float(row, "full_then_crop_total_ms")
    if push_total is None or full_total is None:
        return
    if "pushdown_speedup_vs_full_then_crop" not in row and push_total > 0.0:
        row["pushdown_speedup_vs_full_then_crop"] = f"{full_total / push_total:.6g}"
    if "pushdown_saved_ms_vs_full_then_crop" not in row:
        row["pushdown_saved_ms_vs_full_then_crop"] = f"{full_total - push_total:.6g}"


def split_labels(value: str | None) -> list[str] | None:
    if value is None:
        return None
    labels = [item.strip() for item in value.split(",")]
    return [label for label in labels if label != ""]


def validate_label_options(dataset: str | None, datasets: list[str] | None, image_size: str | None) -> None:
    if dataset is not None and datasets is not None:
        raise ValueError("use either --dataset or --datasets, not both")
    if image_size is not None and image_size == "":
        raise ValueError("--image-size must not be empty")


def mismatched_rows(rows: list[dict[str, str]]) -> list[str]:
    failed: list[str] = []
    for idx, row in enumerate(rows, start=1):
        value = row.get("outputs_match")
        if value != "1":
            failed.append(f"{row.get('dataset', f'row_{idx}')}: outputs_match={value if value is not None else 'missing'}")
    return failed


def missing_required_fields(rows: list[dict[str, str]], fields: list[str]) -> list[str]:
    missing: list[str] = []
    for idx, row in enumerate(rows, start=1):
        dataset = row.get("dataset", f"row_{idx}")
        for field in fields:
            if row.get(field) in (None, ""):
                missing.append(f"{dataset}: missing {field}")
    return missing


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


def write_markdown(rows: list[dict[str, str]], columns: list[str], out: TextIO) -> None:
    out.write("| " + " | ".join(columns) + " |\n")
    out.write("| " + " | ".join("---" for _ in columns) + " |\n")
    for row in rows:
        out.write("| " + " | ".join(markdown_escape(row.get(column, "")) for column in columns) + " |\n")


def write_csv(rows: list[dict[str, str]], columns: list[str], out: TextIO) -> None:
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def collect_rows(
    inputs: Iterable[str],
    dataset: str | None,
    image_size: str | None,
    datasets: list[str] | None,
    image_sizes: list[str] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    input_list = list(inputs)
    if not input_list:
        input_list = ["-"]
    block_index = 0
    for path in input_list:
        inferred_dataset, text = read_text(path)
        current_dataset = dataset if dataset is not None else inferred_dataset
        blocks = parse_blocks(text, current_dataset)
        if len(blocks) > 1 and dataset is None and path == "-":
            for idx, block in enumerate(blocks, start=1):
                block["dataset"] = f"{current_dataset}_{idx}"
        for block in blocks:
            if datasets is not None and block_index < len(datasets):
                block["dataset"] = datasets[block_index]
            if image_sizes is not None and block_index < len(image_sizes):
                block["image_size"] = image_sizes[block_index]
            elif image_size is not None and "image_size" not in block:
                block["image_size"] = image_size
            add_derived_fields(block)
            rows.append(block)
            block_index += 1
    if datasets is not None and len(datasets) != block_index:
        raise ValueError(f"--datasets has {len(datasets)} labels but found {block_index} result blocks")
    if image_sizes is not None and len(image_sizes) != block_index:
        raise ValueError(f"--image-sizes has {len(image_sizes)} labels but found {block_index} result blocks")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="pipeline_benchmark output files, or '-' / stdin")
    parser.add_argument("--dataset", help="dataset label to use instead of file stem")
    parser.add_argument("--image-size", help="image size label, for example 224x224 or varies")
    parser.add_argument("--datasets", help="comma-separated dataset labels for result blocks")
    parser.add_argument("--image-sizes", help="comma-separated image size labels for result blocks")
    parser.add_argument("--require-match", action="store_true", help="fail if any result block has outputs_match != 1")
    parser.add_argument("--require-default-fields", action="store_true", help="fail if any default report field is missing")
    parser.add_argument("--require-fields", help="comma-separated field names that must be present in every result block")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--columns", help="comma-separated output column list")
    args = parser.parse_args()

    columns = DEFAULT_COLUMNS if args.columns is None else [column.strip() for column in args.columns.split(",")]
    datasets = split_labels(args.datasets)
    image_sizes = split_labels(args.image_sizes)
    try:
        validate_label_options(args.dataset, datasets, args.image_size)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        rows = collect_rows(
            args.inputs,
            args.dataset,
            args.image_size,
            datasets,
            image_sizes,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not rows:
        print("no pipeline_benchmark result blocks found", file=sys.stderr)
        return 1
    if args.require_match:
        failures = mismatched_rows(rows)
        if failures:
            print("pipeline benchmark output mismatch detected:", file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
            return 2
    required_fields = DEFAULT_COLUMNS if args.require_default_fields else []
    extra_required_fields = split_labels(args.require_fields)
    if extra_required_fields is not None:
        required_fields = [*required_fields, *extra_required_fields]
    if required_fields:
        missing = missing_required_fields(rows, required_fields)
        if missing:
            print("pipeline benchmark output is missing required fields:", file=sys.stderr)
            for item in missing:
                print(f"  {item}", file=sys.stderr)
            return 3

    if args.format == "csv":
        write_csv(rows, columns, sys.stdout)
    else:
        write_markdown(rows, columns, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
