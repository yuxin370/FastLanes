#!/usr/bin/env python3
"""Artifact writers for PLS schedule experiments."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schedule import PoolWave, ScheduleResult, batch_records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False))
            stream.write("\n")
    os.replace(temporary, path)


def _pool_record(pool: PoolWave, *, include_samples: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "epoch": pool.epoch,
        "pool_index": pool.pool_index,
        "segment_ids": list(pool.segment_ids),
        "source_shard_ids": list(pool.source_shard_ids),
        "sample_count": pool.size,
        "unique_crop_keys": len(set(pool.crop_keys)),
    }
    if include_samples:
        result.update(
            samples_before_shuffle=list(pool.samples_before_shuffle),
            samples_after_shuffle=list(pool.samples_after_shuffle),
            crop_keys=list(pool.crop_keys),
        )
    return result


def write_schedule_artifacts(
    output_dir: Path,
    result: ScheduleResult,
    *,
    manifest_record: Mapping[str, Any],
    trace_detail: str,
) -> None:
    if trace_detail not in ("digests", "full"):
        raise ValueError("trace_detail must be digests or full")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        output_dir / "contract.json",
        {
            "schema_version": "galp-pls-schedule-contract-v1",
            "manifest": dict(manifest_record),
            "config": result.config.as_dict(),
            "trace_detail": trace_detail,
        },
    )
    write_json(output_dir / "summary.json", result.summary)
    write_jsonl(
        output_dir / "segment_trace.jsonl",
        (
            {
                "segment_id": segment.segment_id,
                "sample_count": segment.size,
                "source_shard_ids": list(segment.source_shard_ids),
                "sample_ids": (
                    [sample.logical_sample_id for sample in segment.samples]
                    if trace_detail == "full"
                    else None
                ),
            }
            for segment in result.segments
        ),
    )
    write_jsonl(
        output_dir / "pool_trace.jsonl",
        (_pool_record(pool, include_samples=trace_detail == "full") for pool in result.pools),
    )
    if trace_detail == "full":
        write_jsonl(
            output_dir / "sample_trace.jsonl",
            (
                {
                    "epoch": value.epoch,
                    "epoch_position": value.epoch_position,
                    "pool_index": value.pool_index,
                    "logical_sample_id": value.sample.logical_sample_id,
                    "label": value.sample.label,
                    "galp_image_id": value.sample.galp_image_id,
                    "segment_id": value.segment_id,
                    "source_shard_id": value.source_shard_id,
                    "crop_key": value.crop_key,
                }
                for epoch in result.epochs
                for value in epoch
            ),
        )
    batch_values = list(batch_records(result))
    with (output_dir / "batch_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(batch_values[0]) if batch_values else ["batch_index"])
        writer.writeheader()
        writer.writerows(batch_values)
    (output_dir / "report.md").write_text(render_schedule_report(result.summary), encoding="utf-8")


def render_schedule_report(summary: Mapping[str, Any]) -> str:
    config = summary["config"]
    coverage = summary["epoch_coverage"]
    lines = [
        "# Physical-load-segment schedule report",
        "",
        "## Scope",
        "",
        "This artifact reports the realized schedule. Mixing statistics are explanatory observations, not pass/fail gates.",
        "",
        "## Configuration",
        "",
        f"- Organization: `{config['organization']}`",
        f"- Fixed organization seed: `{config['organization_seed']}`",
        f"- Order: `{config['order_policy']}`",
        f"- Crop: `{config['crop_policy']}`",
        f"- Images per PLS: `{config['segment_images']}`",
        f"- Complete PLSs per pool: `{config['segments_per_pool']}`",
        f"- Optimizer batch: `{config['optimizer_batch_size']}`",
        "",
        "## Realized structure",
        "",
        f"- Samples: `{summary['sample_count']}`; classes: `{summary['class_count']}`",
        f"- PLSs: `{summary['segment_count']}`; pools: `{summary['pool_count']}`",
        f"- Pool-size mean/p50/max: `{summary['pool_size'].get('mean')}` / `{summary['pool_size'].get('p50')}` / `{summary['pool_size'].get('max')}`",
        f"- Optimizer batches: `{summary['optimizer_batch_count']}`",
        "",
        "## Observed batch composition",
        "",
        f"- Unique classes mean/p50: `{summary['unique_classes_per_batch'].get('mean')}` / `{summary['unique_classes_per_batch'].get('p50')}`",
        f"- Label entropy mean/p50: `{summary['label_entropy_bits_per_batch'].get('mean')}` / `{summary['label_entropy_bits_per_batch'].get('p50')}` bits",
        f"- Unique source shards mean/p50: `{summary['unique_source_shards_per_batch'].get('mean')}` / `{summary['unique_source_shards_per_batch'].get('p50')}`",
        f"- Unique crop keys mean/p50: `{summary['unique_crop_keys_per_batch'].get('mean')}` / `{summary['unique_crop_keys_per_batch'].get('p50')}`",
        f"- Single-class batches: `{summary['single_class_batch_count']}`",
        "",
        "## Coverage validity",
        "",
    ]
    for item in coverage:
        lines.append(
            f"- Epoch {item['epoch']}: samples `{item['scheduled_samples']}`, duplicates `{item['duplicate_samples']}`, "
            f"missing `{item['missing_samples']}`, complete `{item['complete']}`"
        )
    lines.extend(
        [
            "",
            "No claim about convergence or final accuracy is made by this schedule-only artifact. Those require the paired model-training runs in the pre-registered matrix.",
            "",
        ]
    )
    return "\n".join(lines)
