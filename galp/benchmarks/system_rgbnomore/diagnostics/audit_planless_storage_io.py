#!/usr/bin/env python3
"""Audit Planless Direct-DCT persistent bytes, reader metadata, and measured I/O."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TORCH_BINDING_DIR = REPO_ROOT / "build/galp/torch"
if DEFAULT_TORCH_BINDING_DIR.is_dir() and str(DEFAULT_TORCH_BINDING_DIR) not in sys.path:
    sys.path.append(str(DEFAULT_TORCH_BINDING_DIR))
TORCH_SOURCE_DIR = REPO_ROOT / "galp/torch"
if str(TORCH_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_SOURCE_DIR))

import _galp_direct_dct as galp_dct
from rgbnomore_dct_profile import RGBNOMORE_VAL_DCT_GRID_TRANSFORM


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _rss_bytes() -> int:
    status = Path("/proc/self/status")
    if not status.is_file():
        return 0
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _fls_reader_open_metadata(path: Path) -> dict[str, int]:
    file_size = path.stat().st_size
    _require(file_size >= 48, f"truncated FLS container: {path}")
    with path.open("rb") as stream:
        header = stream.read(24)
        stream.seek(file_size - 24)
        footer = stream.read(24)
    _require(len(header) == 24 and len(footer) == 24, f"failed to read FLS framing: {path}")
    inline_footer = bool(header[16])
    descriptor_offset, descriptor_size, _magic = struct.unpack("<QQQ", footer)
    _require(inline_footer, f"storage audit requires inline FLS table descriptors: {path}")
    _require(
        descriptor_offset <= file_size
        and descriptor_size <= file_size - descriptor_offset
        and descriptor_offset + descriptor_size <= file_size - 24,
        f"invalid inline FLS table descriptor range: {path}",
    )
    return {
        "file_header_bytes": 24,
        "file_footer_bytes": 24,
        "table_descriptor_bytes": descriptor_size,
        "total_reader_open_bytes": 48 + descriptor_size,
    }


def _parse_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    data = path.read_bytes()
    offset = 0

    def take(fmt: str) -> tuple[int, ...]:
        nonlocal offset
        size = struct.calcsize(fmt)
        _require(offset + size <= len(data), f"truncated GALP shard manifest: {path}")
        values = struct.unpack_from(fmt, data, offset)
        offset += size
        return values

    def take_string() -> str:
        nonlocal offset
        (size,) = take("<I")
        _require(offset + size <= len(data), f"truncated GALP shard manifest string: {path}")
        value = data[offset : offset + size].decode("utf-8")
        offset += size
        return value

    _require(data[:8] == b"GJDCTSH1", f"unexpected GALP shard manifest format: {path}")
    offset = 8
    version, _reserved, rowgroup_vectors, rowgroups_per_shard, image_count = take("<IHIIQ")
    (shard_count,) = take("<I")
    shards: list[dict[str, Any]] = []
    root = path.parent
    for _ in range(shard_count):
        (
            shard_id,
            first_global_image_index,
            shard_image_count,
            real_row_count,
            padding_row_count,
            physical_row_count,
            rowgroup_count,
            block_group_count,
        ) = take("<IQIQQQII")
        fls_file_size, metadata_file_size = take("<QQ")
        fls_name = take_string()
        metadata_name = take_string()
        fls_path = (root / fls_name).resolve()
        metadata_path = (root / metadata_name).resolve()
        _require(fls_path.is_relative_to(root), f"FLS path escapes manifest root: {fls_name}")
        _require(metadata_path.is_relative_to(root), f"metadata path escapes manifest root: {metadata_name}")
        _require(fls_path.stat().st_size == fls_file_size, f"FLS size disagrees with manifest: {fls_path}")
        _require(
            metadata_path.stat().st_size == metadata_file_size,
            f"metadata size disagrees with manifest: {metadata_path}",
        )
        fls_reader_metadata = _fls_reader_open_metadata(fls_path)
        shards.append(
            {
                "shard_id": shard_id,
                "first_global_image_index": first_global_image_index,
                "image_count": shard_image_count,
                "real_row_count": real_row_count,
                "padding_row_count": padding_row_count,
                "physical_row_count": physical_row_count,
                "rowgroup_count": rowgroup_count,
                "block_group_count": block_group_count,
                "fls_file_size": fls_file_size,
                "metadata_file_size": metadata_file_size,
                "fls_file": str(fls_path),
                "metadata_file": str(metadata_path),
                "fls_reader_open_metadata": fls_reader_metadata,
            }
        )
    _require(offset == len(data), f"GALP shard manifest has trailing bytes: {path}")
    return {
        "path": str(path),
        "version": version,
        "rowgroup_vectors": rowgroup_vectors,
        "rowgroups_per_shard": rowgroups_per_shard,
        "image_count": image_count,
        "shards": shards,
        "manifest_bytes": len(data),
    }


def _persistent_audit(path: Path) -> dict[str, Any]:
    manifest = _parse_manifest(path)
    real_rows = sum(int(shard["real_row_count"]) for shard in manifest["shards"])
    fls_container_bytes = sum(int(shard["fls_file_size"]) for shard in manifest["shards"])
    jpeg_image_index_bytes = sum(int(shard["metadata_file_size"]) for shard in manifest["shards"])
    fls_table_descriptor_bytes = sum(
        int(shard["fls_reader_open_metadata"]["table_descriptor_bytes"])
        for shard in manifest["shards"]
    )
    fls_reader_open_metadata_bytes = sum(
        int(shard["fls_reader_open_metadata"]["total_reader_open_bytes"])
        for shard in manifest["shards"]
    )
    fls_format_framing_bytes = fls_reader_open_metadata_bytes - fls_table_descriptor_bytes
    compressed_coefficient_payload_bytes = (
        fls_container_bytes - fls_table_descriptor_bytes - fls_format_framing_bytes
    )
    index_bytes = jpeg_image_index_bytes + fls_table_descriptor_bytes
    manifest_bytes = int(manifest["manifest_bytes"])
    # Every real row carries all 64 signed 16-bit coefficient columns before compression.
    raw_dct_bytes = real_rows * 64 * 2
    execution_metadata_bytes = 0
    auxiliary_bytes = fls_format_framing_bytes
    total_bytes = (
        compressed_coefficient_payload_bytes
        + index_bytes
        + manifest_bytes
        + execution_metadata_bytes
        + auxiliary_bytes
    )
    return {
        "manifest": manifest["path"],
        "manifest_version": manifest["version"],
        "image_count": manifest["image_count"],
        "shard_count": len(manifest["shards"]),
        "real_coefficient_rows": real_rows,
        "raw_dct_bytes": raw_dct_bytes,
        "fls_container_bytes": fls_container_bytes,
        "compressed_coefficient_payload_bytes": compressed_coefficient_payload_bytes,
        "compressed_coefficient_payload_scope": "sum_of_all_compressed_FLS_rowgroup_records",
        "manifest_bytes": manifest_bytes,
        "index_bytes": index_bytes,
        "jpeg_image_index_bytes": jpeg_image_index_bytes,
        "fls_table_descriptor_bytes": fls_table_descriptor_bytes,
        "fls_reader_open_metadata_bytes": fls_reader_open_metadata_bytes,
        "fls_format_framing_bytes": fls_format_framing_bytes,
        "cold_reader_metadata_index_read_bytes": manifest_bytes + index_bytes + fls_format_framing_bytes,
        "execution_metadata_bytes": execution_metadata_bytes,
        "auxiliary_bytes": auxiliary_bytes,
        "total_persistent_bytes": total_bytes,
        "overall_compression_ratio_raw_to_total": raw_dct_bytes / total_bytes if total_bytes else 0.0,
    }


def _reader_audit(manifest: Path, batch_size: int) -> tuple[dict[str, Any], Any]:
    before_rss = _rss_bytes()
    opened = time.perf_counter()
    reader = galp_dct.DirectDctReader(str(manifest.resolve()))
    open_ms = (time.perf_counter() - opened) * 1000.0
    after_open_rss = _rss_bytes()
    count = min(batch_size, int(reader.image_count))
    preview = reader.plan_batch(
        list(range(count)),
        crop=None,
        dct_coeffs="all",
        cache_capacity_mib=0,
        layout="transformed_dct_grid",
        grid_transform=RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
        plan_cache_capacity=0,
    )
    keys = (
        "compact_reader_image_locator_bytes",
        "compact_reader_shard_index_bytes",
        "compact_reader_layout_dictionary_bytes",
        "compact_reader_quant_table_dictionary_bytes",
        "compact_reader_shard_descriptor_bytes",
        "compact_reader_total_bytes",
    )
    return {
        "reader_open_ms": open_ms,
        "process_rss_before_reader_bytes": before_rss,
        "process_rss_after_reader_bytes": after_open_rss,
        "process_rss_reader_delta_bytes": max(0, after_open_rss - before_rss),
        "process_rss_scope": "whole_process_delta_includes_general_manifest_and_metadata_objects",
        "compact_native_allocations": {key: int(preview[key]) for key in keys},
        "compact_shard_index_derived": bool(preview["compact_reader_shard_index_derived"]),
        "preview_planning_ms": float(preview["planning_ms"]),
    }, reader


def _scheduled_trace_audit(
    reader: Any,
    *,
    image_count: int,
    batch_size: int,
    trace: str,
    seed: int,
    cold_reader_metadata_index_bytes: int,
) -> dict[str, Any]:
    image_ids = list(range(image_count))
    if trace == "shuffled":
        random.Random(seed).shuffle(image_ids)
    total_bytes = 0
    total_rowgroups = 0
    for offset in range(0, image_count, batch_size):
        batch_ids = image_ids[offset : offset + batch_size]
        preview = reader.plan_batch(
            batch_ids,
            crop=None,
            dct_coeffs="all",
            cache_capacity_mib=0,
            layout="transformed_dct_grid",
            grid_transform=RGBNOMORE_VAL_DCT_GRID_TRANSFORM,
            plan_cache_capacity=0,
        )
        _require(bool(preview["uses_planless_fixed_transform"]), "storage audit did not use the planless path")
        by_shard: dict[int, list[int]] = {}
        for rowgroup in preview["rowgroups"]:
            by_shard.setdefault(int(rowgroup["shard_id"]), []).append(int(rowgroup["rowgroup_index"]))
        for shard_id, rowgroup_indices in by_shard.items():
            total_bytes += int(reader.rowgroup_storage_bytes(shard_id, rowgroup_indices))
        total_rowgroups += len(preview["rowgroups"])
    total_storage_read_bytes = total_bytes + cold_reader_metadata_index_bytes
    return {
        "trace": trace,
        "seed": seed if trace == "shuffled" else None,
        "images": image_count,
        "batches": (image_count + batch_size - 1) // batch_size,
        "rowgroups": total_rowgroups,
        "rowgroup_storage_bytes": total_bytes,
        "coefficient_payload_bytes_read": total_bytes,
        "metadata_index_bytes_read": cold_reader_metadata_index_bytes,
        "total_storage_read_bytes": total_storage_read_bytes,
        "coefficient_bytes_per_image": total_bytes / image_count if image_count else 0.0,
        "total_bytes_per_image": total_storage_read_bytes / image_count if image_count else 0.0,
        "scope": (
            "exact_scheduled_FLS_rowgroup_records_plus_one_cold_read_of_manifest_JPEG_indexes_"
            "and_each_selected_shard_FLS_header_footer_inline_descriptor"
        ),
    }


def _counter_values(result_path: Path) -> list[int]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for records_key in ("repeat_records", "repeats"):
            repeat_records = payload.get(records_key)
            if isinstance(repeat_records, list):
                values = [
                    int(record.get("native_counters", {}).get("rowgroup_storage_bytes_read", 0))
                    for record in repeat_records
                    if isinstance(record, dict)
                ]
                if values:
                    return values
    if isinstance(payload, dict) and "rowgroup_storage_bytes_read" in payload:
        return [int(payload["rowgroup_storage_bytes_read"])]
    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, list):
        values = [
            int(item["rowgroup_storage_bytes_read"])
            for item in results
            if isinstance(item, dict) and "rowgroup_storage_bytes_read" in item
        ]
        if values:
            return values
    raise RuntimeError(f"result has no native rowgroup_storage_bytes_read counter: {result_path}")


def _trace_audit(
    candidate: Path | None,
    baseline: Path | None,
    *,
    candidate_metadata_index_bytes: int,
    baseline_metadata_index_bytes: int,
    baseline_scheduled_rowgroup_bytes: int,
) -> dict[str, Any] | None:
    if candidate is None and baseline is None:
        return None
    _require(candidate is not None, "trace comparison requires a candidate execution result")
    candidate_values = _counter_values(candidate)
    baseline_values = (
        _counter_values(baseline) if baseline is not None else [baseline_scheduled_rowgroup_bytes]
    )
    candidate_median = float(statistics.median(candidate_values))
    baseline_median = float(statistics.median(baseline_values))
    candidate_total = candidate_median + candidate_metadata_index_bytes
    baseline_total = baseline_median + baseline_metadata_index_bytes
    return {
        "candidate_result": str(candidate.resolve()),
        "baseline_result": str(baseline.resolve()) if baseline is not None else None,
        "baseline_source": "native_execution_counter" if baseline is not None else "exact_scheduled_rowgroup_bytes",
        "candidate_rowgroup_storage_bytes_read": candidate_values,
        "baseline_rowgroup_storage_bytes_read": baseline_values,
        "candidate_median_bytes": candidate_median,
        "baseline_median_bytes": baseline_median,
        "rowgroup_read_amplification_ratio": candidate_median / baseline_median if baseline_median else None,
        "candidate_cold_metadata_index_bytes": candidate_metadata_index_bytes,
        "baseline_cold_metadata_index_bytes": baseline_metadata_index_bytes,
        "candidate_cold_total_storage_read_bytes": candidate_total,
        "baseline_cold_total_storage_read_bytes": baseline_total,
        "read_amplification_ratio": candidate_total / baseline_total if baseline_total else None,
        "counter_scope": (
            "native_actual_compressed_rowgroup_pread_bytes_plus_exact_cold_reader_metadata_index_bytes"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_manifest", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dataset-size", type=int, default=50000)
    parser.add_argument("--shuffle-seed", type=int, default=20260718)
    parser.add_argument("--candidate-sequential-result", type=Path)
    parser.add_argument("--baseline-sequential-result", type=Path)
    parser.add_argument("--candidate-shuffled-result", type=Path)
    parser.add_argument("--baseline-shuffled-result", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--require-measured-execution",
        action="store_true",
        help="Fail unless sequential and shuffled candidate/baseline execution counters are present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require(args.batch_size > 0, "--batch-size must be positive")
    baseline_manifest = args.baseline_manifest or args.candidate_manifest
    candidate = _persistent_audit(args.candidate_manifest)
    baseline = _persistent_audit(baseline_manifest)
    reader_audit, candidate_reader = _reader_audit(args.candidate_manifest, args.batch_size)
    image_count = min(args.dataset_size, int(candidate["image_count"]), int(baseline["image_count"]))
    _require(image_count > 0, "audit dataset is empty")
    same_manifest = args.candidate_manifest.resolve() == baseline_manifest.resolve()
    if same_manifest:
        baseline_reader = candidate_reader
    else:
        _, baseline_reader = _reader_audit(baseline_manifest, args.batch_size)
    persistent_ratio = candidate["total_persistent_bytes"] / baseline["total_persistent_bytes"]
    execution_metadata_ratio = (
        candidate["execution_metadata_bytes"] / candidate["compressed_coefficient_payload_bytes"]
        if candidate["compressed_coefficient_payload_bytes"]
        else 0.0
    )
    traces: dict[str, Any] = {}
    for trace_name, candidate_result, baseline_result in (
        ("sequential", args.candidate_sequential_result, args.baseline_sequential_result),
        ("shuffled", args.candidate_shuffled_result, args.baseline_shuffled_result),
    ):
        candidate_scheduled = _scheduled_trace_audit(
            candidate_reader,
            image_count=image_count,
            batch_size=args.batch_size,
            trace=trace_name,
            seed=args.shuffle_seed,
            cold_reader_metadata_index_bytes=int(candidate["cold_reader_metadata_index_read_bytes"]),
        )
        baseline_scheduled = (
            dict(candidate_scheduled)
            if same_manifest
            else _scheduled_trace_audit(
                baseline_reader,
                image_count=image_count,
                batch_size=args.batch_size,
                trace=trace_name,
                seed=args.shuffle_seed,
                cold_reader_metadata_index_bytes=int(baseline["cold_reader_metadata_index_read_bytes"]),
            )
        )
        scheduled_ratio = (
            candidate_scheduled["total_storage_read_bytes"] / baseline_scheduled["total_storage_read_bytes"]
            if baseline_scheduled["total_storage_read_bytes"]
            else None
        )
        traces[trace_name] = {
            "scheduled": {
                "candidate": candidate_scheduled,
                "baseline": baseline_scheduled,
                "read_amplification_ratio": scheduled_ratio,
            },
            "measured_execution": _trace_audit(
                candidate_result,
                baseline_result,
                candidate_metadata_index_bytes=int(candidate["cold_reader_metadata_index_read_bytes"]),
                baseline_metadata_index_bytes=int(baseline["cold_reader_metadata_index_read_bytes"]),
                baseline_scheduled_rowgroup_bytes=int(baseline_scheduled["rowgroup_storage_bytes"]),
            ),
        }
    scheduled_read_gate = all(
        trace["scheduled"]["read_amplification_ratio"] is not None
        and trace["scheduled"]["read_amplification_ratio"] <= 1.03
        for trace in traces.values()
    )
    measured_traces = [trace["measured_execution"] for trace in traces.values()]
    measured_execution_read_gate: bool | None = None
    if all(trace is not None for trace in measured_traces):
        measured_execution_read_gate = all(
            trace["read_amplification_ratio"] is not None
            and trace["read_amplification_ratio"] <= 1.03
            for trace in measured_traces
            if trace is not None
        )
    gate_results: dict[str, bool | None] = {
        "persistent_bytes": persistent_ratio <= 1.02,
        "execution_metadata": execution_metadata_ratio <= 0.01,
        "scheduled_sequential_and_shuffled_read_amplification": scheduled_read_gate,
        "measured_execution_sequential_and_shuffled_read_amplification": measured_execution_read_gate,
    }
    complete = all(value is not None for value in gate_results.values())
    passed = complete and all(bool(value) for value in gate_results.values())
    payload = {
        "schema_version": 1,
        "candidate": candidate,
        "baseline": baseline,
        "reader_open_and_compact_native_memory": reader_audit,
        "traces": traces,
        "ratios": {
            "candidate_to_baseline_total_persistent_bytes": persistent_ratio,
            "execution_metadata_to_compressed_payload": execution_metadata_ratio,
            "compact_reader_native_to_compressed_payload": (
                reader_audit["compact_native_allocations"]["compact_reader_total_bytes"]
                / candidate["compressed_coefficient_payload_bytes"]
            ),
        },
        "gates": {
            "candidate_to_baseline_total_persistent_bytes_max": 1.02,
            "execution_metadata_to_compressed_payload_max": 0.01,
            "candidate_to_baseline_read_bytes_max": 1.03,
        },
        "gate_results": {**gate_results, "complete": complete, "passed": passed},
    }
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")
    print(output)
    if args.require_measured_execution and not complete:
        return 1
    return 0 if all(value is not False for value in gate_results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
