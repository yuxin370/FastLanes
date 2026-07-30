#!/usr/bin/env python3
"""Run and validate a same-contract three-way GALP crop I/O comparison."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from shared.common import load_contract, sha256_json, source_tree_metadata, write_json


MODES: tuple[tuple[str, str], ...] = (
    ("full_decode", "full-rowgroup-decode"),
    ("crop_rowgroup", "rowgroup-read-selected-decode"),
    ("crop_vector", "vector-range-read-selected-decode"),
)
REQUIRED_COUNTERS = (
    "planned_vector_count",
    "actual_vector_count",
    "full_vector_count",
    "compressed_payload_bytes_read",
    "full_compressed_payload_bytes",
    "pread_count",
    "vector_bundle_rowgroup_count",
    "vector_bundle_envelope_rowgroup_count",
    "vector_bundle_pread_count",
    "requested_source_block_count",
    "source_blocks_transformed",
    "sparse_read_fallback_rowgroup_count",
)


def _require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def _load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result is not an object: {path}")
    return payload


def _first_repeat(result: dict[str, Any], mode: str, failures: list[str]) -> dict[str, Any]:
    repeats = result.get("repeats", [])
    _require(isinstance(repeats, list) and bool(repeats), failures, f"{mode}: repeat records are missing")
    return repeats[0] if isinstance(repeats, list) and repeats else {}


def _artifact_arrays(result: dict[str, Any]) -> dict[str, np.ndarray]:
    artifact = Path(str(result.get("semantic_artifact", "")))
    if not artifact.is_file():
        raise FileNotFoundError(f"semantic artifact is missing: {artifact}")
    with np.load(artifact, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files if name != "metadata_json"}


def _max_abs(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape:
        return float("inf")
    if lhs.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))))


def _mode_metrics(result: dict[str, Any], mode: str, failures: list[str]) -> dict[str, Any]:
    record = _first_repeat(result, mode, failures)
    counters = record.get("native_counters", {})
    properties = record.get("native_properties", {})
    stages = record.get("stage_breakdown_ms", {}).get("native_totals_seconds", {})
    _require(isinstance(counters, dict), failures, f"{mode}: native counters are missing")
    _require(isinstance(properties, dict), failures, f"{mode}: native properties are missing")
    counters = counters if isinstance(counters, dict) else {}
    properties = properties if isinstance(properties, dict) else {}
    for field in REQUIRED_COUNTERS:
        _require(field in counters, failures, f"{mode}: required counter {field} is missing")
    decode_ms = float(stages.get("decode_seconds", 0.0)) * 1000.0 if isinstance(stages, dict) else 0.0
    transform_ms = 0.0
    if isinstance(stages, dict):
        transform_ms = 1000.0 * (
            float(stages.get("device_mapping_seconds", 0.0))
            + float(stages.get("fixed_transform_kernel_seconds", 0.0))
        )
    rowgroup_count = int(counters.get("rowgroups", counters.get("rowgroup_count", 0)))
    sparse_fallback_count = int(counters.get("sparse_read_fallback_rowgroup_count", 0))
    return {
        "crop_execution_mode": result.get("pipeline_config", {}).get("crop_execution_mode"),
        "storage_read_granularity": properties.get("storage_read_granularity"),
        "decode_granularity": properties.get("decode_granularity"),
        "planned_vector_count": int(counters.get("planned_vector_count", 0)),
        "actual_vector_count": int(counters.get("actual_vector_count", 0)),
        "full_vector_count": int(counters.get("full_vector_count", 0)),
        "compressed_payload_bytes_read": int(counters.get("compressed_payload_bytes_read", 0)),
        "full_compressed_payload_bytes": int(counters.get("full_compressed_payload_bytes", 0)),
        "pread_count": int(counters.get("pread_count", 0)),
        "vector_bundle_rowgroup_count": int(counters.get("vector_bundle_rowgroup_count", 0)),
        "vector_bundle_envelope_rowgroup_count": int(
            counters.get("vector_bundle_envelope_rowgroup_count", 0)
        ),
        "vector_bundle_pread_count": int(counters.get("vector_bundle_pread_count", 0)),
        "read_amplification": float(properties.get("read_amplification", 0.0)),
        "requested_source_block_count": int(counters.get("requested_source_block_count", 0)),
        "source_blocks_transformed": int(counters.get("source_blocks_transformed", 0)),
        "sparse_read_supported": properties.get("sparse_read_supported", False),
        "sparse_read_fallback_reason": properties.get("sparse_read_fallback_reason", ""),
        "sparse_read_fallback_rowgroup_count": sparse_fallback_count,
        "sparse_read_fallback_rowgroup_ratio": (
            float(sparse_fallback_count / rowgroup_count) if rowgroup_count > 0 else 0.0
        ),
        "decode_ms": decode_ms,
        "transform_ms": transform_ms,
        "end_to_end_seconds": float(record.get("seconds", 0.0)),
        "throughput_images_per_s": float(record.get("throughput_images_per_s", 0.0)),
        "correct_top1": int(record.get("correct_top1", -1)),
        "correct_top5": int(record.get("correct_top5", -1)),
        "images": int(record.get("images", 0)),
    }


def validate_crop_io_ab_results(
    base_contract: dict[str, Any],
    results: dict[str, dict[str, Any]],
    *,
    semantic_tolerance: float = 1.2e-7,
) -> dict[str, Any]:
    failures: list[str] = []
    metrics = {mode: _mode_metrics(results[mode], mode, failures) for mode, _ in MODES}
    reference_result = results["full_decode"]
    base_galp_config = base_contract["pipelines"]["galp"]
    _require(
        int(base_galp_config.get("cache_capacity_mib", -1)) == 0,
        failures,
        "base contract: decoded-rowgroup cache must be zero",
    )
    _require(
        int(base_galp_config.get("plan_cache_capacity", -1)) == 0,
        failures,
        "base contract: exact-batch plan cache must be zero",
    )
    for mode, execution_mode in MODES:
        result = results[mode]
        _require(result.get("pipeline") == "galp", failures, f"{mode}: result is not the GALP pipeline")
        config = copy.deepcopy(result.get("pipeline_config", {}))
        observed_mode = config.pop("crop_execution_mode", None)
        _require(observed_mode == execution_mode, failures, f"{mode}: crop execution mode is {observed_mode!r}")
        expected_config = copy.deepcopy(base_contract["pipelines"]["galp"])
        expected_config.pop("crop_execution_mode", None)
        _require(config == expected_config, failures, f"{mode}: non-mode GALP contract fields changed")
        _require(result.get("execution") == reference_result.get("execution"), failures, f"{mode}: execution contract changed")
        _require(result.get("model") == reference_result.get("model"), failures, f"{mode}: model/checkpoint changed")
        _require(
            result.get("sample_manifest_sha256") == reference_result.get("sample_manifest_sha256"),
            failures,
            f"{mode}: sample manifest changed",
        )
        _require(
            _first_repeat(result, mode, failures).get("sample_trace")
            == _first_repeat(reference_result, "full_decode", failures).get("sample_trace"),
            failures,
            f"{mode}: measured sample trace changed",
        )

    full = metrics["full_decode"]
    rowgroup = metrics["crop_rowgroup"]
    vector = metrics["crop_vector"]
    _require(full["actual_vector_count"] == full["full_vector_count"] > 0, failures, "full_decode: decode was not full-rowgroup")
    _require(full["storage_read_granularity"] == "rowgroup", failures, "full_decode: storage granularity is not rowgroup")
    for mode, item in (("crop_rowgroup", rowgroup), ("crop_vector", vector)):
        _require(
            0 < item["planned_vector_count"] < item["full_vector_count"],
            failures,
            f"{mode}: planned vectors are not a strict non-empty subset",
        )
        _require(
            0 < item["actual_vector_count"] < item["full_vector_count"],
            failures,
            f"{mode}: actual decoded vectors are not a strict non-empty subset",
        )
        _require(
            item["decode_granularity"] in {"selected-vector", "mixed"},
            failures,
            f"{mode}: selected decode was not reported",
        )
    _require(
        rowgroup["compressed_payload_bytes_read"] == rowgroup["full_compressed_payload_bytes"] > 0,
        failures,
        "crop_rowgroup: physical bytes do not equal the full-rowgroup baseline",
    )
    _require(rowgroup["storage_read_granularity"] == "rowgroup", failures, "crop_rowgroup: storage granularity is not rowgroup")
    _require(
        rowgroup["compressed_payload_bytes_read"] == full["compressed_payload_bytes_read"],
        failures,
        "crop_rowgroup: physical bytes differ from full_decode despite the same rowgroup reads",
    )
    _require(
        full["source_blocks_transformed"]
        == rowgroup["source_blocks_transformed"]
        == vector["source_blocks_transformed"]
        and full["source_blocks_transformed"] > 0,
        failures,
        "three-way A/B changed transform source blocks; storage-read reduction cannot be isolated",
    )
    _require(
        full["requested_source_block_count"]
        == rowgroup["requested_source_block_count"]
        == vector["requested_source_block_count"]
        and full["requested_source_block_count"] > 0,
        failures,
        "three-way A/B changed requested source blocks",
    )
    _require(
        0 < vector["compressed_payload_bytes_read"] < vector["full_compressed_payload_bytes"],
        failures,
        "crop_vector: physical compressed bytes were not reduced",
    )
    _require(
        vector["compressed_payload_bytes_read"] < rowgroup["compressed_payload_bytes_read"],
        failures,
        "crop_vector: physical bytes were not lower than crop_rowgroup",
    )
    _require(
        vector["storage_read_granularity"]
        in {"selected-vector-range", "vector-bundle-range", "vector-bundle-envelope", "mixed"},
        failures,
        "crop_vector: selected-vector-range storage was not reported",
    )
    _require(
        vector["sparse_read_supported"] in {True, "mixed"},
        failures,
        "crop_vector: sparse-read capability was not reported",
    )

    arrays = {mode: _artifact_arrays(results[mode]) for mode, _ in MODES}
    semantic_diffs: dict[str, dict[str, float]] = {}
    for candidate in ("crop_rowgroup", "crop_vector"):
        common = sorted(set(arrays["full_decode"]).intersection(arrays[candidate]))
        numeric_semantic = [key for key in common if key.startswith("input_") or key == "logits"]
        diffs = {key: _max_abs(arrays["full_decode"][key], arrays[candidate][key]) for key in numeric_semantic}
        semantic_diffs[candidate] = diffs
        for key, error in diffs.items():
            _require(error <= semantic_tolerance, failures, f"{candidate}: {key} max_abs={error} exceeds {semantic_tolerance}")

    reference_predictions = arrays["full_decode"].get("top1_predictions", np.asarray([], dtype=np.int64))
    expected_prediction_count = int(base_contract["semantic_validation"]["prediction_agreement_sample_count"])
    _require(
        expected_prediction_count == 50_000,
        failures,
        f"base contract: Top-1 agreement set must contain 50000 samples, configured {expected_prediction_count}",
    )
    _require(
        reference_predictions.size == expected_prediction_count,
        failures,
        f"full_decode: expected {expected_prediction_count} predictions, found {reference_predictions.size}",
    )
    agreements: dict[str, float] = {}
    for candidate in ("crop_rowgroup", "crop_vector"):
        observed = arrays[candidate].get("top1_predictions", np.asarray([], dtype=np.int64))
        agreement = (
            float(np.mean(reference_predictions == observed))
            if observed.shape == reference_predictions.shape and reference_predictions.size
            else 0.0
        )
        agreements[candidate] = agreement
        _require(agreement == 1.0, failures, f"{candidate}: full-set Top-1 agreement is {agreement}")
        reference_counts = [
            (record.get("correct_top1"), record.get("correct_top5"))
            for record in reference_result.get("repeats", [])
        ]
        candidate_counts = [
            (record.get("correct_top1"), record.get("correct_top5"))
            for record in results[candidate].get("repeats", [])
        ]
        _require(candidate_counts == reference_counts, failures, f"{candidate}: correct counts differ from full_decode")

    observed_sources: dict[str, Any] = {}
    for name, expected in base_contract.get("source_revisions", {}).items():
        files = expected.get("runtime_file_sha256", {})
        if not isinstance(files, dict):
            failures.append(f"source {name}: runtime file hashes are missing")
            continue
        observed = source_tree_metadata(Path(str(expected.get("root", ""))), list(files))
        observed_sources[name] = observed
        _require(expected.get("benchmark_source_clean") is True, failures, f"source {name}: base contract was not clean")
        _require(observed.get("benchmark_source_clean") is True, failures, f"source {name}: runtime sources are dirty")
        _require(observed.get("git_commit") == expected.get("git_commit"), failures, f"source {name}: git commit changed")
        _require(
            observed.get("runtime_file_sha256") == files,
            failures,
            f"source {name}: runtime source hashes changed",
        )

    return {
        "schema_version": "galp_crop_io_ab_v1",
        "ok": not failures,
        "failures": failures,
        "base_contract_sha256": sha256_json(base_contract),
        "semantic_tolerance": semantic_tolerance,
        "semantic_max_abs": semantic_diffs,
        "prediction_agreement_sample_count": int(reference_predictions.size),
        "top1_agreement": agreements,
        "correct_counts_identical": all(
            metrics[name]["correct_top1"] == full["correct_top1"]
            and metrics[name]["correct_top5"] == full["correct_top5"]
            for name in ("crop_rowgroup", "crop_vector")
        ),
        "modes": metrics,
        "source_revisions_observed": observed_sources,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GALP crop I/O three-way A/B",
        "",
        f"Validation: **{'PASS' if summary['ok'] else 'FAIL'}**",
        "",
        "| Mode | Requested / transformed blocks | Planned / actual / full vectors | Physical bytes | pread | Sparse fallback rowgroups | Decode ms | Transform ms | E2E s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, _ in MODES:
        item = summary["modes"][mode]
        lines.append(
            f"| {mode} | {item['requested_source_block_count']} / {item['source_blocks_transformed']} | "
            f"{item['planned_vector_count']} / {item['actual_vector_count']} / {item['full_vector_count']} | "
            f"{item['compressed_payload_bytes_read']} | "
            f"{item['pread_count']} | {item['sparse_read_fallback_rowgroup_count']} "
            f"({item['sparse_read_fallback_rowgroup_ratio']:.6f}) | "
            f"{item['decode_ms']:.3f} | {item['transform_ms']:.3f} | "
            f"{item['end_to_end_seconds']:.6f} |"
        )
    lines.extend(["", "## Correctness", ""])
    lines.append(f"- Semantic tolerance: `{summary['semantic_tolerance']}`.")
    lines.append(f"- Prediction samples: `{summary['prediction_agreement_sample_count']}`.")
    for mode, agreement in summary["top1_agreement"].items():
        lines.append(f"- {mode} Top-1 agreement with full decode: `{agreement:.6f}`.")
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {item}" for item in summary["failures"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_modes(base_contract_path: Path, output_dir: Path, python: str) -> dict[str, dict[str, Any]]:
    base_contract = load_contract(base_contract_path)
    pipeline_script = Path(__file__).with_name("pipeline.py")
    binding_dir = str(base_contract["pipelines"]["galp"]["torch_binding_dir"])
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        binding_dir if not existing_pythonpath else os.pathsep.join((binding_dir, existing_pythonpath))
    )
    results: dict[str, dict[str, Any]] = {}
    for mode, execution_mode in MODES:
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        contract = copy.deepcopy(base_contract)
        contract["pipelines"]["galp"]["crop_execution_mode"] = execution_mode
        contract_path = mode_dir / "contract.json"
        result_path = mode_dir / "result.json"
        write_json(contract_path, contract)
        subprocess.run(
            [python, str(pipeline_script), "--pipeline", "galp", "--contract", str(contract_path), "--output", str(result_path)],
            check=True,
            env=child_env,
        )
        results[mode] = _load_result(result_path)
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-run", action="store_true", help="Validate existing MODE/result.json files.")
    parser.add_argument("--semantic-tolerance", type=float, default=1.2e-7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    base_contract = load_contract(args.contract)
    if args.skip_run:
        results = {mode: _load_result(args.output_dir / mode / "result.json") for mode, _ in MODES}
    else:
        results = _run_modes(args.contract, args.output_dir, args.python)
    summary = validate_crop_io_ab_results(base_contract, results, semantic_tolerance=args.semantic_tolerance)
    write_json(args.output_dir / "validation.json", summary)
    _write_markdown(args.output_dir / "report.md", summary)
    print("RESULT_JSON " + json.dumps(summary, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
