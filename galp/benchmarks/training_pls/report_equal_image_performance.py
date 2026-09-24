#!/usr/bin/env python3
"""Compare equal-image GALP, RGB-no-more, DALI, and PyTorch training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from galp.benchmarks.training_audit_policy import validate_audit_policy
from galp.benchmarks.training_pls.contracts import (
    NATIVE_PHYSICAL_BACKEND,
    STANDARD_DCT_BACKEND,
    blocking_code_identity,
    condition_identity_hash,
)
from galp.benchmarks.training_pls.report_convergence_reference import (
    PAIRED_IDENTITY_FIELDS,
)
from galp.benchmarks.training_pls.recipe import sha256_json


EXPECTED_EPOCHS = (1, 2)
EXPECTED_SAMPLES = 1_281_167
EXPECTED_MICROBATCHES = 20_019
EXPECTED_UPDATES = 1_252
STANDARD_PIPELINES = ("d2", "d3", "pytorch")
DCT_PIPELINES = ("native_b6", "rgbnomore_dct")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _read_standard_result(root: Path, pipeline: str) -> dict[str, Any]:
    result_path = root / "runs" / pipeline / "final_result.json"
    if result_path.is_file():
        return _read_json(result_path)

    evidence: list[str] = []
    status_path = root / "runs" / pipeline / "run_status.json"
    if status_path.is_file():
        try:
            status = _read_json(status_path)
            status_description = f"run_status state={status.get('state')!r}"
            if "completed_epoch" in status:
                status_description += (
                    f", completed_epoch={status.get('completed_epoch')!r}"
                )
            if status.get("error_type") or status.get("error"):
                status_description += (
                    f", error={status.get('error_type', 'Error')}: "
                    f"{status.get('error', '')}"
                )
            evidence.append(status_description)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            evidence.append(f"unreadable run_status.json ({error})")

    recorded_failure: Mapping[str, Any] | None = None
    summary_path = root / "results.json"
    if summary_path.is_file():
        try:
            summary = _read_json(summary_path)
            summary_results = summary.get("results", [])
            if isinstance(summary_results, list):
                recorded_failure = next(
                    (
                        row
                        for row in reversed(summary_results)
                        if isinstance(row, Mapping)
                        and row.get("pipeline") == pipeline
                        and row.get("state") == "failed"
                    ),
                    None,
                )
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    failures_path = root / "failures.jsonl"
    if recorded_failure is None and failures_path.is_file():
        try:
            recorded_failure = next(
                (
                    row
                    for row in reversed(_read_jsonl(failures_path))
                    if row.get("pipeline") == pipeline
                ),
                None,
            )
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    if recorded_failure is not None:
        evidence.append(
            "recorded failure="
            f"{recorded_failure.get('error_type', 'Error')}: "
            f"{recorded_failure.get('error', '')}"
        )

    detail = "; ".join(evidence) if evidence else "no run status or failure record"
    raise ValueError(
        f"{pipeline} equal-image run is incomplete: missing {result_path}; {detail}. "
        "Rerun that RGB pipeline successfully before generating the report"
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _epoch_map(records: Sequence[Mapping[str, Any]], source: str) -> dict[int, dict[str, Any]]:
    selected = [
        dict(row)
        for row in records
        if row.get("record_type") == "train"
        and row.get("scope") == "epoch"
        and int(row.get("epoch", -1)) in EXPECTED_EPOCHS
    ]
    result: dict[int, dict[str, Any]] = {}
    for row in selected:
        epoch = int(row["epoch"])
        if epoch in result:
            raise ValueError(f"{source} duplicates epoch {epoch}")
        result[epoch] = row
    if tuple(sorted(result)) != EXPECTED_EPOCHS:
        raise ValueError(f"{source} must contain exact epochs {EXPECTED_EPOCHS}")
    for epoch, row in result.items():
        observed = {
            "samples": int(row["epoch_samples"]),
            "microbatches": int(row["epoch_microbatches"]),
            "updates": int(row["epoch_optimizer_updates"]),
        }
        expected = {
            "samples": EXPECTED_SAMPLES,
            "microbatches": EXPECTED_MICROBATCHES,
            "updates": EXPECTED_UPDATES,
        }
        if observed != expected:
            raise ValueError(
                f"{source} epoch {epoch} workload differs: {observed} != {expected}"
            )
    return result


def _gpu_identity(environment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": environment.get("gpu_name"),
        "uuid": environment.get("gpu_uuid"),
        "hostname": environment.get("hostname"),
        "torch": environment.get("torch"),
        "torch_cuda_build": environment.get("torch_cuda_build"),
    }


def _read_pls_contract(run: Path) -> dict[str, Any]:
    path = run / "run_manifest.json"
    contract = _read_json(path)
    if contract.get("run_manifest_hash") != sha256_json(
        {key: value for key, value in contract.items() if key != "run_manifest_hash"}
    ):
        raise ValueError(f"run-manifest hash mismatch: {path}")
    if contract.get("condition_hash") != condition_identity_hash(contract):
        raise ValueError(f"condition identity hash mismatch: {path}")
    return contract


def _execution_backend(contract: Mapping[str, Any]) -> str:
    mode = str(contract.get("execution_mode", ""))
    if mode == "native_physical_pls":
        return NATIVE_PHYSICAL_BACKEND
    if mode == "standard_rgbnomore_dct":
        return STANDARD_DCT_BACKEND
    return mode


def build_report(
    *,
    standard_root: Path | None = None,
    native_run: Path,
    rgbnomore_run: Path | None = None,
    standard_pipelines: Sequence[str] = STANDARD_PIPELINES,
    standard_roots: Mapping[str, Path] | None = None,
    requested_claim: str = "system-level",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    standard_pipelines = tuple(standard_pipelines)
    if not standard_pipelines or len(set(standard_pipelines)) != len(standard_pipelines):
        raise ValueError("standard_pipelines must be non-empty and unique")
    unsupported = set(standard_pipelines) - set(STANDARD_PIPELINES)
    if unsupported:
        raise ValueError(f"unsupported standard pipelines: {sorted(unsupported)}")
    if requested_claim not in {"system-level", "pipeline-only", "recipe-matched"}:
        raise ValueError("requested_claim must be system-level, pipeline-only, or recipe-matched")
    if requested_claim in {"pipeline-only", "recipe-matched"}:
        raise ValueError(
            f"{requested_claim} claim is invalid for GALP B6 versus RGB pipelines: "
            "model domain, augmentation recipe, crop, and sample order differ"
        )
    if standard_roots is not None and standard_root is not None:
        raise ValueError("provide standard_root or standard_roots, not both")
    if standard_roots is None:
        if standard_root is None:
            raise ValueError("a standard RGB result root is required")
        resolved_standard_roots = {
            pipeline: standard_root.resolve() for pipeline in standard_pipelines
        }
    else:
        if set(standard_roots) != set(standard_pipelines):
            raise ValueError("standard_roots must map every selected RGB pipeline exactly")
        resolved_standard_roots = {
            pipeline: Path(standard_roots[pipeline]).resolve()
            for pipeline in standard_pipelines
        }
    native_run = native_run.resolve()
    rgbnomore_run = rgbnomore_run.resolve() if rgbnomore_run is not None else None
    contracts = {
        pipeline: _read_json(root / "contract.json")
        for pipeline, root in resolved_standard_roots.items()
    }
    contract = contracts[standard_pipelines[0]]
    expected_schedule = {
        "processed_images": 2 * EXPECTED_SAMPLES,
        "total_microbatches": 2 * EXPECTED_MICROBATCHES,
        "total_optimizer_updates": 2 * EXPECTED_UPDATES,
    }
    for pipeline, observed_contract in contracts.items():
        if (
            observed_contract.get("schema_version")
            != "galp-equal-image-rgb-epoch-contract-v3"
            or observed_contract.get("benchmark")
            != "equal-image-epoch-aware-rgb-training-v3"
        ):
            raise ValueError(
                f"{pipeline} root uses a legacy/unsupported equal-image contract"
            )
        expected_contract_hash = sha256_json(
            {
                key: value
                for key, value in observed_contract.items()
                if key != "contract_hash"
            }
        )
        if observed_contract.get("contract_hash") != expected_contract_hash:
            raise ValueError(f"{pipeline} equal-image contract hash mismatch")
        validate_audit_policy(observed_contract.get("audit_policy", {}))
        if bool(observed_contract.get("profiling", {}).get("enabled", False)):
            raise ValueError(
                "profiling-only RGB runs cannot be used as formal performance evidence"
            )
        schedule = observed_contract.get("prefix_schedule", {})
        for key, expected in expected_schedule.items():
            if int(schedule.get(key, -1)) != expected:
                raise ValueError(f"{pipeline} standard contract {key} differs")

    standard_environments = {
        pipeline: _read_json(root / "environment.json")
        for pipeline, root in resolved_standard_roots.items()
    }
    standard_environment = standard_environments[standard_pipelines[0]]
    native_environment = _read_json(native_run / "environment.json")
    native_condition = _read_pls_contract(native_run)
    native_contract_backend = _execution_backend(native_condition)
    if (
        (native_contract_backend and native_contract_backend != NATIVE_PHYSICAL_BACKEND)
        or native_environment.get("execution_mode") != "native_physical_pls"
        or native_environment.get("physical_fls_observed") is not True
        or native_environment.get("physical_gpu_pool") is not True
    ):
        raise ValueError("native run is not the physical PLS/GPU-pool backend")
    if (
        native_condition.get("schema_version") != "galp-pls-condition-contract-v3"
        or native_condition.get("condition_id") != "B6"
        or (
            native_contract_backend
            and native_contract_backend != NATIVE_PHYSICAL_BACKEND
        )
    ):
        raise ValueError("native run condition is not physical B6")
    if native_condition.get("condition_hash") != condition_identity_hash(
        native_condition
    ):
        raise ValueError("native B6 training identity hash mismatch")
    if native_condition.get("run_manifest_hash") != sha256_json(
        {
            key: value
            for key, value in native_condition.items()
            if key != "run_manifest_hash"
        }
    ):
        raise ValueError("native B6 run-manifest integrity hash mismatch")

    rgbnomore_condition: dict[str, Any] | None = None
    rgbnomore_environment: dict[str, Any] | None = None
    native_mapping_hash: str | None = None
    rgbnomore_mapping_hash: str | None = None
    if rgbnomore_run is not None:
        rgbnomore_condition = _read_pls_contract(rgbnomore_run)
        rgbnomore_environment = _read_json(rgbnomore_run / "environment.json")
        if _execution_backend(rgbnomore_condition) != STANDARD_DCT_BACKEND:
            raise ValueError(
                "--rgbnomore-run is not a standard RGB-no-more DCT run"
            )
        if (
            rgbnomore_environment.get("execution_mode")
            != "standard_rgbnomore_dct"
            or rgbnomore_environment.get("physical_fls_observed") is not False
            or rgbnomore_environment.get("physical_gpu_pool") is not False
        ):
            raise ValueError(
                "RGB-no-more run does not identify the standard JPEG-to-DCT backend"
            )
        paired_differences = {
            field: {
                "native_b6": native_condition.get(field),
                "rgbnomore_dct": rgbnomore_condition.get(field),
            }
            for field in PAIRED_IDENTITY_FIELDS
            if native_condition.get(field) != rgbnomore_condition.get(field)
        }
        if paired_differences:
            raise ValueError(
                "GALP/RGB-no-more paired DCT identity differs: "
                f"{sorted(paired_differences)}"
            )
        native_mapping_hash = str(
            native_condition.get("physical_execution", {}).get(
                "premixed_mapping_sha256", ""
            )
        )
        rgbnomore_mapping_hash = str(
            rgbnomore_condition.get("standard_dct_reference", {}).get(
                "premixed_mapping_sha256", ""
            )
        )
        if not native_mapping_hash or native_mapping_hash != rgbnomore_mapping_hash:
            raise ValueError(
                "GALP/RGB-no-more premixed mapping SHA-256 differs: "
                f"native={native_mapping_hash!r}, "
                f"rgbnomore={rgbnomore_mapping_hash!r}"
            )
    native_audit_policy = validate_audit_policy(
        native_condition.get("audit_policy", {})
    )
    standard_model_ids = {
        str(observed["model"]["model_id"]) for observed in contracts.values()
    }
    native_model_id = str(native_condition["model_id"])
    if standard_model_ids != {native_model_id} or not native_model_id:
        raise ValueError(
            "standard/native model-family IDs differ: "
            f"standard={sorted(standard_model_ids)}, native={native_model_id!r}"
        )
    standard_recipe_hashes = {
        str(observed.get("model", {}).get("recipe_hash", ""))
        for observed in contracts.values()
    }
    native_recipe_hash = str(native_condition.get("recipe_hash", ""))
    recorded_recipe_hashes = {
        value
        for value in standard_recipe_hashes | {native_recipe_hash}
        if value
    }
    if recorded_recipe_hashes and standard_recipe_hashes != {native_recipe_hash}:
        raise ValueError("standard/native model recipe hashes differ")
    standard_model_sources = {
        json.dumps(
            observed.get("model", {}).get("source_provenance", []),
            sort_keys=True,
        )
        for observed in contracts.values()
    }
    native_model_sources = json.dumps(
        native_condition.get("model_source_provenance", []), sort_keys=True
    )
    recorded_model_sources = {
        value for value in standard_model_sources | {native_model_sources} if value != "[]"
    }
    if recorded_model_sources and standard_model_sources != {native_model_sources}:
        raise ValueError("standard/native external model-source hashes differ")
    standard_gpu = _gpu_identity(standard_environment)
    native_gpu = _gpu_identity(native_environment)
    all_gpu_identities = {
        pipeline: _gpu_identity(environment)
        for pipeline, environment in standard_environments.items()
    }
    all_gpu_identities["native_b6"] = native_gpu
    if rgbnomore_environment is not None:
        all_gpu_identities["rgbnomore_dct"] = _gpu_identity(
            rgbnomore_environment
        )
    if any(identity != standard_gpu for identity in all_gpu_identities.values()):
        raise ValueError(
            f"standard/native hardware-software identity differs: {all_gpu_identities}"
        )
    if not standard_gpu["uuid"]:
        raise ValueError("formal comparison requires a non-empty GPU UUID")
    required_gpu_names = {
        str(observed.get("required_gpu_name_substring", ""))
        for observed in contracts.values()
    } | {str(native_condition.get("required_gpu_name_substring", ""))}
    if rgbnomore_condition is not None:
        required_gpu_names.add(
            str(rgbnomore_condition.get("required_gpu_name_substring", ""))
        )
    if len(required_gpu_names) != 1:
        raise ValueError("required GPU-name contracts differ")
    required_gpu_name = next(iter(required_gpu_names))
    if required_gpu_name and required_gpu_name.lower() not in str(
        standard_gpu["name"]
    ).lower():
        raise ValueError("observed GPU does not satisfy required_gpu_name_substring")

    source_identities = {
        pipeline: blocking_code_identity(observed["source_identity"])
        for pipeline, observed in contracts.items()
    }
    source_identities["native_b6"] = blocking_code_identity(
        native_condition["code_version"]
    )
    if rgbnomore_condition is not None:
        source_identities["rgbnomore_dct"] = blocking_code_identity(
            rgbnomore_condition["code_version"]
        )
    if len({tuple(sorted(value.items())) for value in source_identities.values()}) != 1:
        raise ValueError("source-code/runtime hashes differ across result roots")

    seeds = {int(observed.get("seed", -1)) for observed in contracts.values()}
    seeds.add(int(native_condition.get("training_seed", -2)))
    if rgbnomore_condition is not None:
        seeds.add(int(rgbnomore_condition.get("training_seed", -3)))
    if len(seeds) != 1:
        raise ValueError("training seed differs across result roots")
    audit_hashes = {
        observed["audit_policy"]["audit_policy_hash"]
        for observed in contracts.values()
    } | {native_audit_policy["audit_policy_hash"]}
    if rgbnomore_condition is not None:
        audit_hashes.add(
            validate_audit_policy(rgbnomore_condition.get("audit_policy", {}))[
                "audit_policy_hash"
            ]
        )
    if len(audit_hashes) != 1:
        raise ValueError("audit policy differs across result roots")

    epoch_maps: dict[str, dict[int, dict[str, Any]]] = {}
    standard_results: dict[str, dict[str, Any]] = {}
    for pipeline in standard_pipelines:
        root = resolved_standard_roots[pipeline]
        result = _read_standard_result(root, pipeline)
        if result.get("state") != "completed" or int(result.get("completed_epoch", -1)) != 2:
            raise ValueError(f"{pipeline} equal-image run is incomplete")
        if (
            result.get("contract_hash") != contracts[pipeline].get("contract_hash")
            or result.get("audit_policy_hash")
            != contracts[pipeline]["audit_policy"]["audit_policy_hash"]
        ):
            raise ValueError(f"{pipeline} result contract/audit identity differs")
        if result.get("model_domain") != "rgb" or int(
            result.get("model_parameter_count", -1)
        ) <= 0:
            raise ValueError(f"{pipeline} result lacks the RGB model identity")
        standard_results[pipeline] = result
        epoch_maps[pipeline] = _epoch_map(
            result.get("epoch_records", []), f"standard {pipeline}"
        )

    native_status = _read_json(native_run / "run_status.json")
    if int(native_status.get("completed_epoch", -1)) < 2:
        raise ValueError("native B6 run has not completed epoch 2")
    if int(native_status.get("seed", -1)) != next(iter(seeds)):
        raise ValueError("standard/native training seed differs")
    native_records = _read_jsonl(native_run / "metrics.jsonl")
    epoch_maps["native_b6"] = _epoch_map(native_records, "native B6")
    if rgbnomore_run is not None:
        rgbnomore_status = _read_json(rgbnomore_run / "run_status.json")
        if int(rgbnomore_status.get("completed_epoch", -1)) < 2:
            raise ValueError("RGB-no-more DCT run has not completed epoch 2")
        if int(rgbnomore_status.get("seed", -1)) != next(iter(seeds)):
            raise ValueError("RGB-no-more DCT training seed differs")
        rgbnomore_records = _read_jsonl(rgbnomore_run / "metrics.jsonl")
        epoch_maps["rgbnomore_dct"] = _epoch_map(
            rgbnomore_records, "RGB-no-more DCT"
        )
        for epoch in EXPECTED_EPOCHS:
            native_epoch = epoch_maps["native_b6"][epoch]
            rgbnomore_epoch = epoch_maps["rgbnomore_dct"][epoch]
            for digest in ("sample_order_digest", "pool_membership_digest"):
                if native_epoch.get(digest) != rgbnomore_epoch.get(digest):
                    raise ValueError(
                        "GALP/RGB-no-more logical schedule differs at "
                        f"epoch {epoch}: {digest}"
                    )
    if {"d2", "pytorch"}.issubset(epoch_maps):
        for epoch in EXPECTED_EPOCHS:
            d2_epoch = epoch_maps["d2"][epoch]
            pytorch_epoch = epoch_maps["pytorch"][epoch]
            if (
                d2_epoch.get("sample_order", {}).get("emitted_order_sha256")
                != pytorch_epoch.get("sample_order", {}).get("emitted_order_sha256")
                or d2_epoch.get("augmentation_decision_sha256")
                != pytorch_epoch.get("augmentation_decision_sha256")
            ):
                raise ValueError(
                    f"D2/PyTorch canonical order or planned augmentation digest differs at epoch {epoch}"
                )

    timing_fields = {
        "epoch_seconds",
        "data_preparation_seconds",
        "exposed_input_wait_seconds",
        "model_forward_backward_optimizer_seconds",
        "audit_seconds",
        "boundary_sync_seconds",
        "pipeline_internal_work_seconds",
        "timing_semantics",
    }
    for pipeline, epochs in epoch_maps.items():
        for epoch, record in epochs.items():
            missing = sorted(timing_fields - set(record))
            if missing:
                raise ValueError(
                    f"{pipeline} epoch {epoch} lacks unified timing fields: {missing}"
                )
            if record["timing_semantics"].get("validation_included") is not False:
                raise ValueError(f"{pipeline} training throughput includes validation")

    rgb_model_only_by_pipeline = {
        pipeline: _read_json(root / "model_only_rgb.json")
        for pipeline, root in resolved_standard_roots.items()
    }
    dct_model_only_by_pipeline = {
        "native_b6": _read_json(native_run / "model_only_dct.json")
    }
    if rgbnomore_run is not None:
        dct_model_only_by_pipeline["rgbnomore_dct"] = _read_json(
            rgbnomore_run / "model_only_dct.json"
        )
    for pipeline, calibration in rgb_model_only_by_pipeline.items():
        if (
            calibration.get("contract_hash") != contracts[pipeline]["contract_hash"]
            or calibration.get("audit_policy_hash") not in audit_hashes
            or calibration.get("initial_model_hash")
            != standard_results[pipeline].get("initial_model_hash")
            or int(calibration.get("model_parameter_count", -1))
            != int(standard_results[pipeline]["model_parameter_count"])
        ):
            raise ValueError(f"{pipeline} model-only calibration identity differs")
    dct_conditions = {"native_b6": native_condition}
    if rgbnomore_condition is not None:
        dct_conditions["rgbnomore_dct"] = rgbnomore_condition
    for pipeline, calibration in dct_model_only_by_pipeline.items():
        observed_condition = dct_conditions[pipeline]
        if (
            calibration.get("condition_hash")
            != observed_condition.get("condition_hash")
            or calibration.get("audit_policy_hash") not in audit_hashes
            or calibration.get("initial_model_hash")
            != observed_condition.get("initial_model_hash")
        ):
            raise ValueError(
                f"{pipeline} DCT model-only calibration identity differs"
            )
    if rgbnomore_condition is not None and (
        rgbnomore_condition.get("initial_model_hash")
        != native_condition.get("initial_model_hash")
    ):
        raise ValueError("GALP/RGB-no-more DCT initial model hash differs")
    rgb_initial_hashes = {
        result["initial_model_hash"] for result in standard_results.values()
    }
    rgb_parameter_counts = {
        int(result["model_parameter_count"]) for result in standard_results.values()
    }
    if len(rgb_initial_hashes) != 1 or len(rgb_parameter_counts) != 1:
        raise ValueError("RGB pipeline model identities differ")
    calibration_contracts = [
        *rgb_model_only_by_pipeline.values(),
        *dct_model_only_by_pipeline.values(),
    ]
    calibration_workloads = {
        (
            int(value.get("microbatch_images", -1)),
            int(value.get("gradient_accumulation", -1)),
            str(value.get("precision")),
            int(value.get("warmup_optimizer_updates", -1)),
            int(value.get("measured_optimizer_updates", -1)),
        )
        for value in calibration_contracts
    }
    expected_calibration_workload = (
        64,
        16,
        str(equal_training_precision := contract["training"]["precision"]),
        5,
        120,
    )
    if calibration_workloads != {expected_calibration_workload}:
        raise ValueError("RGB/DCT model-only calibration workloads differ")

    rows: list[dict[str, Any]] = []
    native_warm_ips = float(epoch_maps["native_b6"][2]["images_per_second"])
    domains = {
        "native_b6": "dct",
        "rgbnomore_dct": "dct",
        "d2": "rgb",
        "d3": "rgb",
        "pytorch": "rgb",
    }
    pipeline_order = (
        "native_b6",
        *(("rgbnomore_dct",) if rgbnomore_run is not None else ()),
        *standard_pipelines,
    )
    for pipeline in pipeline_order:
        for epoch in EXPECTED_EPOCHS:
            source = epoch_maps[pipeline][epoch]
            ips = float(source["images_per_second"])
            model_only = (
                dct_model_only_by_pipeline[pipeline]
                if pipeline in DCT_PIPELINES
                else rgb_model_only_by_pipeline[pipeline]
            )
            epoch_seconds = float(source["epoch_seconds"])
            loader_stages = source.get("loader_stage_seconds", {}) or {}
            native_stats = source.get("native_execution_stats", {}) or {}
            selected_vectors = float(native_stats.get("selected_vector_count", 0))
            full_vectors = float(native_stats.get("full_vector_count", 0))
            compressed_read = float(
                native_stats.get("compressed_payload_bytes_read", 0)
            )
            full_compressed = float(
                native_stats.get("full_compressed_payload_bytes", 0)
            )
            rows.append(
                {
                    "pipeline": pipeline,
                    "input_domain": domains[pipeline],
                    "epoch": epoch,
                    "observation": "cold" if epoch == 1 else "warm-primary",
                    "epoch_samples": int(source["epoch_samples"]),
                    "epoch_microbatches": int(source["epoch_microbatches"]),
                    "epoch_optimizer_updates": int(source["epoch_optimizer_updates"]),
                    "epoch_seconds": epoch_seconds,
                    "images_per_second": ips,
                    "warm_throughput_relative_to_native_b6": (
                        None if epoch == 1 else ips / native_warm_ips
                    ),
                    "data_preparation_seconds": float(
                        source["data_preparation_seconds"]
                    ),
                    "exposed_input_wait_seconds": float(source["exposed_input_wait_seconds"]),
                    "audit_seconds": float(source["audit_seconds"]),
                    "boundary_sync_seconds": float(source["boundary_sync_seconds"]),
                    "pipeline_internal_work_seconds": source[
                        "pipeline_internal_work_seconds"
                    ],
                    "model_only_images_per_second": float(
                        model_only["images_per_second"]
                    ),
                    "model_only_milliseconds_per_microbatch": float(
                        model_only["milliseconds_per_microbatch"]
                    ),
                    "data_preparation_percent_of_epoch": 100.0
                    * float(source["data_preparation_seconds"])
                    / epoch_seconds,
                    "exposed_input_wait_percent_of_epoch": 100.0
                    * float(source["exposed_input_wait_seconds"])
                    / epoch_seconds,
                    "audit_percent_of_epoch": 100.0
                    * float(source["audit_seconds"])
                    / epoch_seconds,
                    "boundary_sync_percent_of_epoch": 100.0
                    * float(source["boundary_sync_seconds"])
                    / epoch_seconds,
                    "loader_data_wait_seconds": loader_stages.get(
                        "loader_data_wait"
                    ),
                    "dali_handoff_seconds": loader_stages.get("dali_handoff"),
                    "loader_read_seconds": loader_stages.get("read"),
                    "loader_decode_seconds": loader_stages.get("decode"),
                    "loader_augmentation_seconds": loader_stages.get(
                        "augmentation"
                    ),
                    "loader_preprocess_seconds": loader_stages.get("preprocess"),
                    "native_selected_vector_fraction": (
                        selected_vectors / full_vectors if full_vectors else None
                    ),
                    "native_compressed_read_fraction": (
                        compressed_read / full_compressed
                        if full_compressed
                        else None
                    ),
                    "native_execution_stats": native_stats or None,
                    "timing_breakdown_is_additive": False,
                }
            )

    equal_training = contract["training"]
    native_recipe_augmentation = {
        "crop": native_condition["crop_policy"],
        "order": native_condition["order_policy"],
        "mixup": native_condition["mixup"],
        "randaugment": native_condition["randaugment"],
    }
    rgb_augmentation = contract["augmentation"]
    matrix_checks = [
        ("same_gpu_name_uuid_hostname", True, standard_gpu),
        ("same_torch_cuda", True, {"torch": standard_gpu["torch"], "cuda": standard_gpu["torch_cuda_build"]}),
        ("same_seed", True, next(iter(seeds))),
        ("same_sample_count", True, EXPECTED_SAMPLES),
        ("same_microbatch", int(equal_training["microbatch_images"]) == int(native_condition["microbatch_size"]), 64),
        ("same_accumulation", int(equal_training["gradient_accumulation"]) == int(native_condition["gradient_accumulation"]), 16),
        ("same_optimizer_updates", True, EXPECTED_UPDATES),
        ("same_precision", equal_training["precision"] == native_condition["precision"], equal_training["precision"]),
        ("same_optimizer", contract["optimizer"] == native_condition["optimizer_configuration"], None),
        ("same_scheduler_family", contract["scheduler"]["type"] == native_condition["scheduler"]["type"] and contract["scheduler"]["warmup_optimizer_updates"] == native_condition["scheduler"]["warmup_optimizer_updates"], None),
        ("same_audit_policy", len(audit_hashes) == 1, next(iter(audit_hashes))),
        ("same_timing_scope", True, sorted(timing_fields)),
        ("same_model_domain", False, {"native_b6": "dct", "rgbnomore_dct": "dct" if rgbnomore_run is not None else "not-run", "rgb": "rgb"}),
        ("same_model_parameter_count", int(dct_model_only_by_pipeline["native_b6"]["model_parameter_count"]) == int(next(iter(rgb_model_only_by_pipeline.values()))["model_parameter_count"]), {"dct": dct_model_only_by_pipeline["native_b6"]["model_parameter_count"], "rgb": next(iter(rgb_model_only_by_pipeline.values()))["model_parameter_count"]}),
        ("same_augmentation", False, {"native_b6": native_recipe_augmentation, "rgb": rgb_augmentation}),
        ("same_sample_order", False, {"native_b6": native_condition["order_policy"], "rgbnomore_dct": rgbnomore_condition["order_policy"] if rgbnomore_condition is not None else "not-run", "d2_pytorch": "canonical", "d3": "DALI-native"}),
        ("same_crop_policy", False, {"native_b6": native_condition["crop_policy"], "d2_pytorch": "per-sample-planned", "d3": "DALI-native"}),
        ("same_mixup_randaugment", False, {"native_b6": {"mixup": True, "randaugment": True}, "rgb": {"mixup": False, "randaugment": False}}),
        ("same_source_code_runtime_hashes", True, source_identities),
        (
            "same_galp_rgbnomore_dct_schedule",
            rgbnomore_condition is None
            or all(
                epoch_maps["native_b6"][epoch].get("sample_order_digest")
                == epoch_maps["rgbnomore_dct"][epoch].get("sample_order_digest")
                and epoch_maps["native_b6"][epoch].get("pool_membership_digest")
                == epoch_maps["rgbnomore_dct"][epoch].get("pool_membership_digest")
                for epoch in EXPECTED_EPOCHS
            ),
            "not-run" if rgbnomore_condition is None else "epoch-1-and-2-match",
        ),
        (
            "same_galp_rgbnomore_premixed_mapping",
            rgbnomore_condition is None
            or (
                bool(native_mapping_hash)
                and native_mapping_hash == rgbnomore_mapping_hash
            ),
            {
                "native_b6": native_mapping_hash,
                "rgbnomore_dct": rgbnomore_mapping_hash,
            },
        ),
        (
            "same_external_model_source_hashes",
            not recorded_model_sources
            or standard_model_sources == {native_model_sources},
            json.loads(native_model_sources),
        ),
        ("same_model_only_calibration_workload", True, next(iter(calibration_workloads))),
    ]
    fairness_matrix = [
        {
            "check": name,
            "matches": bool(matches),
            "evidence": evidence,
            "blocking_for_system_level": name in {
                "same_gpu_name_uuid_hostname",
                "same_torch_cuda",
                "same_seed",
                "same_sample_count",
                "same_microbatch",
                "same_accumulation",
                "same_optimizer_updates",
                "same_precision",
                "same_optimizer",
                "same_scheduler_family",
                "same_audit_policy",
                "same_timing_scope",
                "same_source_code_runtime_hashes",
                "same_galp_rgbnomore_dct_schedule",
                "same_galp_rgbnomore_premixed_mapping",
                "same_external_model_source_hashes",
                "same_model_only_calibration_workload",
            },
        }
        for name, matches, evidence in matrix_checks
    ]
    blocking_failures = [
        row["check"]
        for row in fairness_matrix
        if row["blocking_for_system_level"] and not row["matches"]
    ]
    if blocking_failures:
        raise ValueError(f"system-level fairness checks failed: {blocking_failures}")

    report = {
        "schema_version": "galp-equal-image-performance-report-v4",
        "model_id": native_model_id,
        "precision": equal_training_precision,
        "hardware": standard_gpu,
        "workload": {
            "epochs": [1, 2],
            "epoch_1_role": "cold observation",
            "epoch_2_role": "primary warm observation",
            "samples_per_epoch": EXPECTED_SAMPLES,
            "microbatch_images": 64,
            "gradient_accumulation": 16,
            "microbatches_per_epoch": EXPECTED_MICROBATCHES,
            "optimizer_updates_per_epoch": EXPECTED_UPDATES,
            "validation_excluded_from_training_throughput": True,
        },
        "rows": rows,
        "warm_throughput_ratios": {
            pipeline: float(epoch_maps[pipeline][2]["images_per_second"])
            / native_warm_ips
            for pipeline in pipeline_order
            if pipeline != "native_b6"
        },
        "fairness_matrix": fairness_matrix,
        "requested_claim": requested_claim,
        "model_only_calibration": {
            "rgb_by_pipeline": rgb_model_only_by_pipeline,
            "dct_by_pipeline": dct_model_only_by_pipeline,
            "subtraction_from_e2e_forbidden": True,
        },
        "included_pipelines": list(pipeline_order),
        "claims": {
            "galp_vs_rgbnomore_dct": (
                "direct recipe-matched DCT system comparison with identical model "
                "initialization, sample order, premixed mapping, augmentation recipe, "
                "optimizer, scheduler, precision, and image count"
                if rgbnomore_run is not None
                else "pending; RGB-no-more DCT run is not included"
            ),
            "d2_vs_pytorch": (
                "direct RGB comparison with canonical order and planned crop/flip"
                if {"d2", "pytorch"}.issubset(standard_pipelines)
                else "pending; D2 and PyTorch are not both included"
            ),
            "d3_performance_ceiling": (
                "DALI-native shuffle/crop/flip performance ceiling; order and "
                "augmentation decisions differ from D2/PyTorch"
                if "d3" in standard_pipelines
                else "pending; D3 is not included"
            ),
            "rgb_vs_native_b6": (
                "equal-image full-application system comparison only; not a pure "
                "data-pipeline speedup because model domain, crop/order, RandAugment, "
                "and Mixup differ"
            ),
            "pipeline_only_cross_domain_claim": False,
            "recipe_matched_cross_domain_claim": False,
            "epoch_1_cold_comparison": False,
            "epoch_2_warm_comparison": True,
            "accuracy_or_convergence_claim": False,
            "long_run_throughput_qualification": False,
        },
        "source_artifacts": {
            "standard_roots": {
                pipeline: str(root)
                for pipeline, root in resolved_standard_roots.items()
            },
            "native_run": str(native_run),
            "rgbnomore_run": (
                str(rgbnomore_run) if rgbnomore_run is not None else None
            ),
            "rgbnomore_condition_hash": (
                rgbnomore_condition.get("condition_hash")
                if rgbnomore_condition is not None
                else None
            ),
            "standard_contract_hashes": {
                pipeline: observed["contract_hash"]
                for pipeline, observed in contracts.items()
            },
        },
    }
    return report, rows


def _write_outputs(
    output_dir: Path, report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        path
        for path in (
            output_dir / "equal_image_performance.json",
            output_dir / "equal_image_performance.csv",
        )
        if path.exists()
    ]
    if existing:
        raise ValueError(
            f"report output already exists ({existing}); use a new output directory"
        )
    _atomic_json(output_dir / "equal_image_performance.json", report)
    fields = [
        "pipeline",
        "input_domain",
        "epoch",
        "observation",
        "epoch_samples",
        "epoch_microbatches",
        "epoch_optimizer_updates",
        "epoch_seconds",
        "images_per_second",
        "warm_throughput_relative_to_native_b6",
        "data_preparation_seconds",
        "exposed_input_wait_seconds",
        "audit_seconds",
        "boundary_sync_seconds",
        "pipeline_internal_work_seconds",
        "model_only_images_per_second",
        "model_only_milliseconds_per_microbatch",
        "data_preparation_percent_of_epoch",
        "exposed_input_wait_percent_of_epoch",
        "audit_percent_of_epoch",
        "boundary_sync_percent_of_epoch",
        "loader_data_wait_seconds",
        "dali_handoff_seconds",
        "loader_read_seconds",
        "loader_decode_seconds",
        "loader_augmentation_seconds",
        "loader_preprocess_seconds",
        "native_selected_vector_fraction",
        "native_compressed_read_fraction",
        "timing_breakdown_is_additive",
    ]
    temporary = output_dir / ".equal_image_performance.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    os.replace(temporary, output_dir / "equal_image_performance.csv")
    warm = {row["pipeline"]: row for row in rows if int(row["epoch"]) == 2}
    lines = [
        "# Equal-image training performance and breakdown",
        "",
        f"Model: `{report['model_id']}`; precision: "
        f"`{report['precision']}`.",
        "",
        "Epoch 2 is the registered warm observation. Percentages below are "
        "individual observations against epoch wall time and are not additive.",
        "",
        "| Pipeline | img/s | vs Native B6 | prep % | exposed wait % | audit % | boundary sync % | model-only img/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    native_ips = float(warm["native_b6"]["images_per_second"])
    display_order = (
        "native_b6",
        "rgbnomore_dct",
        *STANDARD_PIPELINES,
    )
    for pipeline in (name for name in display_order if name in warm):
        row = warm[pipeline]
        lines.append(
            f"| {pipeline} | {float(row['images_per_second']):.2f} | "
            f"{float(row['images_per_second']) / native_ips:.3f}x | "
            f"{float(row['data_preparation_percent_of_epoch']):.2f} | "
            f"{float(row['exposed_input_wait_percent_of_epoch']):.2f} | "
            f"{float(row['audit_percent_of_epoch']):.2f} | "
            f"{float(row['boundary_sync_percent_of_epoch']):.2f} | "
            f"{float(row['model_only_images_per_second']):.2f} |"
        )
    native = warm["native_b6"]
    lines.extend(
        [
            "",
            "## Native B6 storage breakdown",
            "",
            f"- Selected vectors / full vectors: `{native.get('native_selected_vector_fraction')}`.",
            f"- Compressed bytes read / full compressed bytes: `{native.get('native_compressed_read_fraction')}`.",
            "",
            "## Interpretation boundary",
            "",
            "Native B6 and RGB-no-more DCT are the direct recipe-matched DCT system "
            "comparison. D2 and PyTorch are the direct RGB comparison. D3 is a "
            "DALI-native performance ceiling. DCT versus RGB paths is a system-level "
            "comparison because input domain, crop/order, RandAugment, and Mixup differ.",
        ]
    )
    (output_dir / "equal_image_performance.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-root", type=Path)
    parser.add_argument("--d2-root", type=Path)
    parser.add_argument("--d3-root", type=Path)
    parser.add_argument("--pytorch-root", type=Path)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument(
        "--rgbnomore-run",
        type=Path,
        help="standard-rgbnomore-dct B6 run stopped after epoch 2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--standard-pipelines",
        default=",".join(STANDARD_PIPELINES),
        help="comma-separated completed RGB pipelines to include",
    )
    parser.add_argument(
        "--requested-claim",
        choices=("system-level", "pipeline-only", "recipe-matched"),
        default="system-level",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    standard_pipelines = tuple(
        item.strip() for item in args.standard_pipelines.split(",") if item.strip()
    )
    explicit_roots = {
        pipeline: root
        for pipeline, root in {
            "d2": args.d2_root,
            "d3": args.d3_root,
            "pytorch": args.pytorch_root,
        }.items()
        if root is not None
    }
    report, rows = build_report(
        standard_root=args.standard_root,
        standard_roots=explicit_roots or None,
        native_run=args.native_run,
        rgbnomore_run=args.rgbnomore_run,
        standard_pipelines=standard_pipelines,
        requested_claim=args.requested_claim,
    )
    _write_outputs(args.output_dir.resolve(), report, rows)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "gpu": report["hardware"],
                "warm_epoch": [row for row in rows if int(row["epoch"]) == 2],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
