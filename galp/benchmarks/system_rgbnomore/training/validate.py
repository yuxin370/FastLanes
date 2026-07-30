#!/usr/bin/env python3
"""Validate training artifacts with independent status dimensions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from training.artifacts import read_json, sha256_file, sha256_json, verify_artifact_hashes, write_json
from training.schema import (
    COMPARISON_GROUPS,
    PIPELINES,
    STATUS_DIMENSIONS,
    STATUS_VALUES,
    TRAINING_CONTRACT_SCHEMA,
    TRAINING_PIPELINE_SCHEMA,
    TRAINING_RESULT_SCHEMA,
    TRAINING_VALIDATION_SCHEMA,
    empty_status,
    validate_training_document,
)


BASE_ARTIFACTS = (
    "contract.json",
    "results.json",
    "commands.json",
    "run_metadata.json",
    "artifact_hashes.json",
    "sample_order.json",
    "augmentation_contract.json",
    "semantic_comparison.json",
    "training_curves.json",
    "initial_states.json",
    "repeat_resets.json",
)


def _load(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except Exception as error:
        failures.append(f"cannot read {path.name}: {type(error).__name__}: {error}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"{path.name} is not a JSON object")
        return {}
    return payload


def _status_document(value: Any, where: str, failures: list[str]) -> None:
    if not isinstance(value, dict):
        failures.append(f"{where} status is not an object")
        return
    for dimension in STATUS_DIMENSIONS:
        if value.get(dimension) not in STATUS_VALUES:
            failures.append(f"{where}.{dimension} has invalid status {value.get(dimension)!r}")


def _all_finite(values: Iterable[Any]) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _validate_repeat(
    pipeline: str,
    phase: str,
    repeat: dict[str, Any],
    contract: dict[str, Any],
    failures: list[str],
) -> None:
    prefix = f"{pipeline}.{phase}.repeat{repeat.get('repeat')}"
    expected = contract["execution"]["phases"][phase]
    mode = contract["execution"].get("mode", "audit")
    if repeat.get("execution_mode", "audit") != mode:
        failures.append(f"{prefix} execution mode mismatch")
    if repeat.get("warmup_steps") != expected["warmup_steps"]:
        failures.append(f"{prefix} warmup step count mismatch")
    if repeat.get("measured_steps") != expected["measured_steps"]:
        failures.append(f"{prefix} measured step count mismatch")
    if repeat.get("optimizer_steps") != repeat.get("warmup_steps", 0) + repeat.get("measured_steps", 0):
        failures.append(f"{prefix} did not execute one optimizer step per train step")
    if len(repeat.get("measured_losses", [])) != repeat.get("measured_steps"):
        failures.append(f"{prefix} loss trace length does not match measured steps")
    if not _all_finite(repeat.get("warmup_losses", [])) or not _all_finite(repeat.get("measured_losses", [])):
        failures.append(f"{prefix} contains non-finite loss")
    reset = repeat.get("reset", {})
    for field in (
        "matches_initial_model",
        "matches_initial_optimizer",
        "matches_initial_scheduler",
        "matches_initial_rng",
    ):
        if reset.get(field) is not True:
            failures.append(f"{prefix} reset gate failed: {field}")
    order = repeat.get("sample_order", {})
    order_validation = order.get("validation", {})
    if order_validation.get("ok") is not True:
        failures.append(f"{prefix} optimizer-consumed sample order gate failed")
    if order.get("prefetch_overrun", 0) < 0:
        failures.append(f"{prefix} prefetch overrun is negative")
    if repeat.get("model_parameters_finite") is not True:
        failures.append(f"{prefix} model parameters are not finite")
    if repeat.get("optimizer_state_finite") is not True:
        failures.append(f"{prefix} optimizer state is not finite")
    if repeat.get("scheduler_progress_correct") is not True:
        failures.append(f"{prefix} scheduler progression gate failed")
    instrumentation = repeat.get("measurement_instrumentation", {})
    synchronization = repeat.get("synchronization_accounting", {})
    if mode == "runtime":
        if instrumentation.get("deep_gradient_scans_in_measured_path") != 0:
            failures.append(f"{prefix} runtime measured path performed deep gradient scans")
        if instrumentation.get("full_parameter_snapshots_in_measured_path") != 0:
            failures.append(f"{prefix} runtime measured path performed full parameter snapshots")
        if instrumentation.get("first_step_probe_in_timing") is not False:
            failures.append(f"{prefix} runtime timing includes the first-step probe")
        explicit_syncs = synchronization.get("explicit_host_device", {})
        expected_sync_count = 2 if str(contract["model"]["device"]).startswith("cuda") else 0
        if explicit_syncs.get("count") != expected_sync_count:
            failures.append(
                f"{prefix} runtime explicit synchronization count is "
                f"{explicit_syncs.get('count')}, expected {expected_sync_count}"
            )
    if repeat.get("failures"):
        failures.extend(f"{prefix}: {value}" for value in repeat["failures"])


def _validate_pipeline(
    pipeline: str,
    payload: dict[str, Any],
    contract: dict[str, Any],
    output_dir: Path,
    failures: list[str],
    warnings: list[str],
) -> None:
    failures.extend(
        f"pipeline_{pipeline}.json: {error}"
        for error in validate_training_document(payload, TRAINING_PIPELINE_SCHEMA)
    )
    if payload.get("pipeline") != pipeline:
        failures.append(f"pipeline_{pipeline}.json names pipeline {payload.get('pipeline')!r}")
    if "mode" in contract.get("execution", {}) and "execution_mode" not in payload:
        failures.append(f"{pipeline} artifact is missing execution mode")
    if payload.get("execution_mode", "audit") != contract["execution"].get("mode", "audit"):
        failures.append(f"{pipeline} execution mode differs from contract")
    expected_domain = "dct" if pipeline in ("galp", "rgbnomore") else "rgb"
    if payload.get("domain") != expected_domain:
        failures.append(f"{pipeline} domain mismatch")
    probe = payload.get("first_step_semantic_probe", {})
    if not probe.get("fresh_clone") or not probe.get("before_warmup"):
        failures.append(f"{pipeline} first-step probe was not a pre-warmup fresh clone")
    if probe.get("formal_repeat_polluted") is not False:
        failures.append(f"{pipeline} first-step probe pollution gate failed")
    if probe.get("probe_policy") != "audit" or probe.get("execution_mode_independent") is not True:
        failures.append(f"{pipeline} first-step probe is not execution-mode independent")
    if probe.get("failures"):
        failures.extend(f"{pipeline}.first_step: {value}" for value in probe["failures"])
    raw_artifact = payload.get("first_step_raw_artifact", {})
    raw_path = Path(raw_artifact.get("path", ""))
    if not raw_path.is_file():
        fallback = output_dir / raw_path.name
        raw_path = fallback if fallback.is_file() else raw_path
    if not raw_path.is_file() or sha256_file(raw_path) != raw_artifact.get("sha256"):
        failures.append(f"{pipeline} first-step raw artifact hash mismatch")
    progress_path = output_dir / f"pipeline_progress_{pipeline}.json"
    if not progress_path.is_file():
        failures.append(f"{pipeline} pipeline progress artifact is missing")
    else:
        progress = _load(progress_path, failures)
        if progress.get("pipeline") != pipeline:
            failures.append(f"{pipeline} pipeline progress identity mismatch")
        if progress.get("contract_sha256") != contract.get("contract_sha256"):
            failures.append(f"{pipeline} pipeline progress contract mismatch")
        if progress.get("execution_mode", "audit") != contract["execution"].get("mode", "audit"):
            failures.append(f"{pipeline} pipeline progress execution-mode mismatch")
        if progress.get("status") != "complete":
            failures.append(f"{pipeline} pipeline progress is not complete")

    phases = payload.get("phase_results", {})
    for phase in ("smoke", "step"):
        phase_contract = contract["execution"]["phases"][phase]
        if not phase_contract["enabled"]:
            if phase in phases:
                warnings.append(f"{pipeline} contains unrequested {phase} results")
            continue
        repeats = phases.get(phase, {}).get("repeats", [])
        if len(repeats) != phase_contract["repeats"]:
            failures.append(f"{pipeline}.{phase} repeat count mismatch")
        if not repeats or repeats[0].get("repeat") != 0:
            failures.append(f"{pipeline}.{phase} does not preserve repeat 0")
        for repeat in repeats:
            _validate_repeat(pipeline, phase, repeat, contract, failures)
        if phase == "step" and repeats:
            aggregate = phases[phase].get("aggregate", {})
            expected_included = [value for value in (1, 2, 3, 4) if value < len(repeats)]
            if aggregate.get("included_repeats") != expected_included:
                failures.append(f"{pipeline}.step aggregate must use repeats 1-4 only")
            cv = aggregate.get("throughput_cv")
            limit = contract["gates"]["throughput_cv_limit"]
            expected_status = "passed" if cv is not None and cv <= limit else "failed"
            if aggregate.get("performance_status") != expected_status:
                failures.append(f"{pipeline}.step performance status does not match throughput CV gate")
            if expected_status == "failed":
                failures.append(f"{pipeline}.step throughput CV performance gate failed")
            # Performance is independent: a CV failure must not rewrite a
            # numerically clean correctness result.
            if expected_status == "failed" and not any(repeat.get("failures") for repeat in repeats):
                if payload.get("status", {}).get("correctness") == "failed":
                    failures.append(f"{pipeline} performance failure incorrectly overwrote correctness")

    convergence_contract = contract["execution"]["phases"]["convergence"]
    if convergence_contract["enabled"]:
        seed_results = phases.get("convergence", {}).get("seeds", [])
        actual_seeds = [result.get("seed") for result in seed_results]
        if actual_seeds != convergence_contract["seeds"]:
            failures.append(f"{pipeline}.convergence seed set mismatch")
        for result in seed_results:
            if result.get("optimizer_steps") != convergence_contract["train_steps"]:
                failures.append(f"{pipeline}.convergence seed {result.get('seed')} step count mismatch")
            if result.get("model_parameters_finite") is not True:
                failures.append(f"{pipeline}.convergence seed {result.get('seed')} model is not finite")
            if result.get("optimizer_state_finite") is not True:
                failures.append(f"{pipeline}.convergence seed {result.get('seed')} optimizer is not finite")
            if result.get("scheduler_progress_correct") is not True:
                failures.append(f"{pipeline}.convergence seed {result.get('seed')} scheduler gate failed")
            classification = result.get("classification")
            expected_classification = {
                "random": "from_scratch_short_convergence",
                "weights": "fine_tuning",
                "full-checkpoint": "resumed_training",
            }[contract["initialization"]["mode"]]
            if classification != expected_classification:
                failures.append(f"{pipeline}.convergence classification mismatch")
            if result.get("failures"):
                failures.extend(
                    f"{pipeline}.convergence.seed{result.get('seed')}: {value}"
                    for value in result["failures"]
                )
            for checkpoint_name in ("final_checkpoint", "best_checkpoint"):
                checkpoint = result.get(checkpoint_name)
                if not checkpoint:
                    failures.append(f"{pipeline}.convergence missing {checkpoint_name}")
                    continue
                path = Path(checkpoint["path"])
                if not path.is_file() or sha256_file(path) != checkpoint.get("sha256"):
                    failures.append(f"{pipeline}.convergence {checkpoint_name} hash mismatch")
    _status_document(payload.get("status"), pipeline, failures)


def validate_output(output_dir: Path, *, write_result: bool = True) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    missing = [name for name in BASE_ARTIFACTS if not (output_dir / name).is_file()]
    checks["base_artifacts"] = {"ok": not missing, "missing": missing}
    failures.extend(f"missing required artifact: {name}" for name in missing)

    contract = _load(output_dir / "contract.json", failures)
    results = _load(output_dir / "results.json", failures)
    commands = _load(output_dir / "commands.json", failures)
    failures.extend(f"contract.json: {error}" for error in validate_training_document(contract, TRAINING_CONTRACT_SCHEMA))
    failures.extend(f"results.json: {error}" for error in validate_training_document(results, TRAINING_RESULT_SCHEMA))
    if contract:
        mode = contract.get("execution", {}).get("mode")
        if mode not in ("audit", "runtime"):
            failures.append(f"contract execution mode is invalid: {mode!r}")
        if "mode" in contract.get("execution", {}) and "execution_mode" not in results:
            failures.append("results are missing execution mode")
        if results.get("execution_mode", "audit") != mode:
            failures.append("results execution mode differs from contract")
        if "mode" in contract.get("execution", {}) and "execution_mode" not in commands:
            failures.append("commands are missing execution mode")
        if commands.get("execution_mode", "audit") != mode:
            failures.append("commands execution mode differs from contract")
        contract_copy = {key: value for key, value in contract.items() if key != "contract_sha256"}
        actual_contract_hash = sha256_json(contract_copy)
        if contract.get("contract_sha256") != actual_contract_hash:
            failures.append("contract hash cannot be recomputed")
        if results.get("contract_sha256") != contract.get("contract_sha256"):
            failures.append("results reference a different contract hash")
        separation = contract.get("datasets", {}).get("separation", {})
        if separation.get("ok") is not True:
            failures.append("train/validation dataset contamination gate failed")
        augmentation = contract.get("augmentation", {})
        if any((augmentation.get("mixup"), augmentation.get("cutmix"), augmentation.get("randaugment"))):
            failures.append("v1 contract unexpectedly enables mixup/cutmix/randaugment")
        if augmentation.get("worker_rng_used") is not False:
            failures.append("augmentation contract depends on worker RNG")
        required_domains = {
            "dct" if pipeline in ("galp", "rgbnomore") else "rgb"
            for pipeline in contract.get("enabled_pipelines", [])
        }
        groups_by_domain = contract.get("optimizer", {}).get("parameter_groups_by_domain", {})
        for domain in sorted(required_domains):
            groups = groups_by_domain.get(domain)
            if not isinstance(groups, list) or [group.get("stable_id") for group in groups] != [
                "decay",
                "no_decay",
            ]:
                failures.append(f"{domain} optimizer parameter groups are missing or unstable")
                continue
            actual_count = sum(int(group.get("parameter_count", -1)) for group in groups)
            expected_count = int(
                contract.get("model", {})
                .get("domains", {})
                .get(domain, {})
                .get("expected_trainable_parameters", -2)
            )
            if actual_count != expected_count:
                failures.append(
                    f"{domain} optimizer parameter groups cover {actual_count} parameters, expected {expected_count}"
                )

    enabled = list(contract.get("enabled_pipelines", []))
    required_groups = list(contract.get("required_comparison_groups", []))
    pipeline_payloads: dict[str, dict[str, Any]] = {}
    for pipeline in PIPELINES:
        path = output_dir / f"pipeline_{pipeline}.json"
        if pipeline in enabled:
            if not path.is_file():
                failures.append(f"enabled pipeline lacks artifact: {pipeline}")
                continue
            payload = _load(path, failures)
            pipeline_payloads[pipeline] = payload
            _validate_pipeline(pipeline, payload, contract, output_dir, failures, warnings)
        elif path.exists():
            failures.append(f"disabled pipeline has fabricated artifact: {pipeline}")

    for group in required_groups:
        missing_group = sorted(set(COMPARISON_GROUPS[group]) - set(enabled))
        if missing_group:
            failures.append(f"required comparison group {group} is incomplete: {missing_group}")
    semantic = _load(output_dir / "semantic_comparison.json", failures)
    for group, pair in COMPARISON_GROUPS.items():
        both = all(pipeline in enabled for pipeline in pair)
        status = semantic.get(group, {}).get("status")
        if both and status not in ("passed", "warning", "failed"):
            failures.append(f"enabled same-domain pair {group} lacks semantic gate")
        if both and status == "failed":
            failures.extend(
                f"{group} semantic gate: {value}"
                for value in semantic[group].get("failures", [])
            )
        if both and status == "warning":
            warnings.extend(
                f"{group} semantic gate: {value}"
                for value in semantic[group].get("warnings", [])
            )
        if not both and status != "not_run":
            failures.append(f"incomplete comparison group {group} must be not_run")

    initial_states = _load(output_dir / "initial_states.json", failures)
    records = initial_states.get("states", [])
    for domain, pair in COMPARISON_GROUPS.items():
        if all(pipeline in enabled for pipeline in pair):
            for field, label in (
                ("initial_model_state_sha256", "model"),
                ("initial_optimizer_state_sha256", "optimizer"),
                ("initial_scheduler_state_sha256", "scheduler"),
            ):
                hashes = {
                    payload.get("first_step_semantic_probe", {}).get(field)
                    for pipeline, payload in pipeline_payloads.items()
                    if pipeline in pair
                }
                if len(hashes) != 1 or None in hashes:
                    failures.append(f"{domain} pair did not share one initial {label} state")
    for record in records:
        artifact = record.get("artifact", {})
        path = Path(artifact.get("path", ""))
        if not path.is_file():
            fallback = output_dir / path.name
            path = fallback if fallback.is_file() else path
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            failures.append(f"initial state artifact hash mismatch for {record.get('domain')} seed {record.get('seed')}")

    artifact_hashes_path = output_dir / "artifact_hashes.json"
    if artifact_hashes_path.is_file():
        hash_payload = _load(artifact_hashes_path, failures)
        hash_errors = verify_artifact_hashes(output_dir, hash_payload)
        failures.extend(hash_errors)
        checks["artifact_hashes"] = {"ok": not hash_errors, "errors": hash_errors}

    pipeline_status = {
        pipeline: (
            dict(pipeline_payloads[pipeline].get("status", empty_status("failed")))
            if pipeline in enabled
            else empty_status("not_run")
        )
        for pipeline in PIPELINES
    }
    artifact_integrity_failed = bool(missing) or any(
        marker in failure.lower()
        for failure in failures
        for marker in (
            "artifact hash",
            "artifact missing",
            "artifact integrity",
            "hash mismatch",
            "cannot read",
            "cannot be recomputed",
            "lacks artifact",
            "fabricated artifact",
            "missing required artifact",
            "pipeline progress",
        )
    )
    if artifact_integrity_failed:
        for pipeline in enabled:
            pipeline_status[pipeline]["artifact"] = "failed"
            pipeline_status[pipeline]["overall"] = "failed"
    comparison_status: dict[str, dict[str, str]] = {}
    for group, pair in COMPARISON_GROUPS.items():
        if all(pipeline in enabled for pipeline in pair):
            comparison_status[group] = empty_status()
            comparison_status[group]["semantic"] = semantic.get(group, {}).get("status", "failed")
            comparison_status[group]["correctness"] = (
                "failed" if any(pipeline_status[name]["correctness"] == "failed" for name in pair) else "passed"
            )
            comparison_status[group]["artifact"] = "passed"
            if artifact_integrity_failed:
                comparison_status[group]["artifact"] = "failed"
            comparison_status[group]["overall"] = (
                "failed"
                if "failed" in comparison_status[group].values()
                else ("warning" if "warning" in comparison_status[group].values() else "passed")
            )
        else:
            comparison_status[group] = empty_status("not_run")

    declared_pipeline_status = results.get("pipeline_status", {})
    for pipeline in PIPELINES:
        if pipeline not in declared_pipeline_status:
            failures.append(f"results.json lacks declared status for {pipeline}")
            continue
        expected_declared = (
            pipeline_payloads[pipeline].get("status")
            if pipeline in enabled and pipeline in pipeline_payloads
            else empty_status("not_run")
        )
        if declared_pipeline_status[pipeline] != expected_declared:
            failures.append(f"results.json status disagrees with pipeline artifact for {pipeline}")
    declared_group_status = results.get("comparison_group_status", {})
    for group in COMPARISON_GROUPS:
        if group not in declared_group_status:
            failures.append(f"results.json lacks declared comparison status for {group}")
    checks["disabled_pipeline_not_run"] = {
        "ok": all(
            declared_pipeline_status.get(name, {}).get("overall") == "not_run"
            for name in PIPELINES
            if name not in enabled
        )
    }
    if not checks["disabled_pipeline_not_run"]["ok"]:
        failures.append("disabled pipeline status is not not_run")
    payload = {
        "schema_version": TRAINING_VALIDATION_SCHEMA,
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
        "pipeline_status": pipeline_status,
        "comparison_group_status": comparison_status,
    }
    if write_result:
        write_json(output_dir / "validation.json", payload)
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = validate_output(args.output_dir, write_result=not args.no_write)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
